#ifndef RDMA_DEVICE_H_
#define RDMA_DEVICE_H_

#include "rdma_common.h"
#include "rdma_thread.h"
#include "spinlock.h"

#define MIN_CQ_LENGTH 32
#define TX_DEPTH 32
#define RX_DEPTH 32
#define MAX_SEND_SGE 16
#define MAX_RECV_SGE 16
#define CQ_POLL_BATCH 16

#define RECV_BUFFER_SIZE (1024 * 1024)

#define MEM_REGION_ALIGNMENT 4096

namespace wolong {

extern void dump_device_attr(struct ibv_context *context);

class RDMADevice;

typedef enum { LOCAL , REMOTE } PrintDataSide;

/* The type of the device */
enum ctx_device {
	DEVICE_ERROR		= -1,
	UNKNOWN			= 0,
	CONNECTX 		= 1,
	CONNECTX2 		= 2,
	CONNECTX3 		= 3,
	CONNECTIB 		= 4,
	LEGACY 			= 5,
	CHELSIO_T4 		= 6,
	CHELSIO_T5 		= 7,
	CONNECTX3_PRO		= 8,
	SKYHAWK			= 9,
	CONNECTX4		= 10,
	CONNECTX4LX		= 11
};

#define MEMCPY_LOCAL_TO_REMOTE 0
#define MEMCPY_REMOTE_TO_LOCAL 1

typedef void (*memcpy_callback_t) (void *arg, enum ibv_wc_status status);
typedef memcpy_callback_t send_callback_t;
typedef void (*rpc_func_t) (void *buff, size_t size, void *arg);

struct rpc_func_entry_t {
	rpc_func_t func;
	void *arg;
};

struct memcpy_req_entry {
	struct ibv_sge sge;
	struct ibv_send_wr swr;
	memcpy_callback_t cb;
	void *arg;
};

struct recv_req_entry {
	struct ibv_sge sge;
	struct ibv_recv_wr rwr;
	struct ibv_mr *mr;
	struct ibv_qp* qp;
};

class RDMAChannel {
 public:
	RDMAChannel(struct ibv_qp * qp) : qp_(qp) {}
	~RDMAChannel() {}

	void Memcpy(
		void *local_addr, 
		struct ibv_mr *local_region, 
		void *remote_addr, 
		unsigned remote_key, 
		size_t size,
		int direction,
		memcpy_callback_t cb,
		void *arg
		);

	void Send(
		void *local_addr, 
		struct ibv_mr *local_region, 
		size_t size,
		send_callback_t cb,
		void *arg
		);

 private:
	struct ibv_qp *qp_;
};

class SenderPeer {
 public:
	SenderPeer(RDMADevice *dev, int num_qps, const std::string &host, int port);

	~SenderPeer() {}

	struct ibv_qp *GetChannel(int thread_id = 0);
	struct ibv_qp *GetChannelWithIdx(int channel_idx) {
		struct ibv_qp *qp = qps_[channel_idx];
		return qp;
	}
	
 private:
	std::atomic<uint64_t> cur_qp_id_;
	RDMADevice *rdma_dev_;
	struct rdma_event_channel *cm_channel_;
	std::vector<struct rdma_cm_id *> cm_ids_;
	std::vector<struct ibv_qp *> qps_;
	std::string peer_host_;
	int peer_port_;
};

class RDMADevice {
 public:
	RDMADevice(
		struct ibv_device *ibdev, 
		int dev_idx, 
		const std::string& dev_name, 
		int conn_type,
		int num_cqs,
		int num_qps_per_peer,
		std::string &host,
		int port) :
		ib_dev_(ibdev), dev_idx_(dev_idx), gid_index_(-1), ib_port_(1), dlid_(0), port_(port),
		inline_size_(DEF_INLINE), out_reads_(0), pkey_index_(0),
		connection_type_(conn_type), num_cqs_(num_cqs), num_qps_per_peer_(num_qps_per_peer),
		cur_cq_idx_(0), pd_(nullptr)
	{
		spinlock_init(&dev_lock_);
		cqs_.clear();
		comp_channels_.clear();
		srv_cm_ids_.clear();
		server_qps_.clear();
		peers_.clear();

		rpc_funcs_.resize(1024);
		for (size_t i = 0; i < rpc_funcs_.size(); i++) {
			rpc_funcs_[i].func = nullptr;
		}

		my_host_ = host;
		side_ = LOCAL;
		name_ = dev_name;
		context_ = ibv_open_device(ib_dev_);
		if (context_ == nullptr) {
			fprintf(stderr, "Couldn't get context for the device %s\n", name_.c_str());
			return;
		}

		fprintf(stderr, "Device:%s\n", name_.c_str());
		dump_device_attr(context_);

		if (!CheckLink()) {
			fprintf(stderr, " Couldn't get link info for the device %s\n", name_.c_str());
			context_ = nullptr;
			return;
		}

		if (!CreateRdmaResources()) {
			fprintf(stderr," Unable to create the needed resources\n");
			context_ = nullptr;
			return;
		}

		struct addrinfo hints;
		char *service;
		struct sockaddr_in sin;
		struct addrinfo *res;

		memset(&hints, 0, sizeof hints);
		hints.ai_flags = AI_PASSIVE;
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;

		int number;
		if (asprintf(&service,"%d", port_) < 0) {
			fprintf(stderr, "Failed to allocate service string\n");
			abort();
		}

		number = getaddrinfo(my_host_.c_str(), service, &hints, &res);
		if (number < 0) {
			fprintf(stderr, "%s for %s:%d\n", gai_strerror(number), my_host_.c_str(), port_);
			abort();
		}
		free(service);
		freeaddrinfo(res);

		sin.sin_addr.s_addr = ((struct sockaddr_in*)res->ai_addr)->sin_addr.s_addr;
		sin.sin_family = PF_INET;
		sin.sin_port = htons((unsigned short)port_);

		if (rdma_bind_addr(control_cm_id_, (struct sockaddr *)&sin)) {
			fprintf(stderr," rdma_bind_addr failed\n");
			abort();
		}

		context_ = control_cm_id_->verbs;
		pd_ = control_cm_id_->pd;

		// Create cqs and completion channels
		for (int i = 0; i < num_cqs_; i++) {
			struct ibv_comp_channel *comp_channel = ibv_create_comp_channel(context_);
			if (!comp_channel) {
				fprintf(stderr, "Couldn't create completion channel\n");
				context_ = nullptr;
				return;
			}
			comp_channels_.push_back(comp_channel);

			struct ibv_cq *cq = ibv_create_cq(context_, MIN_CQ_LENGTH, nullptr, comp_channel, 0);
			if (!cq) {
				fprintf(stderr, "[%s:%d] Failed to create CQ\n", __FILE__, __LINE__);
				abort();
			}
			cqs_.push_back(cq);

			if (ibv_req_notify_cq(cq, 0)) {
				fprintf(stderr, "[%s:%d]ibv_req_notify_cq failed\n", __FILE__, __LINE__);
				abort();
			}
		}

		CreateConnServThread();
		
		cq_polling_threads_.resize(num_cqs_);
		for (int i = 0; i < num_cqs_; i++) {
			CreateCQPollingThread(i);	
		}
	}

	~RDMADevice() {}

	RDMAChannel *GetRDMAChannel(const std::string &servername, int server_port, int thread_id = 0);	
	RDMAChannel *GetRDMAChannelWithIdx(const std::string &servername, int server_port, int idx);	

	struct ibv_mr *AllocateMemRegion(size_t size);
	struct ibv_mr *RegisterMemRegion(void *buf, size_t size);

	void DeallocateMemRegion(struct ibv_mr *mr) {
		void *addr = mr->addr;
		spinlock_acquire(&dev_lock_);
		if (ibv_dereg_mr(mr)) {
			fprintf(stderr, "Failed to deregister memory region\n");
			abort();
		}
		spinlock_release(&dev_lock_);

		if (addr) free(addr);
	}

	// Not thread-safe
	void RegisterRPCFunc(uint32_t cmd, rpc_func_t func, void *arg);

 private:
	friend class RDMADeviceManager;
	friend class SenderPeer;

	bool CheckLink();
	uint8_t SetLinkLayer();
	void SetMaxInline();
	bool CreateRdmaResources();
	void CreateConnServThread();
	void CreateCQPollingThread(int idx);
	void RecvHandler(struct recv_req_entry *recve);
	
	spinlock_t dev_lock_;

	struct ibv_device *ib_dev_;
	int dev_idx_;
	struct ibv_context *context_;
	uint8_t ib_port_;
	enum ibv_transport_type transport_type_;
	uint8_t link_type_;
	int gid_index_;
	int inline_size_;
	int out_reads_;
	int pkey_index_;
	PrintDataSide side_;
	uint16_t dlid_;
	int connection_type_;
	int port_;
	int sockfd_;
	struct ibv_pd *pd_;
	int num_cqs_;
	std::atomic<uint64_t> cur_cq_idx_;
	int num_qps_per_peer_;
	std::unordered_map<std::string, SenderPeer *> peers_;

	std::vector<struct ibv_comp_channel *> comp_channels_;
	std::vector<struct ibv_cq *> cqs_;
	std::vector<struct ibv_qp *> server_qps_;
	std::vector<struct rdma_cm_id *> srv_cm_ids_;

	Thread *conn_serv_thread_;
	std::vector<Thread *> cq_polling_threads_;

	std::string my_host_;

	struct rdma_cm_id *control_cm_id_;
	struct rdma_event_channel *control_cm_channel_;

	std::vector<rpc_func_entry_t> rpc_funcs_;

	std::string name_;
};

}

#endif // RDMA_DEVICE_H_
