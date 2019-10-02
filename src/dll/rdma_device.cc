#include "rdma_device.h"

namespace wolong {

static const char *portStates[] = {"Nop","Down","Init","Armed","","Active Defer"};

enum ctx_device ib_dev_name(struct ibv_context *context)
{
	enum ctx_device dev_fname = UNKNOWN;
	struct ibv_device_attr attr;

	if (ibv_query_device(context,&attr)) {
		dev_fname = DEVICE_ERROR;
	}

	else if (attr.vendor_id == 5157) {

		switch (attr.vendor_part_id >> 12) {
			case 10 :
			case 4  : dev_fname = CHELSIO_T4; break;
			case 11 :
			case 5  : dev_fname = CHELSIO_T5; break;
			default : dev_fname = UNKNOWN; break;
		}

		/* Assuming it's Mellanox HCA or unknown.
		If you want Inline support in other vendor devices, please send patch to gilr@dev.mellanox.co.il
		*/
	} else {

		switch (attr.vendor_part_id) {
			case 4115  : dev_fname = CONNECTX4; break;
			case 4116  : dev_fname = CONNECTX4; break;
			case 4117  : dev_fname = CONNECTX4LX; break;
			case 4118  : dev_fname = CONNECTX4LX; break;
			case 4113  : dev_fname = CONNECTIB; break;
			case 4099  : dev_fname = CONNECTX3; break;
			case 4100  : dev_fname = CONNECTX3; break;
			case 4103  : dev_fname = CONNECTX3_PRO; break;
			case 4104  : dev_fname = CONNECTX3_PRO; break;
			case 26418 : dev_fname = CONNECTX2; break;
			case 26428 : dev_fname = CONNECTX2; break;
			case 26438 : dev_fname = CONNECTX2; break;
			case 26448 : dev_fname = CONNECTX2; break;
			case 26458 : dev_fname = CONNECTX2; break;
			case 26468 : dev_fname = CONNECTX2; break;
			case 26478 : dev_fname = CONNECTX2; break;
			case 25408 : dev_fname = CONNECTX;  break;
			case 25418 : dev_fname = CONNECTX;  break;
			case 25428 : dev_fname = CONNECTX;  break;
			case 25448 : dev_fname = CONNECTX;  break;
			case 1824  : dev_fname = SKYHAWK;  break;
			default	   : dev_fname = UNKNOWN;
		}
	}

	return dev_fname;
}

void dump_device_attr(struct ibv_context *context) {
	struct ibv_device_attr attr;

	if (ibv_query_device(context,&attr)) {
		fprintf(stderr, "[%s:%d] Failed to query device attributes\n", __FILE__, __LINE__);
		abort();
	}

	fprintf(stderr, "  max_mr: %d\n", attr.max_mr);
	fprintf(stderr, "  max_mr_size: %" PRIu64 "\n", attr.max_mr_size);
}

static int ctx_set_out_reads(struct ibv_context *context,int num_user_reads)
{
	int max_reads = 0;
	struct ibv_device_attr attr;

	if (!ibv_query_device(context,&attr)) {
		max_reads = attr.max_qp_rd_atom;
	}

	if (num_user_reads > max_reads) {
		fprintf(stderr," Number of outstanding reads is above max = %d\n",max_reads);
		fprintf(stderr," Changing to that max value\n");
		num_user_reads = max_reads;
	}
	else if (num_user_reads <= 0) {
		num_user_reads = max_reads;
	}

	return num_user_reads;
}

const char *LinkLayerStr(uint8_t link_layer) {
	switch (link_layer) {
		case IBV_LINK_LAYER_UNSPECIFIED:
		case IBV_LINK_LAYER_INFINIBAND:
			return "IB";
		case IBV_LINK_LAYER_ETHERNET:
			return "Ethernet";
		default:
			return "Unknown";
	}
}

////////////////////////////////////////////////////////////////////////////

void RDMAChannel::Memcpy(
	void *local_addr,
	struct ibv_mr *local_region,
	void *remote_addr,
	unsigned remote_key,
	size_t size,
	int direction,
	memcpy_callback_t cb,
	void *arg
	) 
{
	struct memcpy_req_entry *mcpe = (struct memcpy_req_entry *)malloc(sizeof(struct memcpy_req_entry));

	mcpe->sge.addr = (uint64_t)local_addr;
	mcpe->sge.length = size;
	mcpe->sge.lkey = local_region->lkey;

	mcpe->swr.wr_id = (uint64_t)mcpe;
	mcpe->swr.next = nullptr;
	mcpe->swr.sg_list = &mcpe->sge;
	mcpe->swr.num_sge = 1;
	if (direction == MEMCPY_LOCAL_TO_REMOTE)
		mcpe->swr.opcode = IBV_WR_RDMA_WRITE;
	else if (direction == MEMCPY_REMOTE_TO_LOCAL)
		mcpe->swr.opcode = IBV_WR_RDMA_READ;
	else {
		fprintf(stderr, "Unknown rdma memcpy direction\n");
		abort();
	}
	mcpe->swr.send_flags = IBV_SEND_SIGNALED;
	mcpe->swr.imm_data = 0;
	mcpe->swr.wr.rdma.remote_addr = (uint64_t)remote_addr;
	mcpe->swr.wr.rdma.rkey = remote_key;
	mcpe->swr.xrc_remote_srq_num = 0;

	mcpe->cb = cb;
	mcpe->arg = arg;

	struct ibv_send_wr *bad_wr;
	int ret;

	ret = ibv_post_send(qp_, &mcpe->swr, &bad_wr);
	if (ret) {
		fprintf(stderr, "Failed to post send\n");
		abort();
	}
}


void RDMAChannel::Send(
	void *local_addr,
	struct ibv_mr *local_region,
	size_t size,
	send_callback_t cb,
	void *arg
	) 
{
	struct memcpy_req_entry *mcpe = (struct memcpy_req_entry *)malloc(sizeof(struct memcpy_req_entry));

	mcpe->sge.addr = (uint64_t)local_addr;
	mcpe->sge.length = size;
	mcpe->sge.lkey = local_region->lkey;

	mcpe->swr.wr_id = (uint64_t)mcpe;
	mcpe->swr.next = nullptr;
	mcpe->swr.sg_list = &mcpe->sge;
	mcpe->swr.num_sge = 1;
	mcpe->swr.opcode = IBV_WR_SEND;
	mcpe->swr.send_flags = IBV_SEND_SIGNALED;

	mcpe->cb = cb;
	mcpe->arg = arg;

	struct ibv_send_wr *bad_wr;
	int ret;

	ret = ibv_post_send(qp_, &mcpe->swr, &bad_wr);
	if (ret) {
		fprintf(stderr, "Failed to post send\n");
		abort();
	}
}

////////////////////////////////////////////////////////////////////////////

#define NUM_OF_RETRIES (10)

SenderPeer::SenderPeer(RDMADevice *dev, int num_qps, const std::string &host, int port) {
	peer_host_ = host;
	peer_port_ = port;
	rdma_dev_ = dev;
	cur_qp_id_ = 0;
	qps_.resize(num_qps);
	cm_ids_.resize(num_qps);
	for (size_t i = 0; i < qps_.size(); i++) {
		qps_[i] = nullptr;
		cm_ids_[i] = nullptr;
	}

	char *service;
	int num_of_retry= NUM_OF_RETRIES;
	struct sockaddr_in sin;
	struct addrinfo *res;
	struct rdma_cm_event *event;
	struct rdma_conn_param conn_param;
	struct addrinfo hints;
	int number;

	// create event channal for cm_ids
	cm_channel_ = rdma_create_event_channel();
	if (cm_channel_ == nullptr) {
		fprintf(stderr, " rdma_create_event_channel failed\n");
		abort();
	}

	// resolve peer address
	memset(&hints, 0, sizeof hints);
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;

	if (asprintf(&service,"%d", peer_port_) < 0) {
		fprintf(stderr, "Failed to allocate service string\n");
		abort();
	}

	number = getaddrinfo(peer_host_.c_str(), service, &hints, &res);
	if (number < 0) {
		fprintf(stderr, "%s for %s:%d\n", gai_strerror(number), peer_host_.c_str(), peer_port_);
		abort();
	}
	free(service);
	freeaddrinfo(res);

	sin.sin_addr.s_addr = ((struct sockaddr_in*)res->ai_addr)->sin_addr.s_addr;
	sin.sin_family = PF_INET;
	sin.sin_port = htons((unsigned short)peer_port_);

	int is_udp_ps = rdma_dev_->connection_type_ == UD || rdma_dev_->connection_type_ == RawEth;
	enum rdma_port_space port_space = (is_udp_ps) ? RDMA_PS_UDP : RDMA_PS_TCP;

	// establish a connection for each qp
	for (size_t i = 0; i < qps_.size(); i++) {

		// create cm_id
		struct rdma_cm_id *cm_id;
		if (rdma_create_id(cm_channel_, &cm_id , nullptr, port_space)) {
			fprintf(stderr,"rdma_create_id failed\n");
			abort();
		}

		// bind self	
		number = getaddrinfo(rdma_dev_->my_host_.c_str(), nullptr, &hints, &res);
		if (number < 0) {
			fprintf(stderr, "%s for %s\n", gai_strerror(number), rdma_dev_->my_host_.c_str());
			abort();
		}
		freeaddrinfo(res);

		struct sockaddr_in my_sin;
		my_sin.sin_addr.s_addr = ((struct sockaddr_in*)res->ai_addr)->sin_addr.s_addr;
		my_sin.sin_family = PF_INET;
		my_sin.sin_port = htons((unsigned short)0);

		if (rdma_bind_addr(cm_id,(struct sockaddr *)&my_sin)) {
			fprintf(stderr,"[%s:%d]rdma_bind_addr failed\n", __FILE__, __LINE__);
			abort();
		}

		if (cm_id->pd != rdma_dev_->pd_) {
			fprintf(stderr, "[%s:%d]Wrong pd\n", __FILE__, __LINE__);
		}


		while (1) {
			if (num_of_retry == 0) {
				fprintf(stderr, "Received %d times ADDR_ERROR\n",NUM_OF_RETRIES);
				abort();
			}

			if (rdma_resolve_addr(cm_id, nullptr, (struct sockaddr *)&sin, 2000)) {
				fprintf(stderr, "rdma_resolve_addr failed\n");
				abort();
			}

			if (rdma_get_cm_event(cm_channel_, &event)) {
				fprintf(stderr, "rdma_get_cm_events failed\n");
				abort();
			}

			if (event->event == RDMA_CM_EVENT_ADDR_ERROR) {
				num_of_retry--;
				rdma_ack_cm_event(event);
				continue;
			}

			if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
				fprintf(stderr, "[%d]unexpected CM event %d\n", __LINE__, event->event);
				rdma_ack_cm_event(event);
				abort();
			}

			rdma_ack_cm_event(event);
			break;
		}

		while (1) {
			if (num_of_retry <= 0) {
				fprintf(stderr, "Received %d times ADDR_ERROR - aborting\n",NUM_OF_RETRIES);
				abort();
			}

			if (rdma_resolve_route(cm_id, 2000)) {
				fprintf(stderr, "rdma_resolve_route failed\n");
				abort();
			}

			if (rdma_get_cm_event(cm_channel_, &event)) {
				fprintf(stderr, "rdma_get_cm_events failed\n");
				abort();
			}

			if (event->event == RDMA_CM_EVENT_ROUTE_ERROR) {
				num_of_retry--;
				rdma_ack_cm_event(event);
				continue;
			}

			if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
				fprintf(stderr, "unexpected CM event %d\n",event->event);
				rdma_ack_cm_event(event);
				abort();
			}

			rdma_ack_cm_event(event);
			break;
		}

		struct ibv_cq *cq = rdma_dev_->cqs_[rdma_dev_->cur_cq_idx_ % rdma_dev_->num_cqs_];
		rdma_dev_->cur_cq_idx_++;

		struct ibv_qp_init_attr attr;
		memset(&attr, 0, sizeof(struct ibv_qp_init_attr));

		attr.send_cq = cq;
		attr.recv_cq = cq;
		attr.cap.max_send_wr = TX_DEPTH;
		attr.cap.max_send_sge = MAX_SEND_SGE;
		attr.cap.max_inline_data = rdma_dev_->inline_size_;
		attr.srq = NULL;
		attr.cap.max_recv_wr = RX_DEPTH;
		attr.cap.max_recv_sge = MAX_RECV_SGE;

		switch (rdma_dev_->connection_type_) {
			case RC : attr.qp_type = IBV_QPT_RC; break;
			case UC : attr.qp_type = IBV_QPT_UC; break;
			case UD : attr.qp_type = IBV_QPT_UD; break;
			default:  fprintf(stderr, "Unknown connection type \n");
								abort();
		}

		if (rdma_create_qp(cm_id, rdma_dev_->pd_, &attr)) {
			fprintf(stderr, "[%s:%d]Couldn't create rdma QP - %s\n", __FILE__, __LINE__, strerror(errno));
			abort();
		}

		cm_ids_[i] = cm_id;
		qps_[i] = cm_id->qp;	

		if (rdma_dev_->pd_ != cm_id->qp->pd) {
			fprintf(stderr, "[%s:%d] qps not share the same pd in device %s\n", __FILE__, __LINE__, rdma_dev_->name_.c_str());
			abort();
		}

		memset(&conn_param, 0, sizeof conn_param);
		conn_param.responder_resources = rdma_dev_->out_reads_;
		conn_param.initiator_depth = rdma_dev_->out_reads_;
		conn_param.retry_count = 7;
		conn_param.rnr_retry_count = 7;

		if (rdma_connect(cm_id, &conn_param)) {
			fprintf(stderr, "Function rdma_connect failed\n");
			abort();
		}

		if (rdma_get_cm_event(cm_channel_, &event)) {
			fprintf(stderr, "rdma_get_cm_events failed\n");
			abort();
		}

		if (event->event != RDMA_CM_EVENT_ESTABLISHED) {
			rdma_ack_cm_event(event);
			fprintf(stderr, "Unexpected CM event bl blka %d\n", event->event);
			abort();
		}

		rdma_ack_cm_event(event);
	}
}

struct ibv_qp *SenderPeer::GetChannel(int thread_id) {
	uint64_t cqpid = cur_qp_id_ % qps_.size();
	cur_qp_id_++;
	struct ibv_qp *qp = qps_[cqpid];	

	if (!qp) {
		fprintf(stderr, "Cannot get qp from peer\n");
		abort();
	}

	return qp;	
}

//////////////////////////////////////////////////////////////////////

uint8_t RDMADevice::SetLinkLayer() {
	struct ibv_port_attr port_attr;
	uint8_t curr_link;	

	if (ibv_query_port(context_, ib_port_, &port_attr)) {
		fprintf(stderr," Unable to query port attributes\n");
		return LINK_FAILURE;
	}

	int curr_mtu;
	switch (port_attr.active_mtu) {
		case IBV_MTU_256: curr_mtu = 256; break;
		case IBV_MTU_512: curr_mtu = 512; break;
		case IBV_MTU_1024: curr_mtu = 1024; break;
		case IBV_MTU_2048: curr_mtu = 2048; break;
		case IBV_MTU_4096: curr_mtu = 4096; break;
		default: curr_mtu = 0;
	}

	fprintf(stderr, "[%s:%d] current MTU: %d\n", __FILE__, __LINE__, curr_mtu);

	if (port_attr.state != IBV_PORT_ACTIVE) {
		fprintf(stderr," Port number %d state is %s\n"
			,ib_port_
			,portStates[port_attr.state]);
		return LINK_FAILURE;
	}

	curr_link = port_attr.link_layer;
	if (!strcmp(LinkLayerStr(curr_link),"Unknown")) {
		fprintf(stderr," Unable to determine link layer \n");
		return LINK_FAILURE;
	}

	return port_attr.link_layer;
}

void RDMADevice::SetMaxInline() {
	enum ctx_device current_dev = ib_dev_name(context_);
	if (current_dev == UNKNOWN || current_dev == DEVICE_ERROR) {
		if (inline_size_ != DEF_INLINE) {
			fprintf(stderr,"Device not recognized to implement inline feature. Disabling it\n");
		}
		inline_size_ = 0;
		return;
	}

	if (inline_size_ == DEF_INLINE) {
		if (false) {
			// Only latency sensitive load needs inline.
			// Might enable later
			inline_size_ = DEF_INLINE_DC; // The smallest one
		} else {
			inline_size_ = 0;
		}
	}

	return;
}

bool RDMADevice::CheckLink() {
	bool succ = true;
	transport_type_ = context_->device->transport_type;
	link_type_ = SetLinkLayer();

	if (link_type_ == LINK_FAILURE) {
		fprintf(stderr, " Couldn't set the link layer\n");
		succ = false;
		return succ;
	}

	if (link_type_ == IBV_LINK_LAYER_ETHERNET && gid_index_ == -1) {		
		gid_index_ = 0;
	}

	SetMaxInline();

	out_reads_ = ctx_set_out_reads(context_, out_reads_);	

	return succ;
}

bool RDMADevice::CreateRdmaResources() {
	int is_udp_ps = connection_type_ == UD || connection_type_ == RawEth;
	enum rdma_port_space port_space = (is_udp_ps) ? RDMA_PS_UDP : RDMA_PS_TCP;
	struct rdma_cm_id **srv_cm_id = &control_cm_id_;

	control_cm_channel_ = rdma_create_event_channel();
	if (control_cm_channel_ == nullptr) {
		fprintf(stderr, " rdma_create_event_channel failed\n");
		return false;
	}

	if (rdma_create_id(control_cm_channel_, srv_cm_id , nullptr, port_space)) {
		fprintf(stderr,"rdma_create_id failed\n");
		return false;
	}

	return true;
}

struct ibv_mr *RDMADevice::AllocateMemRegion(size_t size) {
	int flags = IBV_ACCESS_LOCAL_WRITE;
	flags |= IBV_ACCESS_REMOTE_WRITE;
	flags |= IBV_ACCESS_REMOTE_READ;
	flags |= IBV_ACCESS_REMOTE_ATOMIC;

	void *buf = memalign(MEM_REGION_ALIGNMENT, size);
	if (!buf) {
		fprintf(stderr, "Failed to allocate memory for region!\n");
		abort();
	}

	struct ibv_mr *mr = ibv_reg_mr(pd_, buf, size, flags);
	if (!mr) {
		fprintf(stderr, "Failed to allocate memory region!\n");
		abort();
	}
	return mr;
}

struct ibv_mr *RDMADevice::RegisterMemRegion(void *buf, size_t size) {
	int flags = IBV_ACCESS_LOCAL_WRITE;
	flags |= IBV_ACCESS_REMOTE_WRITE;
	flags |= IBV_ACCESS_REMOTE_READ;
	flags |= IBV_ACCESS_REMOTE_ATOMIC;

	struct ibv_mr *mr = ibv_reg_mr(pd_, buf, size, flags);
	if (!mr) {
		fprintf(stderr, "Failed to register memory region!  --  %s\n", strerror(errno));
		abort();
	}
	return mr;
}

RDMAChannel *RDMADevice::GetRDMAChannel(const std::string &servername, int server_port, int thread_id) {
	std::string key = servername + std::to_string(server_port);

	spinlock_acquire(&dev_lock_);	
	std::unordered_map<std::string, SenderPeer *>::iterator it = peers_.find(key);
	if (it != peers_.end()) {
		SenderPeer *peer = it->second;
		struct ibv_qp *qp = peer->GetChannel(thread_id);
		spinlock_release(&dev_lock_);
		return new RDMAChannel(qp);
	}

	SenderPeer *peer = new SenderPeer(this, num_qps_per_peer_, servername, server_port);
	peers_[key] = peer;
	struct ibv_qp *qp = peer->GetChannel(thread_id);
	spinlock_release(&dev_lock_);
	return new RDMAChannel(qp);
}
 
RDMAChannel *RDMADevice::GetRDMAChannelWithIdx(const std::string &servername, int server_port, int idx) {
	std::string key = servername + std::to_string(server_port);

	spinlock_acquire(&dev_lock_);	
	std::unordered_map<std::string, SenderPeer *>::iterator it = peers_.find(key);
	if (it != peers_.end()) {
		SenderPeer *peer = it->second;
		struct ibv_qp *qp = peer->GetChannelWithIdx(idx);
		spinlock_release(&dev_lock_);
		return new RDMAChannel(qp);
	}

	SenderPeer *peer = new SenderPeer(this, num_qps_per_peer_, servername, server_port);
	peers_[key] = peer;
	struct ibv_qp *qp = peer->GetChannelWithIdx(idx);
	spinlock_release(&dev_lock_);
	return new RDMAChannel(qp);
}

void RDMADevice::CreateConnServThread() {
	conn_serv_thread_ = StartThread(
		ThreadOptions(), "conn_serv_thread", [this]() {
			struct rdma_cm_event *event;
			struct rdma_conn_param conn_param;

			if (rdma_listen(control_cm_id_, 0)) {
				fprintf(stderr, "rdma_listen failed\n");
				abort();
			}

			// main loop serving connection
			while	(1) {
				if (rdma_get_cm_event(control_cm_channel_, &event)) {
					fprintf(stderr, "rdma_get_cm_events failed\n");
					abort();
				}

				if (event->event == RDMA_CM_EVENT_ESTABLISHED) {
					rdma_ack_cm_event(event);
					continue;
				}

				if (event->event != RDMA_CM_EVENT_CONNECT_REQUEST) {
					fprintf(stderr, "bad event waiting for connect request %d\n",event->event);
					abort();
				}

				struct rdma_cm_id *cm_id = (struct rdma_cm_id*)event->id;
				srv_cm_ids_.push_back(cm_id);

				rdma_ack_cm_event(event);

				// create qp at server side
				struct ibv_qp_init_attr attr;
				struct ibv_qp* qp = nullptr;

				memset(&attr, 0, sizeof(struct ibv_qp_init_attr));
				struct ibv_cq *cq = cqs_[cur_cq_idx_ % num_cqs_];
				cur_cq_idx_++;

				attr.send_cq = cq;
				attr.recv_cq = cq;
				attr.cap.max_send_wr = TX_DEPTH;
				attr.cap.max_send_sge = MAX_SEND_SGE;
				attr.cap.max_inline_data = inline_size_;
				attr.srq = NULL;
				attr.cap.max_recv_wr = RX_DEPTH;
				attr.cap.max_recv_sge = MAX_RECV_SGE;

				switch (connection_type_) {
					case RC : attr.qp_type = IBV_QPT_RC; break;
					case UC : attr.qp_type = IBV_QPT_UC; break;
					case UD : attr.qp_type = IBV_QPT_UD; break;
					default:  fprintf(stderr, "Unknown connection type \n");
										abort();
				}

				if (rdma_create_qp(cm_id, pd_, &attr)) {
					fprintf(stderr, "[%s:%d]Couldn't create rdma QP - %s\n", __FILE__, __LINE__, strerror(errno));
					abort();
				}

				qp = cm_id->qp;	
				server_qps_.push_back(qp);

				if (pd_ != qp->pd) {
					fprintf(stderr, "[%s:%d] qps not share the same pd in device %s\n", __FILE__, __LINE__, name_.c_str());
					abort();
				}

				// accept
				memset(&conn_param, 0, sizeof conn_param);
				conn_param.responder_resources = out_reads_;
				conn_param.initiator_depth = out_reads_;
				conn_param.retry_count = 7;
				conn_param.rnr_retry_count = 7;

				if (rdma_accept(cm_id, &conn_param)) {
					fprintf(stderr, "Function rdma_accept failed\n");
					abort();
				}

				struct ibv_mr *mr = AllocateMemRegion(RECV_BUFFER_SIZE);
				struct recv_req_entry *recve = (struct recv_req_entry *)malloc(sizeof(struct recv_req_entry));
				recve->mr = mr;
				recve->qp = qp;
				recve->sge.addr = (uint64_t)mr->addr;
				recve->sge.length = RECV_BUFFER_SIZE;
				recve->sge.lkey = mr->lkey;

				recve->rwr.next = nullptr;
				recve->rwr.wr_id = (uint64_t)recve;
				recve->rwr.sg_list = &recve->sge;
				recve->rwr.num_sge = 1;

				struct ibv_recv_wr *bad_wr;
				if (ibv_post_recv(recve->qp, &recve->rwr, &bad_wr)) {
					fprintf(stderr, "Function ibv_post_recv failed for RDMA_CM QP\n");
					abort();
				}
			}
		});
}

void RDMADevice::RecvHandler(struct recv_req_entry *recve) {
	uint32_t *buff = (uint32_t *)recve->sge.addr;

	uint32_t cmd = *buff;
	buff++;
	uint32_t size = *buff;
	buff++;

	rpc_func_t rpcfunc;

	if ((cmd >= rpc_funcs_.size()) || ((rpcfunc = rpc_funcs_[cmd].func) == nullptr)) {
		fprintf(stderr, "Unknown rpc command (%d), drop it\n", cmd);
	} else {
		rpcfunc(buff, size, rpc_funcs_[cmd].arg);
	}

	struct ibv_recv_wr *bad_wr;
	if (ibv_post_recv(recve->qp, &recve->rwr, &bad_wr)) {
		fprintf(stderr, "Function ibv_post_recv failed for RDMA_CM QP\n");
		abort();
	}
}

void RDMADevice::RegisterRPCFunc(uint32_t cmd, rpc_func_t func, void *arg) {
	size_t old_size = rpc_funcs_.size();
	if (cmd >= old_size) {
		rpc_funcs_.resize(cmd + 1);
	}

	for (size_t i = old_size; i < rpc_funcs_.size(); i++) {
		rpc_funcs_[i].func = nullptr;
	}

	rpc_funcs_[cmd].func = func;
	rpc_funcs_[cmd].arg = arg;
}

void RDMADevice::CreateCQPollingThread(int idx) {
	struct ibv_comp_channel *comp_channel = comp_channels_[idx];
	struct ibv_cq *cq = cqs_[idx];

	cq_polling_threads_[idx] = StartThread(
		ThreadOptions(), "cq_polling_thread", [this, comp_channel, cq]() {
			struct ibv_wc *wc = (struct ibv_wc *)malloc(sizeof(struct ibv_wc) * CQ_POLL_BATCH);

			struct ibv_cq *ev_cq;
			void *ev_ctx;

			while (1) {
 
				if (ibv_get_cq_event(comp_channel, &ev_cq, &ev_ctx)) {
					fprintf(stderr, "Failed to get cq_event\n");
					abort();
				}

				ibv_ack_cq_events(ev_cq, 1);

				if (ibv_req_notify_cq(ev_cq, 0)) {
					fprintf(stderr, "Couldn't request CQ notification\n");
					abort();
				}

				if (ev_cq != cq) {
					fprintf(stderr, "[%s:%d] CQs do not match\n", __FILE__, __LINE__);
					abort();
				}

				int ne;

				do {
					ne = ibv_poll_cq(ev_cq, CQ_POLL_BATCH, wc);
					//ne = ibv_poll_cq(cq, CQ_POLL_BATCH, wc);

					if (ne > 0) {
						for (int i = 0; i < ne; i++) {
							if (wc[i].opcode == IBV_WC_RECV) {
								struct recv_req_entry *recve = (struct recv_req_entry *)wc[i].wr_id;
								RecvHandler(recve);
							} else if ((wc[i].opcode == IBV_WC_RDMA_READ) || 
													(wc[i].opcode == IBV_WC_RDMA_WRITE) || 
													(wc[i].opcode == IBV_WC_SEND)) {
								struct memcpy_req_entry *mcpe = (struct memcpy_req_entry *)wc[i].wr_id;

								if (wc[i].status != IBV_WC_SUCCESS) {
									fprintf(stderr, "work completion status %s\n", ibv_wc_status_str(wc[i].status));
								}

								mcpe->cb(mcpe->arg, wc[i].status);
								free(mcpe);
							}
						}
					}	else if (ne < 0) {
						fprintf(stderr, "Error in ibv_poll_cq\n");
						abort();
					}
				} while (ne > 0);
			}
		});
}

}
