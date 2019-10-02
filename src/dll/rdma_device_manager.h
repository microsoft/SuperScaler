#ifndef RDMA_DEVICE_MANAGER_H_
#define RDMA_DEVICE_MANAGER_H_

#include "rdma_common.h"
#include "rdma_device.h"

namespace wolong {

class RDMADeviceManager {
 public:
	RDMADeviceManager(
		int num_cqs_per_dev, 
		int num_qps_per_peer_per_device, 
		const std::string &host, 
		int port);
	
	bool Init();

	size_t NumDevices() { return device_vec_.size(); }

	RDMADevice *GetDevice(int dev_idx) {
		if (dev_idx >= device_vec_.size()) return nullptr;
		return device_vec_[dev_idx];
	}

	RDMADevice *GetDevice(std::string dev_name) {
		std::unordered_map<std::string, RDMADevice *>::iterator it;
		it = device_map_.find(dev_name);
		if (it == device_map_.end()) return nullptr;
		return it->second;
	}

	~RDMADeviceManager() {
		for (size_t i = 0; i < device_vec_.size(); i++) {
			if (device_vec_[i]) {
				delete device_vec_[i];
			}
		}
	}

 private:
	bool DiscoverDevices();

	std::vector<RDMADevice *> device_vec_;
	std::unordered_map<std::string, RDMADevice *> device_map_;
	int connection_type_;
	int num_cqs_per_device_;
	int num_qps_per_peer_per_device_;
	std::string my_host_;
	int service_port_;
};

}

#endif // RDMA_DEVICE_MANAGER_H_
