#include "rdma_device_manager.h"

namespace wolong {

RDMADeviceManager::RDMADeviceManager(
	int num_cqs_per_dev, 
	int num_qps_per_peer_per_device,
	const std::string &host,
	int port) :
	num_cqs_per_device_(num_cqs_per_dev),
	num_qps_per_peer_per_device_(num_qps_per_peer_per_device),
	service_port_(port)
{
	my_host_ = host;
	connection_type_ = RC;
	device_vec_.clear();
	device_map_.clear();
}

bool RDMADeviceManager::Init() {
	bool succ = true;
	// find rdma devices
	succ = DiscoverDevices();
	if (!succ) {
		return succ;
	}

	return succ;
}

bool RDMADeviceManager::DiscoverDevices() {
	int num_of_device;
	struct ibv_device **dev_list;
	struct ibv_device *ib_dev = nullptr;
	bool succ = true;
	
	dev_list = ibv_get_device_list(&num_of_device);
	
	if (num_of_device <= 0) {
		fprintf(stderr," Did not detect devices \n");
		fprintf(stderr," If device exists, check if driver is up\n");
		succ = false;
		return succ;
	}

	device_vec_.clear(); //miao resize(num_of_device);
	fprintf(stdout, "Discovered %d RDMA devices:\n", num_of_device);
	for (int i = 0; i < num_of_device; i++) {
                if(i != 3) //miao
                   continue;
		ib_dev = dev_list[i];
		std::string dev_name = ibv_get_device_name(ib_dev);
		fprintf(stderr, "dev_name: %s\n", dev_name.c_str());
		RDMADevice *rdmadev = 
			new RDMADevice(ib_dev, i, dev_name, 
											connection_type_, 
											num_cqs_per_device_, 
											num_qps_per_peer_per_device_,
											my_host_,
											service_port_);
		device_vec_.push_back(rdmadev);
		device_map_[dev_name] = rdmadev;
	}
	fflush(stdout);

	if (device_vec_.size() != device_map_.size()) {
		fprintf(stderr, "Multiple RDMA Devices have same name!!\n");	
		succ = false;
		return succ;
	}

	return succ;
}

}
