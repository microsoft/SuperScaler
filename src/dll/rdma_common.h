#ifndef RDMA_COMMON_H_
#define RDMA_COMMON_H_

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <unistd.h>
#include <malloc.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <functional>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <atomic>
#include <cinttypes>


#define RC  (0)
#define UC  (1)
#define UD  (2)
#define RawEth  (3)
#define XRC (4)
#define DC  (5)

#define LINK_FAILURE (4)
#define DEF_INLINE   (-1)

/* Optimal Values for Inline */
#define DEF_INLINE_WRITE (220)
#define DEF_INLINE_SEND_RC_UC (236)
#define DEF_INLINE_SEND_XRC (236)
#define DEF_INLINE_SEND_UD (188)
#define DEF_INLINE_DC (150)

#define WOLONG_DISALLOW_COPY_AND_ASSIGN(TypeName) \
	TypeName(const TypeName&) = delete;         \
	void operator=(const TypeName&) = delete

#endif // RDMA_COMMON_H_
