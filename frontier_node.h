/*
 * frontier_node.h
 *
 *  Created on: Jul 19, 2011
 *      Author: zhangfan
 */

#ifndef FRONTIER_NODE_H_
#define FRONTIER_NODE_H_

#include "mem_controller.h"

using namespace std;

#define MAX_CANDIDATE_LEN 128

typedef enum{MAIN_MEMORY, GRAPHIC_MEMORY, UNALLOCATED} mem_status;
typedef enum{CPU_TO_GPU, GPU_TO_CPU} mem_transfer_direction;

class job_manager;

class frontier_node {
 public:
  static int vlist_len;
  static int vlist_len_int;
  static int vlist_len_int_16;

 public:
  unsigned int candidate[MAX_CANDIDATE_LEN];
  int candidate_len;
  int support;

  mem_status vlist_location;
  union {unsigned int* g_addr; unsigned int* c_addr; } vlist_mem_ref;

 public:
  frontier_node();
  frontier_node(const frontier_node &fn);
  ~frontier_node();
  frontier_node & operator=(const frontier_node &fn);

  void alloc_vlist_gpu(GPUMemPool * gmc);
  void free_vlist_gpu(GPUMemPool * gmc);

  void alloc_vlist_cpu(CPUMemPool * cmc);
  void free_vlist_cpu(CPUMemPool * cmc);

  void assign_vlist(unsigned int * vlist_cpy);

  void transfer_vlist_ctog(CPUMemPool * cmc,GPUMemPool * gmc);
  void transfer_vlist_gtoc(GPUMemPool * gmc,CPUMemPool * cmc);
  void transfer_vlist_ctoc(CPUMemPool * cmc_src,CPUMemPool * cmc_dst);

  void copy_vlist_ctog(CPUMemPool * cmc,GPUMemPool * gmc);
  void debug();
};

#endif /* FRONTIER_NODE_H_ */
