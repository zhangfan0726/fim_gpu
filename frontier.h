/*
 * frontier.h
 *
 *  Created on: May 24, 2011
 *      Author: zhangfan
 */

#ifndef FRONTIER_H_
#define FRONTIER_H_

#include "vector"
#include "pthread.h"
#include "sched.h"

#include "frontier_node.h"
#include "mem_controller.h"
#include "gpu_interface.h"
#include "cpu_interface.h"
#include "data_interface.h"

using namespace std;

class frontier_stack {
 public:
  frontier_node ** base;
  frontier_node ** stack_pointer;
 public:
  GPUMemPool* gmc;
  CPUMemPool* cmc;
  int data_size;
  int fim_num;
  int epsilon;
  float support_ratio;
  ListUnionGPU lug;
  mem_status vlist_location;

  void init_cpu(CPUMemPool* pcmc, int data_size, float min_sup);

  void copy_to_cpu();
  void copy_to_gpu();
  void transfer_in_cpu(CPUMemPool * new_cmc);

  void expand_gpu(int size, int thread_id);
  void expand_cpu(int size, int thread_id);

  int size();

  void produce_jobs(job_manager &jm);

  void destroy();
  void debug();
};

void split_frontier_stack(frontier_stack & fs_src, frontier_stack * &mfs_dst, int mfs_num);

#endif /* FRONTIER_H_ */
