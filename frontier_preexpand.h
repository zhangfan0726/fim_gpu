#ifndef FRONTIER_PREEXPAND_H_
#define FRONTIER_PREEXPAND_H_

#include "frontier_node.h"
#include "mem_controller.h"
#include "data_interface.h"
#include "job_manager.h"

class frontier_preexpand {
 public:
  frontier_node** base;
  frontier_node** stack_pointer;
 public:
  CPUMemPool * cmc;
  int data_size;

  int fim_num;
  float support_ratio;

  void pre_expand_init(CPUMemPool *pcmc, string file, float min_sup);

  void produce_jobs(job_manager &jm, int threshold);
  void produce_jobs_compact(job_manager &jm);
  void destroy();
  void debug();
};

#endif /* FRONTER_STACK_PREEXPAND_H_ */
