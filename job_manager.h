/*
 * job_manager.h
 *
 *  Created on: Jul 19, 2011
 *      Author: zhangfan
 */

#ifndef JOB_MANAGER_H_
#define JOB_MANAGER_H_

#include "frontier.h"
#include "frontier_preexpand.h"
#include "pthread.h"
#include "sched.h"

typedef enum{BEFORE_PRODUCING,PRODUCING,AFTER_PRODUCING} jm_status;

class job_manager {
 public:
  frontier_stack** job_list;
  int job_num;
  int job_size;
  jm_status status;
  int fim_num;
  pthread_mutex_t job_lock;

  job_manager();
  job_manager(int size);
  ~job_manager();

  void inc_fim_num(int n);
  void set_status(jm_status jst);
  jm_status get_status();
  bool pop_job(frontier_stack* &fs);
  bool push_job(frontier_stack* fs);

  void debug();
};

#endif /* JOB_MANAGER_H_ */
