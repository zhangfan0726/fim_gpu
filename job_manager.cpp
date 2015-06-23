/*
 * job_manager.cpp
 *
 *  Created on: Jul 19, 2011
 *      Author: zhangfan
 */

#include "job_manager.h"
#include "iostream"

using namespace std;

job_manager::job_manager() {
  pthread_mutex_init(&job_lock,NULL);
  job_num = 0;
  job_size = 0;
  job_list = NULL;
  fim_num = 0;
  status = BEFORE_PRODUCING;
}

job_manager::job_manager(int size) {
  pthread_mutex_init(&job_lock,NULL);
  job_num = 0;
  job_list = (frontier_stack **)malloc(sizeof(frontier_stack *)*size);
  job_size = size;
  fim_num = 0;
  status = BEFORE_PRODUCING;
}

job_manager::~job_manager() {
  free(job_list);
  job_num = 0;
  job_size = 0;
  job_list = NULL;
  status = AFTER_PRODUCING;
  pthread_mutex_destroy(&job_lock);
}

void job_manager::inc_fim_num(int n) {
  pthread_mutex_lock(&job_lock);
  fim_num += n;
  pthread_mutex_unlock(&job_lock);
}

void job_manager::set_status(jm_status jst) {
  pthread_mutex_lock(&job_lock);
  status = jst;
  pthread_mutex_unlock(&job_lock);
}

jm_status job_manager::get_status() {
  return status;
}

bool job_manager::pop_job(frontier_stack* &fs) {
  bool ret;
  pthread_mutex_lock(&job_lock);
  if(job_num==0) {
    ret = false;
  } else {
    job_num--;
    fs = job_list[job_num];
    ret = true;
  }
  pthread_mutex_unlock(&job_lock);
  return ret;
}

bool job_manager::push_job(frontier_stack* fs) {
  bool ret;
  pthread_mutex_lock(&job_lock);
  if(job_num == job_size) {
    ret = false;
  } else {
    job_list[job_num] = fs;
    job_num++;
    ret = true;
  }
  pthread_mutex_unlock(&job_lock);
  return ret;
}

void job_manager::debug() {
  cerr << "job_size : " << job_size << " job_num : " << job_num << endl;
  switch(status) {
    case BEFORE_PRODUCING:
      cerr << "before producing" << endl;
      break;
    case PRODUCING:
      cerr << "producing" << endl;
      break;
    case AFTER_PRODUCING:
      cerr << "after producing" << endl;
      break;
  }
  int i;
  for (i = 0; i < job_num; i++) {
    cerr << "JM No." << i << endl;
    if (job_list[i])
      job_list[i]->debug();
    else
      cerr << "-----end_mark-----" << endl;
  }
}