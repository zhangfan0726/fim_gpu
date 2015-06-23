/*
 * test_case.h
 *
 *  Created on: Jul 19, 2011
 *      Author: zhangfan
 */

#ifndef TEST_CASE_H_
#define TEST_CASE_H_

#define MAX_GPU_SPACE 8000
#define MAX_CPU_SPACE 8000
#define MAX_LEN 200
#define MAX_WIDTH 30

void test_gpu_mem_pool(int list_num, int list_size);
void test_cpu_mem_pool(int list_num, int list_size);
void test_ListUnionGPU();
void test_ListUnionCPU();
void test_job_manager();
void test_frontier_preexpand();
void test_gpu_kernel();


#endif /* TEST_CASE_H_ */
