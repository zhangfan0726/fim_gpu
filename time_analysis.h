/*
 * time_analysis.h
 *
 *  Created on: Feb 8, 2012
 *      Author: zhangfan
 */

#ifndef TIME_ANALYSIS_H_
#define TIME_ANALYSIS_H_

#include "pthread.h"

class time_analysis
{
public:

	float t_gpu;
	float t_cpu;
	float t_datatrans;
        float t_init;
	float t_gen;
	float t_sup;
	float t_all;
        int fim_num;

	pthread_mutex_t lock;

public:
	time_analysis();
        void inc_init(float);
	void inc_gen(float);
	void inc_sup(float);
        void inc_all(float);
        void set_fim_num(int n) { fim_num=n; }
	void print();
};

#endif /* TIME_ANALYSIS_H_ */
