/*
 * time_analysis.cpp
 *
 *  Created on: Feb 8, 2012
 *      Author: zhangfan
 */

#include "time_analysis.h"
#include "iostream"

using namespace std;

time_analysis::time_analysis()
{
	pthread_mutex_init(&lock,NULL);
	t_gpu=0;
	t_cpu=0;
	t_datatrans=0;
	t_gen=0;
	t_sup=0;
        t_all=0;
        fim_num=0;
}

void time_analysis::inc_init(float aug)
{
	pthread_mutex_lock(&lock);
	t_init+=aug;
	pthread_mutex_unlock(&lock);
}


void time_analysis::inc_gen(float aug)
{
	pthread_mutex_lock(&lock);
	t_gen+=aug;
	pthread_mutex_unlock(&lock);
}

void time_analysis::inc_sup(float aug)
{
	pthread_mutex_lock(&lock);
	t_sup+=aug;
	pthread_mutex_unlock(&lock);
}

void time_analysis::inc_all(float aug)
{
	pthread_mutex_lock(&lock);
	t_all+=aug;
	pthread_mutex_unlock(&lock);
}

void time_analysis::print()
{
	cout<<"-------------running status-------------"<<endl;
        cout<<"initialization "<<t_init<<" ("<<(t_init/(t_gen+t_sup+t_init))<<"%)"
	cout<<"candidate generation "<<t_gen<<" ("<<(t_gen/(t_gen+t_sup+t_init))<<"%)"<<endl;
	cout<<"support counting "<<t_sup<<" ("<<(t_sup/(t_gen+t_sup+t_init))<<"%)"<<endl;
}
