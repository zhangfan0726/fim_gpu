/*
 * test_case.cpp
 *
 *  Created on: Jul 19, 2011
 *      Author: zhangfan
 */
#include "test_case.h"
#include "frontier.h"
#include "iostream"
#include "vector"
#include "string"
#include "cstring"
#include "gpu_interface.h"
#include "cpu_interface.h"
#include "mem_controller.h"
#include "data_interface.h"
#include "job_manager.h"
#include "frontier_preexpand.h"
#include "global.h"


void test_gpu_mem_pool(int list_num, int list_size) {
  int occ_size=0;
  GPU_mem_pool gmc(0, list_size);
  unsigned int* addr;
  vector<unsigned int *> addr_list;
  time_t begin, end;
  int i=0, j=0;

  begin=clock();
  while(true) {
    int ind=rand()%list_num;
    if(ind>occ_size) {
      addr=gmc.alloc();
      if (addr == NULL) {
	cerr << "memory full" << endl;
      } else {
	occ_size++;
	addr_list.push_back(addr);
      }
      gmc.debug(false);
      for (i = 0; i < addr_list.size(); i++) {
	cerr<<addr_list[i]<<" ";
      }
      cerr<<endl;
      // alloc
    } else {
      vector<unsigned int*>::iterator it;
      it = addr_list.begin() + rand() % addr_list.size();
      if (gmc.free(*it) == NULL) {
	cerr << "Error in mem free"<<endl;
      } else {
	addr_list.erase(it);
	occ_size--;
      }
      gmc.debug(false);
      for (i = 0; i < addr_list.size(); i++) {
	cerr << addr_list[i] << " ";
      }
      cerr << endl;
    }
  //  getchar();
  }
}

/*
void test_cpu_mem_pool(int list_num, int list_size) {
  int occ_size=0;
  CPU_mem_pool cmc;
  unsigned int * addr;
  int add_num=0;
  int direction=0;
  vector<unsigned int *> addr_list;
	time_t begin,end;
	int i=0,j=0;

	cmc.init(list_num,list_size);

	begin=clock();
	while(true)
	{
		//cmc.debug();
		//getchar();
		occ_size=cmc.stack_pointer-cmc.stack;
		int ind=rand()%(cmc.stack_size);
		//if(ind>=occ_size || addr_list.size()==0)
		if(direction==0)
		{
			//cerr<<"alloc"<<endl;
			addr=cmc.alloc();
			for(int i=0;i<list_size;i++)
			{
			    addr[i]=1;
			}
			addr_list.push_back(addr);
			add_num++;
			if(add_num>20000)
			{
				direction=1;
			}
		}
		else
		{
			//cerr<<"free "<<endl;
		    vector<unsigned int *>::iterator it;
		    it=addr_list.begin()+rand()%addr_list.size();
            addr=(*it);
            for(int i=0;i<list_size;i++)
            {
            	addr[i]=0;
            }
			if(cmc.free(*it)==NULL)
			{
				cerr<<"error in mem free"<<endl;
			}
			else
			{
				addr_list.erase(it);
			}
			add_num--;
			if(add_num<10)
			{
				direction=0;
			}
			//cmc.debug();

		}
		//getchar();
	}
}*/

void test_ListUnionGPU() {
  unsigned int* d_dst_vlist,*d_src_vlist_1, *d_src_vlist_2;
  ListUnionGPU lug;
  int i;
  time_t begin,end;
  GPU_mem_pool gmc(0, MAX_WIDTH);
  lug.initialize(MAX_LEN,MAX_WIDTH);
  for (i = 0;i < MAX_LEN; i++) {
    d_src_vlist_1=d_vlist_generator(MAX_WIDTH,&gmc);
    d_src_vlist_2=d_vlist_generator(MAX_WIDTH,&gmc);
    d_dst_vlist=d_vlist_generator(MAX_WIDTH,&gmc);
    lug.add_to_tail(d_src_vlist_1,d_src_vlist_2,d_dst_vlist);
    //    lug.debug(false);
    // getchar();
  }
  //  lug.debug();
  cerr << "finish generation" << endl;
  begin=clock();
  for (i = 0; i < 1; i++) {
    lug.support_counting();
     // cerr<<i<<endl;
  }
  end=clock();
  cerr<<"stop"<<endl;
  float diffTime=(float)(end-begin)/CLOCKS_PER_SEC;
  cerr<<diffTime<<endl;
  lug.debug(false);
}

void test_job_manager()
{
	int i;
	frontier_stack *p;
	job_manager jm(32);
	while(true)
	{
		int tmp=rand()%2;
		//cerr<<tmp<<endl;
		if(tmp==0)
		{
			if(!jm.push_job(p))
			{
				cerr<<"full"<<endl;
			}
		}
		else
		{
			if(!jm.pop_job(p))
			{
				cerr<<"empty"<<endl;
			}
		}
	}
}

void test_gpu_kernel() {

  int test_list_num = 32;
  int test_list_size = 2048;
  GPU_mem_pool gmc(0, test_list_size);

  unsigned int** src1 = (unsigned int**)malloc(sizeof(unsigned int*) * test_list_num);
  unsigned int** src2 = (unsigned int**)malloc(sizeof(unsigned int*) * test_list_num);
  unsigned int** dst = (unsigned int**)malloc(sizeof(unsigned int*) * test_list_num);

  for (int i = 0; i < test_list_num; ++i) {
    src1[i] = (unsigned int*)malloc(sizeof(unsigned int*) * test_list_size);
    src2[i] = (unsigned int*)malloc(sizeof(unsigned int*) * test_list_size);
    dst[i] = (unsigned int*)malloc(sizeof(unsigned int*) * test_list_size);
    
    for (int j = 0; j < test_list_size; ++j) {
      if (j % 2 == 0) {
        src1[i][j] == 0xffffffff;
        src2[i][j] == 0xffffffff;
      } else {
        src1[i][j] == 0;
        src2[i][j] == 0;
      }
      dst[i][j] = 0;
    }
  }
  
}
