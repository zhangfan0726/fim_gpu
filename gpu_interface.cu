#include "cuda.h"
#include "cstdio"
#include "gpu_interface.h"
#include "mem_controller.h"
#include "vector"
#include "iostream"
#include "global.h"
#include "iostream"

using namespace std;

__global__ void kernel_calc( unsigned int** src_list_1,
		             unsigned int** src_list_2,
                             unsigned int** dst_list,
                             unsigned int* result,
                             unsigned int list_len,
                             unsigned int vlist_len,
                             unsigned int old_block_pos) {
  __shared__ unsigned int sup[MAX_THREAD];

  unsigned int* psrc1;
  unsigned int* psrc2;
  unsigned int* pdst;

  unsigned int iter, i, tmp;
  int current_block_pos = blockIdx.x + old_block_pos;
  unsigned int bound;

  if (current_block_pos >= list_len)
    return;

  sup[threadIdx.x] = 0;

  iter = (vlist_len - 1) / blockDim.x + 1;

  psrc1 = src_list_1[current_block_pos];
  psrc2 = src_list_2[current_block_pos];
  pdst = dst_list[current_block_pos];

  __syncthreads();

  for (i = 0; i < iter; i++) {
    int thread_pos = i * blockDim.x + threadIdx.x;
    if (thread_pos >= vlist_len)
      break;

    tmp=psrc1[thread_pos] & psrc2[thread_pos];

    sup[threadIdx.x]+=__popc(tmp);
    pdst[thread_pos]=tmp;

  }
  __syncthreads();

  for (bound = blockDim.x / 2; bound > 0; bound >>= 1) {
    if (threadIdx.x < bound)
      sup[threadIdx.x]+=sup[threadIdx.x+bound];
    __syncthreads();
  }

  __syncthreads();

  if(threadIdx.x == 0) {
    *(result+current_block_pos)=sup[0];
  }
}

void ListUnionGPU::initialize(unsigned int size, unsigned int vlen) {
  list_size=size;
  list_len=0;
  vlist_len=vlen;
  src_list_1=(unsigned int**)malloc(sizeof(unsigned int *)*list_size);
  src_list_2=(unsigned int**)malloc(sizeof(unsigned int *)*list_size);
  dst_list=(unsigned int**)malloc(sizeof(unsigned int *)*list_size);
  result=(unsigned int*)malloc(sizeof(unsigned int)*list_size);
  return;
}

void ListUnionGPU::add_to_tail(unsigned int* psrc1, unsigned int* psrc2, unsigned int* pdst) {
  src_list_1[list_len]=psrc1;
  src_list_2[list_len]=psrc2;
  dst_list[list_len]=pdst;
  list_len++;
  return;
}

void ListUnionGPU::support_counting() {
  unsigned int ** d_src_list_1, **d_src_list_2, **d_dst_list;
  unsigned int * d_result;
  int i = 0;
  time_t begin, end;
  cudaError_t err;

  err = cudaMalloc((void **)&d_src_list_1, sizeof(unsigned int *)*list_len);
  if(err != cudaSuccess) {
    cerr << "error in cuda malloc" << endl;
  }
  err = cudaMalloc((void **)&d_src_list_2, sizeof(unsigned int *)*list_len);
  if(err != cudaSuccess) {
    cerr<<"error in cuda malloc"<<endl;
  }
  err = cudaMalloc((void **)&d_dst_list, sizeof(unsigned int *)*list_len);
  if(err != cudaSuccess) {
    cerr<<"error in cuda malloc"<<endl;
  }
  err = cudaMalloc((void **)&d_result, sizeof(unsigned int)*list_len);
  if(err != cudaSuccess) {
    cerr<<"error in cuda malloc"<<endl;
  }
  err = cudaMemcpy(d_src_list_1, src_list_1, sizeof(unsigned int *)*list_len,
                   cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    cerr<<"error in cuda mempcy"<<endl;
  }
  err = cudaMemcpy(d_src_list_2, src_list_2, sizeof(unsigned int *)*list_len,
                   cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    cerr<<"error in cuda mempcy"<<endl;
  }
  err = cudaMemcpy(d_dst_list, dst_list, sizeof(unsigned int *)*list_len,
                   cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    cerr<<"error in cuda mempcy"<<endl;
  }
  int iter = 0, pos = 0;
  iter = (list_len - 1) / MAX_BLOCK + 1;

  begin = clock(); 
  for (i = 0; i < iter; i++) {
    kernel_calc<<<MAX_BLOCK, MAX_THREAD>>>(d_src_list_1,d_src_list_2,d_dst_list,
                                           d_result,list_len,vlist_len,pos);

    pos+=MAX_BLOCK;
  }
  cudaThreadSynchronize();
  end = clock();
  time_support_counting += (float)(end - begin);
  err = cudaMemcpy(result, d_result, sizeof(unsigned int)*list_len, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    cerr<<"error in cuda mempcy"<<endl;
  }

  err = cudaFree(d_src_list_1);
  if (err != cudaSuccess) {
    cerr<<"error in cuda free"<<endl;
  }
  err = cudaFree(d_src_list_2);
  if (err != cudaSuccess) {
    cerr<<"error in cuda free"<<endl;
  }
  err = cudaFree(d_dst_list);
  if (err != cudaSuccess) {
    cerr<<"error in cuda free"<<endl;
  }
  err = cudaFree(d_result);
  if (err != cudaSuccess) {
    cerr<<"error in cuda free"<<endl;
  }

}

void ListUnionGPU::clear() {
  list_len = 0;
}

void ListUnionGPU::destroy() {
  list_size=0;
  list_len=0;
  vlist_len=0;
  free(src_list_1);
  free(src_list_2);
  free(dst_list);
  free(result);
  return;
}

unsigned int * d_vlist_generator(int vlist_len, GPUMemPool * gmc) {
  unsigned int * d_res,* res;
  int i;
  cudaError_t err;
  res=(unsigned int *)malloc(sizeof(unsigned int)*vlist_len);
  for (i = 0; i < vlist_len; i++) {
    res[i] = rand() % 16;
  }

  d_res = gmc->alloc();
  if (d_res == NULL)
    cerr << "GPU memory full"<<endl;

  err = cudaMemcpy(d_res, res, sizeof(unsigned int) * vlist_len, cudaMemcpyHostToDevice);
  if (err != 0) {
    cerr<<"cuda call error in random generator"<<endl;
  }
  free(res);
  return d_res;
}

void ListUnionGPU::debug(bool verification=false) {
  int i, j, k;
  unsigned int * h_vlist;
  cudaError_t err;
  int sum = 0;
  cerr << "list size : " << list_size << " list len : " << list_len
       << " vlist len : " << vlist_len << endl;

  if (verification == false) {
    h_vlist = (unsigned int *)malloc(sizeof(unsigned int)*vlist_len);
    cerr << "list_size = " << list_size << "," << "list_len = " 
         << list_len << "," << "vlist_len = " << vlist_len << endl;

    cerr << "lists" << endl;
    for (i = 0; i < list_len; i++) {
      cerr << "list item " << dec << i << endl;
      err = cudaMemcpy(h_vlist,src_list_1[i], sizeof(unsigned int)*vlist_len,
                       cudaMemcpyDeviceToHost);
      if (err != 0) {
	cerr << "cuda call error in debug, code = " << err << endl;
      }
      for (j = 0; j < vlist_len; j++) {
	int cur_pos = h_vlist[j];
	for (k = 31; k >= 0; k--) {
	  cerr << ((cur_pos >> k) & 0x00000001) << ",";
	}
	// cerr<<hex<<"("<<cur_pos<<"),";
      }
      cerr << endl;
      err = cudaMemcpy(h_vlist,src_list_2[i], sizeof(unsigned int)*vlist_len, cudaMemcpyDeviceToHost);
      if (err != 0) {
	cerr << "cuda call error in debug, code = " << err << endl;
      }
      for (j = 0; j < vlist_len; j++) {
	int cur_pos = h_vlist[j];
	for (k = 31; k >= 0; k--) {
	  cerr << ((cur_pos>>k) & 0x00000001)<<",";
	}
      }
      cerr << endl;
      err = cudaMemcpy(h_vlist, dst_list[i], sizeof(unsigned int)*vlist_len,
                       cudaMemcpyDeviceToHost);
      if (err != 0) {
	cerr << "cuda call error in debug, code = " << err << endl;
      }
      sum = 0;
      for (j = 0; j < vlist_len; j++) {
	int cur_pos = h_vlist[j];
	for (k = 31; k >= 0; k--) {
	  cerr << ((cur_pos >> k) & 0x00000001) << ",";
	  sum += ((cur_pos >> k) & 0x00000001);
        }
      }
      cerr << endl;
      cerr << "Results : " << result[i] << endl;
      cerr << "Verification : " << sum << endl;
    }

    cerr << "result: " << endl;
    for (i = 0; i < list_len; i++) {
      cerr << dec << result[i] << ",";
    }
    cerr << endl;
    free(h_vlist);
  } else if (verification == true) {
    h_vlist=(unsigned int *)malloc(sizeof(unsigned int)*vlist_len);

    for (i = 0; i < list_len; i++) {
      err = cudaMemcpy(h_vlist, src_list_1[i], sizeof(unsigned int)*vlist_len,
                       cudaMemcpyDeviceToHost);
      if (err != 0) {
	cerr << "cuda call error in debug, code = " << err << endl;
      }

      err = cudaMemcpy(h_vlist, src_list_2[i], sizeof(unsigned int)*vlist_len,
                       cudaMemcpyDeviceToHost);
      if (err != 0) {
	cerr << "cuda call error in debug, code = " << err << endl;
      }

      err = cudaMemcpy(h_vlist, dst_list[i], sizeof(unsigned int)*vlist_len,
                       cudaMemcpyDeviceToHost);
      if (err != 0) {
	cerr << "cuda call error in debug, code = " << err << endl;
      }
      sum = 0 ;
      for (j = 0; j < vlist_len; j++) {
	int cur_pos = h_vlist[j];
	for (k = 31; k >= 0; k--) {
	  sum += ((cur_pos >> k) & 0x00000001);
	}
      }
      if (result[i] != sum) {
	cerr << "inconsistent result at " << i << endl;
      }
    }
    free(h_vlist);
  }
  cerr<<"------------------------finish debugging lug-------------------------\n";
}
