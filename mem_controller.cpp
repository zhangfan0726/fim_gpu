#include "mem_controller.h"
#include "cuda_runtime.h"
#include "global.h"
#include "cstdlib"
#include "cstring"
#include "iostream"
#include "unistd.h"
#include "sys/sysinfo.h"

using namespace std;

MemChecker global_checker;

GPUMemPool::GPUMemPool() {
  stack = NULL;
  stack_pointer = 0;
  base = NULL;
}

GPUMemPool::GPUMemPool(int p_vlist_num,int p_vlist_size) {
  init(p_vlist_size);
}

GPUMemPool::~GPUMemPool() {
  destroy();
}

unsigned long GPUMemPool::get_mem_occupancy() {
  return sizeof(unsigned int) * vlist_num;
}

void GPUMemPool::init(int p_vlist_size) {
  cudaError_t err;
  int vlist_num_int;
  int i;
  int device_id;

  cudaDeviceProp deviceProp;

  vlist_size = p_vlist_size;

  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&deviceProp, device_id);

  long max_mem_size = deviceProp.totalGlobalMem;

  long max_vlist_num = max_mem_size/(sizeof(unsigned int)*vlist_size);

  pthread_mutex_init(&mem_lock,NULL);

  long min_vlist_num = 0;
  long latest_vlist_num;
  long mid_vlist_num = (min_vlist_num + max_vlist_num)/2;

  while (true) {
    mid_vlist_num = (min_vlist_num + max_vlist_num)/2;
    err = cudaMalloc((void **)&base, sizeof(unsigned int) * mid_vlist_num * vlist_size);
    if (err != cudaSuccess) {
      max_vlist_num=mid_vlist_num;
    } else {
      cudaFree(base);
      if (max_vlist_num - min_vlist_num < 32) {
        break;
      } else {
        min_vlist_num = mid_vlist_num;
      }
    }
  }
  
  vlist_num = mid_vlist_num * 7 / 8;

  err = cudaMalloc((void **)&base, sizeof(unsigned int) * vlist_num * vlist_size);
  if (err != cudaSuccess) {
    cerr << "Error occurs in GPU memory pool allocation" << endl;
    exit(1);
  }
  
  err = cudaMemset(base, 0, sizeof(unsigned int) * vlist_num * vlist_size);
  if (err != cudaSuccess) {
    cerr << "Error occurs in clearing GPU memory pool" << endl;
    exit(0);
  }

  stack = (unsigned int **)malloc(sizeof(unsigned int*) * vlist_num);
  if (!stack) {
    cerr << "Not enough memory on CPU"<<endl;
    exit(1);
  }
  
  for (i = 0; i < vlist_num; i++) {
    stack[i] = base + (i * vlist_size);
  }
  
  stack_pointer = stack + i;
  global_checker.push_back(this);
}

unsigned int* GPUMemPool::alloc() {
  unsigned int* ret;

  if (stack_pointer == stack) {
    cerr<<"Not enough spaces on GPU mem pool"<<endl;
    ret=NULL;
  }
  else {
    stack_pointer--;
    ret=*stack_pointer;
  }

  return ret;
}

unsigned int* GPUMemPool::free(unsigned int * addr) {
  unsigned int* ret;

  if (stack_pointer == stack + vlist_num)
    ret = NULL;
  else {
    ret = addr;
    *stack_pointer = addr;
    stack_pointer++;
  }

  return ret;
}

unsigned int GPUMemPool::free_space() {
  return (unsigned int)(stack_pointer - stack);
}

void GPUMemPool::destroy() {
  cudaError_t err;
  if (base) {
    err = cudaFree(base);
    if (err != cudaSuccess) {
      cerr << "In memory pool free : error in free graphic memory" << endl;
      exit(0);
    }
    base = NULL;
  }
  pthread_mutex_destroy(&mem_lock);
  if (stack) {
    std::free(stack);
    stack = NULL;
  }
  global_checker.remove((MemPool *)this);
}

void GPUMemPool::debug(bool check) {
  int i = 0;
  if (check == false) {
    for (i = 0; i < stack_pointer - stack; i++) {
      cerr << stack[i] << " ";
    }
    cerr << endl;
  } else {
    cerr << "Free vlist num : " << stack_pointer - stack << endl;
  }
}

/////////////////////////////////////////////////////////////////////////
/// CPU MEMORY POOL
/////////////////////////////////////////////////////////////////////////
CPUMemPool::CPUMemPool() {
  stack = NULL;
  stack_pointer = 0;
  base_list.clear();
}

CPUMemPool::CPUMemPool(int p_vlist_size) {
  init(p_vlist_size);
}

CPUMemPool::~CPUMemPool() {
  destroy();
}

void CPUMemPool::init(int p_vlist_size) {
  int vlist_num_int;
  int i;
  unsigned int * base;

  base_size = INIT_CPU_MEM_SIZE;
  stack_size = base_size;
  vlist_size = p_vlist_size;

  //cerr<<"cmc init"<<endl;
  pthread_mutex_init(&mem_lock,NULL);

  base = (unsigned int *)malloc(
      sizeof(unsigned int) * base_size * vlist_size);
  if (!base) {
    cerr << "Not enough CPU memory" << endl;
    exit(1);
  }
  memset((void *)base, 0 ,
         sizeof(unsigned int) * base_size * vlist_size);
  base_list.push_back(base);

  stack = (unsigned int **)malloc(
      sizeof(unsigned int *) * stack_size);
  if (!stack) {
    cerr << "Not enough CPU memory" << endl;
    exit(1);
  }
  for (i = 0; i < stack_size; i++) {
    stack[i] = base + (i * vlist_size);
  }
  stack_pointer = stack + i;
  global_checker.push_back(this);
}

void CPUMemPool::double_space_base() {
  int i;
  unsigned int ** cursor;
  unsigned int * new_base;
  unsigned int new_base_size;
  unsigned int ** new_stack;
  unsigned int ** new_stack_pointer;

  new_base = (unsigned int *)malloc(
      sizeof(unsigned int) * base_size * vlist_size);
  if (new_base == NULL) {
    cerr << "Not enough memory on CPU (base)" << endl;
    exit(1);
  }
  memset((void *)new_base, 0, 
         sizeof(unsigned int) * base_size * vlist_size);
  base_list.push_back(new_base);

  for (i = 0; i < base_size; i++) {
    if (stack_pointer >= stack + stack_size)
      double_space_stack();

    *(stack_pointer++) = new_base + i * (vlist_size);
  }
  base_size = 2 * base_size;

  if (!global_checker.check()) {
    cerr << "Program run out of system memory, quiting" << endl;
    exit(1);
  }
}

void CPUMemPool::double_space_stack() {
  unsigned int ** new_stack;
  unsigned int new_stack_size;
  unsigned int ** new_stack_pointer;
  
  new_stack_size=stack_size * 2;
  new_stack = (unsigned int **)malloc(
      sizeof(unsigned int *) * new_stack_size);
  if (!new_stack) {
    cerr << "Not enough memory on CPU" << endl;
    exit(1);
  }
  memcpy((void *)new_stack,(void *)stack,
         sizeof(unsigned int *) * stack_size);
  new_stack_pointer = new_stack + (stack_pointer - stack);

  std::free(stack);

  stack = new_stack;
  stack_pointer = new_stack_pointer;
  stack_size = new_stack_size;

  if (!global_checker.check()) {
    cerr << "Program run out of system memory, quiting" << endl;
    exit(1);
  }
}

void test_pool(unsigned int * vlist, int vlist_size) {
  for (int i = 0; i < vlist_size; i++) {
    vlist[i]++;
    vlist[i]--;
  }
}

unsigned int * CPUMemPool::alloc() {
  unsigned int * ret;

  pthread_mutex_lock(&mem_lock);
  if (stack_pointer <= stack) {
    double_space_base();
  }

  stack_pointer--;
  ret = *stack_pointer;

  pthread_mutex_unlock(&mem_lock);

  return ret;
}

unsigned int * CPUMemPool::free(unsigned int * addr) {
  unsigned int * ret;

  pthread_mutex_lock(&mem_lock);
  if (stack_pointer >= stack + stack_size)
    double_space_stack();

  ret = addr;
  *stack_pointer = addr;
  stack_pointer++;

  pthread_mutex_unlock(&mem_lock);

  return ret;
}

void CPUMemPool::destroy() {
  unsigned int * base;
  pthread_mutex_destroy(&mem_lock);
  while (!base_list.empty()) {
    base = base_list.back();
    base_list.pop_back();
    if (base) {
      std::free(base);
    }
  }
  if (stack) {
    std::free(stack);
    stack = NULL;
  }
  base_list.clear();
  global_checker.remove((MemPool *)this);
}

unsigned int CPUMemPool::free_space() {
  return (unsigned int)(stack_pointer - stack);
}

unsigned long CPUMemPool::get_mem_occupancy() {
  return sizeof(unsigned int) * ((unsigned long)vlist_size *
      (base_size) + stack_size);
}

void CPUMemPool::debug(bool check) {
  int i = 0, j = 0;
  if (check == false) {
    cerr << "vlist size = " << vlist_size
         << ", base size = " << base_size
         << " , stack size = " << stack_size
         << " ,free = " << stack_pointer - stack << endl;
    for (i = 0; i < stack_pointer - stack; i++) {
      cerr << stack[i] << " ";
    }
    cerr << endl;
  } else {
    for (i = 0; i < stack_pointer - stack; i++) {
      for (j = 0; j < vlist_size; j++) {
        stack[i][j] = 0;
      }
    }
  }
}

/////////////////////////////////////////////////////
// MemChecker
//////////////////////////////////////////////////////
MemChecker::MemChecker() {
  struct sysinfo si;
  pthread_mutex_init(&mem_lock,NULL);
  sysinfo(&si);
  sys_mem_size = si.totalram;
  cout << "Checking system memory, size="
       << sys_mem_size << " bytes" << endl;
}

MemChecker::~MemChecker() {
  pthread_mutex_destroy(&mem_lock);
  clear();
}

bool MemChecker::check() {
  unsigned long size = 0;
  pthread_mutex_lock(&mem_lock);
  list<MemPool *>::iterator it;
  for (it = begin(); it != end(); it++) {
    size = size + (*it)->get_mem_occupancy();
  }

  pthread_mutex_unlock(&mem_lock);
  return(size < sys_mem_size);
}
