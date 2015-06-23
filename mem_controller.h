////////////////////////////////////////////////////////////////
// This is the implementation for GPU/CPU memory pool class.
// The vertical lists are allocated from the memory pool
////////////////////////////////////////////////////////////////
#ifndef _GPU_MEM_CONTROLLER_
#define _GPU_MEM_CONTROLLER_

#include "list"
#include "cstdlib"
#include "pthread.h"

using namespace std;

#define INIT_CPU_MEM_SIZE 64

class MemPool {
 public:
  int vlist_size;
  unsigned int** stack;
  unsigned int** stack_pointer;
  pthread_mutex_t mem_lock;
 public:
  virtual unsigned long get_mem_occupancy() = 0;
  virtual void init(int p_vlist_size) = 0;
  virtual unsigned int* alloc() = 0;
  virtual unsigned int* free(unsigned int* addr) = 0;
  virtual void destroy() = 0;
  virtual void debug(bool check) = 0;
};

class GPUMemPool : public MemPool {
 private:
  unsigned int * base;
  int vlist_num;

 public:
  GPUMemPool();
  GPUMemPool(int, int);
  ~GPUMemPool();

  virtual unsigned long get_mem_occupancy();
  virtual void init(int p_vlist_size);

  virtual unsigned int* alloc();
  virtual unsigned int* free(unsigned int* addr);
  unsigned int free_space();

  virtual void destroy();
  virtual void debug(bool check);
};

class CPUMemPool : public MemPool {
 public:
  list<unsigned int*> base_list;
  int base_size;
  int stack_size;

  void double_space_base();
  void double_space_stack();

 public:
  CPUMemPool();
  CPUMemPool(int);
  ~CPUMemPool();

  void init(int p_vlist_size);

  virtual unsigned long get_mem_occupancy();

  virtual unsigned int* alloc();
  virtual unsigned int* free(unsigned int* addr);
  unsigned int free_space();

  virtual void destroy();
  virtual void debug(bool check);
};

class MemChecker : public list<MemPool*> {
 private:
  unsigned long sys_mem_size;
  pthread_mutex_t mem_lock;
 public:
  MemChecker();
  ~MemChecker();
  bool check();
};

extern MemChecker global_checker;

#endif
