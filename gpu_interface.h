#ifndef _GPU_INTERFACE_H_
#define _GPU_INTERFACE_H_

#include "mem_controller.h"

class ListUnionGPU {
 public:
  unsigned int ** src_list_1;
  unsigned int ** src_list_2;
  unsigned int ** dst_list;
  unsigned int * result;
  int list_len;
  int list_size;
  int vlist_len;
 public:
  void initialize(unsigned int, unsigned int);
  void add_to_tail(unsigned int* psrc1, unsigned int* psrc2, unsigned int* pdst);
  void support_counting();

  void clear();
  void destroy();

  void debug(bool);
};

unsigned int* d_vlist_generator(int vlist_len, GPUMemPool* gmc);

#endif
