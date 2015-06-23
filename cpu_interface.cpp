#include "cstdlib"

#include "global.h"

using namespace std;

void single_vlist_intersection_cpu(unsigned int* src1,
				   unsigned int* src2,
				   unsigned int* dst,
		                   int& support,
				   unsigned int vlist_len) {
  int i;
  support = 0;
  for (i = 0; i < vlist_len; i++) {
    dst[i] = src1[i] & src2[i];
    support += bitcnt(dst[i]);
  }
}
