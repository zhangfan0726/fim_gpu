/*
 * global.cpp
 *
 *  Created on: May 24, 2011
 *      Author: zhangfan
 */
#include "global.h"

float time_support_counting = 0;
float time_memory_operation = 0;
float time_candidate_generation = 0;
float time_init = 0;
float time_expansion = 0;

candidate_collection cc[MAX_CORE];
candidate_collection cc_pre;

void set_bit(unsigned int * vlist, int index, int data) {
  int seg = index / (sizeof(unsigned int) * 8);
  int offset = index % (sizeof(unsigned int) * 8);
  unsigned int bit_mask = 0x80000000;
  bit_mask = bit_mask>>offset;
  if (data == 1) {
    vlist[seg] = vlist[seg] | bit_mask;
  } else if (data == 0) {
    vlist[seg] = vlist[seg] & (~bit_mask);
  }
}

int get_bit(unsigned int * vlist, int index) {
  int seg = index / (sizeof(unsigned int) * 8);
  int offset = index % (sizeof(unsigned int) * 8);

  unsigned int bit_mask = 0x80000000;
  bit_mask = bit_mask >> offset;

  if ((vlist[seg] & bit_mask) == 0) {
    return 0;
  } else {
    return 1;
  }
}

int bitcnt(unsigned int src) {
  src = (src & 0x55555555) + ((src >> 1) & 0x55555555);
  src = (src & 0x33333333) + ((src >> 2) & 0x33333333);
  src = (src & 0x0f0f0f0f) + ((src >> 4) & 0x0f0f0f0f);
  src = (src & 0x00ff00ff) + ((src >> 8) & 0x00ff00ff);
  src = (src & 0x0000ffff) + ((src >> 16) & 0x0000ffff);
  return src;
}

bool test_candidate(unsigned int * candidate, int candidate_len) {
  int i = 0;
  for (i = 1; i < candidate_len; i++) {
    if (candidate[i] <= candidate[i-1])
      return false;
  }
  return true;
}

bool test_sibling(unsigned int * c1, int c1_len, unsigned int * c2, int c2_len) {
  int i;
  if (c1_len != c2_len)
    return false;

  for (i = 0; i < c1_len - 1; i++) {
    if (c1[i] != c2[i])
      return false;
  }
  return true;
}

int calc_support(vector<vector<int> > & data, unsigned int * candidate, int candidate_len) {
  int i, j, k, size = 0;
  int sup = 0;

  bool isfind = false;

  for (i = 0;i < data.size(); i++) {
    isfind = true;
    for (j = 1;j < candidate_len; j++) {
      for (k = 0; k < data[i].size(); k++) {
        if (candidate[j] == data[i][k]) {
          break;
        }
      }
      if (k >= data[i].size()) {
        isfind = false;
        break;
      }
    }
    if (isfind == true) {
      sup++;
    }  
  }
  return sup;
}