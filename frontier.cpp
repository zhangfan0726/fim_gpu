#include "frontier.h"
#include "gpu_interface.h"
#include "global.h"

#include "cuda_runtime.h"
#include "string"
#include "fstream"
#include "iostream"
#include "ctime"
#include "cstdlib"
#include "cstring"

using namespace std;

void frontier_stack::init_cpu(CPUMemPool *pcmc, int pdata_size,
                              float min_sup) {
  cmc = pcmc;
  data_size = pdata_size;
  support_ratio = min_sup;
  fim_num = 0;

  base = (frontier_node **)malloc(sizeof(frontier_node *) * MAX_FRONTIER_SIZE);
  stack_pointer = base;

  vlist_location = MAIN_MEMORY;
}

void frontier_stack::copy_to_cpu() {
  frontier_node** p;

  for (p = base; p < stack_pointer; p++) {
    if (*p != NULL) {
      (*p)->transfer_vlist_gtoc(gmc,cmc);
    }
  }
  vlist_location = MAIN_MEMORY;
}

void frontier_stack::copy_to_gpu() {
  int i = 0;
  frontier_node** p;

  for (p = base; p < stack_pointer; p++) {
    if (*p != NULL) {
      (*p)->transfer_vlist_ctog(cmc, gmc);
    }
  }
  vlist_location = GRAPHIC_MEMORY;
}

void frontier_stack::transfer_in_cpu(CPUMemPool* new_cmc) {
  frontier_node** p;

  for (p = base; p < stack_pointer; p++) {
    if (*p != NULL) {
      (*p)->transfer_vlist_ctoc(cmc,new_cmc);
    }
  }
  cmc = new_cmc;
  vlist_location = MAIN_MEMORY;
}

int frontier_stack::size() {
  int sum = 0;
  frontier_node** p;
  for (p = base; p < stack_pointer; p++) {
    if (*p)
      sum++;
  }
  return sum;
}

typedef enum{END_DLIST_EMPTY, END_DLIST_NONEMPTY, EXPANDING,
             REACH_EXPANDING_LIMIT} expand_status;

void frontier_stack::expand_gpu(int size, int thread_id) {
  int i, j, k = 0;
  expand_status es;

  frontier_node* dst_list[MAX_EXPAND_SIZE];
  int dst_list_len = 0;
  frontier_node * current_node, * next_node;
  frontier_node * tmp_node;
  frontier_node ** cursor_current;
  frontier_node ** cursor_next;
  int accu_size = 0;
  int vlist_len_int;
  bool add_flag = false;
  time_t begin, end;
  time_t begin_a, end_a;

  static int new_count = 0;
  
  vlist_len_int = frontier_node::vlist_len_int_16;
  lug.clear();

  cursor_current = stack_pointer - 1;

  while (true) {

    if (cursor_current < base) {
      if (dst_list_len > 0) {
        es = END_DLIST_NONEMPTY;
        break;
      } else {
        es = END_DLIST_EMPTY;
        break;
      }
    } else if (accu_size > size) {
      es = REACH_EXPANDING_LIMIT;
      accu_size = 0;
      break;
    } else if (*(cursor_current) == NULL) {
      cursor_current--;
    } else {
      es = EXPANDING;
      current_node = *cursor_current;
      add_flag = false;
      for (cursor_next = cursor_current - 1;
           (*cursor_next) != NULL && cursor_next > base; cursor_next--) {
        next_node = (*cursor_next);

        tmp_node = new frontier_node;

        memcpy((void *)tmp_node->candidate, (void *)current_node->candidate,
               sizeof(unsigned int)*current_node->candidate_len);

        tmp_node->candidate[k] = next_node->candidate[k - 1];
        tmp_node->candidate_len = k + 1;

        tmp_node->alloc_vlist_gpu(gmc);
        dst_list[dst_list_len++] = tmp_node;
        lug.add_to_tail(current_node->vlist_mem_ref.g_addr,
                        next_node->vlist_mem_ref.g_addr,
                        tmp_node->vlist_mem_ref.g_addr);

        accu_size++;
        add_flag = true;
      }
      if (add_flag == true) {
        dst_list[dst_list_len++] = NULL;
      }
      cursor_current--;
    }
  }

  lug.support_counting();

  for (i = 0, j = 0; i < dst_list_len && j < lug.list_len;) {
    if (dst_list[i]) {
      dst_list[i]->support = lug.result[j];
      i++;
      j++;
    } else {
      i++;
    }
  }

  while (stack_pointer > cursor_current + 1) {
    stack_pointer--;
    if ((*stack_pointer)) {
      (*stack_pointer)->free_vlist_gpu(gmc);
      fim_num++;
      delete (*stack_pointer);
    }
  }

  if (dst_list_len != 0) {
    *(stack_pointer++) = dst_list[dst_list_len - 1];
    for (i = dst_list_len - 2; i >= 0; i--) {
      if (dst_list[i] == NULL) {
        if (*(stack_pointer - 1) != NULL)
          *(stack_pointer++) = dst_list[i];
      } else if (dst_list[i]->support >= support_ratio * data_size) {
        *(stack_pointer++) = dst_list[i];
      } else {
        dst_list[i]->free_vlist_gpu(gmc);
        delete dst_list[i];
      }
    }
  }
}

void frontier_stack::expand_cpu(int size, int thread_id) {
  int i,j,k = 0;
  expand_status es;

  frontier_node * dst_list[MAX_EXPAND_SIZE];
  int dst_list_len = 0;
  frontier_node * current_node, * next_node;
  frontier_node * tmp_node;
  frontier_node ** cursor_current;
  frontier_node ** cursor_next;
  int accu_size = 0;
  int vlist_len_int;
  bool add_flag = false;
  time_t begin, end;

  vlist_len_int = (frontier_node::vlist_len + 31) / 32;
  vlist_len_int = ((vlist_len_int + 15) / 16) * 16;

  cursor_current=stack_pointer-1;

  while (true) {
    if (cursor_current < base) {
      if (dst_list_len > 0) {
        es = END_DLIST_NONEMPTY;
        break;
      } else {
        es = END_DLIST_EMPTY;
        break;
      }
    } else if (accu_size > size) {
      es = REACH_EXPANDING_LIMIT;
      accu_size = 0;
      break;
    } else if (*(cursor_current) == NULL) {
      cursor_current--;
    } else {
      es = EXPANDING;
      current_node = *cursor_current;
      add_flag = false;

      for (cursor_next = cursor_current - 1; 
          (*cursor_next) != NULL && cursor_next > base;
          cursor_next--) {

        next_node=(*cursor_next);

        tmp_node=new frontier_node();

        for (k = 0; k < current_node->candidate_len; k++) {
          tmp_node->candidate[k] = current_node->candidate[k];
        }
        tmp_node->candidate[k] = next_node->candidate[k - 1];
        tmp_node->candidate_len = k + 1;
        tmp_node->alloc_vlist_cpu(cmc);
        dst_list[dst_list_len++]=tmp_node;
        begin = clock();
        single_vlist_intersection_cpu(
            current_node->vlist_mem_ref.c_addr, 
            next_node->vlist_mem_ref.c_addr,
            tmp_node->vlist_mem_ref.c_addr,
            tmp_node->support, frontier_node::vlist_len_int);
        end = clock();
        time_support_counting += (float)(end - begin);
        accu_size++;
        add_flag=true;
      }

      if (add_flag == true) {
        dst_list[dst_list_len++] = NULL;
      }
      cursor_current--;
    }
  }

  while (stack_pointer > cursor_current + 1) {
    stack_pointer--;
    if ((*stack_pointer)) {
      (*stack_pointer)->free_vlist_cpu(cmc);
      fim_num++;
      delete *stack_pointer;
    }
  }

  if (dst_list_len != 0) {
    *(stack_pointer++) = dst_list[dst_list_len - 1];

  for (i = dst_list_len - 2; i >= 0;i--) {
    if (dst_list[i] == NULL) {
      if (*(stack_pointer - 1) != NULL)
        *(stack_pointer++) = dst_list[i];
      } else if (dst_list[i]->support >=
                     support_ratio * data_size) {
        *(stack_pointer++)=dst_list[i];
      } else {
        dst_list[i]->free_vlist_cpu(cmc);
        delete dst_list[i];
      }
    }
  }
}

void frontier_stack::destroy() {
  frontier_node ** p;
  for (p = base; p < stack_pointer; p++) {
    if (*p) {
      if ((*p)->vlist_location == MAIN_MEMORY) {
        (*p)->free_vlist_cpu(cmc);
      } else if ((*p)->vlist_location == GRAPHIC_MEMORY) {
        (*p)->free_vlist_gpu(gmc);
      } else {
      }
      delete (*p);
    }
  }
  free(base);
}

void frontier_stack::debug() {
  int i, j;
  cerr << "vlist_len : " << frontier_node::vlist_len
       << " support_ratio : " << support_ratio 
       << " FIM number : "<< fim_num << endl;

  if (vlist_location == MAIN_MEMORY) {
    cerr << "vlist is on main memory" << endl;
  } else if (vlist_location == GRAPHIC_MEMORY) {
    cerr << "vlist is on graphic memory" << endl;
  } else {
    cerr << "vlist is not allocated" << endl;
  }

  i = 0;
  for (i = 0; i < stack_pointer - base; i++) {
    cerr << "No." << i << endl;
    if (base[i]) {
      cerr << base[i] << endl;
      base[i]->debug();
    } else {
      cerr << "---divider---" << endl;
    }
  }
}
