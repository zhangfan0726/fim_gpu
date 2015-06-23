#include "frontier_preexpand.h"

#include "global.h"
#include "iostream"
#include "map"
#include "fstream"
#include "unistd.h"

using namespace std;
void frontier_preexpand::pre_expand_init(CPUMemPool* pcmc, string file, float min_sup) {
  frontier_node * tmp_node;
  int i = 0, j = 0, k = 0;

  cmc = pcmc;
  support_ratio = min_sup;

  fim_num = 0;

  map<int,vector<int> > freq_tracer;
  map<int,vector<int> >::iterator fti,old_fti;
  vector<int> vec_zero;
  int item_num, max_children_index, trans_num = 0, max_children_num = 0, candidate_num = 0;
  int v;
  string data_tmp;
  ifstream if_data;

  if_data.open(file.c_str());
  if (!if_data.is_open()) {
    cerr <<" Error in reading file" << endl;
    exit(1);
  }

  while (!if_data.eof()) {
    std::getline(if_data,data_tmp); 
    int p1 = 0, p2 = 0;
    char dig[64];
    int item;
    while (true) {
      if (data_tmp[p2] == ' ') {
        for (i = p1, j = 0; i < p2; i++, j++) {
          dig[j] = data_tmp[i];
        }
        dig[j] = '\0';
        item = atoi(dig);
        fti = freq_tracer.find(item);
        if (fti == freq_tracer.end()) {
          freq_tracer[item] = vec_zero;
          freq_tracer[item].push_back(trans_num);
        } else {
          fti->second.push_back(trans_num);
        }
        p2++;
        p1 = p2;
      } else if (p2 >= data_tmp.length()) {
        if (p2 != p1) {
          for (i = p1, j=0; i < p2; i++, j++) {
            dig[j] = data_tmp[i];
          }
          dig[j] = '\0';
          item = atoi(dig);
          fti = freq_tracer.find(item);
          if (fti == freq_tracer.end()) {
            freq_tracer[item] = vec_zero;
            freq_tracer[item].push_back(trans_num);
          } else {
            fti->second.push_back(trans_num);
          }
        }
        break;
      } else if (data_tmp[p2] != ' ') {
        p2++;
      }
    }
    trans_num++;
  }

  data_size = trans_num - 1;

  frontier_node::vlist_len = data_size;
  frontier_node::vlist_len_int = (frontier_node::vlist_len + 31) / 32;
  frontier_node::vlist_len_int_16 = ((frontier_node::vlist_len_int + 15) / 16) * 16;

  cmc->init(frontier_node::vlist_len_int);

  for (fti = freq_tracer.begin(); fti != freq_tracer.end();) {
    if (fti->second.size() < trans_num * min_sup) {
      old_fti = fti;
      fti++;
      freq_tracer.erase(old_fti);
    } else {
      fti++;
      fim_num++;
    }
  }

  candidate_num = freq_tracer.size() + 1;

  base = (frontier_node**)malloc(sizeof(frontier_node*) * candidate_num);
  stack_pointer = base + candidate_num - 1;

  for (fti = freq_tracer.begin(); fti != freq_tracer.end(); fti++) {
    *(stack_pointer) = new frontier_node;
    (*stack_pointer)->candidate[0] = 0;
    (*stack_pointer)->candidate[1 ] =fti->first;
    (*stack_pointer)->candidate_len = 2;
    (*stack_pointer)->support = fti->second.size();
    for (j = 0; j < fti->second.size(); j++) {
      if ((*stack_pointer)->vlist_location == UNALLOCATED) {
        (*stack_pointer)->alloc_vlist_cpu(cmc);
      }
      set_bit((*stack_pointer)->vlist_mem_ref.c_addr, fti->second[j], 1);
    }
    stack_pointer--;
  };
  (*stack_pointer) = NULL;
  stack_pointer = base + candidate_num;
}

typedef enum{END_CURRENT_JOB, END_PRODUCE, SKIP_DILIM, JM_FULL, PRODUCE} LOOP_STATUS;

void frontier_preexpand::produce_jobs(job_manager &jm, int threshold) {
  LOOP_STATUS ls = PRODUCE;
  bool add_flag = false;
  bool end_flag = false;
  int job_intensity = 0, local_intensity = 0;
  int i,j,k,num;
  frontier_stack * current_stack;
  frontier_node * tmp;
  frontier_node ** cursor;
  frontier_node * current_node,* next_node,* tmp_node;
  vector<int> vec_cand;

  current_stack = NULL;
  jm.set_status(PRODUCING);

  while (true) {
    if (jm.job_num >= jm.job_size) {
      ls = JM_FULL;
      sleep(0);
    } else if (ls == END_PRODUCE) {
      jm.set_status(AFTER_PRODUCING);
      break;
    } else if (stack_pointer <= base) {
      ls = END_PRODUCE;
      if (current_stack) {
        num = current_stack->stack_pointer - current_stack->base;
        for (i = 0; i < num / 2; i++) {
          tmp = current_stack->base[i];
          current_stack->base[i] = current_stack->base[num - 1 - i];
          current_stack->base[num - 1 - i] = tmp;
        }
    	jm.push_job(current_stack);
    	current_stack = NULL;
      }
    } else if (job_intensity > threshold) {
      ls = END_CURRENT_JOB;
      job_intensity = 0;
      num = current_stack->stack_pointer - current_stack->base;
      for (i = 0; i < num / 2; i++) {
        tmp = current_stack->base[i];
        current_stack->base[i] = current_stack->base[num - 1 - i];
        current_stack->base[num - 1 - i]=tmp;
      }
      jm.push_job(current_stack);
      current_stack = NULL;
    } else if (*(stack_pointer - 1) == NULL) {
      ls = SKIP_DILIM;
      stack_pointer--;
    } else {
      ls = PRODUCE;
      if (!current_stack) {
        current_stack = new frontier_stack;
        current_stack->init_cpu(cmc, data_size, support_ratio);
      }

      stack_pointer--;
      current_node = *stack_pointer;
      
      add_flag = false;
      local_intensity = 0;
      for (cursor = stack_pointer - 1; (*cursor) != NULL && cursor > base; cursor--) {
        next_node = (*cursor);

        tmp_node = new frontier_node;
        for (k = 0; k < current_node->candidate_len; k++) {
          tmp_node->candidate[k] = current_node->candidate[k];
        }
        tmp_node->candidate[k] = next_node->candidate[k - 1];
        tmp_node->candidate_len = k + 1;
        tmp_node->alloc_vlist_cpu(cmc);
        for (j = 0; j < frontier_node::vlist_len_int; j++) {
          tmp_node->vlist_mem_ref.c_addr[j] =
              current_node->vlist_mem_ref.c_addr[j] & next_node->vlist_mem_ref.c_addr[j];
          tmp_node->support += bitcnt(tmp_node->vlist_mem_ref.c_addr[j]);
        }

        if (tmp_node->support >= data_size * support_ratio) {
          *(current_stack->stack_pointer++) = tmp_node;
          add_flag = true;
          local_intensity++;
        } else {
          tmp_node->free_vlist_cpu(cmc);
          delete tmp_node;
        }
      }
      if (add_flag == true) {
        *(current_stack->stack_pointer++) = NULL;
      }
      current_node->free_vlist_cpu(cmc);
      jm.fim_num++;
      cc_pre[current_node] = ((float)(current_node->support)) / data_size;
      job_intensity += local_intensity*(local_intensity - 1) / 2;
    }
  }
}

void frontier_preexpand::destroy() {
  frontier_node ** p;
  for (p = base; p < stack_pointer; p++) {
    if ((*p)) {
      if ((*p)->vlist_location == MAIN_MEMORY) {
        (*p)->free_vlist_cpu(cmc);
      }
      delete (*p);
    }
  }
  free(base);
}

void frontier_preexpand::debug() {
  int i,j;
  cerr << "vlist_len : " << frontier_node::vlist_len
       << " support_ratio : " << support_ratio << " FIM number : "
       << fim_num << endl;

  i = 0;
  for (i = 0; i < stack_pointer - base; i++) {
    cerr << "No." << i << endl;
    if (base[i]) {
      cerr << base[i] << endl;
      base[i]->debug();
    } else {
      cerr<<"---divider---"<<endl;
    }
  }
}
