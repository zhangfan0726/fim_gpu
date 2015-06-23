#include "frontier.h"
#include "iostream"
#include "vector"
#include "string"
#include "omp.h"
#include "pthread.h"
#include "sched.h"
#include "errno.h"
#include "time.h"
#include "cstring"
#include "gpu_interface.h"
#include "cpu_interface.h"
#include "mem_controller.h"
#include "data_interface.h"
#include "job_manager.h"
#include "cuda_runtime.h"
#include "time_analysis.h"
#include "global.h"
#include "unistd.h"

using namespace std;

job_manager jm(JOB_SIZE);

// Worker thread function
void* thread(void* param) {
  int i;
  time_t t_begin, t_end;
  time_t t_begin_s, t_end_s;
  frontier_stack* current_stack;
  int index;
  GPUMemPool gmc;
  CPUMemPool cmc;
  cudaError_t err;

  // Extracts thread information. thread_type = 0 means this is a GPU worker.
  // thread_type = 1 means this is a CPU worker.
  unsigned int thread_type = (unsigned long)param >> 32;
  unsigned int thread_id = (unsigned long)param;

  cout << "starting thread " << thread_id
       << ", thread_type = " << thread_type << endl;

  t_begin = clock();

  if (thread_type == 0) {
    err = cudaSetDevice(thread_id);

    if (err != cudaSuccess) {
      cout << "Failed to initiate cuda device." << endl;
      exit(1);
    }

    gmc.init(frontier_node::vlist_len_int_16);

    while (true) {
      while(jm.status == BEFORE_PRODUCING);

        if (jm.pop_job(current_stack)) {
       	  current_stack->gmc = &gmc;
          current_stack->lug.initialize(
              MAX_LUG_SIZE, frontier_node::vlist_len_int_16);

          current_stack->copy_to_gpu();

          t_begin_s = clock();
          while (current_stack->stack_pointer > current_stack->base) {
            current_stack->expand_gpu(MAX_BLOCK, thread_id);
          }
          t_end_s = clock();

          current_stack->lug.destroy();
          jm.inc_fim_num(current_stack->fim_num);
          current_stack->destroy();
          delete current_stack;
        } else if (jm.status == PRODUCING) {
          sleep(0);
          continue;
        } else if (jm.status == AFTER_PRODUCING) {
          break;
        }
      }
      gmc.destroy();
    } else {
      cmc.init(frontier_node::vlist_len_int);
      while (jm.get_status() == BEFORE_PRODUCING);
      while (true) {

        if (jm.pop_job(current_stack)) {
    	  current_stack->transfer_in_cpu(&cmc);
 
          t_begin_s = clock();
    	  while (current_stack->stack_pointer > current_stack->base) {
    	    current_stack->expand_cpu(1, thread_id);
    	  }
          t_end_s = clock();

          jm.inc_fim_num(current_stack->fim_num);
 
          current_stack->destroy();
          delete current_stack;
 
    	} else if (jm.get_status() == PRODUCING) {
    	  sleep(0);
    	  continue;
    	} else if (jm.get_status() == AFTER_PRODUCING) {
    	  break;
    	}
      }
      cmc.destroy();
    }
    t_end = clock();
    time_expansion += (float)(t_end - t_begin);
    cout << "Ending thread " << thread_id
         << " ,thread_type = " << thread_type 
         << ", expansion time = " << (float)(t_end - t_begin) / CLOCKS_PER_SEC << endl;
    return NULL;
}

int fim_thread(string in_file_name, string out_file_name,
               float support_ratio, int gpu_num, int cpu_num) {
  pthread_t id[MAX_CORE];
  pthread_attr_t attr[MAX_CORE];
  cpu_set_t mask[MAX_CORE];
  struct sched_param param[MAX_CORE];
  int ret;

  string in_file, out_file, compute_mode;
  float minsup;
  int actual_gpu_num, actual_cpu_num, thread_num;

  frontier_stack ft;
  frontier_preexpand fp;

  data_interface di;

  GPUMemPool gmc_pre_expand;
  CPUMemPool cmc_pre_expand;

  int fim_sum = 0;
  time_t begin, end;
  int i = 0, j = 0;

  begin = clock();

  actual_cpu_num = sysconf(_SC_NPROCESSORS_ONLN);
  cudaGetDeviceCount(&actual_gpu_num);

  actual_cpu_num = (actual_cpu_num > actual_gpu_num) ?
      (actual_cpu_num - actual_gpu_num) : 0;

  if (cpu_num > actual_cpu_num)
    cpu_num = actual_cpu_num;
  if (gpu_num > actual_gpu_num)
    gpu_num = actual_gpu_num;

  cout << "Found " << actual_cpu_num << " available cpu, "
       << actual_gpu_num << " available gpu" << endl;

  thread_num = cpu_num + gpu_num;

  cout << "Candidate preexpansion" << endl;

  fp.pre_expand_init(&cmc_pre_expand, in_file_name, support_ratio);
  end = clock();
  time_init += (float)(end - begin);
  cout << "Starting multi-thread FIM, number of gpu threads = "
       << gpu_num << ", number of cpu threads = " << cpu_num << endl;
  for (i = 0; i < thread_num; i++) {
    unsigned long thread_param = 0;
    if (i < gpu_num) {
      thread_param = i;
    } else {
      thread_param = 1;
      thread_param = thread_param << 32;
      thread_param = thread_param | (i - gpu_num);
    }
    ret = pthread_create(&id[i], NULL, thread, (void*)thread_param);
    if (ret != 0) {
      cerr << "Failed to create worker" << endl;
      exit(1);
    }
  }

  fp.produce_jobs(jm, JOB_INTENSITY);

  for (i = 0; i < thread_num; i++) {
    pthread_join(id[i], NULL);
  }

  cc_pre.print_candidate(out_file_name);
  for (i = 0; i < thread_num; i++) {
    cc[i].append_candidate(out_file_name);
  }
  cout << "Number of frequent itemsets: " << jm.fim_num << endl;

  time_candidate_generation = time_expansion - time_support_counting;

  cout << "Time of setup: " << (time_init) / CLOCKS_PER_SEC << "s" << endl;
  cout << "Time of candidate generation: "
       << (time_candidate_generation / CLOCKS_PER_SEC) << "s" << endl;
  cout << "Time of support counting: "
       << (time_support_counting / CLOCKS_PER_SEC) << "s" << endl;
  return 0;
}

int main(int argc, char ** argv) {
  string in_file, out_file;
  int gpu_num, cpu_num;
  float minsup;
  if (argc != 6) {
    cout << "Usage : <program> <input> <output> min_sup(%) gpu_num cpu_num"
         << endl;
    return 0;
  }
  in_file = argv[1];
  out_file = argv[2];
  minsup = (atof(argv[3])) / 100;
  gpu_num = atoi(argv[4]);
  cpu_num = atoi(argv[5]);

  cout << "Starting fim_frontier_expansion, number of gpu = "
       << gpu_num << " , number of cpu = " << cpu_num << endl;

  return fim_thread(in_file, out_file, minsup, gpu_num, cpu_num);
}
