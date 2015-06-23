#ifndef _CPU_INTERFACE_H_
#define _CPU_INTERFACE_H_

void single_vlist_intersection_cpu(unsigned int* src1,
				   unsigned int* src2,
				   unsigned int* dst,
		                   int& support,
				   unsigned int vlist_len);
#endif
