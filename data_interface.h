/*
 * data_interface.h
 *
 *  Created on: Jul 13, 2011
 *      Author: zhangfan
 */

#ifndef DATA_INTERFACE_H_
#define DATA_INTERFACE_H_

#include "vector"
#include "string"

using namespace std;

class data_interface : public vector<vector<int> > {
 public:
  int item_num;
  void init_from_file(string filename);
  void debug();
};

#endif /* DATA_INTERFACE_H_ */
