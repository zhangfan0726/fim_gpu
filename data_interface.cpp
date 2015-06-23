/*
 * data_interface.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: zhangfan
 */
#include "data_interface.h"

#include "fstream"
#include "iostream"

#include "frontier.h"

using namespace std;

void data_interface::init_from_file(string file_name) {
  int max_children_index = 0;
  string data_tmp;
  ifstream if_data;
  int i = 0, j = 0, k = 0;

  if_data.open(file_name.c_str());
  if (!if_data.is_open()) {
    cerr << "Error in reading file" << endl;
  }
  
  while (!if_data.eof()) {
    vector<int> vec_tmp; 
    std::getline(if_data,data_tmp);

    int p1 = 0, p2 = 0;
    char dig[64];
    int item;
    while(true) {
      if (data_tmp[p2] == ' ') {
        for (i = p1, j = 0; i < p2; i++, j++) {
	  dig[j] = data_tmp[i];
	}
	dig[j] = '\0';
	item = atoi(dig);
	vec_tmp.push_back(item);
	if (item > max_children_index) {
	  max_children_index = item;
	}
        p2++;
        p1 = p2;
      }
      else if (p2 >= data_tmp.length()) {
        if (p2 != p1) {
          for (i = p1, j = 0; i < p2; i++, j++) {
            dig[j] = data_tmp[i];
	  }
          dig[j] = '\0';
          item = atoi(dig);
	  vec_tmp.push_back(item);
	  if (item > max_children_index) {
	    max_children_index = item;
	  }
	}
	break;
      } else if (data_tmp[p2] != ' ') {
        p2++;
      }
    }
    if (vec_tmp.size() != 0) {
      push_back(vec_tmp);
    }
  }
  
  item_num = max_children_index + 1;
  frontier_node::vlist_len = size();
  frontier_node::vlist_len_int = (frontier_node::vlist_len + 31) / 32;
  frontier_node::vlist_len_int_16 = ((frontier_node::vlist_len_int + 15) / 16) * 16;
}

void data_interface::debug() {
  int i,j;
  cerr << "item number : " << item_num << endl;
  for (i = 0; i < size(); i++) {
    for (j = 0;j < (*this)[i].size(); j++) {
      cerr << (*this)[i][j] << " ";
    }
    cerr << endl;
  }
}