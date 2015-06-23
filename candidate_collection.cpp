#include "candidate_collection.h"

#include "fstream"
#include "iostream"

using namespace std;

class frontier_node;

candidate_collection::~candidate_collection() {
  map<frontier_node*, float>::iterator it;
  for (it = begin(); it != end(); it++) {
    delete it->first;
 }
 clear();
}

void candidate_collection::print_candidate(string filename) {
  int i,j;
  map<frontier_node*, float>::iterator it;
  ofstream of(filename.c_str(),ios::out);

  for (it = begin(); it != end(); it++) {
    for (j = 1; j < it->first->candidate_len; j++) {
      of << it->first->candidate[j] << " ";
    }
    of << "(" << it->second << ")" << endl;
  }
  of.close();
}

void candidate_collection::append_candidate(string filename) {
  int i,j;
  map<frontier_node*, float>::iterator it;
  ofstream of(filename.c_str(),ios::out|ios::app);

  for (it = begin(); it != end(); it++) {
    for (j = 1; j < it->first->candidate_len; j++) {
      of << it->first->candidate[j] << " ";
    }
    of << "(" << it->second << ")" << endl;
  }
  of.close();
}