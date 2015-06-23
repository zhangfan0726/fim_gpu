#ifndef CANDIDATE_COLLECTION_
#define CANDIDARE_COLLECTION_

#include "string"
#include "map"

#include "frontier_node.h"

using namespace std;

class candidate_collection : public map<frontier_node*, float> {
 public:
  virtual ~candidate_collection();
  void print_candidate(string filename);
  void append_candidate(string filename);
};

#endif // CANDIDATE_COLLECTION_