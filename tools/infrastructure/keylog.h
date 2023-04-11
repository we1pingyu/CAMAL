//
//  core_workload.h
//  YCSB-C
//
//  Created by Jinglei Ren on 12/9/14.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//  
//  Modified by Yongping Luo on 3/4/22
//  Copyright (c) 2022 Yongping Luo <ypluo18@qq.com>
//

#ifndef YCSB_C_BASIC_DB_H_
#define YCSB_C_BASIC_DB_H_

#include <fstream>
#include <string>
#include <vector>
#include <cassert>

using std::string;
using std::ofstream;
using std::endl;

namespace ycsbc {

class KeyLog {
 public:
    KeyLog(const string & filename) {
      fout.open(filename.c_str(), std::ios::out);
      assert(fout.good());
    }

    ~KeyLog() {
      fout.close();
    }

 public:
  void log_key(const std::string &key) {
    fout << key << endl;
  }

private:
  ofstream fout;
};

} // ycsbc

#endif // YCSB_C_BASIC_DB_H_

