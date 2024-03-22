#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <string>

#include "spdlog/spdlog.h"
#include "nlohmann/json.hpp"

namespace tmpdb
{

    typedef enum
    {
        ENTRIES = 0,
        LEVELS = 1
    } bulk_load_type;

    typedef enum
    {
        INCREASING = 0,
        FIXED = 1,
        BUFFER = 2
    } file_size_policy;

    class CompactorOptions
    {
    public:
        int size_ratio = 2;            //> (T)
        int K = 1;                     //> (K)
        int Z = 1;                     //> (Z)
        size_t buffer_size = 1048576;  //> bytes (B) defaults 1 MB
        size_t entry_size = 1024;      //> bytes (E)
        double bits_per_element = 5.0; //> bits per element per bloom filter at all levels (h)
        bulk_load_type bulk_load_opt = ENTRIES;
        file_size_policy file_size_policy_opt = INCREASING;
        uint64_t fixed_file_size = std::numeric_limits<uint64_t>::max(); //> default MAX size

        size_t num_entries = 0;
        size_t new_num_entries = 0;
        size_t levels = 0;

        size_t file_size = std::numeric_limits<size_t>::max();

        CompactorOptions(){};

        CompactorOptions(std::string config_path);

        bool read_config(std::string config_path);

        bool write_config(std::string config_path);
    };

} /* namespace tmpdb */

#endif /* TIERED_OPTIONS_H_ */