#include "tmpdb/compactor_options.hpp"

using namespace tmpdb;
using json = nlohmann::json;

CompactorOptions::CompactorOptions(std::string config_path)
{
    this->read_config(config_path);
}

bool CompactorOptions::read_config(std::string config_path)
{
    json cfg;
    std::ifstream read_cfg(config_path);
    if (!read_cfg.is_open())
    {
        spdlog::warn("Unable to read file: {}", config_path);
        spdlog::warn("Using default tiered options");
        return false;
    }
    read_cfg >> cfg;

    this->size_ratio = cfg["size_ratio"];
    this->buffer_size = cfg["buffer_size"];
    this->entry_size = cfg["entry_size"];
    this->bits_per_element = cfg["bits_per_element"];
    this->bulk_load_opt = cfg["bulk_load_opt"];
    this->num_entries = cfg["num_entries"];
    this->levels = cfg["levels"];
    this->fixed_file_size = cfg["fixed_file_size"];
    this->file_size_policy_opt = cfg["file_size_policy_opt"];

    return true;
}

bool CompactorOptions::write_config(std::string config_path)
{
    json cfg;
    cfg["size_ratio"] = this->size_ratio;
    cfg["buffer_size"] = this->buffer_size;
    cfg["entry_size"] = this->entry_size;
    cfg["bits_per_element"] = this->bits_per_element;
    cfg["bulk_load_opt"] = this->bulk_load_opt;
    cfg["levels"] = this->levels;
    cfg["num_entries"] = this->num_entries;
    cfg["fixed_file_size"] = this->fixed_file_size;
    cfg["file_size_policy_opt"] = this->file_size_policy_opt;

    std::ofstream out_cfg(config_path);
    if (!out_cfg.is_open())
    {
        spdlog::error("Unable to create or open file: {}", config_path);
        return false;
    }
    out_cfg << cfg.dump(4) << std::endl;
    out_cfg.close();
    spdlog::info("Writing configuration file at {}", config_path);

    return true;
}