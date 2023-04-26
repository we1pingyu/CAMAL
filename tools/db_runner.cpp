#include <chrono>
#include <iostream>
#include <ctime>
#include <filesystem>
#include <unistd.h>

#include "clipp.h"
#include "spdlog/spdlog.h"

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/env.h"
#include "rocksdb/iostats_context.h"
#include "rocksdb/perf_context.h"
#include "rocksdb/compaction_filter.h"
#include "tmpdb/tiered_compactor.hpp"
#include "infrastructure/data_generator.hpp"
using namespace ROCKSDB_NAMESPACE;
// #define BATCH_SIZE 100
typedef struct environment
{
    std::string db_path;

    double non_empty_reads = 0.25;
    double empty_reads = 0.25;
    double range_reads = 0.25;
    double writes = 0.25;
    size_t prime_reads = 0;

    size_t steps = 10;

    std::string compaction_style = "level";
    // Build mode
    double T = 10;

    size_t B = 1 << 20; //> 1 MiB
    size_t E = 1 << 10; //> 1 KiB
    double bits_per_element = 5.0;
    size_t N = 1e6;
    size_t L = 0;

    int verbose = 0;
    bool destroy_db = true;

    int max_rocksdb_levels = 16;
    int parallelism = 1;

    int seed = 0;
    tmpdb::file_size_policy file_size_policy_opt = tmpdb::file_size_policy::INCREASING;
    uint64_t fixed_file_size = std::numeric_limits<uint64_t>::max();

    std::string dist_mode = "zipfian";
    double skew = 0.5;

    size_t cache_cap = 0;
    bool use_cache = true;

    std::string key_log_file;
    bool use_key_log = true;

} environment;

environment parse_args(int argc, char *argv[])
{
    using namespace clipp;
    using std::to_string;

    size_t minimum_entry_size = 32;

    environment env;
    bool help = false;

    auto general_opt = "general options" % ((option("-v", "--verbose") & integer("level", env.verbose)) % ("Logging levels (DEFAULT: INFO, 1: DEBUG, 2: TRACE)"),
                                            (option("-h", "--help").set(help, true)) % "prints this message");

    auto build_opt = ("build options:" % ((value("db_path", env.db_path)) % "path to the db",
                                          (option("-N", "--entries") & integer("num", env.N)) % ("total entries, default pick [default: " + to_string(env.N) + "]"),
                                          (option("-T", "--size-ratio") & number("ratio", env.T)) % ("size ratio, [default: " + fmt::format("{:.0f}", env.T) + "]"),
                                          (option("-B", "--buffer-size") & integer("size", env.B)) % ("buffer size (in bytes), [default: " + to_string(env.B) + "]"),
                                          (option("-E", "--entry-size") & integer("size", env.E)) % ("entry size (bytes) [default: " + to_string(env.E) + ", min: 32]"),
                                          (option("-b", "--bpe") & number("bits", env.bits_per_element)) % ("bits per entry per bloom filter [default: " + fmt::format("{:.1f}", env.bits_per_element) + "]"),
                                          (option("-c", "--compaction") & value("mode", env.compaction_style)) % "set level or tier compaction",
                                          (option("-d", "--destroy").set(env.destroy_db)) % "destroy the DB if it exists at the path"));

    auto run_opt = ("run options:" % ((option("-e", "--empty_reads") & number("num", env.empty_reads)) % ("empty queries, [default: " + to_string(env.empty_reads) + "]"),
                                      (option("-r", "--non_empty_reads") & number("num", env.non_empty_reads)) % ("non-empty queries, [default: " + to_string(env.non_empty_reads) + "]"),
                                      (option("-q", "--range_reads") & number("num", env.range_reads)) % ("range reads, [default: " + to_string(env.range_reads) + "]"),
                                      (option("-w", "--writes") & number("num", env.writes)) % ("writes, [default: " + to_string(env.writes) + "]"),
                                      (option("-s", "--steps") & integer("num", env.steps)) % ("steps, [default: " + to_string(env.steps) + "]"),
                                      (option("--dist") & value("mode", env.dist_mode)) % ("distribution mode ['uniform', 'zipf']"),
                                      (option("--skew") & number("num", env.skew)) % ("skewness for zipfian [0, 1)"),
                                      (option("--cache").set(env.use_cache, true) & number("cap", env.cache_cap)) % "use block cache",
                                      (option("--key-log-file").set(env.use_key_log, true) & value("file", env.key_log_file)) % "use keylog to record each key"));

    auto minor_opt = ("minor options:" % ((option("--max_rocksdb_level") & integer("num", env.max_rocksdb_levels)) % ("limits the maximum levels rocksdb has [default: " + to_string(env.max_rocksdb_levels) + "]"),
                                          (option("--parallelism") & integer("num", env.parallelism)) % ("parallelism for writing to db [default: " + to_string(env.parallelism) + "]"),
                                          (option("--seed") & integer("num", env.seed)) % "seed for generating data [default: random from time]"));

    auto file_size_policy_opt =
        ("file size policy (pick one)" %
         one_of(
             (
                 option("--increasing_files").set(env.file_size_policy_opt, tmpdb::file_size_policy::INCREASING) % "file size will match run size as LSM tree grows (default)",
                 (option("--fixed_files").set(env.file_size_policy_opt, tmpdb::file_size_policy::FIXED) & opt_integer("size", env.fixed_file_size)) % "fixed file size specified after fixed_files flag [default size MAX uint64]",
                 option("--buffer_files").set(env.file_size_policy_opt, tmpdb::file_size_policy::BUFFER) % "file size matches the buffer size")));

    auto cli = (general_opt,
                build_opt,
                run_opt,
                minor_opt,
                file_size_policy_opt);

    if (!parse(argc, argv, cli))
        help = true;

    if (env.E < minimum_entry_size)
    {
        help = true;
        spdlog::error("Entry size is less than {} bytes", minimum_entry_size);
    }

    if (help)
    {
        auto fmt = doc_formatting{}.doc_column(42);
        std::cout << make_man_page(cli, "db_builder", fmt);
        exit(EXIT_FAILURE);
    }

    return env;
}

size_t estimate_levels(size_t N, double T, size_t E, size_t B)
{
    if ((N * E) < B)
    {
        return 1;
    }

    size_t estimated_levels = std::ceil(std::log((N * E / B) + 1) / std::log(T));

    return estimated_levels;
}

void print_db_status(rocksdb::DB *db)
{
    spdlog::debug("Files per level");
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::vector<std::string> file_names;
    int level_idx = 1;
    for (auto &level : cf_meta.levels)
    {
        std::string level_str = "";
        for (auto &file : level.files)
        {
            level_str += file.name + ", ";
        }
        level_str = level_str == "" ? "EMPTY" : level_str.substr(0, level_str.size() - 2);
        spdlog::debug("Level {} : {} Files : {}", level_idx, level.files.size(), level_str);
        level_idx++;
    }
}

int main(int argc, char *argv[])
{
    spdlog::set_pattern("[%T.%e]%^[%l]%$ %v");
    environment env = parse_args(argc, argv);

    if (env.verbose == 1)
    {
        spdlog::info("Log level: DEBUG");
        spdlog::set_level(spdlog::level::debug);
    }
    else if (env.verbose == 2)
    {
        spdlog::info("Log level: TRACE");
        spdlog::set_level(spdlog::level::trace);
    }
    else
    {
        spdlog::set_level(spdlog::level::info);
    }

    if (env.destroy_db)
    {
        spdlog::info("Destroying DB: {}", env.db_path);
        rocksdb::DestroyDB(env.db_path, rocksdb::Options());
    }

    spdlog::info("Building DB: {}", env.db_path);
    rocksdb::Options rocksdb_opt;
    // rocksdb_opt.memtable_factory.reset(rocksdb::NewHashSkipListRepFactory());
    // rocksdb_opt.db = rocksdb::NewInMemoryDbOptions();

    rocksdb_opt.create_if_missing = true;
    rocksdb_opt.error_if_exists = true;
    rocksdb_opt.IncreaseParallelism(env.parallelism);
    rocksdb_opt.compression = rocksdb::kNoCompression;
    rocksdb_opt.bottommost_compression = kNoCompression;
    rocksdb_opt.use_direct_reads = true;
    rocksdb_opt.use_direct_io_for_flush_and_compaction = true;
    rocksdb_opt.max_open_files = 512;
    rocksdb_opt.advise_random_on_open = false;
    rocksdb_opt.random_access_max_buffer_size = 0;
    rocksdb_opt.avoid_unnecessary_blocking_io = true;
    tmpdb::TieredCompactor *tiered_compactor = nullptr;
    tmpdb::TieredOptions tiered_opt;

    if (env.compaction_style == "level")
    {
        rocksdb_opt.write_buffer_size = env.B;
        rocksdb_opt.level0_file_num_compaction_trigger = 4;
        rocksdb_opt.num_levels = env.max_rocksdb_levels;
        rocksdb_opt.max_bytes_for_level_multiplier = env.T;
        rocksdb_opt.max_bytes_for_level_base =
            rocksdb_opt.write_buffer_size * rocksdb_opt.max_bytes_for_level_multiplier;
    }
    else
    {
        rocksdb_opt.compaction_style = rocksdb::kCompactionStyleNone;
        rocksdb_opt.write_buffer_size = env.B;
        rocksdb_opt.disable_auto_compactions = true;
        // rocksdb_opt.target_file_size_base = UINT64_MAX;
        // rocksdb_opt.target_file_size_multiplier = UINT64_MAX;
        // rocksdb_opt.max_background_jobs = 1;
        tiered_opt.size_ratio = env.T;
        tiered_opt.buffer_size = env.B;
        tiered_opt.entry_size = env.E;
        tiered_opt.bits_per_element = env.bits_per_element;
        tiered_opt.num_entries = env.N;
        tiered_opt.levels = tmpdb::TieredCompactor::estimate_levels(env.N, env.T, env.E, env.B) + 1;
        rocksdb_opt.num_levels = (env.T - 1) * tiered_opt.levels + 1;
        tiered_compactor = new tmpdb::TieredCompactor(tiered_opt, rocksdb_opt);
        rocksdb_opt.listeners.emplace_back(tiered_compactor);
    }
    rocksdb::BlockBasedTableOptions table_options;
    if (env.compaction_style == "level")
    {
        table_options.filter_policy.reset(
            rocksdb::NewMonkeyFilterPolicy(
                env.bits_per_element,
                rocksdb_opt.max_bytes_for_level_multiplier,
                tiered_opt.levels));
    }
    else
    {
        table_options.filter_policy.reset(
            rocksdb::NewMonkeyFilterPolicy(
                env.bits_per_element,
                tiered_opt.size_ratio,
                tiered_opt.levels));
    }
    // table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(env.bits_per_element));
    if (env.cache_cap == 0)
        table_options.no_block_cache = true;
    else
        table_options.block_cache = rocksdb::NewLRUCache(env.cache_cap);
    rocksdb_opt.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    rocksdb_opt.statistics = rocksdb::CreateDBStatistics();
    rocksdb::DB *db = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(rocksdb_opt, env.db_path, &db);
    db->SetDBOptions({{"stats_level", "kExceptDetailedTimers"}});
    if (!status.ok())
    {
        spdlog::error("Problems opening DB");
        spdlog::error("{}", status.ToString());
        delete db;
        exit(EXIT_FAILURE);
    }
    std::map<std::string, uint64_t> stats;
    uint64_t num_running_compactions, num_pending_compactions, num_running_flushes, num_pending_flushes;
    KeyLog *key_log = new KeyLog(env.key_log_file);
    rocksdb::WriteOptions write_opt;
    write_opt.low_pri = true; //> every insert is less important than compaction
    write_opt.disableWAL = true;
    DataGenerator *data_gen = new YCSBGenerator(env.N, "uniform", 0.0);
    std::pair<std::string, std::string> key_value;
    auto write_time_start = std::chrono::high_resolution_clock::now();
    // size_t batch_size = 10;
    // for (size_t entry_num = 0; entry_num < env.N; entry_num += batch_size)
    // {
    //     // spdlog::info("{}", entry_num);
    //     rocksdb::WriteBatch batch(0, UINT64_MAX);
    //     for (int i = 0; i < (int)batch_size; i++)
    //     {
    //         key_value =
    //             data_gen->gen_kv_pair(env.E);
    //         batch.Put(key_value.first, key_value.second);
    //     }
    //     db->Write(write_opt, &batch);
    // }
    for (size_t entry_num = 0; entry_num < env.N; entry_num += 1)
    {
        // spdlog::info("{}", entry_num);
        key_value = data_gen->gen_kv_pair(env.E);
        db->Put(write_opt, key_value.first, key_value.second);
    }
    spdlog::info("Waiting for all compactions to finish before running");
    rocksdb::FlushOptions flush_opt;
    // db->Flush(flush_opt);
    if (env.compaction_style == "level")
    {
        while (true)
        {
            db->GetIntProperty(DB::Properties::kNumRunningFlushes,
                               &num_running_flushes);
            db->GetIntProperty(DB::Properties::kMemTableFlushPending,
                               &num_pending_flushes);
            db->GetIntProperty(DB::Properties::kNumRunningCompactions,
                               &num_running_compactions);
            db->GetIntProperty(DB::Properties::kCompactionPending,
                               &num_pending_compactions);
            if (num_running_compactions == 0 && num_pending_compactions == 0 && num_running_flushes == 0 && num_pending_flushes == 0)
                break;
        }
    }
    else
    {
        while (true)
        {
            db->GetIntProperty(DB::Properties::kNumRunningFlushes,
                               &num_running_flushes);
            db->GetIntProperty(DB::Properties::kMemTableFlushPending,
                               &num_pending_flushes);
            if (num_running_flushes == 0 && num_pending_flushes == 0)
                break;
        }
        while (tiered_compactor->compactions_left_count > 0)
            ;
        // while (tiered_compactor->requires_compaction(db))
        // {
        //     while (tiered_compactor->compactions_left_count > 0)
        //         ;
        // }
    }
    auto write_time_end = std::chrono::high_resolution_clock::now();
    auto write_time = std::chrono::duration_cast<std::chrono::milliseconds>(write_time_end - write_time_start).count();
    spdlog::info("(init_time) : ({})", write_time);
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::string run_per_level = "[";
    for (auto &level : cf_meta.levels)
    {
        run_per_level += std::to_string(level.files.size()) + ", ";
    }
    run_per_level = run_per_level.substr(0, run_per_level.size() - 2) + "]";
    spdlog::info("files_per_level_build : {}", run_per_level);

    std::string size_per_level = "[";
    for (auto &level : cf_meta.levels)
    {
        size_per_level += std::to_string(level.size) + ", ";
    }
    size_per_level = size_per_level.substr(0, size_per_level.size() - 2) + "]";
    spdlog::info("size_per_level_build : {}", size_per_level);

    rocksdb_opt.statistics->Reset();
    rocksdb::get_iostats_context()->Reset();
    rocksdb::get_perf_context()->Reset();
    std::mt19937 engine;
    std::uniform_real_distribution<double> dist(0, 1);
    double p[] = {env.empty_reads, env.non_empty_reads, env.range_reads, env.writes};
    double cumprob[] = {p[0], p[0] + p[1], p[0] + p[1] + p[2], 1.0};
    std::string value, key, limit;
    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);
    rocksdb::Iterator *it = db->NewIterator(rocksdb::ReadOptions());
    auto time_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < env.steps; i++)
    {
        double r = dist(engine);
        int outcome = 0;
        for (int j = 0; j < 4; j++)
        {
            if (r < cumprob[j])
            {
                outcome = j;
                break;
            }
        }
        switch (outcome)
        {
        case 0:
        {
            key = data_gen->gen_new_dup_key();
            key_log->log_key(key);
            status = db->Get(rocksdb::ReadOptions(), key, &value);
            break;
        }
        case 1:
        {
            key = data_gen->gen_existing_key();
            key_log->log_key(key);
            status = db->Get(rocksdb::ReadOptions(), key, &value);
            break;
        }
        case 2:
        {
            key = data_gen->gen_existing_key();
            key_log->log_key(key);
            limit = std::to_string(stoi(key) + 2);
            for (it->Seek(rocksdb::Slice(key)); it->Valid() && it->key().ToString() < limit; it->Next())
            {
                value = it->value().ToString();
            }
            break;
        }
        case 3:
        {
            key_value = data_gen->gen_new_kv_pair(tiered_opt.entry_size);
            key_log->log_key(key_value.first);
            db->Put(write_opt, key_value.first, key_value.second);
            break;
        }
        default:
            break;
        }
    }
    delete it;
    // spdlog::info("Flushing DB...");
    // db->Flush(flush_opt);
    spdlog::info("Waiting for all remaining background compactions to finish");
    if (env.compaction_style == "level")
    {
        while (true)
        {
            db->GetIntProperty(DB::Properties::kNumRunningFlushes,
                               &num_running_flushes);
            db->GetIntProperty(DB::Properties::kMemTableFlushPending,
                               &num_pending_flushes);
            db->GetIntProperty(DB::Properties::kNumRunningCompactions,
                               &num_running_compactions);
            db->GetIntProperty(DB::Properties::kCompactionPending,
                               &num_pending_compactions);
            if (num_running_compactions == 0 && num_pending_compactions == 0 && num_running_flushes == 0 && num_pending_flushes == 0)
                break;
        }
    }
    else
    {
        while (true)
        {
            db->GetIntProperty(DB::Properties::kNumRunningFlushes,
                               &num_running_flushes);
            db->GetIntProperty(DB::Properties::kMemTableFlushPending,
                               &num_pending_flushes);
            if (num_running_flushes == 0 && num_pending_flushes == 0)
                break;
        }
        while (tiered_compactor->compactions_left_count > 0)
            ;
        // while (tiered_compactor->requires_compaction(db))
        // {
        //     while (tiered_compactor->compactions_left_count > 0)
        //         ;
        // }
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

    db->GetColumnFamilyMetaData(&cf_meta);
    run_per_level = "[";
    for (auto &level : cf_meta.levels)
    {
        run_per_level += std::to_string(level.files.size()) + ", ";
    }
    run_per_level = run_per_level.substr(0, run_per_level.size() - 2) + "]";
    spdlog::info("files_per_level : {}", run_per_level);
    rocksdb_opt.statistics->getTickerMap(&stats);

    spdlog::info("(l0, l1, l2plus) : ({}, {}, {})",
                 stats["rocksdb.l0.hit"],
                 stats["rocksdb.l1.hit"],
                 stats["rocksdb.l2andup.hit"]);
    spdlog::info("(bf_true_neg, bf_pos, bf_true_pos) : ({}, {}, {})",
                 stats["rocksdb.bloom.filter.useful"],
                 stats["rocksdb.bloom.filter.full.positive"],
                 stats["rocksdb.bloom.filter.full.true.positive"]);
    spdlog::info("(bytes_written, compact_read, compact_write, flush_write) : ({}, {}, {}, {})",
                 stats["rocksdb.bytes.written"],
                 stats["rocksdb.compact.read.bytes"],
                 stats["rocksdb.compact.write.bytes"],
                 stats["rocksdb.flush.write.bytes"]);
    spdlog::info("(write_io) : ({})", (stats["rocksdb.bytes.written"] +
                                       stats["rocksdb.compact.read.bytes"] +
                                       stats["rocksdb.compact.write.bytes"] +
                                       stats["rocksdb.flush.write.bytes"]) /
                                          4096);
    spdlog::info("(read_io) : ({})", rocksdb::get_perf_context()->block_read_count);
    spdlog::info("(total_latency) : ({})", latency);
    double cache_hit_rate = stats["rocksdb.block.cache.miss"] == 0 ? 0 : double(stats["rocksdb.block.cache.hit"]) / double(stats["rocksdb.block.cache.hit"] + stats["rocksdb.block.cache.miss"]);
    if (cache_hit_rate < 1e-3)
        spdlog::info("(cache_hit_rate) : ({})", 0.0);
    else
        spdlog::info("(cache_hit_rate) : ({})", cache_hit_rate);
    spdlog::info("(cache_hit) : ({})", stats["rocksdb.block.cache.hit"]);
    spdlog::info("(cache_miss) : ({})", stats["rocksdb.block.cache.miss"]);
    db->GetColumnFamilyMetaData(&cf_meta);

    run_per_level = "[";
    for (auto &level : cf_meta.levels)
    {
        run_per_level += std::to_string(level.files.size()) + ", ";
    }
    run_per_level = run_per_level.substr(0, run_per_level.size() - 2) + "]";
    spdlog::info("files_per_level : {}", run_per_level);

    size_per_level = "[";
    for (auto &level : cf_meta.levels)
    {
        size_per_level += std::to_string(level.size) + ", ";
    }
    size_per_level = size_per_level.substr(0, size_per_level.size() - 2) + "]";
    spdlog::info("size_per_level : {}", size_per_level);
    // spdlog::info("size_ratio : {}", rocksdb_opt.max_bytes_for_level_multiplier);

    db->Close();
    delete db;
    delete key_log;
    delete data_gen;
    return EXIT_SUCCESS;
}