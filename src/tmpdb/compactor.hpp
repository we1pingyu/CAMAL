#ifndef COMPACTOR_H_
#define COMPACTOR_H_

#include <cmath>
#include <set>
#include <mutex>
#include <vector>
#include <atomic>

#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/listener.h"

#include "spdlog/spdlog.h"
#include "tmpdb/compactor_options.hpp"

namespace tmpdb
{

    class BaseCompactor;

    typedef struct CompactionTask
    {
        rocksdb::DB *db;
        BaseCompactor *compactor;
        const std::string &column_family_name;
        std::vector<std::string> input_file_names;
        int output_level;
        rocksdb::CompactionOptions compact_options;
        size_t origin_level_id;
        bool retry_on_fail;
        bool is_a_retry;

        /**
         * @brief Construct a new Compaction Task object
         *
         * @param db
         * @param compactor
         * @param column_family_name
         * @param input_file_names
         * @param output_level
         * @param compact_options
         * @param origin_level_id
         * @param retry_on_fail
         * @param is_a_retry
         */
        CompactionTask(
            rocksdb::DB *db, BaseCompactor *compactor,
            const std::string &column_family_name,
            const std::vector<std::string> &input_file_names,
            const int output_level,
            const rocksdb::CompactionOptions &compact_options,
            const size_t origin_level_id,
            bool retry_on_fail,
            bool is_a_retry)
            : db(db),
              compactor(compactor),
              column_family_name(column_family_name),
              input_file_names(input_file_names),
              output_level(output_level),
              compact_options(compact_options),
              origin_level_id(origin_level_id),
              retry_on_fail(retry_on_fail),
              is_a_retry(is_a_retry) {}
    } CompactionTask;

    class BaseCompactor : public ROCKSDB_NAMESPACE::EventListener
    {
    public:
        CompactorOptions compactor_opt;
        rocksdb::Options rocksdb_opt;
        rocksdb::CompactionOptions rocksdb_compact_opt;
        std::vector<bool> level_being_compacted;

        /**
         * @brief Construct a new Compactor object
         *
         * @param compactor_opt
         * @param rocksdb_opt
         */
        BaseCompactor(const CompactorOptions compactor_opt, const rocksdb::Options rocksdb_opt);

        /**
         * @brief Picks and returns a compaction task given the specified DB and column family.
         * It is the caller's responsibility to destroy the returned CompactionTask.
         *
         * @param db An open database
         * @param cf_name Names of the column families
         * @param level Target level id
         *
         * @returns CompactionTask Will return a "nullptr" if it cannot find a proper compaction task.
         */
        virtual CompactionTask *PickCompaction(rocksdb::DB *db, const std::string &cf_name, const size_t level) = 0;

        /**
         * @brief Picks and returns a compaction task given the specified DB and column family.
         * It is the caller's responsibility to destroy the returned CompactionTask.
         *
         * @param db An open database
         * @param cf_name Names of the column families
         * @param level Target level id
         *
         * @returns CompactionTask Will return a "nullptr" if it cannot find a proper compaction task.
         */
        virtual CompactionTask *PickLevelCompaction(rocksdb::DB *db, const std::string &cf_name, const size_t level) = 0;
        /**
         * @brief Schedule and run the specified compaction task in background.
         *
         * @param task
         */
        virtual void ScheduleCompaction(CompactionTask *task) = 0;
    };

    class Compactor : public BaseCompactor
    {
    public:
        std::mutex compactions_left_mutex;
        std::mutex meta_data_mutex;
        std::atomic<int> compactions_left_count;

        /**
         * @brief Construct a new Compactor object
         *
         * @param compactor_opt
         * @param rocksdb_opt
         */
        Compactor(const CompactorOptions compactor_opt, const rocksdb::Options rocksdb_opt)
            : BaseCompactor(compactor_opt, rocksdb_opt), compactions_left_count(0){};

        /**
         * @brief
         *
         * @param db
         * @return int
         */
        int largest_occupied_level(rocksdb::DB *db) const;

        /// @brief
        /// @param db
        /// @param compactor_opt
        /// @param num_entries
        /// @return
        std::vector<size_t> calculate_level_capacity(CompactorOptions compactor_opt);

        /**
         * @brief
         *
         * @param db
         * @param cf_name
         * @param level
         * @return CompactionTask*
         */
        CompactionTask *PickCompaction(rocksdb::DB *db, const std::string &cf_name, const size_t level) override;
        /**
         * @brief
         *
         * @param db
         * @param cf_name
         * @param level
         * @return CompactionTask*
         */
        CompactionTask *PickLevelCompaction(rocksdb::DB *db, const std::string &cf_name, const size_t level) override;
        /**
         * @brief
         *
         * @param db
         * @param info
         */
        void OnFlushCompleted(rocksdb::DB *db, const ROCKSDB_NAMESPACE::FlushJobInfo &info) override;

        /**
         * @brief
         *
         * @param arg
         */
        static void CompactFiles(void *arg);

        /**
         * @brief
         *
         * @param task
         */
        void ScheduleCompaction(CompactionTask *task) override;

        bool requires_compaction(rocksdb::DB *db);

        /**
         * @brief Estimates the number of levels needed based on
         *
         * @param N Total number of entries
         * @param T Size ratio
         * @param E Entry size
         * @param B Buffer size
         * @return size_t Number of levels
         */
        static size_t estimate_levels(size_t N, double T, size_t E, size_t B);

        /**
         * @brief Calculates the nubmer of elements assuming a tree with the
         *        respective parameters is full.
         *
         * @param T size ratio
         * @param E entry size
         * @param B buffer size
         * @param L number of levels
         */
        static size_t calculate_full_tree(double T, size_t E, size_t B, size_t L);
        void updateT(int T);
        void updateM(size_t M);
    };

} /* namespace tmpdb */

#endif /* COMPACTOR_H_ */