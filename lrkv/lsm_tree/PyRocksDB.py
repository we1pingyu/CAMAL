"""
Python API for RocksDB
"""
import logging
import os
import re
import shutil
import subprocess
import numpy as np

THREADS = 28


class RocksDB(object):
    """
    Python API for RocksDB
    """

    def __init__(self, config):
        """
        Constructor

        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
        self.level_hit_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(l0, l1, l2plus\) : '
            r'\((-?\d+), (-?\d+), (-?\d+)\)'
        )
        self.bf_count_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(bf_true_neg, bf_pos, bf_true_pos\) : '
            r'\((-?\d+), (-?\d+), (-?\d+)\)'
        )
        self.compaction_bytes_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(bytes_written, compact_read, compact_write, flush_write\) : '
            r'\((-?\d+), (-?\d+), (-?\d+), (-?\d+)\)'
        )
        self.read_io_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(read_io\) : ' r'\((-?\d+)\)'
        )
        self.files_per_level_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] files_per_level : ' r'(\[[0-9,\s]+\])'
        )
        self.size_per_level_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] size_per_level : ' r'(\[[0-9,\s]+\])'
        )
        self.total_latency_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(total_latency\) : ' r'\((-?\d+)\)'
        )
        self.cache_hit_rate_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(cache_hit_rate\) : ' r'\((\d+(\.\d+)?)\)'
        )
        self.cache_hit_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(cache_hit\) : ' r'\((-?\d+)\)'
        )
        self.cache_miss_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(cache_miss\) : ' r'\((-?\d+)\)'
        )
        self.init_time_prog = re.compile(
            r'\[[0-9:.]+\]\[info\] \(init_time\) : ' r'\((-?\d+)\)'
        )

    def options_from_config(self):
        db_settings = {}
        db_settings['path_db'] = self.config['app']['DATABASE_PATH']
        db_settings['N'] = self.config['lsm_tree_config']['N']
        db_settings['B'] = self.config['lsm_tree_config']['B']
        db_settings['E'] = self.config['lsm_tree_config']['E']
        db_settings['M'] = self.config['lsm_tree_config']['M']
        db_settings['P'] = self.config['lsm_tree_config']['P']
        db_settings['is_leveling_policy'] = self.config['lsm_tree_config'][
            'is_leveling_policy'
        ]

        # Defaults
        db_settings['db_name'] = 'default'
        db_settings['h'] = 5
        db_settings['T'] = 10

        return db_settings

    def estimate_levels(self):
        mbuff = self.M - (self.h * self.N)
        l = np.ceil((np.log((self.N * self.E) / mbuff) + 1) / np.log(self.T))

        return l

    def run(
        self,
        db_name,
        path_db,
        h,
        T,
        N,
        E,
        M,
        num_z0,
        num_z1,
        num_q,
        num_w,
        dist,
        skew,
        steps,
        is_leveling_policy=True,
        cache_cap=0,
        key_log='',
    ):
        """
        Runs a set of queries on the database

        :param num_z0: empty reads
        :param num_z1: non-empty reads
        :param num_w: writes
        """
        self.path_db = path_db
        self.db_name = db_name
        self.h, self.T = h, int(np.ceil(T))
        self.N, self.M = int(N), int(M)
        self.E = E >> 3  # Converts bits -> bytes
        if is_leveling_policy:
            self.compaction_style = 'level'
        else:
            self.compaction_style = 'tier'
        os.makedirs(os.path.join(self.path_db, self.db_name), exist_ok=True)
        mbuff = int(self.M - (self.h * self.N)) >> 3
        db_dir = os.path.join(self.path_db, self.db_name)
        cmd = [
            self.config['app']['EXECUTION_PATH'],
            db_dir,
            f'-N {self.N}',
            f'-T {self.T}',
            f'-B {mbuff}',
            f'-E {self.E}',
            f'-b {self.h}',
            f'-e {num_z0}',
            f'-r {num_z1}',
            f'-q {num_q}',
            f'-w {num_w}',
            f'-s {steps}',
            f'-c {self.compaction_style}',
            f'--parallelism {THREADS}',
            f'--dist {dist}',
            f'--skew {skew}',
            f'--cache {cache_cap}',
            f'--key-log-file {key_log}',
        ]

        cmd = ' '.join(cmd)
        self.logger.debug(f'{cmd}')
        self.logger.info(f'{cmd}')

        proc = subprocess.Popen(
            cmd,
            # stdin=None,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
        )

        results = {}

        try:
            timeout = 10 * 60 * 60
            proc_results, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.logger.warn('Timeout limit reached. Aborting')
            proc.kill()
            results['l0_hit'] = 0
            results['l1_hit'] = 0
            results['l2_plus_hit'] = 0
            results['filter_neg'] = 0
            results['filter_pos'] = 0
            results['filter_pos_true'] = 0
            results['bytes_written'] = 0
            results['compact_read'] = 0
            results['compact_write'] = 0
            results['flush_written'] = 0
            results['read_io'] = 0
            results['files_per_level'] = 0
            results['size_per_level'] = 0
            results['total_latency'] = 0
            results['cache_hit_rate'] = 0
            results['cache_hit'] = 0
            results['cache_miss'] = 0
            results['init_time'] = 0
            return results
        try:
            level_hit_results = [int(result) for result in self.level_hit_prog.search(proc_results).groups()]  # type: ignore
            bf_count_results = [int(result) for result in self.bf_count_prog.search(proc_results).groups()]  # type: ignore
            compaction_results = [int(result) for result in self.compaction_bytes_prog.search(proc_results).groups()]  # type: ignore
            read_io_result = [int(result) for result in self.read_io_prog.search(proc_results).groups()]  # type: ignore
            files_per_level = self.files_per_level_prog.findall(proc_results)[0]
            size_per_level = self.size_per_level_prog.findall(proc_results)[0]
            total_latency_result = [
                int(result)
                for result in self.total_latency_prog.search(proc_results).groups()
            ]

            cache_hit_rate_result = [
                float(result)
                for result in self.cache_hit_rate_prog.search(proc_results).groups()
            ]
            cache_hit_result = [
                int(result)
                for result in self.cache_hit_prog.search(proc_results).groups()
            ]
            cache_miss_result = [
                int(result)
                for result in self.cache_miss_prog.search(proc_results).groups()
            ]
            init_time_result = [
                int(result)
                for result in self.init_time_prog.search(proc_results).groups()
            ]
            results['l0_hit'] = level_hit_results[0]
            results['l1_hit'] = level_hit_results[1]
            results['l2_plus_hit'] = level_hit_results[2]

            results['filter_neg'] = bf_count_results[0]
            results['filter_pos'] = bf_count_results[1]
            results['filter_pos_true'] = bf_count_results[2]

            results['bytes_written'] = compaction_results[0]
            results['compact_read'] = compaction_results[1]
            results['compact_write'] = compaction_results[2]
            results['flush_written'] = compaction_results[3]

            results['read_io'] = read_io_result[0]

            results['files_per_level'] = files_per_level.strip()
            results['size_per_level'] = size_per_level.strip()

            results['total_latency'] = total_latency_result[0]
            results['cache_hit_rate'] = cache_hit_rate_result[0]
            results['cache_hit'] = cache_hit_result[0]
            results['cache_miss'] = cache_miss_result[0]

            results['init_time'] = init_time_result[0]
            return results
        except:
            self.logger.warn('Log errors')
            proc.kill()
            results['l0_hit'] = 0
            results['l1_hit'] = 0
            results['l2_plus_hit'] = 0
            results['z0_ms'] = 0
            results['z1_ms'] = 0
            results['q_ms'] = 0
            results['w_ms'] = 0
            results['filter_neg'] = 0
            results['filter_pos'] = 0
            results['filter_pos_true'] = 0
            results['bytes_written'] = 0
            results['compact_read'] = 0
            results['compact_write'] = 0
            results['flush_written'] = 0
            results['read_io'] = 0
            results['files_per_level'] = 0
            results['size_per_level'] = 0
            results['total_latency'] = 0
            results['cache_hit_rate'] = 0
            results['cache_hit'] = 0
            results['cache_miss'] = 0
            results['init_time'] = 0
            return results
