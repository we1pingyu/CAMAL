import numpy as np
import pandas as pd
import subprocess
import sys
import logging
import os
import re
import yaml
import copy
import random
import pickle as pkl

sys.path.append("./lrkv")
from runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import CostFunction
from lsm_tree.tunner import NominalWorkloadTuning
from utils import model_lr
from utils import model_xgb
from utils.distribution import dist_regression

np.set_printoptions(suppress=True)

config_yaml_path = os.path.join("lrkv/config/config.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
scaling = config["lsm_tree_config"]["scaling"]
scaling = 10
E = config["lsm_tree_config"]["E"] / 8
Q = int(config["lsm_tree_config"]["Q"])
B = int(4000 / E)
S = 2
M = int(config["lsm_tree_config"]["M"] * scaling)
N = int(config["lsm_tree_config"]["N"] * scaling)
sel = config["lsm_tree_config"]["s"]
workloads = [
    (0.01, 0.01, 0.01, 0.97),
    (0.05, 0.05, 0.15, 0.75),
    (0.45, 0.05, 0.05, 0.45),
    (0.33, 0.33, 0.01, 0.33),
    (0.75, 0.05, 0.05, 0.15),
    (0.45, 0.05, 0.05, 0.45),
    (0.05, 0.05, 0.15, 0.75),
]

dists = ["uniform"]

class Optimizer(object):
    def __init__(self, config):
        self.db_id = 0
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
        self.latency_per_workload_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] latency_per_workload : " r"(\[[0-9,\s]+\])"
        )

    def get_optimal_param(self):
        test_cases = []
        for workload in workloads:
            dist = "uniform"
            skew = 0.0
            row = self.config["lsm_tree_config"].copy()
            row["db_name"] = "level_optimizer"
            row["path_db"] = self.config["app"]["DATABASE_PATH"]
            z0, z1, q, w = workload
            row["N"] = N
            row["queries"] = Q
            row["dist"] = dist
            row["skew"] = skew
            row["z0"] = z0
            row["z1"] = z1
            row["q"] = q
            row["w"] = w

            # learned xgb optimizer
            row = copy.deepcopy(row)
            row["optimizer"] = "xgb"
            level_cost_models = pkl.load(
                open(self.config["xgb_model"]["level_xgb_cost_model"], "rb")
            )
            (
                best_T,
                best_h,
                best_ratio,
                best_var,
                best_cost,
            ) = model_xgb.traverse_var_optimizer_uniform(
                level_cost_models,
                1,
                z0,
                z1,
                q,
                w,
                E,
                M / scaling,
                N / scaling,
            )
            test_cases.append([z0, z1, q, w, best_T, best_h, best_ratio])

        return test_cases
    
    def gen_test_workloads(self, cases):
        workloads_file = open("workloads.in", "w")
        for case in cases:
            z0, z1, q, w, best_T, best_h, best_ratio = case
            workloads_file.write(f"{z0} {z1} {q} {w} {best_T} {best_h} {best_ratio}\n")

    def start_db_runner(self, tuning_T, tuning_h, default_config):
        cmd = [
            "build/db_runner_dynamic",
            f"/tmp/level_test_{self.db_id}",
            f"-N {N}",
            f"-M {M}",
            f"-E 1000",
            f"-s {Q}",
            f"--sel {sel}",
            f"--scaling {scaling}",
            f"--parallelism 28",
            f"--dist uniform",
            f"--skew 0.0",
            f"--cache 0.0",
            f"--key-log-file optimizer_data/{self.db_id}.dat",
        ]
        if tuning_T:
            cmd.append("--tuning-T")
        if tuning_h:
            cmd.append("--tuning-h")
        if default_config:
            cmd.append("--default-config")
        cmd = " ".join(cmd)
        self.db_id += 1
        self.logger.debug(f"{cmd}")
        self.logger.info(f"{cmd}")
        proc = subprocess.Popen(
            cmd,
            # stdin=None,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
        )
        results = []

        try:
            timeout = 10 * 60 * 60
            proc_results, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.logger.warn("Timeout limit reached. Aborting")
            proc.kill()
            return results
        try:
            latency_per_workload = self.latency_per_workload_prog.findall(proc_results)[0]
            results = latency_per_workload.strip()
            return results
        except:
            self.logger.warn("Log errors")
            proc.kill()
            return results


    def run(self):
        cases = self.get_optimal_param()
        self.gen_test_workloads(cases)

        latency_ht = self.start_db_runner(tuning_T=True, tuning_h=True, default_config=False)
        latency_t = self.start_db_runner(tuning_T=True, tuning_h=False, default_config=False)
        latency_h = self.start_db_runner(tuning_T=False, tuning_h=True, default_config=False)
        latency_default = self.start_db_runner(tuning_T=False, tuning_h=False, default_config=True)

        print("latency_ht: ", latency_ht)
        print("latency_t: ", latency_t)
        print("latency_h: ", latency_h)
        print("latency_default: ", latency_default)

        df = []
        for i in range(len(cases) - 1):
            row = {}
            row["source_z0"], row["source_z1"], row["source_q"], row["source_w"] = cases[i][: 4]
            row["target_z0"], row["target_z1"], row["target_q"], row["target_w"] = cases[i + 1][: 4]
            row["N"] = N
            row["M"] = M
            row["s"] = Q
            row["T"], row["h"], row["ratio"] = cases[i + 1][4: ]
            row["latency_tuning_ht"] = latency_ht[i]
            row["latency_tuning_T"] = latency_t[i]
            row["latency_tuning_h"] = latency_h[i]
            row["latency_rocksdb"] = latency_default[i]
            df.append(row)

        pd.DataFrame(df).to_csv("optimizer_data/dynamic_tuning_results.csv")

if __name__ == "__main__":
    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join("lrkv/config/config.yaml")

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = Runner(config)
    driver.run(Optimizer(config))
