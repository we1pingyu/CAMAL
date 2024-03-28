import numpy as np
import pandas as pd
import subprocess
import sys
import logging
import os
import re
import yaml
import pickle as pkl
from multiprocessing import Process

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
scaling = 1
E = config["lsm_tree_config"]["E"] / 8
Q = int(config["lsm_tree_config"]["Q"] * scaling)
B = int(4000 / E)
S = 2
M = int(config["lsm_tree_config"]["M"] * scaling)
N = int(config["lsm_tree_config"]["N"] * scaling)
sel = config["lsm_tree_config"]["s"]
workloads = [
    (0.91, 0.03, 0.03, 0.03),
    (0.75, 0.15, 0.05, 0.05),
    (0.60, 0.30, 0.05, 0.05),
    (0.45, 0.45, 0.05, 0.05),
    (0.30, 0.60, 0.05, 0.05),
    (0.15, 0.75, 0.05, 0.05),
    (0.03, 0.91, 0.03, 0.03),
    (0.05, 0.75, 0.15, 0.05),
    (0.05, 0.60, 0.30, 0.05),
    (0.05, 0.45, 0.45, 0.05),
    (0.05, 0.30, 0.60, 0.05),
    (0.05, 0.15, 0.75, 0.05),
    (0.03, 0.03, 0.91, 0.03),
    (0.05, 0.05, 0.75, 0.15),
    (0.05, 0.05, 0.60, 0.30),
    (0.05, 0.05, 0.45, 0.45),
    (0.05, 0.05, 0.30, 0.60),
    (0.05, 0.05, 0.15, 0.75),
    (0.03, 0.03, 0.03, 0.91),
    (0.15, 0.05, 0.05, 0.75),
    (0.30, 0.05, 0.05, 0.60),
    (0.45, 0.05, 0.05, 0.45),
    (0.60, 0.05, 0.05, 0.30),
    (0.75, 0.05, 0.05, 0.15),
]

test_workloads_filename = "test_workloads.in"
if os.path.exists(test_workloads_filename):
    os.remove(test_workloads_filename)

dists = ["uniform"]

def model_server(model="xgb"):
    if model == "xgb":
        level_cost_models = pkl.load(
            open(config["xgb_model"]["level_xgb_cost_model"], "rb")
        )
        while True:
            if not os.path.exists('workloads.in'):
                continue

            f_in = open('workloads.in', "r")
            workload = f_in.readline().strip().split(' ')
            z0, z1, q, w = [float(x) for x in workload]
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
            optimal_params_file = open("optimal_params.in", "w")
            optimal_params_file.write(f"{best_T} {best_h} {best_ratio}\n")
            optimal_params_file.close()
            os.remove('workloads.in')
    else:        
        level_cost_models = pkl.load(
            open(config["lr_model"]["level_lr_cost_model"], "rb")
        )
        level_cache_models = pkl.load(
            open(config["lr_model"]["level_lr_cache_model"], "rb")
        )

        while True:
            if not os.path.exists('workloads.in'):
                continue

            f_in = open('workloads.in', "r")
            workload = f_in.readline().strip().split(' ')
            z0, z1, q, w = [float(x) for x in workload]
            (
                best_T,
                best_h,
                best_ratio,
                best_var,
                best_cost,
            ) = model_lr.traverse_var_optimizer_uniform(
                level_cache_models,
                level_cost_models,
                z0,
                z1,
                q,
                w,
                "level",
                E,
                M / scaling,
                N / scaling,
            )
            optimal_params_file = open("optimal_params.in", "w")
            optimal_params_file.write(f"{best_T} {best_h} {best_ratio}\n")
            optimal_params_file.close()
            os.remove('workloads.in')


class Optimizer(object):
    def __init__(self, config):
        self.db_id = 0
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
        self.latency_per_workload_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] latency_per_workload : " r"(\[[0-9,\s]+\])"
        )

    def generate_workloads(self, workload):
        test_workloads_file = open(test_workloads_filename, "a")
        z0, z1, q, w = workload
        test_workloads_file.write(f"{z0} {z1} {q} {w}\n")
        return [z0, z1, q, w]


    def start_db_runner(self, tuning_T, tuning_h, default_config, w=10000, r=0.05):
        server = Process(target=model_server, args=("xgb",))
        server.start()

        cmd = [
            "build/db_runner_dynamic",
            f"/tmp/level_test_{self.db_id}",
            f"-N {N}",
            f"-M {M}",
            f"-E 1000",
            f"-s {Q}",
            f"-w {w}",
            f"-r {r}",
            f"--sel {sel}",
            f"--scaling {scaling}",
            f"--parallelism 1",
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
        try:
            latency_per_workload = self.latency_per_workload_prog.findall(proc_results)[
                0
            ]
            latency_per_workload = latency_per_workload.strip()
            results = latency_per_workload.strip("][").split(", ")
            results = [int(r) for r in results]
        except:
            self.logger.warn("Log errors")
            proc.kill()

        server.terminate()
        return results

    def run(self):
        cases = []
        for workload in workloads:
            case = self.generate_workloads(workload)
            cases.append(case)

        latency_ht = self.start_db_runner(
            tuning_T=True, tuning_h=True, default_config=False, 
            w=10000, r=0.05
        )
        latency_t = self.start_db_runner(
            tuning_T=True, tuning_h=False, default_config=False,
            w=10000, r=0.05
        )
        # latency_h = self.start_db_runner(
        #     tuning_T=False, tuning_h=True, default_config=False
        # )
        latency_default = self.start_db_runner(
            tuning_T=False, tuning_h=False, default_config=True
        )

        print("latency_ht: ", latency_ht)
        print("latency_t: ", latency_t)
        # print("latency_h: ", latency_h)
        print("latency_default: ", latency_default)

        df = []
        for i in range(len(cases) - 1):
            row = {}
            (
                row["source_z0"],
                row["source_z1"],
                row["source_q"],
                row["source_w"],
            ) = cases[i]
            (
                row["target_z0"],
                row["target_z1"],
                row["target_q"],
                row["target_w"],
            ) = cases[i + 1]
            row["N"] = N
            row["M"] = M
            row["s"] = Q
            # row["T"], row["h"], row["ratio"] = cases[i + 1][4:]
            row["latency_tuning_ht"] = latency_ht[i]
            # row["latency_tuning_T"] = latency_t[i]
            # row["latency_tuning_h"] = latency_h[i]
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
