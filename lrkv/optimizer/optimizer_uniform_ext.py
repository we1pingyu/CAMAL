import numpy as np
import pandas as pd
import sys
import logging
import os
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
from utils import model_xgb_ext
from utils.distribution import dist_regression

np.set_printoptions(suppress=True)

config_yaml_path = os.path.join("lrkv/config/config.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
scaling = config["lsm_tree_config"]["scaling"]
scaling = 10
E = config["lsm_tree_config"]["E"] / 8
Q = int(config["lsm_tree_config"]["Q"] * scaling)
B = int(4000 / E)
S = 2
M = config["lsm_tree_config"]["M"] * scaling
N = config["lsm_tree_config"]["N"] * scaling
sel = config["lsm_tree_config"]["s"]
workloads = [
    (0.25, 0.25, 0.25, 0.25),
    (0.97, 0.01, 0.01, 0.01),
    (0.01, 0.97, 0.01, 0.01),
    (0.01, 0.01, 0.97, 0.01),
    (0.01, 0.01, 0.01, 0.97),
    (0.49, 0.49, 0.01, 0.01),
    (0.49, 0.01, 0.49, 0.01),
    (0.49, 0.01, 0.01, 0.49),
    (0.01, 0.49, 0.49, 0.01),
    (0.01, 0.49, 0.01, 0.49),
    (0.01, 0.01, 0.49, 0.49),
    (0.33, 0.33, 0.33, 0.01),
    (0.33, 0.33, 0.01, 0.33),
    (0.33, 0.01, 0.33, 0.33),
    (0.01, 0.33, 0.33, 0.33),
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
# workloads = [
#     (0.01, 0.01, 0.01, 0.97),
# ]

dists = ["uniform"]


class Optimizer(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")

    def run(self):
        i = -1
        df = []
        # workloads = [(0.01, 0.01, 0.97, 0.01), (0.01, 0.01, 0.01, 0.97)]
        non_t = []
        non_cache_t = []
        rocksdb_t = []
        lr_t = []
        xgb_t = []
        for workload in workloads:
            i += 1
            # workload = [random.random() for _ in range(4)]
            # workload = [w / sum(workload) for w in workload]
            dist = "uniform"
            skew = 0.0
            # nominal optimizer
            row = self.config["lsm_tree_config"].copy()
            row["optimizer"] = "nominal"
            row["db_name"] = "level_optimizer"
            row["path_db"] = self.config["app"]["DATABASE_PATH"]
            z0, z1, q, w = workload

            cf = CostFunction(
                N,
                1,
                sel / N,
                B,
                E,
                M,
                True,
                z0,
                z1,
                q,
                w,
            )
            nominal = NominalWorkloadTuning(cf)
            nominal_design = nominal.get_nominal_design(is_leveling_policy=True)
            print(f"nominal_optimizer: {nominal_design}")
            row["T"] = int(nominal_design["T"])
            row["N"] = N
            row["queries"] = Q
            row["M"] = M
            row["h"] = nominal_design["M_filt"] / N
            row["dist"] = dist
            row["skew"] = skew
            row["cache_cap"] = 0.0
            row["is_leveling_policy"] = nominal_design["is_leveling_policy"]
            row["mbuf"] = nominal_design["M_buff"] / 8
            row["z0"] = z0
            row["z1"] = z1
            row["q"] = q
            row["w"] = w
            key_path = "key_log_lr_optimizer_w"
            if not os.path.exists(key_path):
                os.makedirs(key_path)
            key_log = key_path + "/{}.dat".format(i)
            row["key_log"] = key_log
            self.logger.info(f"Building DB at size : {N}")
            self.config = config
            db = RocksDB(self.config)
            results = db.run(
                row["db_name"],
                row["path_db"],
                row["h"],
                row["T"],
                N,
                row["E"],
                row["M"],
                z0,
                z1,
                q,
                w,
                dist,
                skew,
                Q,
                sel,
                is_leveling_policy=row["is_leveling_policy"],
                cache_cap=0,
                key_log=key_log,
                scaling=scaling,
            )
            for key, val in results.items():
                self.logger.info(f"{key} : {val}")
                row[f"{key}"] = val
            row["write_io"] = (
                row["bytes_written"]
                + row["compact_read"]
                + row["compact_write"]
                + row["flush_written"]
            ) / 4096
            self.logger.info("write_io: {}".format(row["write_io"]))
            row["read_model_io"] = Q * cf.calculate_read_cost(row["h"], row["T"])
            row["write_model_io"] = Q * cf.calculate_write_cost(row["h"], row["T"])
            row["model_io"] = row["read_model_io"] + row["write_model_io"]
            self.logger.info("mbuf: {}".format(row["mbuf"]))
            self.logger.info("read_model_io: {}".format(row["read_model_io"]))
            self.logger.info("write_model_io: {}".format(row["write_model_io"]))
            self.logger.info("model_io: {}".format(row["model_io"]))
            # print(row)
            df.append(row)
            pd.DataFrame(df).to_csv(self.config["optimizer_path"]["ckpt"])
            non_t.append(row["total_latency"])

            # nominal + default cache optimizer
            row = copy.deepcopy(row)
            row["optimizer"] = "nominal_cache"
            row["db_name"] = "level_optimizer"
            row["path_db"] = self.config["app"]["DATABASE_PATH"]
            z0, z1, q, w = workload
            row["cache_cap"] = 0.125 * M / 8
            row["M"] = M - row["cache_cap"] * 8
            cf = CostFunction(
                N,
                1,
                sel / N,
                B,
                E,
                row["M"],
                True,
                z0,
                z1,
                q,
                w,
            )
            nominal = NominalWorkloadTuning(cf)
            nominal_design = nominal.get_nominal_design(is_leveling_policy=True)
            print(f"nominal_cache_optimizer: {nominal_design}")
            row["T"] = int(nominal_design["T"])
            row["N"] = N
            row["queries"] = Q
            row["h"] = nominal_design["M_filt"] / N
            row["dist"] = dist
            row["skew"] = skew
            row["is_leveling_policy"] = nominal_design["is_leveling_policy"]
            row["mbuf"] = nominal_design["M_buff"] / 8
            row["z0"] = z0
            row["z1"] = z1
            row["q"] = q
            row["w"] = w
            key_path = "key_log_xgb_optimizer"
            if not os.path.exists(key_path):
                os.makedirs(key_path)
            key_log = key_path + "/{}.dat".format(i)
            row["key_log"] = key_log
            self.logger.info(f"Building DB at size : {N}")
            self.config = config
            db = RocksDB(self.config)
            results = db.run(
                row["db_name"],
                row["path_db"],
                row["h"],
                row["T"],
                N,
                row["E"],
                row["M"],
                z0,
                z1,
                q,
                w,
                dist,
                skew,
                Q,
                sel,
                is_leveling_policy=row["is_leveling_policy"],
                cache_cap=row["cache_cap"],
                key_log=key_log,
                scaling=scaling,
            )
            for key, val in results.items():
                self.logger.info(f"{key} : {val}")
                row[f"{key}"] = val
            row["write_io"] = (
                row["bytes_written"]
                + row["compact_read"]
                + row["compact_write"]
                + row["flush_written"]
            ) / 4096
            self.logger.info("write_io: {}".format(row["write_io"]))
            row["read_model_io"] = Q * cf.calculate_read_cost(row["h"], row["T"])
            row["write_model_io"] = Q * cf.calculate_write_cost(row["h"], row["T"])
            row["model_io"] = row["read_model_io"] + row["write_model_io"]
            self.logger.info("mbuf: {}".format(row["mbuf"]))
            self.logger.info("read_model_io: {}".format(row["read_model_io"]))
            self.logger.info("write_model_io: {}".format(row["write_model_io"]))
            self.logger.info("model_io: {}".format(row["model_io"]))
            # print(row)
            df.append(row)
            pd.DataFrame(df).to_csv(self.config["optimizer_path"]["ckpt"])
            non_cache_t.append(row["total_latency"])

            # rocksdb default setting
            row = copy.deepcopy(row)
            row["optimizer"] = "rocksdb"
            row["is_leveling_policy"] = True
            row["T"] = 10
            row["h"] = 10
            best_h = 10
            row["cache_cap"] = 0.33 * (M - best_h * N) / 8
            row["M"] = M - row["cache_cap"] * 8
            self.logger.info(f"Building DB at size : {N}")
            db = RocksDB(self.config)
            results = db.run(
                row["db_name"],
                row["path_db"],
                row["h"],
                row["T"],
                N,
                row["E"],
                row["M"],
                z0,
                z1,
                q,
                w,
                dist,
                skew,
                Q,
                sel,
                is_leveling_policy=row["is_leveling_policy"],
                cache_cap=row["cache_cap"],
                key_log=key_log,
                scaling=scaling,
            )
            for key, val in results.items():
                self.logger.info(f"{key} : {val}")
                row[f"{key}"] = val
            row["write_io"] = (
                row["bytes_written"]
                + row["compact_read"]
                + row["compact_write"]
                + row["flush_written"]
            ) / 4096
            self.logger.info("write_io: {}".format(row["write_io"]))
            self.logger.info("mbuf: {}".format(row["mbuf"]))
            # print(row)
            df.append(row)
            pd.DataFrame(df).to_csv(self.config["optimizer_path"]["ckpt"])
            rocksdb_t.append(row["total_latency"])

            # learned lr optimizer
            # row = copy.deepcopy(row)
            # row["optimizer"] = "lr"
            # level_cost_models = pkl.load(
            #     open(self.config["lr_model"]["level_lr_cost_model"], "rb")
            # )
            # level_cache_models = pkl.load(
            #     open(self.config["lr_model"]["level_lr_cache_model"], "rb")
            # )
            # tier_cost_models = pkl.load(
            #     open(self.config["lr_model"]["tier_lr_cost_model"], "rb")
            # )
            # tier_cache_models = pkl.load(
            #     open(self.config["lr_model"]["tier_lr_cache_model"], "rb")
            # )
            # (
            #     best_T,
            #     best_h,
            #     best_ratio,
            #     best_var,
            #     best_cost,
            # ) = model_lr.traverse_var_optimizer_uniform(
            #     level_cache_models,
            #     level_cost_models,
            #     z0,
            #     z1,
            #     q,
            #     w,
            #     "level",
            #     E,
            #     M / scaling,
            #     N / scaling,
            # )
            # row["is_leveling_policy"] = True
            # (
            #     tier_best_T,
            #     tier_best_h,
            #     tier_best_ratio,
            #     tier_best_var,
            #     tier_best_cost,
            # ) = model_lr.traverse_var_optimizer_uniform(
            #     tier_cache_models,
            #     tier_cost_models,
            #     z0,
            #     z1,
            #     q,
            #     w,
            #     "tier",
            #     E,
            #     M / scaling,
            #     N / scaling,
            # )
            # print(
            #     f"level_optimizer: best_T: {best_T}, best_h: {best_h}, best_ratio: {best_ratio},best_var: {best_var}, best_cost:{best_cost*Q}"
            # )
            # print(
            #     f"tier_optimizer: best_T: {tier_best_T}, best_h: {tier_best_h}, best_ratio: {tier_best_ratio}, best_var: {tier_best_var}, best_cost:{tier_best_cost*Q}"
            # )
            # policy = row["is_leveling_policy"]
            # print(
            #     f"lr_optimizer: best_T: {best_T}, best_h: {best_h}, best_ratio: {best_ratio}, is_leveling_policy: {policy}, best_cost:{best_cost*Q}"
            # )
            # row["T"] = int(best_T)
            # row["h"] = best_h * best_ratio
            # row["M"] = best_ratio * M
            # row["mbuf"] = best_ratio * (M - best_h * N) / 8
            # row["cache_cap"] = (1 - best_ratio) * M / 8
            # self.logger.info(f"Building DB at size : {N}")
            # db = RocksDB(self.config)
            # results = db.run(
            #     row["db_name"],
            #     row["path_db"],
            #     row["h"],
            #     row["T"],
            #     row["N"],
            #     row["E"],
            #     row["M"],
            #     z0,
            #     z1,
            #     q,
            #     w,
            #     dist,
            #     skew,
            #     Q,
            #     sel,
            #     is_leveling_policy=row["is_leveling_policy"],
            #     cache_cap=row["cache_cap"],
            #     key_log=key_log,
            #     scaling=scaling,
            # )
            # for key, val in results.items():
            #     self.logger.info(f"{key} : {val}")
            #     row[f"{key}"] = val
            # row["write_io"] = (
            #     row["bytes_written"]
            #     + row["compact_read"]
            #     + row["compact_write"]
            #     + row["flush_written"]
            # ) / 4096
            # self.logger.info("write_io: {}".format(row["write_io"]))
            # self.logger.info("mbuf: {}".format(row["mbuf"]))
            # # print(row)
            # df.append(row)
            # pd.DataFrame(df).to_csv(self.config["optimizer_path"]["ckpt"])
            # lr_t.append(row["total_latency"])

            # learned xgb optimizer
            row = copy.deepcopy(row)
            row["optimizer"] = "xgb"
            cost_models = pkl.load(
                open(self.config["xgb_model"]["ext_xgb_cost_model"], "rb")
            )

            (
                best_T,
                best_K,
                best_h,
                best_ratio,
                best_fs,
                best_var,
                best_cost,
            ) = model_xgb_ext.traverse_var_optimizer_uniform(
                cost_models,
                z0,
                z1,
                q,
                w,
                E,
                M / scaling,
                N / scaling,
            )
            print(
                f"xgb_optimizer: best_T: {best_T}, best_K: {best_K}, best_h: {best_h}, best_ratio: {best_ratio}, best_fs: {best_fs}, best_cost:{best_cost*Q}"
            )
            row["T"] = int(best_T)
            row["K"] = int(best_K)
            row["h"] = best_h * best_ratio
            row["M"] = best_ratio * M
            row["mbuf"] = best_ratio * (M - best_h * N) / 8
            row["cache_cap"] = (1 - best_ratio) * M / 8
            self.logger.info(f"Building DB at size : {N}")
            db = RocksDB(self.config)
            results = db.run(
                row["db_name"],
                row["path_db"],
                row["h"],
                row["T"],
                row["N"],
                row["E"],
                row["M"],
                z0,
                z1,
                q,
                w,
                dist,
                skew,
                Q,
                sel,
                K=best_K,
                f=best_fs,
                is_leveling_policy=row["is_leveling_policy"],
                auto_compaction=True,
                cache_cap=row["cache_cap"],
                key_log=key_log,
                scaling=scaling,
            )
            for key, val in results.items():
                self.logger.info(f"{key} : {val}")
                row[f"{key}"] = val
            row["write_io"] = (
                row["bytes_written"]
                + row["compact_read"]
                + row["compact_write"]
                + row["flush_written"]
            ) / 4096
            self.logger.info("write_io: {}".format(row["write_io"]))
            self.logger.info("mbuf: {}".format(row["mbuf"]))
            # print(row)
            df.append(row)
            pd.DataFrame(df).to_csv(self.config["optimizer_path"]["ckpt"])
            xgb_t.append(row["total_latency"])
            print("non_t: ", np.mean(non_t))
            print("non_cache_t: ", np.mean(non_cache_t))
            print("rocksdb_t: ", np.mean(rocksdb_t))
            # print("lr_t: ", np.mean(lr_t))
            print("xgb_t: ", np.mean(xgb_t))
        self.logger.info("Exporting data from lr optimizer")
        pd.DataFrame(df).to_csv(self.config["optimizer_path"]["final"])
        self.logger.info("Finished optimizer\n")


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
