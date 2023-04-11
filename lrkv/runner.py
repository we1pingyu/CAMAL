"""
This class implements the driver program for the robust-lsm-trees
project
"""

import os
import logging
import sys
import yaml

# from jobs.create_workload_uncertainty_tunings import CreateWorkloadUncertaintyTunings
# from jobs.create_workload_nominal_tunings import CreateNominalWorkloadTunings
# from jobs.sample_uncertain_workloads import SampleUncertainWorkloads
# from jobs.run_experiments import ExperimentDriver


class Runner(object):
    """
    This class implements the driver program for the robust-lsm-trees
    project.
    """

    def __init__(self, config):
        """
        Constructor
        :param config_:
        """

        self.config = config
        # Initialize the logging
        if self.config['app']['app_logging_level'] == 'DEBUG':
            logging_level = logging.DEBUG
        elif self.config['app']['app_logging_level'] == 'INFO':
            logging_level = logging.INFO
        else:
            logging_level = logging.INFO

        logging.basicConfig(
            format="LOG: %(asctime)-15s:[%(filename)s]: %(message)s",
            datefmt='%m/%d/%Y %I:%M:%S %p',
        )

        self.logger = logging.getLogger("rlt_logger")
        self.logger.setLevel(logging_level)

    def run(self, work):
        self.logger.info("Starting app: {}".format(self.config['app']['app_name']))
        work.run()
        self.logger.info("Finished")
