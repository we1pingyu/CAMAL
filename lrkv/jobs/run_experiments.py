"""
This class runs the experiments
"""
import logging
from data_collection.level_cache import LevelCache
from data_collection.level_cost import LevelCost


class ExperimentDriver(object):
    """
    Creates experiment driver
    """

    def __init__(self, config):
        """
        Constructor
        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("rlt_logger")

    def run(self):
        self.logger.info("Starting experiments")

        # Get experiment list
        expt_list = self.config['experiments']['expt_list']

        # Run experiments
        for expt_name in expt_list:
            if expt_name == "LevelCache":
                expt = LevelCache(self.config)
                expt.run()
            if expt_name == "LevelCost":
                expt = LevelCost(self.config)
                expt.run()

        self.logger.info("Finished experiments")
