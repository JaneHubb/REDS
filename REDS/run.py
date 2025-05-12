import time
import sys
import tqdm
import logging
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch import nn
from torch.nn.parallel import DataParallel
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from model import REDS
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

if __name__ == '__main__':

    config = Config(model=REDS, config_file_list=['config.yaml'])
    config['seed'] = random.randint(0,2**20-1)
    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # calculate global item freqeuncy 
    item_freq = dataset.inter_feat['item_id'].value_counts()
    item_freq[0] = 0
    sorted_freq = torch.tensor(item_freq.sort_index())
    
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # load model 
    model = REDS(config, train_data.dataset, sorted_freq).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)
    
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )
    
    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
