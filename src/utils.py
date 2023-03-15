import os
import torch
import random
import numpy as np
import pandas as pd
import math 
import time
import os
from tqdm.auto import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedGroupKFold



def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



def create_folds(train, cfg):
    kfold = StratifiedGroupKFold(n_splits = cfg.dataset.num_folds, shuffle = True, random_state = cfg.environment.seed)
    for num, (train_index, val_index) in enumerate(kfold.split(train, train['target'], train['topics_ids'])):
        train.loc[val_index, 'fold'] = int(num)
    train['fold'] = train['fold'].astype(int) 
    return train

# =========================================================================================
# Get max length
# =========================================================================================
def get_max_length(train, cfg):
    lengths = []
    for text in tqdm(train['text'].fillna("").values, total = len(train)):
        length = len(cfg.tokenizer(text, add_special_tokens = False)['input_ids'])
        lengths.append(length)
    return max(lengths) + 2 # cls & sep
    

# =========================================================================================
# F2 score metric
# =========================================================================================
# =========================================================================================
# Get best threshold
# =========================================================================================
def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0
    best_threshold = None
    for thres in np.arange(0.001, 0.1, 0.001):
        x_val['predictions'] = np.where(val_predictions > thres, 1, 0)
        x_val1 = x_val[x_val['predictions'] == 1]
        x_val1 = x_val1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
        x_val1['content_ids'] = x_val1['content_ids'].apply(lambda x: ' '.join(x))
        x_val1.columns = ['topic_id', 'predictions']
        x_val0 = pd.Series(x_val['topics_ids'].unique())
        x_val0 = x_val0[~x_val0.isin(x_val1['topic_id'])]
        x_val0 = pd.DataFrame({'topic_id': x_val0.values, 'predictions': ""})
        x_val_r = pd.concat([x_val1, x_val0], axis = 0, ignore_index = True)
        x_val_r = x_val_r.merge(correlations, how = 'left', on = 'topic_id')
        score = f2_score(x_val_r['content_ids'], x_val_r['predictions'])
        if score > best_score:
            best_score = score
            best_threshold = thres
    return best_score, best_threshold


def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    # logger.info(f'f2-score: {f2.mean():<.4f}')
    return round(f2.mean(), 4)




# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def get_logger(cfg):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    
    if hasattr(cfg.architecture, "save_name"):
        filename=f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.save_name.replace('/', '-')}/train"
    else:
        filename=f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/train"

    
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    filename=filename
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


### setup wand
def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

def init_wandb(cfg):
    import wandb
    try:
        wandb.login(key="39a298fe785a51ae22d755b11a9f9fff01321796")
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')
        
    run = wandb.init(project=cfg.wandb.project_name, 
                     name=f"{cfg.experiment_name}-{cfg.architecture.model_name}",
                     config=class2dict(cfg),
                     group=cfg.architecture.model_name,
                     job_type="train",
                     anonymous=anony)

    return run


