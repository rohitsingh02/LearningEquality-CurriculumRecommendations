# =========================================================================================
# Libraries
# =========================================================================================
import os
import gc
import time
import sys
import yaml
from types import SimpleNamespace
import utils
from asyncio.log import logger
import wandb
import re
import argparse
import importlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from awp import AWP
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


sys.path.append("models")
sys.path.append("datasets")


# =========================================================================================
# get model
# ========================================================================================= 
def get_model(cfg):
    if cfg.architecture.pretrained_weights != "":
        Net = importlib.import_module(cfg.model_class).Net2
    else:
        Net = importlib.import_module(cfg.model_class).Net
    return Net(cfg)
    
# =========================================================================================
# Collate function for training
# =========================================================================================
def collate(inputs):
    mask_len = int(inputs["attention_mask1"].sum(axis=1).max())
    for k, v in inputs.items():
        if k != "target":
            inputs[k] = inputs[k][:,:mask_len]
    return inputs


def load_checkpoint(cfg, model, fold=0):
    weight = f"{cfg.architecture.pretrained_weights}/checkpoint_{fold}.pth"
    d =  torch.load(weight, map_location="cpu")
    if "model" in d:
        model_weights = d["model"]
    else:
        model_weights = d

    try:
        model.load_state_dict(model_weights, strict=True)
    except Exception as e:
        print("removing unused pretrained layers")
        for layer_name in re.findall("size mismatch for (.*?):", str(e)):
            model_weights.pop(layer_name, None)
        model.load_state_dict(model_weights, strict=False)

    print(f"Weights loaded from: {cfg.architecture.pretrained_weights}")


# =========================================================================================
# Train function loop
# =========================================================================================

def valid_fn(valid_loader, model, cfg):
    losses = utils.AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, data_dict in enumerate(valid_loader):
        inputs = collate(data_dict)
        # inputs = data_dict
        inputs = cfg.CustomDataset.batch_to_device(inputs, cfg.device)
        target = inputs['target']
        batch_size = target.size(0)
        
        with torch.no_grad():
            output_dict = model(inputs)
            loss = output_dict["loss"]


        if cfg.training.grad_accumulation > 1:
            loss = loss / cfg.training.grad_accumulation
        losses.update(loss.item(), batch_size)
        
        
        preds.append(output_dict["logits"].sigmoid().squeeze().to('cpu').numpy().reshape(-1))
        
        end = time.time()
        if step % cfg.training.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=utils.timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds, axis=0)
    return losses.avg, predictions


# =========================================================================================
# Train & Evaluate
# =========================================================================================
def train_loop(df, correlations, fold, cfg):
    print(' ')
    print(f"========== fold: {fold} training ==========")
    
    # Split train & validation
    train_df = df[df['fold'] != fold]
    val_df = df[df['fold'] == fold]
    print(train_df.shape, val_df.shape)    
    
    if cfg.debug: 
        print("DEBUG MODE")
        train_df = train_df.head(50)
    
    train_dataset = cfg.CustomDataset(train_df, "train", cfg)
    valid_dataset = cfg.CustomDataset(val_df, "val", cfg)    

    train_loader = DataLoader(
        train_dataset, 
        batch_size = cfg.training.batch_size, 
        shuffle = True, 
        num_workers = cfg.environment.num_workers, 
        pin_memory = True, 
        drop_last = True,
    )

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size = cfg.training.batch_size * 2, 
        shuffle = False, 
        num_workers = cfg.environment.num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    
    # Get model
    model = get_model(cfg) 
    torch.save(model.config, f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/config.pth")

    if hasattr(cfg.architecture, "pretrained_weights") and cfg.architecture.pretrained_weights != "":
        print("START LAODING PRETRAIED WEIGHTS.....")
        try:
            load_checkpoint(cfg, model, fold)
        except:
            print("WARNING: could not load checkpoint")
    
    model.to(cfg.device)
    
    # Optimizer
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay = 0.0):
        # model = model.module
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
            'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters
    
    optimizer_parameters = get_optimizer_params(
        model, 
        encoder_lr = cfg.optimizer.encoder_lr, 
        decoder_lr = cfg.optimizer.decoder_lr,
        weight_decay = cfg.optimizer.weight_decay
    )
    
    optimizer = AdamW(
        optimizer_parameters, 
        lr = cfg.optimizer.encoder_lr, 
        eps = cfg.optimizer.eps, 
        betas = (0.9, 0.999)
    )
    
    num_train_steps = int(len(train_df) / cfg.training.batch_size * cfg.training.epochs)
    num_warmup_steps = num_train_steps * cfg.scheduler.warmup_ratio
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps, 
        num_training_steps = num_train_steps, 
        num_cycles = cfg.scheduler.num_cycles
    )
    
    # ====================================================
    # Setting Up AWP Training 
    # ====================================================
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.environment.mixed_precision)
    awp = AWP(model,
        optimizer,
        adv_lr=0.0001,
        adv_eps=0.001,
        start_epoch= 2, #(cfg.training.epochs * (total_steps // cfg.training.batch_size))/cfg.training.epochs,
        scaler=scaler
    )
    
    # Training & Validation loop
    best_score = 0
    step_val = 0
        
    for epoch in range(cfg.training.epochs):
        cfg.epoch = epoch
        start_time = time.time()
        
        model.train()
        losses = utils.AverageMeter()
        start = end = time.time()
        global_step = 0
        for step, data_dict in enumerate(train_loader):
            if step_val: step_val += 1  
            inputs = collate(data_dict)
            # inputs = data_dict
            inputs = cfg.CustomDataset.batch_to_device(inputs, device)
            batch_size = data_dict['target'].size(0)
            with torch.cuda.amp.autocast(enabled=cfg.environment.mixed_precision):
                output_dict = model(inputs)                
                loss = output_dict["loss"]                
        
            if hasattr(cfg, "awp") and cfg.awp.enable and epoch >= cfg.awp.start_epoch:
                awp.attack_backward(inputs, step_val)

            if cfg.training.grad_accumulation > 1:
                loss = loss / cfg.training.grad_accumulation
                
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            if (step + 1) % cfg.training.grad_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if cfg.training.batch_scheduler:
                    scheduler.step()
                    
            end = time.time()  
            
            
            if  step % cfg.training.print_freq == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    'Grad: {grad_norm:.4f}  '
                    'LR: {lr:.8f}  '
                    .format(epoch+1, step, len(train_loader), 
                            remain=utils.timeSince(start, float(step+1)/len(train_loader)),
                            loss=losses,
                            grad_norm=grad_norm,
                            lr=scheduler.get_lr()[0]))
                
                
            if cfg.wandb.enable:
                wandb.log({f"[fold{fold}] loss": losses.val,
                        f"[fold{fold}] lr": scheduler.get_lr()[0]})
                
                
        avg_loss = losses.avg
        avg_val_loss, predictions = valid_fn(valid_loader, model, cfg)     
        score, threshold, recall = utils.get_best_threshold(val_df, predictions, correlations)   
        
        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {score}, Threshold: {threshold}, Recall: {recall}')
        
        if cfg.wandb.enable:
            wandb.log({f"[fold{fold}] epoch": epoch+1, 
                       f"[fold{fold}] avg_train_loss": avg_loss, 
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] Recall": recall,
                       f"[fold{fold}] score": score})

        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(
                {'model': model.state_dict(),
                'predictions': predictions},
                f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/checkpoint_{fold}.pth"
            )    
           
    predictions = torch.load(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/checkpoint_{fold}.pth", 
                            map_location=torch.device('cpu'))['predictions']   
           
           
    val_df[f"pred_{cfg.dataset.target_col}"] = predictions
    print("*"*100)

    torch.cuda.empty_cache()
    gc.collect()
    
    return val_df



# setting up config

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())
for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)
print(cfg)


os.makedirs(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}", exist_ok=True)


cfg.CustomDataset = importlib.import_module(cfg.dataset_class).CustomDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.device = device

if __name__ == "__main__":
    LOGGER = utils.get_logger(cfg)
    cfg.logger = LOGGER
    
    if cfg.wandb.enable: utils.init_wandb(cfg)
    
    cfg.environment.seed=np.random.randint(1_000_000) if cfg.environment.seed < 0 else np.random.randint(1_000_000)
    utils.set_seed(cfg.environment.seed)

    train_df = pd.read_csv(cfg.dataset.train_dataframe)
    train_df = train_df.drop_duplicates(subset=['topics_ids', "content_ids"])    
    train_df['title1'].fillna("Title does not exist", inplace = True)
    train_df['title2'].fillna("Title does not exist", inplace = True)

    correlations = pd.read_csv(cfg.dataset.correlations)
    topics = pd.read_csv(cfg.dataset.topic_df)    
    contents =  pd.read_csv(cfg.dataset.content_df)
    source_ids = topics.loc[topics.category == 'source', 'id'].values
    no_content_ids = topics.loc[topics.has_content == False, 'id'].values
    
    topics = topics.loc[topics.has_content != False].reset_index(drop=True)
    
    train_df = train_df.loc[~train_df.topics_ids.isin(no_content_ids)]
    print(len(source_ids), topics.shape)
    folds_csv = pd.read_csv(cfg.dataset.folds_csv)
    val_topics = folds_csv.loc[folds_csv.fold==cfg.dataset.fold].topic_id.values


    # remove later
    train_df['fold'] = 1
    train_df.loc[train_df.topics_ids.isin(val_topics), 'fold'] = 0
    train_df = train_df.reset_index(drop=True)    
    # cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.uns_model)  
        
    tokenizer = AutoTokenizer.from_pretrained(cfg.architecture.model_name)  

    tokens =  ["[TITLE]", "[DESC]", "[CATEGORY]", "[LEVEl]", "[PARENT]", "[TEXT]", "[KIND]"] #spc tokens
    special_tokens_dict = {'additional_special_tokens': tokens}
    # special_tokens_dict = {'additional_special_tokens': ['</s>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    cfg.tokenizer = tokenizer
    
    
    print(f"max_len: {cfg.dataset.max_len}")   
    cfg.tokenizer.save_pretrained(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/tokenizer/")
    
    # exit()
    oof_df = pd.DataFrame()
    for fold in [0]:
        _oof_df = train_loop(train_df, correlations, fold, cfg)
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== fold: {fold} result ==========")
        # utils.get_result(oof_df, cfg, LOGGER)
        score, threshold, recall = utils.get_best_threshold(oof_df, oof_df[f"pred_{cfg.dataset.target_col}"], correlations)    
        LOGGER.info(f'Score: {score:.4f}, Threshold: {threshold}, Recall: {recall}')
        oof_df.to_csv(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/oof_df{fold}.csv")
        break

    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info(f"========== CV ==========")
    
    LOGGER.info(f'Score: {score:.4f}, Threshold: {threshold}')
    if cfg.wandb.enable: wandb.finish()
