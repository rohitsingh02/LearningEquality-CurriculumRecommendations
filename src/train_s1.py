# =========================================================================================
# Libraries
# =========================================================================================
import sys
import math
import yaml
import torch
import argparse
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.readers import InputExample
from BinaryClassificationEvaluator import BinaryClassificationEvaluator
# from sentence_transformers.evaluation import BinaryClassificationEvaluator

from types import SimpleNamespace
from asyncio.log import logger
import utils
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")



tokens =  ["[TITLE]", "[DESC]", "[CATEGORY]", "[LEVEl]", "[PARENT]", "[TEXT]", "[KIND]"] #spc tokens


# setting up config
parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())
for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.device = device
model_save_path = f'{cfg.output_dir}/{cfg.experiment_name}'
top_n = 50


if __name__ == "__main__":
    
    cfg.environment.seed=np.random.randint(1_000_000) if cfg.environment.seed < 0 else np.random.randint(1_000_000)
    utils.set_seed(cfg.environment.seed)

    word_embedding_model = models.Transformer(cfg.architecture.model_name, max_seq_length=cfg.dataset.max_len)
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=cfg.architecture.pool == "Mean",
        pooling_mode_cls_token=cfg.architecture.pool == "CLS",
        pooling_mode_max_tokens=cfg.architecture.pool == "Max"
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    topics_df = pd.read_csv(cfg.dataset.topic_df)    
    content_df = pd.read_csv(cfg.dataset.content_df)
    train_df = pd.read_csv(cfg.dataset.train_dataframe)    
    train_df['title1'].fillna("Title does not exist", inplace = True)
    train_df['title2'].fillna("Title does not exist", inplace = True)
    correlations = pd.read_csv(cfg.dataset.correlations)
    folds_csv = pd.read_csv(cfg.dataset.folds_csv)
    val_topics = folds_csv.loc[folds_csv.fold==cfg.dataset.fold].topic_id.values        
    no_content_ids = topics_df.loc[topics_df.has_content == False, 'id'].values    
    train_df = train_df.loc[~train_df.topics_ids.isin(no_content_ids)]
    print(' ')
    print('-' * 50)
    print(f"train.shape: {train_df.shape}")
    print(f"correlations.shape: {correlations.shape}")
    print(train_df.target.value_counts())
    train_df = train_df.reset_index(drop=True)
    print(train_df.columns)
    
    train_df['fold'] = 1
    train_df.loc[train_df.topics_ids.isin(val_topics), 'fold'] = 0
        
    if cfg.dataset.use_only_pos:
        train_df = train_df.loc[train_df.target != 0]
        
    train_df = train_df.reset_index(drop=True)
    val_df = train_df.loc[train_df.fold == 0].reset_index(drop=True)
    train_df = train_df.loc[train_df.fold != 0].reset_index(drop=True)
    
    topics_df = topics_df.loc[topics_df['id'].isin(val_df.topics_ids.values)] # remove later
    
    train_samples = []
    dev_samples = []

    for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
        score = float(row['target'])   # Normalize score to range 0 ... 1
        inp_example = InputExample(
            texts=[row['title1'], row['title2']], label=score
        )
        train_samples.append(inp_example)
        
    
    for index, row in tqdm(val_df.iterrows(), total=len(val_df)):
        score = float(row['target'])   # Normalize score to range 0 ... 1
        inp_example = InputExample(
            texts=[row['title1'], row['title2']], label=score
        )
        dev_samples.append(inp_example)
          
    
    print(len(train_samples), len(dev_samples))    
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=cfg.training.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)    
    evaluator = BinaryClassificationEvaluator(topics_df=topics_df, content_df=content_df, correlations=correlations, model = model, tokenizer=model.tokenizer, top_n=top_n, max_len = cfg.dataset.max_len, name='sts-dev')
    warmup_steps = math.ceil(len(train_dataloader) * cfg.training.epochs  * (cfg.training.warmup_pct/100)) #10% of train data for warm-up

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=cfg.training.epochs,
        evaluation_steps=cfg.training.eval_steps,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        save_best_model=cfg.training.save_best,
        use_amp=cfg.training.fp16
    )

    model.save(model_save_path + "_last")
