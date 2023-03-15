# =========================================================================================
# Libraries
# =========================================================================================
import gc
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import  DataCollatorWithPadding
from sklearn.neighbors import NearestNeighbors
import argparse
from tqdm.auto import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    num_workers = 4
#     model = "../output/mpnet/exp_v3_64_new_f0"
    model = "/home/rohits/pv1/learning_equality/output/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/exp_v3_64_new_f0"
    tokenizer = AutoTokenizer.from_pretrained(model)
    fold=0
    batch_size = 64 #128
    top_n = 50
    seed = 42
    max_len=64


# define some helper functions and classes to aid with data traversal

from pathlib import Path
data_dir = Path('../data')


ADDITIONAL_SPECIAL_TOKENS = [
    "[TITLE]", "[DESC]", "[CATEGORY]", "[LEVEl]", "[PARENT]", "[TEXT]", "[KIND]"
]



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default="",
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0.0,
    )
    args = parser.parse_args()
    return args



def print_markdown(md):
    display(Markdown(md))

class Topic:
    def __init__(self, topic_id):
        self.id = topic_id

    @property
    def parent(self):
        parent_id = topics_df.loc[self.id].parent
        if pd.isna(parent_id):
            return None
        else:
            return Topic(parent_id)

    @property
    def ancestors(self):
        ancestors = []
        parent = self.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        return ancestors

    @property
    def siblings(self):
        if not self.parent:
            return []
        else:
            return [topic for topic in self.parent.children if topic != self]

    @property
    def content(self):
        if self.id in correlations_df.index:
            return [ContentItem(content_id) for content_id in correlations_df.loc[self.id].content_ids.split()]
        else:
            return tuple([]) if self.has_content else []

    def get_breadcrumbs(self, separator=" >> ", include_self=True, include_root=True):
        ancestors = self.ancestors
        if include_self:
            ancestors = [self] + ancestors
        if not include_root:
            ancestors = ancestors[:-1]
        return separator.join(reversed([a.title for a in ancestors]))

    @property
    def children(self):
        return [Topic(child_id) for child_id in topics_df[topics_df.parent == self.id].index]

    def subtree_markdown(self, depth=0):
        markdown = "  " * depth + "- " + self.title + "\n"
        for child in self.children:
            markdown += child.subtree_markdown(depth=depth + 1)
        for content in self.content:
            markdown += ("  " * (depth + 1) + "- " + "[" + content.kind.title() + "] " + content.title) + "\n"
        return markdown

    def __eq__(self, other):
        if not isinstance(other, Topic):
            return False
        return self.id == other.id

    def __getattr__(self, name):
        return topics_df.loc[self.id][name]

    def __str__(self):
        return self.title
    
    def __repr__(self):
        return f"<Topic(id={self.id}, title=\"{self.title}\")>"


class ContentItem:
    def __init__(self, content_id):
        self.id = content_id

    @property
    def topics(self):
        return [Topic(topic_id) for topic_id in topics_df.loc[correlations_df[correlations_df.content_ids.str.contains(self.id)].index].index]

    def __getattr__(self, name):
        return content_df.loc[self.id][name]

    def __str__(self):
        return self.title
    
    def __repr__(self):
        return f"<ContentItem(id={self.id}, title=\"{self.title}\")>"

    def __eq__(self, other):
        if not isinstance(other, ContentItem):
            return False
        return self.id == other.id

    def get_all_breadcrumbs(self, separator=" >> ", include_root=True):
        breadcrumbs = []
        for topic in self.topics:
            new_breadcrumb = topic.get_breadcrumbs(separator=separator, include_root=include_root)
            if new_breadcrumb:
                new_breadcrumb = new_breadcrumb + separator + self.title
            else:
                new_breadcrumb = self.title
            breadcrumbs.append(new_breadcrumb)
        return breadcrumbs


def get_breadcrumbs(row):
    crumbs = ""
    if row.has_content:
        topic = Topic(row.id)
        crumbs = topic.get_breadcrumbs()
    return crumbs


# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_input(text, cfg):
    
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=cfg.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

# =========================================================================================
# Unsupervised dataset
# =========================================================================================
class uns_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['title1'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        return inputs
    
# =========================================================================================
# Mean pooling class
# =========================================================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# =========================================================================================
# Get embeddings
# =========================================================================================
def get_embeddings(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    return preds

# =========================================================================================
# Get the amount of positive classes based on the total
# =========================================================================================
def get_pos_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)





# text
def preprocess(text):
    # remove pattern 2.3.4.5 or 2.3.4.5:
    text = re.sub(r'\w*\s*\d*\.\d*\s*\:*\s*\-*', ' ', text)
    text = re.sub(r'^\d*\:\d*\s*', ' ', text)
    return text

def get_topic_text(row):
    title = row.title1.strip()
    if title != "":
        title = "[TITLE]" + title
    
    if row.description != "":
        desc = row.description.split(".")[0]
        title += "[DESC]" + " ".join(desc.split()[:25]) if len(desc.split()) > 25 else "[DESC]" + desc
        
    if row.category != "":
        title += "[CATEGORY]" + row.category
    
    if row.level != "":
        title += "[LEVEl]" + str(row.level)
        
    if row.t1 != "":
        title += "[PARENT]" + row.t1.replace(" >> ", "[PARENT]")
        
    return title


def get_content_text(row):
    title = row.title1.strip()
    if title != "":
        title = "[TITLE]" + title
    if row.description != "":
        desc = row.description.split(".")[0]
        title += "[DESC]" + " ".join(desc.split()[:25]) if len(desc.split()) > 25 else "[DESC]" + desc
    
    if row.text != "":
        text = row.text.split(".")[0]
        title += "[TEXT]" + " ".join(text.split()[:25]) if len(text.split()) > 25 else "[TEXT]" + text
    
    if row.kind != "":
        title += "[KIND]" + row.kind    
        
    return title






# =========================================================================================
# Build our training set
# =========================================================================================
def build_training_set(topics, content, cfg):
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    targets = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row['id']
        topics_title = row['title1'] 
        topics_lng = row['language']
        predictions = row['predictions'].split(' ')
        ground_truth = row['content_ids'].split(' ')
        predictions.extend(ground_truth)        
        predictions = list(set(predictions))
                
        for pred in predictions:
            content_pred = content.loc[pred]
            content_lng = content_pred['language']
            if topics_lng != content_lng:
                continue                

            # If pred is in ground truth, 1 else 0
            if pred in ground_truth:
                targets.append(1)
            else:
                targets.append(0)
                
            content_title = content_pred['title1']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)

                
    # Build training dataset
    train = pd.DataFrame(
        {'topics_ids': topics_ids, 
         'content_ids': content_ids, 
         'title1': title1, 
         'title2': title2, 
         'target': targets,
        }
    )
    # Release memory
    del topics_ids, content_ids, title1, title2, targets
    gc.collect()
    return train


# =========================================================================================
# Get neighbors
# =========================================================================================
def get_neighbors(topics, content, cfg):
    topics_dataset = uns_dataset(topics, cfg)
    content_dataset = uns_dataset(content, cfg)
    topics_loader = DataLoader(
        topics_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    content_loader = DataLoader(
        content_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = False
        )
    # Create unsupervised model to extract embeddings
    model = uns_model(cfg)
    model.to(device)

    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)
    content_preds = get_embeddings(content_loader, model, device)
    # Transfer predictions to gpu
    topics_preds_gpu = np.array(topics_preds)
    content_preds_gpu = np.array(content_preds)
    # Release memory
    torch.cuda.empty_cache()
    del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
    gc.collect()
    # KNN model
    print(' ')    
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors=cfg.top_n, metric = 'cosine')
    neighbors_model.fit(content_preds_gpu)
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance = False)
    predictions = []
    for k in tqdm(range(len(indices))):
        pred = indices[k]        
        # p = ' '.join([content.loc[ind, 'id'] for ind in pred.get()])
        p = ' '.join([content.loc[ind, 'id'] for ind in pred])
        predictions.append(p)
    topics[f'predictions'] = predictions
    
    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    return topics, content



class uns_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model)
        self.config.output_hidden_states = True    
        self.model = AutoModel.from_pretrained(cfg.model, config = self.config)
        self.pool = MeanPooling()
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature


if __name__ == '__main__':

    args = parse_args()
    CFG.model = args.model 
    CFG.fold = args.fold
    

    # # load the data into pandas dataframes
    topics_df = pd.read_csv(data_dir / "topics.csv", index_col=0).fillna({"title": "", "description": ""})
    topics_df['title1'] = topics_df.title.apply(lambda x: preprocess(x).strip())

    # topics_df.fillna("", inplace=True)
    topics_df['id'] = topics_df.index.values

    content_df = pd.read_csv(data_dir / "content.csv", index_col=0).fillna("")
    content_df['title1'] = content_df.title.apply(lambda x: preprocess(x).strip())
    content_df['id'] = content_df.index.values
    correlations_df = pd.read_csv(data_dir / "correlations.csv")


    topics_df['context'] = topics_df.progress_apply(lambda x: get_breadcrumbs(x), axis=1)
    topics_df.rename(columns = {'context': 't1'}, inplace=True)
    topics_df.fillna("", inplace=True)
    topics_df['title1'] = topics_df.progress_apply(lambda x: get_topic_text(x), axis=1)
    content_df['title1'] = content_df.progress_apply(lambda x: get_content_text(x), axis=1)


    topics_df.reset_index(drop=True, inplace=True)

    topics_df = topics_df.merge(
        correlations_df, left_on='id', right_on='topic_id', how='left'
    )

    topics_df.reset_index(drop=True, inplace=True)
    content_df.reset_index(drop=True, inplace=True)
    topics_df = topics_df.loc[topics_df.has_content==True]
    source_ids = topics_df.loc[topics_df.category == 'source'].id.values


    # Run nearest neighbors
    topics, content = get_neighbors(topics_df, content_df, CFG)
    pos_score = get_pos_score(topics['content_ids'], topics['predictions'])
    print(f'Our max positive score is {pos_score}')
    # We can delete correlations
    # del correlations
    gc.collect()
    # Set id as index for content
    content.set_index('id', inplace = True)

    # Build training set   
    train = build_training_set(topics, content, CFG)
    print(f'Our training set has {len(train)} rows')
    # Save train set to disk to train on another notebook
    train.to_csv(f"train_ret_mpnet_50_f{CFG.fold}_64.csv", index=False)


