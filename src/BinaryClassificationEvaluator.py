import logging
import os
import csv
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import  DataCollatorWithPadding
import gc
from sklearn.neighbors import NearestNeighbors
logger = logging.getLogger(__name__)




class SentenceEvaluator:
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """
        pass






# =========================================================================================
# F2 score metric
# =========================================================================================
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


# Get best threshold
# =========================================================================================
def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0
    best_threshold = None
    # for thres in np.arange(0.001, 0.5, 0.001):
    for thres in np.arange(0.001, 0.99, 0.01):
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
            
        preds.append(y_preds['sentence_embedding'].to('cpu').numpy())
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


# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_input(text, tokenizer, max_len=64):
    
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=max_len,
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
    def __init__(self, df, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.texts = df['title1'].values
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.tokenizer, self.max_len)
        return inputs


# =========================================================================================
# Get neighbors
# =========================================================================================
def get_neighbors(topics, content, model, tokenizer, top_n, max_len):
    topics_dataset = uns_dataset(topics, tokenizer, max_len=max_len)
    content_dataset = uns_dataset(content, tokenizer, max_len=max_len)
    topics_loader = DataLoader(
        topics_dataset, 
        batch_size = 128, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'longest'),
        num_workers = 4, 
        pin_memory = True, 
        drop_last = False
    )
    
    content_loader = DataLoader(
        content_dataset, 
        batch_size = 128, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = tokenizer, padding = 'longest'),
        num_workers = 4, 
        pin_memory = True, 
        drop_last = False
        )
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    topics_preds = get_embeddings(topics_loader, model, device)
    content_preds = get_embeddings(content_loader, model, device)
    # Transfer predictions to gpu
    # topics_preds_gpu = cp.array(topics_preds)
    # content_preds_gpu = cp.array(content_preds)
    topics_preds_gpu = np.array(topics_preds)
    content_preds_gpu = np.array(content_preds)
    
    # Release memory
    torch.cuda.empty_cache()
    del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
    gc.collect()
    # KNN model
    print(' ')
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors=top_n, metric = 'cosine')
    neighbors_model.fit(content_preds_gpu)
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance = False)
    predictions = []
    for k in tqdm(range(len(indices))):
        pred = indices[k]        
        # p = ' '.join([content.loc[ind, 'id'] for ind in pred.get()])
        p = ' '.join([content.loc[ind, 'id'] for ind in pred])
        predictions.append(p)
    topics['predictions'] = predictions
    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices
    gc.collect()
    return topics, content



class BinaryClassificationEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    """

    def __init__(self, topics_df, content_df, correlations, model, tokenizer, top_n, max_len,  name: str = '', batch_size: int = 32, show_progress_bar: bool = False, write_csv: bool = True):
     
        self.topics_df = topics_df
        self.content_df = content_df
        self.correlations = correlations
        self.model = model
        self.tokenizer = tokenizer
        self.top_n = top_n
        self.max_len = max_len

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "binary_classification_evaluation" + ("_"+name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "pos-50"]


    @classmethod
    def from_input_examples(cls, **kwargs):
        return cls(**kwargs)
    

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Binary Accuracy Evaluation of the model on " + self.name + " dataset" + out_txt)

        
        topics, content = get_neighbors(self.topics_df, self.content_df, self.model, self.tokenizer, self.top_n, self.max_len)
        pos_score = get_pos_score(topics['content_ids'], topics['predictions'])
        scores = pos_score
        main_score = pos_score
        file_output_data = [epoch, steps, scores]

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        return main_score