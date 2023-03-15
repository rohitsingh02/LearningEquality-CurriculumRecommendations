from torch.utils.data import Dataset, DataLoader
import torch
import collections
import numpy as np



class CustomDataset(Dataset):
    def __init__(self, df, mode, cfg):
        self.df = df
        self.mode = mode
        self.cfg = cfg
        self.tokenizer = cfg.tokenizer
        self.texts1 = self.df[cfg.dataset.text_column1].values
        self.texts2 = self.df[cfg.dataset.text_column2].values
        self.targets = self.df[cfg.dataset.target_col].values


    def __len__(self):
        return len(self.texts1)


    def batch_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, collections.abc.Mapping):
            return {
                key: CustomDataset.batch_to_device(value, device)
                for key, value in batch.items()
            }
        elif isinstance(batch, collections.abc.Sequence):
            return [CustomDataset.batch_to_device(value, device) for value in batch]
        else:
            raise ValueError(f"Can not move {type(batch)} to device.")


    def encode(self, text, idx):
        sample = dict()
        encodings = self.tokenizer.encode_plus(
            text, 
            return_tensors=None, 
            add_special_tokens=True, 
            max_length=self.cfg.dataset.max_len,
            pad_to_max_length=True,
            truncation=True
        )

        sample[f"input_ids{idx}"] =  torch.tensor(encodings["input_ids"], dtype=torch.long) 
        sample[f"attention_mask{idx}"] = torch.tensor(encodings["attention_mask"], dtype=torch.long) 
        return sample


    def _read_data(self, idx, sample):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        sample.update(self.encode(text1, idx=1))
        sample.update(self.encode(text2, idx=2))
        return sample

    def _read_label(self, idx, sample):
    
        sample["target"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return sample


    def __getitem__(self, idx):
        sample = dict()
        sample = self._read_data(idx=idx, sample=sample)
        if self.targets is not None:
            sample = self._read_label(idx=idx, sample=sample)

        return sample