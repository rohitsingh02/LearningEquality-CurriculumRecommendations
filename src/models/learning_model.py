import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import  AutoConfig, AutoModel
from loss import ContrastiveLoss



class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class WeightedDenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)

        if weights is not None:
            loss = loss * weights
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor, weights=None):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor.argmax(dim=1),
            weight=self.weight,
            reduction=self.reduction,
        )



class RMSELoss(nn.Module):
    """
    Code taken from Y Nakama's notebook (https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
    """
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, predictions, targets):
        loss = torch.sqrt(self.mse(predictions, targets) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


##################################################################################

################################    Poolings    ##################################

##################################################################################

class MeanPooling(nn.Module):
    def __init__(self, dim, cfg):
        super(MeanPooling, self).__init__()
        self.feat_mult = 1
        
    def forward(self, x, attention_mask, input_ids, cfg):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MeanMaxPooling(nn.Module):
    def __init__(self, dim, cfg):
        super(MeanMaxPooling, self).__init__()
        self.feat_mult = 1
        
    def forward(self, x, attention_mask, input_ids, cfg):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask 

        embeddings = x.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        mean_max_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)

        return mean_max_embeddings
    


class MaxPooling(nn.Module):
    def __init__(self, dim, cfg):
        super(MaxPooling, self).__init__()
        self.feat_mult = 1
        
    def forward(self, x, attention_mask, input_ids, cfg):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        embeddings = x.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings
    


class MinPooling(nn.Module):
    def __init__(self, dim, cfg,):
        super(MinPooling, self).__init__()
        self.feat_mult = 1
        
    def forward(self, x, attention_mask, input_ids, cfg):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        embeddings = x.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim = 1)
        return min_embeddings



class GeMText(nn.Module):
    def __init__(self, dim=1, cfg=None, p=3, eps=1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.feat_mult = 1
        # x seeems last hidden state

    def forward(self, x, attention_mask, input_ids, cfg):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
        x = (x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, features):
        ft_all_layers = features['all_layer_embeddings']

        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        features.update({'token_embeddings': weighted_average})
        return features



class AttentionPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_fc):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()



    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q.to("cuda"), h.transpose(-2, -1).to("cuda")).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0).to("cuda"), v_temp.to("cuda")).squeeze(2)
        return v



class NLPPoolings:
    _poolings = {
        # "All [CLS] token": NLPAllclsTokenPooling,
        "GeM": GeMText,
        "Mean": MeanPooling,
        "Max": MaxPooling,
        "Min": MinPooling,
        "MeanMax": MeanMaxPooling,
        "WLP": WeightedLayerPooling,
        "ConcatPool":MeanPooling,
        "AP": AttentionPooling
    }

    @classmethod
    def get(cls, name):
        return cls._poolings.get(name)



    
class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
          
        self.config = AutoConfig.from_pretrained(cfg.uns_model)
        self.model = AutoModel.from_pretrained(cfg.uns_model, config = self.config)
        self.model.resize_token_embeddings(len(cfg.tokenizer))
        self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling(dim=1, cfg=cfg)        
        self.loss_fn = ContrastiveLoss(self.model)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def feature(self, attention_mask, input_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, attention_mask, input_ids, self.cfg)
        return feature
    
    
    def forward(self, inputs, calculate_loss=True):
        x1 = self.feature(inputs["attention_mask1"], inputs["input_ids1"])
        x2 = self.feature(inputs["attention_mask2"], inputs["input_ids2"])
        outputs = {}
        
        if "target" in inputs:
            outputs["target"] = inputs["target"]

        if calculate_loss:
            targets = inputs["target"]
            loss, logits = self.loss_fn(x1, x2, targets) 
            outputs["loss"] = loss
            outputs['logits'] = logits
        
        return outputs
    
    
class Net2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.architecture.model_name, output_hidden_states=True)

        if cfg.architecture.custom_intermediate_dropout:
            self.config.hidden_dropout = cfg.architecture.intermediate_dropout
            self.config.hidden_dropout_prob = cfg.architecture.intermediate_dropout
            self.config.attention_dropout = cfg.architecture.intermediate_dropout
            self.config.attention_probs_dropout_prob = cfg.architecture.intermediate_dropout
              
        self.model = AutoModel.from_pretrained(cfg.architecture.model_name, config=self.config)
        self.model.resize_token_embeddings(len(cfg.tokenizer))
        self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling(dim=1, cfg=cfg)        
        self.loss_fn = ContrastiveLoss(self.model)
        
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def feature(self, attention_mask, input_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, attention_mask, input_ids, self.cfg)
        return feature
    
    
    def forward(self, inputs, calculate_loss=True):
        x1 = self.feature(inputs["attention_mask1"], inputs["input_ids1"])
        x2 = self.feature(inputs["attention_mask2"], inputs["input_ids2"])
        outputs = {}
        
        if "target" in inputs:
            outputs["target"] = inputs["target"]

        if calculate_loss:
            targets = inputs["target"]
            loss, logits = self.loss_fn(x1, x2, targets) 
            outputs["loss"] = loss
            outputs['logits'] = logits
        
        return outputs