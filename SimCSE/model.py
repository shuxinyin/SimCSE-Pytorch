import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer


class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout=0.3):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]


def simcse_unsup_loss(y_pred, device, temp=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1

    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    # [batch_size * 2, 1, 768] * [1, batch_size * 2, 768] = [batch_size * 2, batch_size * 2]
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)

    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    sim = sim / temp  # 相似度矩阵除以温度系数

    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)
