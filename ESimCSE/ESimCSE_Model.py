import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer


class ESimcseModel(nn.Module):

    def __init__(self, pretrained_model, pooling, dropout=0.3):
        super(ESimcseModel, self).__init__()
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


class MomentumEncoder(ESimcseModel):
    """ MomentumEncoder """

    def __init__(self, pretrained_model, pooling):
        super(MomentumEncoder, self).__init__(pretrained_model, pooling)


class MultiNegativeRankingLoss(nn.Module):
    # code reference: https://github.com/zhoujx4/NLP-Series-sentence-embeddings
    def __init__(self):
        super(MultiNegativeRankingLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def multi_negative_ranking_loss(self, embed_src, embed_pos, embed_neg, scale=20.0):
        '''
        scale is a temperature parameter
        '''

        if embed_neg is not None:
            embed_pos = torch.cat([embed_pos, embed_neg], dim=0)

        # print(embed_src.shape, embed_pos.shape)
        scores = self.cos_sim(embed_src, embed_pos) * scale

        labels = torch.tensor(range(len(scores)),
                              dtype=torch.long,
                              device=scores.device)  # Example a[i] should match with b[i]

        return self.cross_entropy_loss(scores, labels)

    def cos_sim(self, a, b):
        """ the function is same with torch.nn.F.cosine_similarity but processed the problem of tensor dimension
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))


if __name__ == '__main__':
    import numpy as np

    input1 = torch.randn(100, 128)
    input2 = torch.randn(100, 128)
    output = F.cosine_similarity(input1, input2)
    print(output.shape)

    embed_src = torch.tensor(np.random.randn(32, 768))  # (batch_size, 768)
    embed_pos = torch.tensor(np.random.randn(32, 768))
    embed_neg = torch.tensor(np.random.randn(160, 768))

    ESimCSELoss = MultiNegativeRankingLoss()
    esimcse_loss = ESimCSELoss.multi_negative_ranking_loss

    res = esimcse_loss(embed_src, embed_pos, embed_neg)
