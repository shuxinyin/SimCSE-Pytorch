import argparse
import sys

from tqdm import tqdm
from loguru import logger

import numpy as np
from scipy.stats import spearmanr
from transformers import BertModel, BertConfig, BertTokenizer

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ESimCSE_dataloader import TrainDataset, TestDataset, CollateFunc, load_sts_data, load_sts_data_unsup
from ESimCSE_Model import ESimcseModel, MomentumEncoder, MultiNegativeRankingLoss


def get_bert_input(source, device):
    input_ids = source.get('input_ids').to(device)
    attention_mask = source.get('attention_mask').to(device)
    token_type_ids = source.get('token_type_ids').to(device)
    return input_ids, attention_mask, token_type_ids


def train(model, momentum_encoder, train_dl, dev_dl, optimizer, loss_func, device, save_path, gamma=0.95):
    """模型训练函数"""
    model.train()
    best = 0
    for batch_idx, (batch_src_source, batch_pos_source, batch_neg_source) in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        input_ids_src, attention_mask_src, token_type_ids_src = get_bert_input(batch_src_source, device)
        input_ids_pos, attention_mask_pos, token_type_ids_pos = get_bert_input(batch_pos_source, device)

        neg_out = None
        if batch_neg_source:
            input_ids_neg, attention_mask_neg, token_type_ids_neg = get_bert_input(batch_neg_source, device)
            neg_out = momentum_encoder(input_ids_neg, attention_mask_neg, token_type_ids_neg)
            # print(neg_out.shape)

        src_out = model(input_ids_src, attention_mask_src, token_type_ids_src)
        pos_out = model(input_ids_pos, attention_mask_pos, token_type_ids_pos)

        loss = loss_func(src_out, pos_out, neg_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #  Momentum Contrast Encoder Update
        for encoder_param, moco_encoder_param in zip(model.parameters(), momentum_encoder.parameters()):
            # print("--", moco_encoder_param.data.shape, encoder_param.data.shape)
            moco_encoder_param.data = gamma \
                                      * moco_encoder_param.data \
                                      + (1. - gamma) * encoder_param.data

        if batch_idx % 5 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = evaluation(model, dev_dl, device)
            model.train()
            if best < corrcoef:
                best = corrcoef
                # torch.save(model.state_dict(), save_path)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")


def evaluation(model, dataloader, device):
    """模型评估函数
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def main(args):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_path_sp = args.data_path + "cnsd-sts-train.txt"
    train_path_unsp = args.data_path + "cnsd-sts-train_unsup.txt"
    dev_path_sp = args.data_path + "cnsd-sts-dev.txt"
    test_path_sp = args.data_path + "cnsd-sts-test.txt"
    # pretrain_model_path = "/data/Learn_Project/Backup_Data/macbert_chinese_pretrained"

    test_data_source = load_sts_data(test_path_sp)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    train_data_source = load_sts_data_unsup(train_path_unsp)
    train_sents = [data[0] for data in train_data_source]
    train_dataset = TrainDataset(train_sents)

    train_call_func = CollateFunc(tokenizer, max_len=args.max_length, q_size=args.q_size, dup_rate=args.dup_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12,
                                  collate_fn=train_call_func)

    test_dataset = TestDataset(test_data_source, tokenizer, max_len=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"]
    model = ESimcseModel(pretrained_model=args.pretrain_model_path, pooling=args.pooler, dropout=args.dropout).to(
        args.device)
    momentum_encoder = MomentumEncoder(args.pretrain_model_path, args.pooler).to(args.device)

    ESimCSELoss = MultiNegativeRankingLoss()
    esimcse_loss = ESimCSELoss.multi_negative_ranking_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train(model, momentum_encoder,
          train_dataloader, test_dataloader,
          optimizer, esimcse_loss,
          args.device, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu', help="gpu or cpu")
    parser.add_argument("--save_path", type=str, default='./model_save')
    parser.add_argument("--un_supervise", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--dup_rate", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--q_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=50, help="max length of input sentences")
    parser.add_argument("--data_path", type=str, default="../data/STS-B/")
    parser.add_argument("--pretrain_model_path", type=str,
                        default="/data/Learn_Project/Backup_Data/bert_chinese")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='first-last-avg', help='which pooler to use')

    args = parser.parse_args()
    logger.add("../log/train.log")
    logger.info("run run run")
    logger.info(args)
    main(args)
