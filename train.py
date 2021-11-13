import argparse
from tqdm import tqdm
from loguru import logger

import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataloader import TrainDataset, TestDataset, load_sts_data
from model import SimcseModel, simcse_unsup_loss
from transformers import BertModel, BertConfig, BertTokenizer


def train(model, train_dl, dev_dl, optimizer, device, save_path):
    """模型训练函数"""
    model.train()
    best = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(device)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(device)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(device)

        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_unsup_loss(out, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    train_path_sp = "./data/STS-B/" + "cnsd-sts-train.txt"
    dev_path_sp = "./data/STS-B/" + "cnsd-sts-dev.txt"
    pretrain_model_path = "/data/Learn_Project/Backup_Data/macbert_chinese_pretrained"

    train_data_source = load_sts_data(train_path_sp)
    test_data_source = load_sts_data(dev_path_sp)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    train_sents = [data[0] for data in train_data_source] + [data[1] for data in train_data_source]
    train_dataset = TrainDataset(train_sents, tokenizer, max_len=args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    test_dataset = TestDataset(test_data_source, tokenizer, max_len=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"]
    model = SimcseModel(pretrained_model=args.pretrain_model_path, pooling=args.pooler, dropout=args.dropout).to(
        args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train(model, train_dataloader, test_dataloader, optimizer, args.device, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu', help="gpu or cpu")
    parser.add_argument("--save_path", type=str, default='./model_save')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=float, default=32)
    parser.add_argument("--max_length", type=int, default=64, help="max length of input sentences")
    parser.add_argument("--data_path", type=str, default="./data/STS-B/")
    parser.add_argument("--pretrain_model_path", type=str,
                        default="/data/Learn_Project/Backup_Data/macbert_chinese_pretrained")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='cls', help='which pooler to use')

    args = parser.parse_args()
    logger.add("./log/train.log")
    logger.info("run run run")
    main(args)
