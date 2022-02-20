import random
import jieba
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, BertTokenizer



def load_sts_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data_source = list()
        for line in f:
            line_split = line.strip().split("||")
            data_source.append((line_split[1], line_split[2], line_split[3]))
        return data_source


def load_sts_data_unsup(path):
    with open(path, 'r', encoding='utf-8') as f:
        data_source = list()
        for line in f:
            line_split = line.strip().split("\n")
            data_source.append(line_split)
        return data_source


class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]

        return text


class CollateFunc(object):
    def __init__(self, tokenizer, max_len=256, q_size=160, dup_rate=0.15):
        self.q = []
        self.q_size = q_size
        self.max_len = max_len
        self.dup_rate = dup_rate
        self.tokenizer = tokenizer

    def word_repetition_normal(self, batch_text):
        dst_text = list()
        for text in batch_text:
            actual_len = len(text)
            dup_len = random.randint(a=0, b=max(
                2, int(self.dup_rate * actual_len)))
            dup_word_index = random.sample(
                list(range(1, actual_len)), k=dup_len)

            dup_text = ''
            for index, word in enumerate(text):
                dup_text += word
                if index in dup_word_index:
                    dup_text += word
            dst_text.append(dup_text)
        return dst_text

    def word_repetition_chinese(self, batch_text):
        ''' span duplicated for chinese
        '''
        dst_text = list()
        for text in batch_text:
            cut_text = jieba.cut(text, cut_all=False)
            text = list(cut_text)

            actual_len = len(text)
            dup_len = random.randint(a=0, b=max(
                2, int(self.dup_rate * actual_len)))
            dup_word_index = random.sample(
                list(range(1, actual_len)), k=dup_len)

            dup_text = ''
            for index, word in enumerate(text):
                dup_text += word
                if index in dup_word_index:
                    dup_text += word
                dst_text.append(dup_text)
            return dup_text

    def negative_samples(self, batch_src_text):
        batch_size = len(batch_src_text)
        negative_samples = None
        if len(self.q) > 0:
            negative_samples = self.q[:self.q_size]
            # print("size of negative_samples", len(negative_samples))

        if len(self.q) + batch_size >= self.q_size:
            del self.q[:batch_size]
        self.q.extend(batch_src_text)

        return negative_samples

    def __call__(self, batch_text):
        '''
        input: batch_text: [batch_text,]
        output: batch_src_text, batch_dst_text, batch_neg_text
        '''
        batch_pos_text = self.word_repetition_normal(batch_text)
        batch_neg_text = self.negative_samples(batch_text)
        # print(len(batch_pos_text))

        batch_tokens = self.tokenizer(batch_text, max_length=self.max_len,
                                      truncation=True, padding='max_length', return_tensors='pt')
        batch_pos_tokens = self.tokenizer(batch_pos_text, max_length=self.max_len,
                                          truncation=True, padding='max_length', return_tensors='pt')

        batch_neg_tokens = None
        if batch_neg_text:
            batch_neg_tokens = self.tokenizer(batch_neg_text, max_length=self.max_len,
                                              truncation=True, padding='max_length', return_tensors='pt')

        return batch_tokens, batch_pos_tokens, batch_neg_tokens


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text2id(self, text):
        return self.tokenizer(text, max_length=self.max_len,
                              truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index):
        da = self.data[index]
        return self.text2id([da[0]]), self.text2id([da[1]]), int(da[2])


if __name__ == "__main__":
    import numpy as np

    train_path_sp = "./data/STS-B/" + "cnsd-sts-train.txt"
    dev_path_sp = "./data/STS-B/" + "cnsd-sts-dev.txt"
    pretrain_model_path = "/Learn_Project/Backup_Data/macbert_chinese_pretrained"

    train_data_source = load_sts_data(train_path_sp)
    test_data_source = load_sts_data(dev_path_sp)
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    train_sents = [data[0] for data in train_data_source] + [data[1] for data in train_data_source]
    train_dataset = TrainDataset(train_sents)

    train_call = CollateFunc(tokenizer, max_len=256, q_size=160, dup_rate=0.15)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1,
                                  collate_fn=train_call)

    for batch_idx, (batch_tokens, batch_pos_tokens, batch_neg_tokens) in enumerate(train_dataloader, start=1):
        print("--", batch_tokens.shape)
