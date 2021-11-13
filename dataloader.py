from torch.utils.data import Dataset, DataLoader


def load_sts_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data_source = list()
        for line in f:
            line_split = line.strip().split("||")
            data_source.append((line_split[1], line_split[2], line_split[3]))
        return data_source


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        # 同一个text重复两次， 通过bert编码互为正样本
        tokens = self.tokenizer([text, text], max_length=self.max_len,
                                truncation=True, padding='max_length', return_tensors='pt')
        return tokens


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text2id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len,
                              truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        da = self.data[index]
        return self.text2id([da[0]]), self.text2id([da[1]]), int(da[2])
