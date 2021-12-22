import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset
from utils.utils import create_logger

logger = create_logger()



class JPIterDataset(IterableDataset):
    def __init__(self,
                 tokenizer,
                 max_len: int=512,
                 random_state: int = 42,
                 mode='train',
                 root: str = './',
                 file_name: str = 'japanese_text_sum.csv') -> None:

        super(JPIterDataset, self).__init__()

        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = pd.read_csv(os.path.join(root, file_name))
        if mode == 'train':
            self.data = self.data[self.data.is_train]
            self.data = self.data.sample(frac=1, random_state=random_state)
        elif mode == 'valid':
            self.data = self.data[self.data.is_val]
        else:
            self.data = self.data[self.data.is_test]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        for source, target in zip(self.data.source, self.data.target):
            sep_idx = None
            input_ids = []
            attention_mask = [1]*self.max_len
            source_tokens, target_tokens = self.tokenizer.encode(source)[:-1], self.tokenizer.encode(target)

            if len(source_tokens) >= self.max_len:
                # additional <sep_token> and <eos_token>
                input_ids = source_tokens[:self.max_len - 2] + [self.tokenizer.sep_token_id] + [self.tokenizer.eos_token_id]
                sep_idx = self.max_len-1
                yield torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(sep_idx)
            else:
                input_ids = source_tokens + [self.tokenizer.sep_token_id] + target_tokens
                sep_idx = len(source_tokens)
                if len(input_ids) >= self.max_len:
                    input_ids = input_ids[:self.max_len-1] + [self.tokenizer.eos_token_id]
                    yield torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(sep_idx)
                else:
                    attention_mask = [1]*len(input_ids) + [0]*(self.max_len-len(input_ids))
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_len-len(input_ids))
                    yield torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(sep_idx)


def dataset(tokenizer,
            mode='train',
            root: str = './data',
            file_name: str = 'jp_text_sum.csv',
            max_len: int=512,
            batch_size: int = 4) -> DataLoader:

    if mode not in ['train', 'valid', 'test']:
        raise ValueError(
            "`mode` must be in the values: 'train', 'valid', or 'test'")

    logger.info(f"Creating {mode} iter dataset...")
    tensors = JPIterDataset(tokenizer=tokenizer,
                            max_len=max_len,
                            mode=mode,
                            root=root,
                            file_name=file_name)
    logger.info(f'Creating {mode} loader...')
    iterator = DataLoader(tensors, batch_size=batch_size)
    logger.info("Done!")
    return iterator

# def dataset(tokenizer,
#             root: str = './',
#             file_name: str = 'jp_text_summary.csv',
#             mode: str = 'train',
#             batch_size: int = 2,
#             max_seq_len: int = 512):

#     global logger
#     if mode not in ['train', 'test']:
#         raise ValueError("`mode` must be in the values: 'train' or 'test'")

#     logger.info('reading dataset...')
#     data = pd.read_csv(os.path.join(root, file_name))

#     if mode == 'train':
#         logger.info("creating train and test loader...")
#         train_data = data[data.is_train]
#         valid_data = data[data.val_data]

#         train_data = [row.text+tokenizer.sep_token +
#                       row.abstract for row in train_data.itertuples()]
#         valid_data = [row.text+tokenizer.sep_token +
#                       row.abstract for row in valid_data.itertuples()]

#         train_token = tokenizer.batch_encode_plus(
#             train_data, return_tensors='pt', return_token_type_ids=True, max_length=max_seq_len, padding=True, truncation=True)
#         valid_token = tokenizer.batch_encode_plus(
#             valid_data, return_tensors='pt', return_token_type_ids=True, max_length=max_seq_len, padding=True, truncation=True)

#         train_token = TensorDataset(
#             train_token['input_ids'], train_token['attention_mask'])
#         valid_token = TensorDataset(
#             valid_token['input_ids'], valid_token['attention_mask'])

#         train_iter = DataLoader(
#             train_token, batch_size=batch_size, shuffle=True)
#         valid_iter = DataLoader(
#             valid_token, batch_size=batch_size, shuffle=False)

#         return train_iter, valid_iter
#     else:
#         logger.info("creating test loader...")
#         test_data = data[data.is_test]
#         test = [row.text+tokenizer.sep_token +
#                 row.abstract for row in test_data.itertuples()]
#         test_token = tokenizer.batch_encode_plus(
#             test, return_tensors='pt', return_token_type_ids=True, max_length=max_seq_len, padding=True, truncation=True)
#         test_token = TensorDataset(
#             test_token['input_ids'], test_token['attention_mask'])

#         test_iter = DataLoader(test_token, batch_size=batch_size, shuffle=True)

#         return test_iter

# class JPDataset(Dataset):

#     def __init__(self, root: str='./', file_name: str='./jp_text_summary.csv', mode: str='train') -> None:
#         super(JPDataset).__init__()

#         self.mode = mode
#         self.root = root
#         self.file_name = file_name

#         if mode not in ['train', 'valid', 'test']:
#             raise ValueError("`mode` must be in the values: 'train', 'valid', or 'test'")

#         self.data = pd.read_csv(os.path.join(self.root, self.file_name))
#         if mode=='train':
#             self.data = self.data[self.data.is_train]
#         elif mode=='valid':
#             self.data = self.data[self.data.is_val]
#         else:
#             self.data = self.data[self.data.is_test]

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, index):
#         tmp = self.data.iloc[index]

#         return {'text': tmp.text, 'abstract': tmp.abstract}