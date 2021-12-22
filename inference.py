import random, os
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
import pandas as pd
import argparse
from transformers import T5Tokenizer, top_k_top_p_filtering, AutoModelForCausalLM
import torch
import torch.nn as nn

from utils.utils import create_logger

logger = create_logger()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"using seed: {seed}")


def summary(text: str,
            device,
            model: AutoModelForCausalLM,
            tokenizer: T5Tokenizer,
            max_seq_len: int = 512,
            summary_max_len: int = 64) -> str:
    
    tokens = tokenizer.encode(text=text,
                            return_tensors='pt',
                            max_length=max_seq_len,
                            truncation=True)[:-1].to(device) + [tokenizer.sep_token_id]
    # separated token
    # only decode to string over tokens are generated from system: token_idx>sep_idx
    sep_idx = len(tokens)-1
    with torch.no_grad():
        for _ in range(summary_max_len):
            last_logit = model(tokens).logits[:, -1]

            filter = top_k_top_p_filtering(last_logit, top_k=50, top_p=1.0)
            props = nn.functional.softmax(filter, dim=-1)
            final_token = torch.multinomial(props, num_samples=1)
            if final_token[0, 0].cpu().numpy() == tokenizer.eos_token_id:
                return tokenizer.decode(tokens.tolist()[0][sep_idx:])

            tokens = torch.cat([tokens, final_token], dim=-1)
        return tokenizer.decode(tokens.tolist()[0][sep_idx:])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--summary_max_len', type=int, default=64, help='number of summary tokens will be generated')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--root_dir', type=str, default='./data')
    parser.add_argument('--file_name', type=str, default='jp_text_sum.csv')
    args = parser.parse_args()

    SUMMARY_MAX_LEN = args.summary_max_len
    CHECK_POINT = args.checkpoint
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed()

    config_file = os.path.join('/'.join(CHECK_POINT.split('/')[:-1]), 'config.pt')
    logger.info(f"Loading config file from: {config_file}...")
    config = torch.load(config_file)

    # data = pd.read_csv(os.path.join(args.root_dir, args.file_name))
    # test_data = data[data.is_test].sample(frac=0.02, random_state=42)
    
    logger.info(f"Loading T5Tokenizer for 'rinna/japanese-gpt2-medium' model...")
    tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')
    tokenizer.do_lower_case = True
    logger.info("Loading 'rinna/japanese-gpt2-medium' model...")
    model = AutoModelForCausalLM.from_pretrained('rinna/japanese-gpt2-medium')
    model.to(DEVICE)
    logger.info(f'Loading checkpoint from: {CHECK_POINT}')
    ckp = torch.load(CHECK_POINT)
    model.load_state_dict(ckp['model_state_dict'])


    # sys_summaries = []
    # for artl in tqdm(test_data.text, desc='Generating text summary', ncols=100, nrows=5):
    #     sys_summary = summary(text=artl,
    #                     device = DEVICE,
    #                     model=model,
    #                     tokenizer=tokenizer,
    #                     summary_max_len=SUMMARY_MAX_LEN)
    #     sys_summaries.append(sys_summary)
    # logger.info(f"Saving system summaries to {os.path.join(args.root_dir, 'sys_summaries.csv')}")
    # df = pd.DataFrame({'articles': test_data.text.tolist() ,'reference_sum': test_data.abstract.tolist(), 'sys_sum': sys_summaries}, columns=['text', 'reference_sum', 'sys_sum'])
    # df.to_csv(os.path.join(args.root_dir, 'sys_summaries.csv'))
    
    
    while True:
        text = input("Enter text here: ")
        s = summary(text=text,
                    device = DEVICE,
                    model=model,
                    tokenizer=tokenizer,
                    max_seq_len=config['max_len'],
                    summary_max_len=SUMMARY_MAX_LEN)
        print(s)
        # tokens = tokenizer(text=text,
        #                    return_attention_mask=True,
        #                    return_tensors='pt',
        #                    padding='max_length',
        #                    truncation=True,
        #                    max_length=max_len)
        # with torch.no_grad():
        #     for _ in range(SUMMARY_MAX_LEN):
        #         last_logits = model(**tokens).logits[:, -1]
        #         filter = top_k_top_p_filtering(
        #             last_logits, top_k=50, top_p=1.0)
        #         props = nn.functional.softmax(filter, dim=-1)
        #         final_token = torch.multinomial(props, num_samples=1)
        #         if final_token[0, 0].numpy() == tokenizer.eos_token_id:
        #             break
        #         tokens = torch.cat([tokens, final_token], dim=-1)
        # summay = tokenizer.decode(tokens.tolist()[0])
        # print(summay)


if __name__ == '__main__':
    main()
