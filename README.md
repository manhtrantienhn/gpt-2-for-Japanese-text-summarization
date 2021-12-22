# Fine-tune GPT-2 for Japanese text summarization

## Directory
    ├── checkpoint/
    ├── log/
    ├── data/
    │   ├── jp_text_sum_extend.csv
    ├── utils/
    │   ├── dataset.py
    │   ├── utils.py
    ├── train.py
    ├── test.py
    └── inference.py

## Install dependencies and make necessary folders:
```shell
cd gpt-2-for-Japanese-text-summarization-main
mkdir checkpoint log data
```
First, you must install dependencies, run the following command:
```shell
pip install -r requirements.txt
```

## Data
You can download processed data from [here](https://drive.google.com/file/d/1X7zq1oqIaZo-gcToXqxqwD_sYjdo6e6w/view?usp=sharing), or raw text from [here](https://drive.google.com/file/d/1ZaKB5q6UN_3XGCj-jo-9Q-j-koUqaDol/view?usp=sharing), then put them to ``├──data/`` folder
## Training

For training from scratch, you can run command like this:

```shell
python3 train.py --root_dir ./data/ --file_name jp_text_sum_extend.csv --batch_size 2 --max_seq_len 512 --epochs 20 --lr 5e-5 --checkpoint ./checkpoint/ --num_freeze_layers 18 --patience 5 --delta 1e-6
```

For resume with the checkpoint, code may be:
```shell
python3 train.py --root_dir ./data/ --file_name jp_text_sum_extend.csv --batch_size 2 --max_seq_len 512 --epochs 20 --lr 5e-5 --checkpoint ./checkpoint/ --num_freeze_layers 18 --patience 5 --delta 1e-6 --resume path-to-the-checkpoint-is-resumed
```

## Testing

For evaluation, the command may like this:

```shell
python3 test.py --root_dir ./data/ --file_name jp_text_sum_extend.csv --checkpoint path-to-the-best-checkpoint
```

## Inference
Generate text with your model:
```shell
python3 inference.py --root_dir ./data/ --file_name jp_text_sum_extend.csv --checkpoint path-to-the-best-checkpoint --summary_max_len 64
```