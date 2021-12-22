import argparse
import os
import time
from torch.nn.modules import loss
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import numpy as np

from utils.dataset import dataset
from utils.utils import create_logger, EarlyStopping, set_seed
logger = create_logger()


def initialize(device,
               ignore_index: int,
               len_train_iter: int,
               num_freeze_layers: int = 18,
               epochs: int = 20,
               lr: float = 1e-5):

    logger.info('Loading Rinna-GPT2-medium model...')
    model = AutoModelForCausalLM.from_pretrained('rinna/japanese-gpt2-medium')
    model.to(device)
    logger.info('Loaded Rinna-GPT2-medium model successfully!\n')

    freeze_layers = [
        f'transformer.h.{str(idx)}' for idx in range(num_freeze_layers)]
    for name, param in model.named_parameters():
        if param.requires_grad and any(freeze_layer in name for freeze_layer in freeze_layers):
            param.requires_grad = False

    no_decay = ['bias', 'ln_1.weight', 'ln_2.weight']
    param_optimizer = [[name, param] for name,
                       param in model.named_parameters() if param.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
         'weight_decay': 0.015},
        {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    n_steps = len_train_iter * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_training_steps=n_steps, num_warmup_steps=100)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    return model, optimizer, scheduler, criterion, epochs


def step(model, criterion, optimizer, scheduler, batch, device):

    input_ids, attention_mask, _ = tuple(t.to(device) for t in batch)
    target_ids = input_ids.detach().clone()
    # batch_size = input_ids.size(0)
    # input_ids, target_ids size of: [batch_size, seq_len]

    optimizer.zero_grad()
    # logits size of: [batch_size, seq_len, vocab_size]
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))

    loss = criterion(logits, target_ids[:, 1:].contiguous().view(-1))

    # logits = torch.cat(tuple(filter(lambda t: t.numel() > 0, [logits[idx][sep_idx[idx]:-1] for idx in range(batch_size)])))
    # target_ids = torch.cat(tuple(filter(lambda t: t.numel() > 0, [target_ids[idx][sep_idx[idx]+1:] for idx in range(batch_size)])))
    # or
    # logits = torch.cat([logits[idx][sep_idx[idx]:-1] for idx in range(batch_size)]).contiguous().view(-1, logits.size(-1))
    # target_ids = torch.cat([target_ids[idx][sep_idx[idx]+1:] for idx in range(batch_size)]).contiguous().view(-1)
    # logits size of: [*, vocab_size]
    # target_ids size of: [*]

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    optimizer.step()
    scheduler.step()

    return loss.item() 


def validate(model, criterion, val_iter, device):
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for batch in val_iter:
            input_ids, attention_mask, _ = tuple(t.to(device) for t in batch)
            target_ids = input_ids.detach().clone()
            # batch_size = input_ids.size(0)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))

            loss = criterion(logits, target_ids[:, 1:].contiguous().view(-1))

            running_loss += loss.item()
    val_loss = running_loss/(len(val_iter))
    perplexity = torch.exp(torch.tensor(val_loss))

    return val_loss, perplexity


def train(model, criterion, optimizer, scheduler, train_iter, val_iter, check_point, epochs, device,
        patience: int = 5,
        delta: float = 1e-6):

    early_stopping = EarlyStopping(patience=patience, delta=delta)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        start = time.time()
        model.train()
        running_loss = 0.0

        for idx, batch in enumerate(train_iter):

            loss = step(model, criterion, optimizer, scheduler, batch, device)
            running_loss += loss

            if (idx+1) % 50 == 0 or idx == 0:
                print("Epoch: {}/{} - iter: {}/{} - train_loss: {}".format(epoch + 1, epochs, idx+1, len(train_iter), running_loss/(idx+1)))
        else:
            train_loss = running_loss/(len(train_iter))
            print("Epoch: {}/{} - iter: {}/{} - train_loss: {}\n".format(epoch +
                  1, epochs, idx + 1, len(train_iter), train_loss))

            print('Evaluating...')
            val_loss, perplexity = validate(model, criterion, val_iter, device)
            print("    Val loss: {} - perplexity: {}\n".format(val_loss, perplexity))

            logger.info(f"Saving model to {os.path.join(check_point, 'cp'+str(epoch+1)+'.pt')}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss
            }, os.path.join(check_point, 'cp'+str(epoch+1)+'.pt'))

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info(
                    f"Early stopping. Saving log loss in {os.path.join(check_point, 'log_loss.txt')}")
                break
            logger.info(f"Total time per epoch: {time.time()-start} seconds")
    train_losses, val_losses = np.array(train_losses).reshape(-1, 1), np.array(val_losses).reshape(-1, 1)
    np.savetxt(os.path.join(check_point, 'log_loss.txt'), np.hstack((train_losses, val_losses)), delimiter='#')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./data', help='data root dir')
    parser.add_argument('--file_name', type=str, default='jp_text_sum.csv', help='data file name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--max_seq_len', type=int, default=512, help='max sequence length')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='dir to save checkpoints')
    parser.add_argument('--num_freeze_layers', type=int, default=18, help='number of freeze layers in GPT-2 model')
    parser.add_argument('--patience', type=int, default=5, help='Patience early stopping')
    parser.add_argument('--delta', type=float, default=1e-6, help='Delta early stopping')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    ROOT_DIR = args.root_dir
    FILE_NAME = args.file_name
    BATCH_SIZE = args.batch_size
    MAX_LEN = args.max_seq_len
    EPOCHS = args.epochs
    LR = args.lr
    CHECKPOINT = args.checkpoint
    NUM_FREEZE_LAYERS = args.num_freeze_layers
    PATIENCE = args.patience
    DELTA = args.delta

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed()

    logger.info(f"Loading T5Tokenizer for 'rinna/japanese-gpt2-medium' model...")
    tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')
    tokenizer.do_lower_case = True
    ignore_index = tokenizer.pad_token_id

    train_iter = dataset(tokenizer=tokenizer,
                         mode='train',
                         root=ROOT_DIR,
                         file_name=FILE_NAME,
                         max_len=MAX_LEN,
                         batch_size=BATCH_SIZE)

    val_iter = dataset(tokenizer=tokenizer,
                       mode='valid',
                       root=ROOT_DIR,
                       file_name=FILE_NAME,
                       max_len=MAX_LEN,
                       batch_size=BATCH_SIZE)

    logger.info('Initializing model...\n')
    model, optimizer, scheduler, criterion, epochs = initialize(device=DEVICE,
                                                                ignore_index=ignore_index,
                                                                len_train_iter=len(train_iter),
                                                                num_freeze_layers=NUM_FREEZE_LAYERS,
                                                                epochs=EPOCHS,
                                                                lr=LR)
    if args.resume:
        ckp = torch.load(args.resume)
        model.load_state_dict(ckp['model_state_dict'])
        optimizer.load_state_dict(ckp['optimizer_state_dict'])
        scheduler.load_state_dict(ckp['scheduler_state_dict'])

    logger.info(f"Saving config to: {os.path.join(CHECKPOINT, 'config.pt')}")
    torch.save({
        'max_len': MAX_LEN,
        'batch_size': BATCH_SIZE,
        'ignore_index': ignore_index,
        'len_train_iter': len(train_iter),
        'num_freeze_layers': NUM_FREEZE_LAYERS,
        'epochs': epochs,
        'lr': LR
    }, os.path.join(CHECKPOINT, 'config.pt'))

    logger.info('Training model...\n')
    start = time.time()
    train(model=model,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          train_iter=train_iter,
          val_iter=val_iter,
          check_point=CHECKPOINT,
          epochs=EPOCHS,
          device=DEVICE,
          patience=PATIENCE,
          delta=DELTA)
    print(f'Total time: {(time.time()-start)} seconds')


if __name__ == '__main__':
    main()
