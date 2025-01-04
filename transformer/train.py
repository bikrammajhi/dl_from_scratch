import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import BilingualDataset, casual_mask

from dataset import load_dataset
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentense(ds, lang):
    for i in range(len(ds)):
        yield ds[i][lang]

def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_path'] = '..tokenizers/tokenizer_{}.json'
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentense(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["src_lang"]}-{config["tgt_lang"]}', split='train')
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tgt_lang'])
    
    # Keep 90% of the dataset for training and 10% for validation
    train_ds_size = int(len(ds_raw)*0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['seq_len'], config['lang_src'], config['lang_tgt'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['seq_len'], config['lang_src'], config['lang_tgt'])
    
    max_len_seq = 0
    max_len_tgt = 0
    
    # Find the maximum sequence length in the training dataset
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        
        max_len_seq = max(max_len_seq, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Maximum sequence length in the source language: {max_len_seq}")
    print(f"Maximum sequence length in the target language: {max_len_tgt}")
        
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
