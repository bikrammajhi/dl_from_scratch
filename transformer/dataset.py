import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang)->None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # We will use a fixed sequence length for both the encoder and decoder
        self.sos_token = torch.Tensor(self.tokenizer_tgt.token_to_id("[SOS]"), dtype=torch.long) 
        self.eos_token = torch.Tensor(self.tokenizer_tgt.token_to_id("[EOS]"), dtype=torch.long)
        self.pad_token = torch.Tensor(self.tokenizer_tgt.token_to_id("[PAD]"), dtype=torch.long)
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: any) -> any:
        src_target_pair = self.ds[index] # {'translation': {'en': 'Hello', 'fr': 'Bonjour'}}
        src_text = src_target_pair['translation'][self.src_lang] # 'Hello'
        tgt_text = src_target_pair['translation'][self.tgt_lang] # 'Bonjour'
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # List of token ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids # List of token ids
        
        # Find out how many padding tokens are needed
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 for SOS and EOS tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # -1 EOS tokens
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0: # If the input sequence is too long
            raise ValueError("Input sequence is too long")

        # Add SOS and EOS tokens to the source input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(enc_input_tokens),
                self.eos_token,
                torch.Tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
            ])
        
        # Add EOS token to the target input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(dec_input_tokens, dtype=torch.int64),
                torch.Tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ])
        
        # Add EOS token to the target label (What we expect as output from the decoder)
        label = torch.cat([
            torch.Tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.Tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
        ])
        
        assert encoder_input.size(0) == self.seq_len 
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return{
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1, seq_len) & (1, Seq_len, Seq_len)
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def casual_mask(seq_len):
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int)
    return mask==0 # (1, seq_len, seq_len)
        