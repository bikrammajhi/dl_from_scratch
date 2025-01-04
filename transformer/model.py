import torch
import torch.nn as nn   
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sine to the even postions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # Add a batch dimension, (1, seq_len, d_model)
        
        # Register the positional encoding as a buffer, so it will be saved in the model's state_dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False) # Slice the positional encoding to match the input shape
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative parameter
        self.bias = nn.Parameter(torch.zeros(1)) # Additive parameter
        
    def forward(self, x):
        mean =  x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * [(x - mean) / (std + self.eps)] + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float)->None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and B2
    
    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return x

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        
        # Create the weight matrices for Q, K, V and the output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout, mask=None):
        d_k = query.shape[-1]
        
        # (batch_size, num_heads, seq_len, d_k) x (batch_size, num_heads, d_k, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        x = torch.matmul(attention_scores, value)
        
        return x, attention_scores
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        # Linearly transform the q, k, v and output
        query = self.W_q(q) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.W_k(k)   # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        value = self.W_v(v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        
        # Split the Q, K, V and output into num_heads
        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, num_heads, d_k) -> (batch_size, num_heads, seq_length, d_k)
        query = query.view(batch_size, query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, value.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        
        x, self.attn = self.attention(query, key, value, self.d_k, mask=mask, dropout=self.dropout)
        
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return self.W_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, sublayer: nn.Module, dropout: float)->None:
        super().__init__()
        self.sublayer = sublayer
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return x + self.dropout(self.sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttentionBlock, Feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.Feed_forward_block = Feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.Feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int)-> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Batch_size, seq_len, vocab_size -> Batch_size, seq_len, d_model
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_pos(self.src_embedding(src)) # Add positional encoding to the input  
        return self.encoder(src, src_mask)        # Encode the input
    
    def decoder(self, encoder_output, tgt, src_mask, tgt_mask):
        src = self.tgt_pos(self.tgt_embedding(tgt))     # Add positional encoding to the input
        return self.decoder(src, encoder_output, src_mask, tgt_mask)    # Decode the input
    
    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff = 2048)->Transformer:
    # Create the input embeddings
    src_embedding = InputEmbedding(src_vocab_size, d_model)
    trg_embedding = InputEmbedding(tgt_vocab_size, d_model)
    
    # Create the positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embedding, trg_embedding, src_pos, trg_pos, projection_layer)

    # Initialize the weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer