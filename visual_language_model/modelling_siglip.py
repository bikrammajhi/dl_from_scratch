from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
        self,  
        hidden_size: int = 768,             # hidden size of the transformer
        intermediate_size: int = 3072,      # feedforward NN size
        num_hidden_layers: int = 12,        # number of transformer layers
        num_attention_heads: int = 12,      # number of attention heads
        num_channels: int = 3,              # number of image channels
        image_size: int = 224,              # image size
        patch_size: int = 16,               # patch size
        layer_norm_eps: float = 1e-6,       # layer norm epsilon
        attention_dropout: float = 0.0,     # attention dropout
        num_image_tokens: int = None,       # number of image tokens
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.path_size = patch_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.path_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,    # We are mapping num_channels to embed_dims
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,        # Each 16x16 pixels will map to a single pixel in feature map
            stride=self.patch_size,
            padding='valid',                    # No padding is added
        )

        self.num_patches = (self.image_size/self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            'position_ids',
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape  # [Batch_size, channels, height, width]
        
        # Convole the 'patch_size' kernel over the image, with no overlapping patches since the stride size is the same as the kerne_size
        # The output of the convolution will have shape [Batch_size, Embed_dim, Num_patches_H, Num_patches_W]
        # Where Num_patches_H = height // patch_size and Num_patches_W = height // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        
        # [Batch_size, Embed_dim, Num_patches_H, Num_patches_W] ->  [Batch_size, Embed_dim, Num_patches] 
        embeddings = patch_embeds.flatten(2)
        
        # [Batch_size, Embed_dim, Num_Patches] -> [Batch_size, Num_patches, Embed_dim]
        embeddings = embeddings.transpose(1, 2)
        
        # Add position embeddings to each patch. Each positional embedding is a vector of size [Embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        # [Batch_size, Num_patches, Embed_dim]
        return embeddings

class SiglipAttention(nn.Module):
    """ Multi-headed attention from "Attention Is All You Need' paper" """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _  = hidden_states.size()
        # query_states: [Batch_size, Num_patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_size, Num_patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_size, Num_patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        
        # query_states: [Batch_size, Num_patches, Embed_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f"{attn_weights.size()}"
            )
        
        # Apply softmax row-wise. attn_weights: [Batch_size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropouts only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f" `attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f"{attn_output.size()}"
            )
        
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous() # It stores the input in a continues memory so that reshape operation can be done without any computational overhead
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights
        


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_size, Num_patches, Embed_Dim] -> [Batch_size, Num_patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # [Batch_size, Num_patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_size, Num_patches, Intermediate_Size] -> [Batch_size, Num_patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [Batch_size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_size, Num_Patches, Embed_Dim] -> [Batch_size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_size, Num_Patches, Embed_Dim] -> [Batch_size, Num_Patches, Embed_Dim]
        hidden_states = self.self_attn(hidden_states = hidden_states)
        # [Batch_size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        # [Batch_size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_size, Num_Patches, Embed_Dim] -> [Batch_size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_size, Num_Patches, Embed_Dim] -> [Batch_size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_size, Num_Patches, Embed_Dim] -> [Batch_size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(self, input_embeds: torch.Tensor)-> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = input_embeds
        
        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)
        return hidden_states
  
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm =  nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_size, channels, Height, Weight] -> [Batch_size, no_pathches, Embed_size]
        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_states = self.encoder(input_embeds = hidden_states)
        
        last_hidden_states = self.post_layernorm(last_hidden_states)
        
        return last_hidden_states
        
        
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values)-> Tuple:
        # [Batch_size, Channels, Height, Weight] -> [Batch_size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)            