import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """	Split the image inot patches and then embed them.

    Paramaters:
    ----------------
    img_size: int
        The size of the image (it is a square).
    patch_size: int
        The size of the patch (it is a square).
    in_chans: int
        The number of input channels.
    embed_dim: int
        The dimension of the embedding (output dimension).
  
    Attributes
    ----------------
    n_patches: int
        The number of patches in the image.
    proj: nn.Conv2d
        The convolutional layer that does the both splitting and embedding.
    """
    def __init__(self, img_size, path_size, in_chans, embed_dim):
        super(PatchEmbed, self).__init__()
        self.imag_size = img_size
        self.path_size = path_size
        self.n_patches = (img_size // path_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=path_size,
            stride=path_size,
        ) # height, and width of the output image will be n_patches ** 0.5.
        
    
    def forward(self, x):
        """ Run forward pass
        
        Parameters
        ----------------
        x: torch.Tensor
           Shape '(n_samples, in_chans, img_size, img_size)'. 
        
        Returns
        ----------------
        torch.Tensor
            Shape '(n_samples, n_patches, embed_dim)'.
        """
        x = self.proj(x)        # (n_samples, embed_dim, n_pathches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)        # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (n_samples, n_patches, embed_dim)
        return x    
    
class MultiHeadAttention(nn.Module):
    """ Attention mechanism that can use multiple heads to attend to different positions.
    
    Parameters
    ----------------
    d_model: int
        The dimension of input and output embeddings. (512, 1024 etc..)
        
    n_heads: int
        Number of attention heads.
        
    qkv_bias: bool
        if True, then we include bias to the query, key, and value projections.
        
    attn_p: float
        Dropout probability applied to the query, key, and value tensors. 
    
    proj_p: float
        Dropout probability applied to the output tensor. 
        
    Attributes
    ----------------
    scale: float
        Normalizing constant for the dot product.
        
    qkv: nn.Linear
        Linear projection for the query, key and value.
    
    proj: nn.Linear
        Linear mapping that takes in the concatenated output of all attention heads and maps it into a new space.
    
    attn_drop, proj_drop: nn.Dropout
        Dropout layers.    
    """	
    
    def __init__(self, d_model, n_heads, qkv_bias=False, attn_p=0., proj_p=0.):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        """ Run forward pass.	
        
        Parameters
        ----------------
        x : torch.Tensor
            Shape '(batch_size, n_patches + 1, d_model)'.
        
        Returns
        ----------------
        torch.Tensor
            Shape '(batch_size, n_patches + 1, d_model)'.
        """
        
        batch_size, n_tokens, d_model = x.shape     # n_tokens = n_patches + 1
        
        if d_model != self.d_model:
            raise ValueError(f'Input embedding dimension {d_model} should match layer embedding dimension {self.d_model}')
        
        qkv = self.qkv(x) # (batch_size, n_patches + 1, 3 * d_model)
        qkv = qkv.reshape(
            batch_size, n_tokens, 3, self.n_heads, self.head_dim
            ) # (batch_size, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
            ) # (3, batch_size, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        k_t = k.transpose(-2, -1)   # (batch_size, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale # (batch_size, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)   # (batch_size, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)
        
        weighted_avg = attn @ v # (batch_size, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (batch_size, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)      # (batch_size, n_patches + 1, d_model)
        
        x = self.proj(weighted_avg) # (batch_size, n_patches + 1, d_model)
        x = self.proj_drop(x)       # (batch_size, n_patches + 1, d_model)
        
        return x
        
        
        
        