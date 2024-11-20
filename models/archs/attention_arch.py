import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple, Union
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, 
                query_dim:int, 
                inner_dim:int, 
                out_dim:int
                ):
        super(SelfAttention, self).__init__()
        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(query_dim, inner_dim)
        self.to_v = nn.Linear(query_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, out_dim)
        
    def forward(self, x):
        # implement self-attention
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        att = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        out = self.to_out(q)
        return out

class Attention(nn.Module):
    def __init__(self,
                query_dim: int,
                cross_attenion_dim: bool=False,
                dim_head: int=  64,
                heads: int=  8,
                kv_heads: Optional[int] = None,
                out_dim: int = None,
                bias: bool = False,
                out_bias: bool = True,
                dropout: float = 0.0,
    ):
        super(Attention, self).__init__()
        self.query_dim = query_dim
        self.cross_attenion_dim = cross_attenion_dim if cross_attenion_dim is not None else query_dim # 支持同token，或kv/q不同token
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads #q/kv 支持不同头数，单头维度相同
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.heads = out_dim//dim_head if out_dim is not None else heads
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attenion_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(cross_attenion_dim, self.inner_kv_dim, bias=bias)
        
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
                                    nn.Dropout(dropout))

    def prepare_attention_mask(self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                encoder_hidden_states: Optional[torch.Tensor]=None,
                attention_mask: Optional[torch.Tensor]=None,
                ):
        r"""
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch, sequence_length, hidden_size)`):
                Input to the layer of shape `(batch, sequence_length, hidden_size)`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch, sequence_length, hidden_size)`, *optional*):

            attention_mask (`torch.FloatTensor` of shape `(batch, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:            
            `torch.Tensor`: The output of the layer after the attention weight has been applied.

        """
        # hidden_states(q): (B, HW, C)
        # encoder_hidden_states(kv): (B, H'W', C)
        # attention_mask: (B, HW, HW)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape(batch_size, channels, height * width).transpose(1, 2) # (B, HW, C)
        
        batch_size, sequence_length, hidden_states_dim = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])
        
        query = self.to_q(hidden_states)# (B, HW, inner_dim)
        
        # selfattention
        if encoder_hidden_states is None:
            key = self.to_k(hidden_states) 
            value = self.to_v(hidden_states)
        else:
            key = self.to_k(encoder_hidden_states)# (B, HW, inner_kv_dim)
            value = self.to_v(encoder_hidden_states)
        
        # Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]`
        # `heads` is the number of heads initialized while constructing the `Attention` class.
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).permute(0, 2, 1, 3) # (B, heads, HW, head_dim)
        
        key = key.view(batch_size, -1, self.heads, head_dim).permute(0, 2, 1, 3) # (B, heads, HW, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim).permute(0, 2, 1, 3) # (B, heads, HW, head_dim)
        
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False)
        
        # linear proj
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads*head_dim) # (B, HW, dim)
        hidden_states = self.to_out(hidden_states)
        
        if input_ndim==4:
            hidden_states = hidden_states.transpose(-1,-2).reshape(batch_size, channels, height, width)
            
        return hidden_states
        
if __name__ == '__main__':
    model = Attention(query_dim=64,
                    cross_attenion_dim=768,
                    dim_head=64,
                    heads=8,
                    )
    query = torch.randn(1, 64, 256, 256)
    text_embedding = torch.randn(1, 77, 768)
    out = model(query, text_embedding)
    print(out.shape)
    
        
