import torch
import torch.nn as nn
import torch.nn.functional as F

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000, sigma: float = 0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)

class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)

        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        # b, nS, k, d
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient) / ctx.num_samples / ctx.sigma
        )
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None] * 5)

class Time2Vec(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.dim = embedding_dim

        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))
        
        self.w = nn.Parameter(torch.randn(1, self.dim - 1))
        self.b = nn.Parameter(torch.randn(1, self.dim - 1))

    def forward(self, t):
        t = t.unsqueeze(-1)
        
        linear = t * self.w0 + self.b0
        periodic = torch.sin(t * self.w + self.b)
        
        return torch.cat([linear, periodic], dim=-1)

class FinEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, temporal_embedding_dim):
        super().__init__()

        self.dim = embedding_dim + temporal_embedding_dim

        self.linear = nn.Linear(input_dim, self.dim - temporal_embedding_dim)
        self.time_embed = Time2Vec(temporal_embedding_dim)
    
    def forward(self, x, t):
        
        stock_vector = self.linear(x) # [batch_size, 30, 60, embedding_dim]
        time_vector = self.time_embed(t) # [60, temporal_embedding_dim]
    
        # [B, 60, T_dim] -> [B, 1, 60, T_dim]
        time_vector = time_vector.unsqueeze(1)
            
        # Expand into [batch_size, 30, 60, temporal_embedding_dim]
        batch_size = x.size(0)
        num_stocks = x.size(1)
        time_vector = time_vector.expand(batch_size, num_stocks, -1, -1)
        
        return torch.cat([stock_vector, time_vector], dim=-1)

class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1, is_causal=False):
        super().__init__()

        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.is_causal = is_causal
    
    def forward(self, x, y, mask_x=None, mask_y=None):

        orig_shape = x.shape  # (B, 30, 60, model_dim)
        
        # [B, 30, 60, 512] -> [B * 30, 60, 512] (example only)
        x = x.flatten(0, 1)
        y = y.flatten(0, 1)
        
        # [B, 30, 60] -> [B * 30, 60]
        if mask_x is not None:
            mask_x = mask_x.flatten(0, 1)
        if mask_y is not None:
            mask_y = mask_y.flatten(0, 1)

        norm_x = self.norm_q(x)
        norm_y = self.norm_kv(y)

        # check which batch items have fully masked y
        if mask_y is not None:
            all_masked_y = mask_y.all(dim=-1)  # [B], True = entire y is masked
            # temporarily unmask one position to avoid nan
            safe_mask_y = mask_y.clone()
            safe_mask_y[all_masked_y, 0] = False
        else:
            all_masked_y = None
            safe_mask_y = None
        
        attn_mask = None
        if self.is_causal:
            sz = x.size(1)
            attn_mask = torch.triu(torch.full((sz, sz), float('-inf'), device=x.device), diagonal=1)

        attn_out, _ = self.attention(
            norm_x, norm_y, norm_y,
            key_padding_mask=safe_mask_y,
            attn_mask=attn_mask
        )
        attn_out = self.dropout(attn_out)

        # zero out output for fully masked cases
        if all_masked_y is not None:
            attn_out[all_masked_y] = 0.0
        if mask_x is not None:
            attn_out[mask_x] = 0.0
        
        out = x + attn_out

        return out.view(orig_shape)

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, expansion=4, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * expansion, embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        norm_x = self.norm(x)
        ffn_out = self.ff(norm_x)

        if mask is not None:
            ffn_out[mask.unsqueeze(-1).expand_as(ffn_out)] = 0.0
        
        return x + ffn_out

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, expansion=4, dropout=0.1, is_causal=False):
        super().__init__()

        self.attn_blk = AttentionBlock(embedding_dim, num_heads, dropout, is_causal)
        self.ffnn = FeedForward(embedding_dim, expansion, dropout)


    def forward(self, x, y, mask_x=None, mask_y=None):
        x = self.attn_blk(x, y, mask_x, mask_y)
        x = self.ffnn(x, mask_x)
        return x

class TransformerLayers(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers=1, expansion=4, dropout=0.1, is_causal=False):
        super().__init__()

        self.transformer = nn.ModuleList([
            TransformerLayer(embedding_dim, num_heads, expansion, dropout, is_causal)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, y, mask_x=None, mask_y=None):
        for layer in self.transformer:
            x = layer(x, y, mask_x, mask_y)
        return x

class StockTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        temporal_embedding_dim,
        num_heads,
        num_layers=1,
        expansion=4,
        dropout=0.1
    ):
        super().__init__()

        self.fin_embed = FinEmbedding(input_dim, embedding_dim, temporal_embedding_dim)

        model_dim = self.fin_embed.dim
        self.time_series_transformer = TransformerLayers(
            model_dim, num_heads, num_layers,
            expansion, dropout, True
        )
        self.inter_stock_transformer = TransformerLayers(
            model_dim, num_heads, 1,
            expansion, dropout
        )
        self.projection = nn.Linear(model_dim, 2)
    
    def forward(self, x, t, mask):
        x = self.fin_embed(x, t)
        x = self.time_series_transformer(x, x, mask, mask)

        x = x.transpose(-3, -2).contiguous() # Becomes [B, 60, 30, 512]
        perm_mask = mask.transpose(-2, -1).contiguous() if mask is not None else None
        x = self.inter_stock_transformer(x, x, perm_mask, perm_mask)

        if mask is not None:
            # [B, 30, 60] -> [B, 60, 30]
            t_mask = mask.transpose(-2, -1) 
            active_mask = ~t_mask  # True = Active Data
            
            time_indices = torch.arange(x.size(1), device=x.device).view(1, -1, 1) # Shape: [1, 60, 1]
            masked_indices = active_mask * time_indices # Shape: [B, 60, 30]

            last_active_idx = masked_indices.argmax(dim=1) # Shape: [B, 30]
            
            has_activity = active_mask.any(dim=1) # Shape: [B, 30]
            last_active_idx = torch.where(has_activity, last_active_idx, 0)
            
            # [B, 30] -> [B, 1, 30]
            gather_idx = last_active_idx.unsqueeze(1)
            
            # [B, 1, 30] -> [B, 1, 30, 1]
            gather_idx = gather_idx.unsqueeze(-1)
            
            # [B, 1, 30, model_dim]
            model_dim = x.size(-1)
            gather_idx = gather_idx.expand(-1, -1, -1, model_dim)
            
            # B, 1, 30, model_dim] -> [B, 30, model_dim]
            last_timestamp = torch.gather(x, dim=1, index=gather_idx).squeeze(1)

        else:
            last_timestamp = x[:, -1, :, :] # Shape: [B, 30, model_dim]

        out = self.projection(last_timestamp) # [B, 30, 2]

        return out

class DynamicSelection(nn.Module):
    def __init__(self, input_dim, K):

        self.down_project = nn.Linear(input_dim, input_dim // 2)
        self.score = nn.Linear(2 * (input_dim // 2), 1)
        self.topk = PerturbedTopK
    
    def forward(self, x, mask):    # Expecting (batch_size, max_num_embeddings, embedding_dim)
        masked_embeddings = self.down_project(x) * mask
        embeddings_sum = torch.sum(masked_embeddings, dim=-2)
        valid_count = torch.sum(mask, dim=-1).clamp(min=1)
        masked_mean = embeddings_sum / valid_count

        masked_mean = masked_mean.unsqueeze(0).unsqueeze(0).expand_as(masked_embeddings)

        masked_embeddings = torch.cat([masked_mean, masked_embeddings], dim=-1) * mask

        scores = self.score(masked_embeddings) * mask




if __name__ == "__main__":
    # Fake configuration matching your specs
    B, num_stocks, num_timestamps, input_features = 16, 30, 60, 10
    embed_dim, temp_dim = 512, 32
    
    model = StockTransformer(
        input_dim=input_features,
        embedding_dim=embed_dim,
        temporal_embedding_dim=temp_dim,
        num_heads=8,
        num_layers=2
    )
    
    # Generate dummy input variables
    dummy_x = torch.randn(B, num_stocks, num_timestamps, input_features)
    dummy_t = torch.randn(B, num_timestamps) # Timeline vectors
    dummy_mask = torch.zeros(B, num_stocks, num_timestamps).bool() # Empty padding mask
    
    # Run a forward pass execution
    preds = model(dummy_x, dummy_t, dummy_mask)
    print("Execution Successful! Prediction matrix shape:", preds.shape) 
    # Should cleanly output: torch.Size([16, 30, 2])