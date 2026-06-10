import torch
import torch.nn as nn
import torch.nn.functional as F

def top_k_one_hot(tensor, k, dim=-1):
    one_hot = torch.zeros_like(tensor)
    top_k_indices = torch.topk(tensor, k, dim=dim).indices
    one_hot.scatter_(dim, top_k_indices, 1)
    return one_hot

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000, sigma: float = 0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)
    
    def set_sigma(self, sigma: float):
        self.sigma = max(sigma, 1e-6)  # clamp to avoid division by zero in backward

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

        # shape of x: (batch_size, num_stocks, window_size, num_features)
        # shape of t: (batch_size, num_stocks, window_size)
        
        stock_vector = self.linear(x) # (batch_size, num_stocks, window_size, embedding_dim)
        time_vector = self.time_embed(t) # (batch_size, num_stocks, window_size, temporal_embedding_dim]
        
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
            mask_x = mask_x.flatten(0, 1).bool()
        if mask_y is not None:
            mask_y = mask_y.flatten(0, 1).bool()

        norm_x = self.norm_q(x)
        norm_y = self.norm_kv(y)

        # check which batch items have fully masked y
        if mask_y is not None:
            all_masked_y = mask_y.all(dim=-1).bool()  # [B], True = entire y is masked
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
            ffn_out[mask.unsqueeze(-1).expand_as(ffn_out).bool()] = 0.0
        
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

        self.dim = self.fin_embed.dim
        self.time_series_transformer = TransformerLayers(
            self.dim, num_heads, num_layers,
            expansion, dropout, True
        )
        self.inter_stock_transformer = TransformerLayers(
            self.dim, num_heads, 1,
            expansion, dropout
        )
        self.projection = nn.Linear(self.dim, 2)
    
    def time_series_transform(self, x, t, mask):
        x = self.fin_embed(x, t)
        x = self.time_series_transformer(x, x, mask, mask)
        return x
    
    def inter_stock_transform(self, x, mask):
        x = x.transpose(-3, -2).contiguous() # Becomes [B, 60, 30, 512]
        perm_mask = mask.transpose(-2, -1).contiguous().bool() if mask is not None else None
        x = self.inter_stock_transformer(x, x, perm_mask, perm_mask)

        if mask is not None:
            # [B, 30, 60] -> [B, 60, 30]
            t_mask = mask.transpose(-2, -1).bool()
            active_mask = ~t_mask  # True = Active Data
            
            time_indices = torch.arange(x.size(1), device=x.device).view(1, -1, 1) # Shape: [1, 60, 1]
            masked_indices = active_mask * time_indices # Shape: [B, 60, 30]

            last_active_idx = masked_indices.argmax(dim=1) # Shape: [B, 30]
            
            has_activity = active_mask.any(dim=1).bool() # Shape: [B, 30]
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
    
    def forward(self, t, x, mask):
        x = self.time_series_transform(x, t, mask)
        return self.inter_stock_transform(x, mask)

class NewsEmbedding(nn.Module):
    def __init__(self, embedding_dim, temporal_embedding_dim):
        super().__init__()

        self.dim = embedding_dim + temporal_embedding_dim

        self.time_embed = Time2Vec(temporal_embedding_dim)
    
    def forward(self, x, t):
        time_vector = self.time_embed(t) # [B, N, temporal_embedding_dim]
        
        return torch.cat([x, time_vector], dim=-1) # [B, N, embedding_dim, temporal_embedding_dim]

class DynamicSelection(nn.Module):
    def __init__(self, input_dim, K):
        super().__init__()
        self.down_project = nn.Linear(input_dim, input_dim // 2)
        self.score = nn.Linear(2 * (input_dim // 2), 1)
        self.topk = PerturbedTopK(K, 500, 0.05)

    def forward(self, x, mask):  # x: (B, N, D), mask: (B, N)
        mask_3d = mask.unsqueeze(-1)  # (B, N, 1)

        projected = self.down_project(x) # (B, N, D // 2)
        masked_projected = projected * mask_3d # (B, N, D // 2)

        embeddings_sum = masked_projected.sum(dim=1) # (B, D // 2)
        valid_count = mask.sum(dim=-1, keepdim=True).clamp(min=1) # (B, 1)
        masked_mean = (embeddings_sum / valid_count) # (B, D // 2)
        masked_mean = masked_mean.unsqueeze(1).expand_as(masked_projected)  # (B, N, D // 2)

        combined = torch.cat([masked_mean, masked_projected], dim=-1) * mask_3d  # (B, N, D)

        scores = self.score(combined).squeeze(-1)                 # (B, N)
        scores = scores.masked_fill(mask == 0, float('-inf'))     # mask out padding before topk

        indicators = self.topk(scores)                            # (B, K, N)

        selected = torch.einsum("bkn,bnd->bkd", indicators, x)   # (B, K, D)

        return selected

class StockNewsTransformer(StockTransformer):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        temporal_embedding_dim,
        num_heads,
        K,
        num_layers=1,
        expansion=4,
        dropout=0.1
    ):
        super().__init__(
            input_dim,
            embedding_dim,
            temporal_embedding_dim,
            num_heads,
            num_layers,
            expansion,
            dropout
        )

        self.news_embed = NewsEmbedding(embedding_dim, temporal_embedding_dim)
        self.news_selection = DynamicSelection(self.dim, K)
        self.topk = self.news_selection.topk

        self.news_fusion_layer = TransformerLayers(
            self.dim, num_heads, num_layers,
            expansion, dropout, True
        )
    
    def news_fusion_transform(self, x, news, t_news, x_mask, news_mask):
        news = self.news_embed(news, t_news) # (B, N, D)
        news = self.news_selection(news, news_mask) # (B, N, D) -> (B, K, D)

        news = news.unsqueeze(1) # (B, K, D) -> # (B, 1, K, D)

        num_stocks = x.size(1)
        news = news.expand(-1, num_stocks, -1, -1) # (B, 1, K, D) -> (B, 30, K, D)

        return self.news_fusion_layer(x, news, x_mask)
    
    def forward(self, t, t_news, x, news, x_mask, news_mask):
        x = self.time_series_transform(x, t, x_mask)
        x = self.news_fusion_transform(x, news, t_news, x_mask, news_mask)
        return self.inter_stock_transform(x, x_mask)

class SigmaAnnealer:
    def __init__(self, model: StockNewsTransformer, sigma_start=0.05, sigma_end=1e-4, num_epochs=50):
        self.topk = model.topk
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.num_epochs = num_epochs

    def step(self, epoch: int):
        t = epoch / self.num_epochs
        sigma = self.sigma_start * (self.sigma_end / self.sigma_start) ** t

        self.topk.set_sigma(sigma)
        return sigma

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