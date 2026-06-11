import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

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
        self.sigma = max(sigma, 1e-6)

class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d))
        perturbed_x = x[:, None, :] + noise * sigma
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices
        indices = torch.sort(indices, dim=-1).values
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma
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

def _nan_check(name, tensor):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    print(f"  [{name}] nan={has_nan}, inf={has_inf}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")
    return has_nan or has_inf

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
        _nan_check("time2vec input t", t)

        linear = t * self.w0 + self.b0
        _nan_check("time2vec linear", linear)

        periodic = torch.sin(t * self.w + self.b)
        _nan_check("time2vec periodic", periodic)

        out = torch.cat([linear, periodic], dim=-1)
        _nan_check("time2vec output", out)
        return out

class FinEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, temporal_embedding_dim):
        super().__init__()
        self.dim = embedding_dim + temporal_embedding_dim
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.time_embed = Time2Vec(temporal_embedding_dim)
    
    def forward(self, x, t):
        _nan_check("fin_embed input x", x)
        _nan_check("fin_embed input t", t)

        stock_vector = self.linear(x)
        _nan_check("fin_embed after linear", stock_vector)

        time_vector = self.time_embed(t)
        _nan_check("fin_embed after time2vec", time_vector)

        out = torch.cat([stock_vector, time_vector], dim=-1)
        _nan_check("fin_embed output", out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads

        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def _expand(self, t, x, y, transpose=False):
        tensor = (
            t
            .unsqueeze(2)
            .unsqueeze(2)
            .expand(
                -1, -1, self.num_heads,
                *((y.size(2), x.size(2)) if transpose else (x.size(2), y.size(2)))
            )
            .flatten(0, 2)
        )
        if transpose:
            return tensor.transpose(-2, -1)
        return tensor
    
    def forward(self, tx, ty, x, y, mask_x=None, mask_y=None):
        orig_shape = x.shape

        norm_x = self.norm_q(x.flatten(0, 1))     # (b * n, x_seq, e)
        norm_y = self.norm_kv(y.flatten(0, 1))    # (b * n, y_seq, e)

        print(tx.shape)
        
        tx_copies = self._expand(tx, x, y, transpose=True)
        ty_copies = self._expand(ty, x, y)
        
        attn_mask = tx_copies + 1e-6 < ty_copies\

        if mask_x is not None:
            attn_mask = attn_mask | self._expand(mask_x, x, y, transpose=True)
        if mask_y is not None:
            attn_mask = attn_mask | self._expand(mask_y, x, y)
        
        all_masked_y = attn_mask.all(dim=-1)
        
        attn_mask[all_masked_y, 0] = False
        print(attn_mask[0])

        print(f'norm_x: {norm_x}')
        print(f'norm_y: {norm_y}')
        print(f'attn_mask: {attn_mask}')

        attn_out, attn_weights = self.attention(
            norm_x, norm_y, norm_y,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False
        )
        print(attn_weights)
        _nan_check("attn output", attn_out)

        attn_out = self.dropout(attn_out)

        all_masked_y = all_masked_y.view(6, self.num_heads, attn_out.shape[2])[:, 0, :]
        
        attn_out[all_masked_y] = 0.0
        if mask_x is not None:
            attn_out[mask_x.flatten(0, 1).bool()] = 0.0
        
        out = x + attn_out
        _nan_check("attn residual output", out)

        return out.view(orig_shape), attn_weights.to(torch.float32)

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
        _nan_check("ffn after norm", norm_x)

        ffn_out = self.ff(norm_x)
        _nan_check("ffn output", ffn_out)

        if mask is not None:
            ffn_out[mask.unsqueeze(-1).expand_as(ffn_out).bool()] = 0.0
        
        out = x + ffn_out
        _nan_check("ffn residual output", out)
        return out

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, expansion=4, dropout=0.1, is_causal=False):
        super().__init__()
        self.attn_blk = AttentionBlock(embedding_dim, num_heads, dropout, is_causal)
        self.ffnn = FeedForward(embedding_dim, expansion, dropout)

    def forward(self, x, y, mask_x=None, mask_y=None):
        x, attn_weights = self.attn_blk(x, y, mask_x, mask_y)
        x = self.ffnn(x, mask_x)
        return x, attn_weights

class TransformerLayers(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers=1, expansion=4, dropout=0.1, is_causal=False):
        super().__init__()
        self.transformer = nn.ModuleList([
            TransformerLayer(embedding_dim, num_heads, expansion, dropout, is_causal)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, y, mask_x=None, mask_y=None):
        attn_blocks = []
        for i, layer in enumerate(self.transformer):
            x, attn_weights = layer(x, y, mask_x, mask_y)
            _nan_check(f"transformer layer {i} output", x)
            attn_blocks.append(attn_weights)
        return x, torch.stack(attn_blocks, dim=0)

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
            self.dim, num_heads, num_layers, expansion, dropout, True
        )
        self.inter_stock_transformer = TransformerLayers(
            self.dim, num_heads, 1, expansion, dropout
        )
        self.projection = nn.Linear(self.dim, 2)
    
    def time_series_transform(self, x, t, mask):
        print("\n-- time_series_transform --")
        x = self.fin_embed(x, t)
        _nan_check("after fin_embed", x)

        x, attn_weights = self.time_series_transformer(x, x, mask, mask)
        _nan_check("after tst", x)
        return x, attn_weights
    
    def inter_stock_transform(self, x, mask):
        print("\n-- inter_stock_transform --")
        x = x.transpose(-3, -2).contiguous()
        perm_mask = mask.transpose(-2, -1).contiguous().bool() if mask is not None else None
        x, attn_weights = self.inter_stock_transformer(x, x, perm_mask, perm_mask)
        _nan_check("after ist", x)

        if mask is not None:
            t_mask = mask.transpose(-2, -1).bool()
            active_mask = ~t_mask
            time_indices = torch.arange(x.size(1)).view(1, -1, 1)
            masked_indices = active_mask * time_indices
            last_active_idx = masked_indices.argmax(dim=1)
            has_activity = active_mask.any(dim=1).bool()
            print(f"  [ist] stocks with no activity: {(~has_activity).sum().item()}")
            last_active_idx = torch.where(has_activity, last_active_idx, 0)
            gather_idx = last_active_idx.unsqueeze(1).unsqueeze(-1)
            model_dim = x.size(-1)
            gather_idx = gather_idx.expand(-1, -1, -1, model_dim)
            last_timestamp = torch.gather(x, dim=1, index=gather_idx).squeeze(1)
        else:
            last_timestamp = x[:, -1, :, :]

        _nan_check("last_timestamp", last_timestamp)

        out = self.projection(last_timestamp)
        _nan_check("after projection", out)
        return out, attn_weights
    
    def forward(self, t, x, mask):
        print("\n===== FORWARD PASS =====")
        _nan_check("input x", x)
        _nan_check("input t", t)

        x, tst_attn_weights = self.time_series_transform(x, t, mask)
        x, ist_attn_weights = self.inter_stock_transform(x, mask)
        return x, tst_attn_weights, ist_attn_weights

class NewsEmbedding(nn.Module):
    def __init__(self, embedding_dim, temporal_embedding_dim):
        super().__init__()
        self.dim = embedding_dim + temporal_embedding_dim
        self.time_embed = Time2Vec(temporal_embedding_dim)
    
    def forward(self, x, t):
        time_vector = self.time_embed(t)
        return torch.cat([x, time_vector], dim=-1)

class DynamicSelection(nn.Module):
    def __init__(self, input_dim, K):
        super().__init__()
        self.down_project = nn.Linear(input_dim, input_dim // 2)
        self.score = nn.Linear(2 * (input_dim // 2), 1)
        self.topk = PerturbedTopK(K, 500, 0.05)

    def forward(self, x, mask):
        mask_3d = mask.unsqueeze(-1)
        projected = self.down_project(x)
        masked_projected = projected * mask_3d
        embeddings_sum = masked_projected.sum(dim=1)
        valid_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        masked_mean = (embeddings_sum / valid_count)
        masked_mean = masked_mean.unsqueeze(1).expand_as(masked_projected)
        combined = torch.cat([masked_mean, masked_projected], dim=-1) * mask_3d
        scores = self.score(combined).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        indicators = self.topk(scores)
        selected = torch.einsum("bkn,bnd->bkd", indicators, x)
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
            input_dim, embedding_dim, temporal_embedding_dim,
            num_heads, num_layers, expansion, dropout
        )
        self.news_embed = NewsEmbedding(embedding_dim, temporal_embedding_dim)
        self.news_selection = DynamicSelection(self.dim, K)
        self.topk = self.news_selection.topk
        self.news_fusion_layer = TransformerLayers(
            self.dim, num_heads, num_layers, expansion, dropout, True
        )
    
    def news_fusion_transform(self, x, news, t_news, x_mask, news_mask):
        news = self.news_embed(news, t_news)
        news = self.news_selection(news, news_mask)
        news = news.unsqueeze(1)
        num_stocks = x.size(1)
        news = news.expand(-1, num_stocks, -1, -1)
        x, attn_weights = self.news_fusion_layer(x, news, x_mask)
        return x, attn_weights
    
    def forward(self, t, t_news, x, news, x_mask, news_mask):
        x, tst_attn_weights = self.time_series_transform(x, t, x_mask)
        x, nft_attn_weights = self.news_fusion_transform(x, news, t_news, x_mask, news_mask)
        x, ist_attn_weights = self.inter_stock_transform(x, x_mask)
        return x, tst_attn_weights, nft_attn_weights, ist_attn_weights

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
    B, num_stocks, num_timestamps, input_features = 16, 30, 60, 10
    embed_dim, temp_dim = 512, 32
    
    model = StockTransformer(
        input_dim=input_features,
        embedding_dim=embed_dim,
        temporal_embedding_dim=temp_dim,
        num_heads=8,
        num_layers=2
    )
    
    dummy_x = torch.randn(B, num_stocks, num_timestamps, input_features)
    dummy_t = torch.randn(B, num_timestamps)
    dummy_mask = torch.zeros(B, num_stocks, num_timestamps).bool()
    
    preds = model(dummy_t, dummy_x, dummy_mask)
    print("Execution Successful! Prediction matrix shape:", preds[0].shape)