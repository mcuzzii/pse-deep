import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        b, c, d = x.shape                                                                       # b = B, d = Tn
        noise = torch.normal(mean=0.0, std=1.0, size=(b, c, num_samples, d)).to(x.device)       # (B, Ts, num_samples, Tn)
        perturbed_x = x[:, :, None, :] + noise * sigma                                          # (B, Ts, num_samples, Tn) noise-perturbed scores
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)                       # indices: (B, Ts, num_samples, K)
        indices = topk_results.indices
        indices = torch.sort(indices, dim=-1).values                                            # (B, Ts, num_samples, K)
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()          # (B, Ts, num_samples, K, Tn)
        indicators = perturbed_output.mean(dim=2)                                               # (B, Ts, K, Tn) - probability scores of every article for each ordinal place
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

class Time2Vec(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.dim = embedding_dim
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(self.dim - 1))
        self.b = nn.Parameter(torch.randn(self.dim - 1))

    def forward(self, t):
        t_vec = t.unsqueeze(-1)

        linear = t_vec * self.w0 + self.b0

        periodic = torch.sin(t_vec * self.w + self.b)

        t_vec = torch.cat([linear, periodic], dim=-1)
        return t_vec

class FinEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, temporal_embedding_dim, dropout=0.1):
        super().__init__()
        self.dim = embedding_dim + temporal_embedding_dim
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.time_embed = Time2Vec(temporal_embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, t):

        stock_vector = self.linear(x)
        time_vector = self.time_embed(t)
        out = torch.cat([stock_vector, time_vector], dim=-1)

        out = self.dropout(out)

        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1, is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal

        self.norm_qkv = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        orig_shape = x.shape

        norm_x = self.norm_qkv(x.flatten(0, 1))     # (b * n, x_seq, e)

        attn_mask = None
        if self.is_causal:
            num_t = x.size(2)
            attn_mask = torch.triu(
                torch.ones(num_t, num_t, dtype=bool),
                diagonal=1,
                device=x.device
            )

        attn_out, attn_weights = self.attention(      
            norm_x, norm_x, norm_x,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False
        )

        attn_out = self.dropout(attn_out)
        
        out = x.flatten(0, 1) + attn_out

        return out.contiguous().view(orig_shape), attn_weights.to(torch.float32)

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
    
    def forward(self, x):
        norm_x = self.norm(x)
        ffn_out = self.ff(norm_x)
        
        out = x + ffn_out
        return out

class SelfAttnTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, expansion=4, dropout=0.1, is_causal=False):
        super().__init__()
        self.attn_blk = SelfAttentionBlock(embedding_dim, num_heads, dropout, is_causal)
        self.ffnn = FeedForward(embedding_dim, expansion, dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attn_blk(x)
        ffn_out = self.ffnn(attn_out)
        return ffn_out, attn_weights

class SelfAttnTransformerLayers(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers=1, expansion=4, dropout=0.1, is_causal=False):
        super().__init__()
        self.transformer = nn.ModuleList([
            SelfAttnTransformerLayer(embedding_dim, num_heads, expansion, dropout, is_causal)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        attn_blocks = []
        str_out = x.clone()
        for _, layer in enumerate(self.transformer):
            str_out, attn_weights = layer(str_out)
            attn_blocks.append(attn_weights)
        return str_out, torch.stack(attn_blocks, dim=0)

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
        self.fin_embed = FinEmbedding(input_dim, embedding_dim, temporal_embedding_dim, dropout)
        self.dim = self.fin_embed.dim
        self.time_series_transformer = SelfAttnTransformerLayers(
            self.dim, num_heads, num_layers, expansion, dropout, True
        )
        self.inter_stock_transformer = SelfAttnTransformerLayers(
            self.dim, num_heads, 1, expansion, dropout
        )
        self.projection = nn.Linear(self.dim, 2)
    
    def time_series_transform(self, x, t):
        embeddings = self.fin_embed(x, t)

        tst_out, attn_weights = self.time_series_transformer(embeddings)
        return tst_out, attn_weights
    
    def inter_stock_transform(self, x):
        x_transposed = x.transpose(-3, -2).contiguous()
        ist_out, attn_weights = self.inter_stock_transformer(x_transposed)

        pooled_vectors = ist_out.mean(dim=1)

        out = self.projection(pooled_vectors)
        return out, attn_weights
    
    def forward(self, t, x, return_weights=False):

        tst_out, tst_attn_weights = self.time_series_transform(x, t)
        ist_out, ist_attn_weights = self.inter_stock_transform(tst_out)

        if return_weights:
            return ist_out, tst_attn_weights, ist_attn_weights
        else:
            return ist_out

class NewsEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, temporal_embedding_dim, time_vec_model, dropout=0.1):
        super().__init__()
        self.dim = embedding_dim + temporal_embedding_dim
        self.time_embed = time_vec_model
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.dim)
    
    def forward(self, x, t):
        time_vector = self.time_embed(t)
        news_vector = self.linear(x)
        combined_embedding = torch.cat([news_vector, time_vector], dim=-1)
        combined_embedding = self.dropout(combined_embedding)
        norm_embedding = self.norm(combined_embedding)

        return norm_embedding

class DynamicSelection(nn.Module):
    def __init__(self, input_dim, K):
        super().__init__()
        self.down_project = nn.Linear(input_dim, input_dim // 2)
        self.score = nn.Linear(2 * (input_dim // 2), 1)
        self.topk = PerturbedTopK(K, 500, 0.05)

    def forward(self, x, news, t, t_news, mask):

        stock_timestamps = t[:, 0, :].unsqueeze(-1)                                     # (B, Ts, 1)
        news_timestamps = t_news.unsqueeze(1)                                           # (B, 1, Tn)
        news_mask = news_timestamps <= stock_timestamps                                 # (B, Ts, Tn)
        news_mask = news_mask * mask.unsqueeze(1)                                       # (B, Ts, Tn) * (B, 1, Tn) = (B, Ts, Tn)
        news_mask_4d = news_mask.unsqueeze(-1)                                          # (B, Ts, Tn, 1)

        projected = self.down_project(news)                                             # (B, Tn, En) -> (B, Tn, En/2)
        projected = projected.unsqueeze(1).expand(-1, t.size(2), -1, -1)                # (B, Ts, Tn, En/2)
        masked_projected = projected * news_mask_4d                                     # (B, Ts, Tn, En/2)

        embeddings_sum = masked_projected.sum(dim=2)                                    # (B, Ts, En/2)
        valid_count = news_mask.sum(dim=-1, keepdim=True).clamp(min=1)                  # (B, Ts, 1)
        masked_mean = embeddings_sum / valid_count                                      # (B, Ts, En/2)
        masked_mean = masked_mean.unsqueeze(2).expand_as(masked_projected)              # (B, Ts, Tn, En/2)
        combined = torch.cat([masked_mean, masked_projected], dim=-1) * news_mask_4d    # (B, Ts, Tn, En)
        scores = self.score(combined).squeeze(-1)                                       # (B, Ts, Tn, En) -> (B, Ts, Tn, 1) -> (B, Ts, Tn)
        scores = scores.masked_fill(news_mask == 0, float('-inf'))                      # (B, Ts, Tn)
        indicators = self.topk(scores)                                                  # (B, Ts, K, Tn)
        news_4d = news.unsqueeze(1).expand_as(combined)                                 # (B, Ts, Tn, En)
        selected = torch.einsum("btkn,btnd->btkd", indicators, news_4d)                 # (B, Ts, K, En)
        return selected

class CrossAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(embedding_dim)
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        # x: (B, S, Ts, Es)
        # y: (B, Tn, En)

        x_norm = self.norm_q(x.flatten(0, 1))                                       # (B*S, Ts, Es)
        y_norm = self.norm_kv(y.flatten(0, 1))                                      

        B, T_tgt, D = x_norm.shape                                                  # T_tgt = Ts
        _, T_src, _ = y_norm.shape                                                  # T_src = K
        
        q = self.q_proj(x_norm)                                                     # (B*S, Ts, Es)
        k = self.k_proj(y_norm)                                                     # (B*S, Ts, K, En)
        v = self.v_proj(y_norm)                                                     # (B*S, Ts, K, En)
        
        q = q.view(B, T_tgt, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_src, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_src, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Calculate Scaled Dot-Product Attention Scores
        # Q shape: (B, H, T_tgt, d_k) | K^T shape: (B, H, d_k, T_src)
        # Product matrix shape: (B, H, T_tgt, T_src)
        scaling_factor = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scaling_factor
        
        # 4. Apply Optional Masking (e.g., filtering out padded tokens or routing alignments)
        if attn_mask is not None:
            # If mask is boolean, fill False/True values with negative infinity
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
            
        # 5. Softmax to create probability distribution along the source timeline
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 6. Weight the Values: (B, H, T_tgt, T_src) x (B, H, T_src, d_k) -> (B, H, T_tgt, d_k)
        context = torch.matmul(attn_weights, v)
        
        # 7. Concatenate heads back together
        # Transpose back: (B, H, T_tgt, d_k) -> (B, T_tgt, H, d_k)
        # Flatten H and d_k: -> (B, T_tgt, D)
        context = context.transpose(1, 2).contiguous().view(B, T_tgt, D)
        
        # 8. Output projection head
        output = self.out_proj(context)
        
        return output, attn_weights

class StockNewsTransformer(StockTransformer):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        temporal_embedding_dim,
        num_heads,
        K=500,
        num_layers=1,
        expansion=4,
        dropout=0.1
    ):
        super().__init__(
            input_dim, embedding_dim, temporal_embedding_dim,
            num_heads, num_layers, expansion, dropout
        )
        self.news_embed = NewsEmbedding(embedding_dim, temporal_embedding_dim, self.fin_embed.time_embed, dropout)
        self.news_selection = DynamicSelection(self.dim, K)
        self.topk = self.news_selection.topk
        self.news_fusion_layer = SelfAttnTransformerLayers(
            self.dim, num_heads, num_layers, expansion, dropout, True
        )
    
    def news_fusion_transform(self, x, news, t, t_news, mask):
        news = self.news_embed(news, t_news)                # (B, N, E)
        news = self.news_selection(x, news, t, t_news, mask)         # (B, K, E)
    
    def forward(self, t, t_news, x, news, news_mask, return_weights=False):
        x, tst_attn_weights = self.time_series_transform(x, t)
        x, nft_attn_weights = self.news_fusion_transform(x, news, t, t_news, news_mask)
        x, ist_attn_weights = self.inter_stock_transform(x)

        if return_weights:
            return x, tst_attn_weights, nft_attn_weights, ist_attn_weights
        else:
            return x

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