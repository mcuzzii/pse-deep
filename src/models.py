import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000, sigma: float = 0.05, chunk_size: int = 100):
        super().__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k
        self.chunk_size = chunk_size

    def forward(self, x):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma, self.chunk_size)

    def set_sigma(self, sigma: float):
        self.sigma = max(sigma, 1e-6)


class PerturbedTopKFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05, chunk_size: int = 100):
        orig_shape = x.shape            # B, S, Tn
        d = orig_shape[-1]              # Tn
        x_flat = x.reshape(-1, d)       # B*S*Ts, Tn
        N = x_flat.size(0)              # B*S*Ts

        indicators_flat = torch.zeros(N * k, d, device=x.device, dtype=x.dtype)         # B*S*Ts*K, Tn
        expected_gradient_flat = torch.zeros(N * k, d, device=x.device, dtype=x.dtype)  # B*S*Ts*K, Tn

        done = 0
        while done < num_samples:
            s = min(chunk_size, num_samples - done)

            noise = torch.normal(0.0, 1.0, size=(N, s, d), device=x.device, dtype=x.dtype)      # B*S*Ts, s, Tn
            perturbed_x = x_flat[:, None, :] + noise * sigma          # (N, s, d)               # B*S*Ts, s, Tn

            indices = torch.topk(perturbed_x, k=k, dim=-1, sorted=False).indices   # (N, s, K)  # B*S*Ts, s, K
            gathered = torch.gather(noise, dim=2, index=indices)                    # (N, s, K) — the only part of noise we need

            idx_for_scatter = indices.permute(0, 2, 1).reshape(N * k, s)            # (N*K, s)

            indicators_flat.scatter_add_(
                1, idx_for_scatter, torch.ones_like(idx_for_scatter, dtype=x.dtype)
            )
            expected_gradient_flat.scatter_add_(
                1, idx_for_scatter, gathered.permute(0, 2, 1).reshape(N * k, s)
            )

            done += s
            del noise, perturbed_x, indices, gathered, idx_for_scatter   # free chunk before next iteration

        indicators = (indicators_flat / num_samples).reshape(N, k, d)
        expected_gradient = (expected_gradient_flat / num_samples / sigma).reshape(N, k, d)

        ctx.orig_shape = orig_shape
        ctx.save_for_backward(expected_gradient)     # (N, K, d) — independent of num_samples now

        return indicators.reshape(*orig_shape[:-1], k, d)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None
        (expected_gradient,) = ctx.saved_tensors     # (N, K, d)
        N, K, d = expected_gradient.shape
        grad_output_flat = grad_output.reshape(-1, K, d)
        grad_input = torch.einsum("nkd,nkd->nd", grad_output_flat, expected_gradient)
        return grad_input.reshape(ctx.orig_shape), None, None, None, None

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
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads

        self.norm_qkv = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=False):
        orig_shape = x.shape

        norm_x = self.norm_qkv(x.flatten(0, 1))

        attn_out, attn_weights = self.attention(
            norm_x, norm_x, norm_x,
            need_weights=return_weights,
            average_attn_weights=return_weights
        )

        attn_out = self.dropout(attn_out)
        
        out = x.flatten(0, 1) + attn_out
        out = out.contiguous().view(orig_shape)

        if attn_weights is not None:
            print(attn_weights.shape)
            attn_weights = attn_weights.reshape(orig_shape[0], orig_shape[1], orig_shape[1])
            return out, attn_weights.to(torch.float32)
        else:
            return out

class LastTimestepAttnBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads

        self.norm_qkv = nn.LayerNorm(embedding_dim)

        self.head_dim = embedding_dim // num_heads
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_weights=False):
        B, S, Ts, Es = x.shape                        # B, S, Ts, Es

        norm_x = self.norm_qkv(x)                       # B, S, Ts, Es

        q = self.q_proj(norm_x)                                                     # (B, S, Ts, Es)
        k = self.k_proj(norm_x)                                                     # (B, S, Ts, Es)
        v = self.v_proj(norm_x)                                                     # (B, S, Ts, Es)

        q = q.view(B, S, Ts, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)[:, :, :, -1, :]  # (B, H, S, Es/H)
        k = k.view(B, S, Ts, self.num_heads, self.head_dim).transpose(2, 3)                         # (B, S, H, Ts, Es/H)
        v = v.view(B, S, Ts, self.num_heads, self.head_dim).transpose(2, 3)                         # (B, S, H, Ts, Es/H)

        scaling_factor = math.sqrt(self.head_dim)
        scores = torch.einsum("bhse,bshte->bhst", q, k) / scaling_factor                            # (B, H, S, Ts)

        attn_weights = F.softmax(scores, dim=-1)                                                    # (B, H, S, Ts)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.einsum("bhst,bshte->bshe", attn_weights, v)                                 # (B, S, H, Es/H)
        context = context.contiguous().view(B, S, Es)                                               # (B, S, H, Es/H) -> (B, S, Es)

        out = self.out_proj(context)
        out = self.dropout(out)
        out = x[:, :, -1, :] + out

        if return_weights:
            attn_weights = attn_weights.mean(dim=1)
            return out, attn_weights.to(torch.float32)
        else:
            return out

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
    def __init__(self, embedding_dim, num_heads, expansion=4, dropout=0.1):
        super().__init__()
        self.attn_blk = SelfAttentionBlock(embedding_dim, num_heads, dropout)
        self.ffnn = FeedForward(embedding_dim, expansion, dropout)

    def forward(self, x, return_weights=False):
        if return_weights:
            attn_out, attn_weights = self.attn_blk(x, return_weights=True)
            ffn_out = self.ffnn(attn_out)
            return ffn_out, attn_weights
        else:
            attn_out = self.attn_blk(x)
            ffn_out = self.ffnn(attn_out)
            return ffn_out

class LastTimestepAttnTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, expansion=4, dropout=0.1):
        super().__init__()
        self.attn_blk = LastTimestepAttnBlock(embedding_dim, num_heads, dropout)
        self.ffnn = FeedForward(embedding_dim, expansion, dropout)

    def forward(self, x, return_weights=False):
        if return_weights:
            attn_out, attn_weights = self.attn_blk(x, return_weights=True)
            ffn_out = self.ffnn(attn_out)
            return ffn_out, attn_weights
        else:
            attn_out = self.attn_blk(x)
            ffn_out = self.ffnn(attn_out)
            return ffn_out

class SelfAttnTransformerLayers(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers=1, expansion=4, dropout=0.1):
        super().__init__()
        self.transformer = nn.ModuleList([
            SelfAttnTransformerLayer(embedding_dim, num_heads, expansion, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, return_weights=False):
        attn_blocks = []
        str_out = x
        for layer in self.transformer:
            if return_weights:
                str_out, attn_weights = layer(str_out, return_weights=True)
                attn_blocks.append(attn_weights)
            else:
                str_out = layer(str_out)
        if return_weights:
            return str_out, torch.stack(attn_blocks, dim=0).mean(dim=0)
        return str_out

class LastTimestepAttnTransformerLayers(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers=1, expansion=4, dropout=0.1):
        super().__init__()
        self.transformer = nn.ModuleList([
            LastTimestepAttnTransformerLayer(embedding_dim, num_heads, expansion, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, return_weights=False):
        attn_blocks = []
        str_out = x
        for layer in self.transformer:
            if return_weights:
                str_out, attn_weights = layer(str_out, return_weights=True)
                attn_blocks.append(attn_weights)
            else:
                str_out = layer(str_out)
        if return_weights:
            return str_out, torch.stack(attn_blocks, dim=0).mean(dim=0)
        return str_out

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
        self.time_series_transformer = LastTimestepAttnTransformerLayers(
            self.dim, num_heads, num_layers, expansion, dropout,
        )
        self.inter_stock_transformer = SelfAttnTransformerLayers(
            self.dim, num_heads, num_layers, expansion, dropout
        )
        self.projection = nn.Linear(self.dim, 2)
    
    def time_series_transform(self, x, t, return_weights=False):
        embeddings = self.fin_embed(x, t)

        if return_weights:
            tst_out, attn_weights = self.time_series_transformer(embeddings, return_weights=True)
            return tst_out, attn_weights
        tst_out = self.time_series_transformer(embeddings)      # B, S, Ts, Es
        return tst_out                                          # B, S, Es
    
    def inter_stock_transform(self, x, return_weights=False):

        if return_weights:
            ist_out, attn_weights = self.inter_stock_transformer(x, return_weights=True)
        else:
            ist_out = self.inter_stock_transformer(x)           # B, S, Es

        out = self.projection(ist_out)                          # B, S, 2

        if return_weights:
            return out, attn_weights
        return out
    
    def forward(self, t, x, return_weights=False):

        if return_weights:
            tst_out, tst_attn_weights = self.time_series_transform(x, t, return_weights=True)
            ist_out, ist_attn_weights = self.inter_stock_transform(tst_out, return_weights=True)
            return ist_out, tst_attn_weights, ist_attn_weights
        else:
            tst_out = self.time_series_transform(x, t)
            ist_out = self.inter_stock_transform(tst_out)
            return ist_out

class NewsEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, temporal_embedding_dim, time_vec_model, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dim = embedding_dim + temporal_embedding_dim
        self.time_embed = time_vec_model
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, t):
        time_vector = self.time_embed(t)
        news_vector = self.linear(x)
        combined_embedding = torch.cat([news_vector, time_vector], dim=-1)
        combined_embedding = self.dropout(combined_embedding)

        return combined_embedding

class SocialEmbedding(nn.Module):
    def __init__(
        self,
        social_input_dim,
        text_input_dim,
        social_embedding_dim,
        text_embedding_dim,
        temporal_embedding_dim,
        time_vec_model,
        dropout=0.1
    ):
        super().__init__()
        self.input_dim = text_input_dim
        self.dim = social_embedding_dim + text_embedding_dim + temporal_embedding_dim
        self.time_embed = time_vec_model
        self.text_linear = nn.Linear(text_input_dim, text_embedding_dim)
        self.social_linear = nn.Linear(social_input_dim, social_embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, s, t):
        time_vector = self.time_embed(t)
        text_embedding = self.text_linear(x)
        social_embedding = self.social_linear(s)
        combined_embedding = torch.cat([text_embedding, social_embedding, time_vector], dim=-1)
        combined_embedding = self.dropout(combined_embedding)

        return combined_embedding

class DynamicSelection(nn.Module):
    def __init__(self, input_dim, K, num_samples, sigma):
        super().__init__()
        self.norm_news = nn.LayerNorm(input_dim)
        self.norm_stock = nn.LayerNorm(input_dim)
        self.down_project_news = nn.Linear(input_dim, input_dim // 2)
        self.down_project_stock = nn.Linear(input_dim, input_dim // 2)
        self.score = nn.Linear(2 * (input_dim // 2), 1)
        self.topk = PerturbedTopK(K, num_samples, sigma)

    def forward(self, x, news, mask):
        # x:      (B, S, Es)
        # news:   (B, Tn, En)
        # mask:   (B, Tn)

        # expand mask across stocks: (B, S, Tn)
        news_mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)                         # (B, S, Tn)

        # project news once, broadcast over stocks
        news_proj = self.down_project_news(self.norm_news(news))                        # (B, Tn, En/2)
        news_proj = news_proj.unsqueeze(1).expand(
            -1, x.size(1), -1, -1
        )                                                                               # (B, S, Tn, En/2)

        # project stock query per (stock, timestep), broadcast over Tn
        stock_proj = self.down_project_stock(self.norm_stock(x))                        # (B, S, Es/2)
        stock_proj = stock_proj.unsqueeze(2).expand(-1, -1, news.size(1), -1)           # (B, S, Tn, Es/2)

        combined = torch.cat([stock_proj, news_proj], dim=-1)                           # (B, S, Tn, En)

        scores = self.score(combined).squeeze(-1)                                       # (B, S, Tn)
        scores = scores.masked_fill(news_mask == 0, float('-inf'))                      # (B, S, Tn)

        indicators = self.topk(scores)                                                  # (B, S, K, Tn)

        return indicators

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
        
        self.attn_dropout = nn.Dropout(dropout)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y, indicators, return_weights=False):
        # x: (B, S, Es)
        # y: (B, Tn, En)
        # indicators: (B, S, K, Tn)
        # news_mask: (B, S, Tn)

        y_stock = y.unsqueeze(1).expand(-1, x.size(1), -1, -1)                      # (B, S, Tn, En)

        x_norm = self.norm_q(x)                                       # (B, S, Es)
        y_norm = self.norm_kv(y_stock)                                # (B, S, Tn, En)

        B, S, _ = x_norm.shape                                                  # T_tgt = Ts
        _, _, Tn, En = y_norm.shape                                                  # T_src = Tn
        
        q = self.q_proj(x_norm)                                                     # (B, S, Es)
        k = self.k_proj(y_norm)                                                     # (B, S, Tn, En)
        v = self.v_proj(y_norm)                                                     # (B, S, Tn, En)
        
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)             # (B, H, S, Es/H)
        k = k.view(B, S, Tn, self.num_heads, self.head_dim).transpose(2, 3)         # (B, S, H, Tn, En/H)
        v = v.view(B, S, Tn, self.num_heads, self.head_dim).transpose(2, 3)         # (B, S, H, Tn, En/H)

        k_selected = torch.einsum("bskt,bshte->bshke", indicators, k)               # (B, S, H, K, En/H)
        v_selected = torch.einsum("bskt,bshte->bshke", indicators, v)               # (B, S, H, K, En/H)

        scaling_factor = math.sqrt(self.head_dim)
        scores = torch.einsum("bhse,bshke->bshk", q, k_selected) / scaling_factor   # (B, S, H, K)

        attn_weights = F.softmax(scores, dim=-1)                                    # (B, S, H, K)
        attn_weights = self.attn_dropout(attn_weights)                              # (B, S, H, K)

        context = torch.einsum("bshk,bshke->bshe", attn_weights, v_selected)        # (B, S, H, En/H)
        context = context.contiguous().view(B, S, En)                               # (B, S, En)

        output = self.out_proj(context)                                             # (B, S, En; note that En = Es)
        output = self.dropout(output)
        output = x + output                                                         # Residual connection

        if return_weights:
            return output, attn_weights.mean(dim=2)                                 # B, S, K
        return output

class CrossAttnTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, expansion=4, dropout=0.1):
        super().__init__()
        self.attn_blk = CrossAttentionBlock(embedding_dim, num_heads, dropout)
        self.ffnn = FeedForward(embedding_dim, expansion, dropout)

    def forward(self, x, y, indicators, return_weights=False):
        if return_weights:
            attn_out, attn_weights = self.attn_blk(x, y, indicators, return_weights=True)
            ffn_out = self.ffnn(attn_out)
            return ffn_out, attn_weights
        else:
            attn_out = self.attn_blk(x, y, indicators)
            ffn_out = self.ffnn(attn_out)
            return ffn_out

class CrossAttnTransformerLayers(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers=1, expansion=4, dropout=0.1):
        super().__init__()
        self.transformer = nn.ModuleList([
            CrossAttnTransformerLayer(embedding_dim, num_heads, expansion, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, y, indicators, return_weights=False):
        attn_blocks = []
        str_out = x
        for layer in self.transformer:
            if return_weights:
                str_out, attn_weights = layer(str_out, y, indicators, return_weights=True)
                attn_blocks.append(attn_weights)
            else:
                str_out = layer(str_out, y, indicators)
        if return_weights:
            return str_out, torch.stack(attn_blocks, dim=0).mean(dim=0)
        return str_out

class StockNewsTransformer(StockTransformer):
    def __init__(
        self,
        input_dim,
        news_input_dim,
        embedding_dim,
        temporal_embedding_dim,
        num_heads,
        K=30,
        num_samples=100,
        sigma=5e-2,
        num_layers=1,
        expansion=4,
        dropout=0.1
    ):
        super().__init__(
            input_dim, embedding_dim, temporal_embedding_dim,
            num_heads, num_layers, expansion, dropout
        )
        self.text_embedding_dim = news_input_dim
        self.news_embed = NewsEmbedding(news_input_dim, embedding_dim, temporal_embedding_dim, self.fin_embed.time_embed, dropout)
        self.news_selection = DynamicSelection(self.dim, K, num_samples, sigma)
        self.K = K
        self.sigma = sigma
        self.topk = self.news_selection.topk
        self.news_fusion_layer = CrossAttnTransformerLayers(
            self.dim, num_heads, num_layers, expansion, dropout
        )
    
    def news_fusion_transform(self, x, news, t_news, mask, return_weights=False):
        news_embeddings = self.news_embed(news, t_news)                                               # (B, Tn, En)
        indicators = self.news_selection(x, news_embeddings, mask)

        if return_weights:
            nft_out, nft_attn_weights = self.news_fusion_layer(x, news_embeddings, indicators, return_weights=True)               # (B, S, Es)
            return nft_out, nft_attn_weights, indicators
        else:
            nft_out = self.news_fusion_layer(x, news_embeddings, indicators)
            return nft_out
    
    def forward(self, t, t_news, x, news, news_mask, return_weights=False):

        if return_weights:
            tst_out, tst_attn_weights = self.time_series_transform(x, t, return_weights=True)
            nft_out, nft_attn_weights, indicators = self.news_fusion_transform(
                tst_out, news, t_news, news_mask, return_weights=True
            )
            ist_out, ist_attn_weights = self.inter_stock_transform(nft_out, return_weights=True)
            return ist_out, tst_attn_weights, nft_attn_weights, indicators, ist_attn_weights
        else:
            tst_out = self.time_series_transform(x, t)
            nft_out = self.news_fusion_transform(
                tst_out, news, t_news, news_mask
            )
            ist_out = self.inter_stock_transform(nft_out)
            return ist_out

class StockSocialTransformer(StockTransformer):
    def __init__(
        self,
        input_dim,                  # 100
        social_input_dim,           # 5
        text_input_dim,             # 1024
        social_embedding_dim,       # 16
        embedding_dim,              # 128
        temporal_embedding_dim,     # 16
        num_heads,                  # 8
        K=30,
        num_samples=100,
        sigma=5e-2,
        num_layers=1,
        expansion=4,
        dropout=0.1
    ):
        super().__init__(
            input_dim, embedding_dim, temporal_embedding_dim,
            num_heads, num_layers, expansion, dropout
        )
        self.social_input_dim = social_input_dim
        self.text_embedding_dim = text_input_dim
        self.social_embed = SocialEmbedding(
            social_input_dim, text_input_dim, social_embedding_dim, embedding_dim - social_embedding_dim,
            temporal_embedding_dim, self.fin_embed.time_embed, dropout
        )
        self.social_selection = DynamicSelection(self.dim, K, num_samples, sigma)
        self.K = K
        self.sigma = sigma
        self.topk = self.social_selection.topk
        self.social_fusion_layer = CrossAttnTransformerLayers(
            self.dim, num_heads, num_layers, expansion, dropout
        )
    
    def social_fusion_transform(self, x, s, es, ts, mask, return_weights=False):
        social_embeddings = self.social_embed(es, s, ts)                                               # (B, Tn, En)
        indicators = self.social_selection(x, social_embeddings, mask)

        if return_weights:
            sft_out, sft_attn_weights = self.social_fusion_layer(x, social_embeddings, indicators, return_weights=True)          # (B, S, Es)
            return sft_out, sft_attn_weights, indicators
        else:
            sft_out = self.social_fusion_layer(x, social_embeddings, indicators)
            return sft_out
    
    def forward(self, t, ts, x, s, es, m, return_weights=False):

        if return_weights:
            tst_out, tst_attn_weights = self.time_series_transform(x, t, return_weights=True)
            sft_out, sft_attn_weights, indicators = self.social_fusion_transform(tst_out, s, es, ts, m, return_weights=True)
            ist_out, ist_attn_weights = self.inter_stock_transform(sft_out, return_weights=True)
            return ist_out, tst_attn_weights, sft_attn_weights, indicators, ist_attn_weights
        else:
            tst_out = self.time_series_transform(x, t)
            sft_out = self.social_fusion_transform(tst_out, s, es, ts, m)
            ist_out = self.inter_stock_transform(sft_out)
            return ist_out

class StockNewsSocialTransformer(StockNewsTransformer):
    def __init__(
        self,
        input_dim,                  # 100
        social_input_dim,           # 5
        text_input_dim,             # 1024
        social_embedding_dim,       # 16
        embedding_dim,              # 128
        temporal_embedding_dim,     # 16
        num_heads,                  # 8
        K=30,
        num_samples=100,
        sigma=5e-2,
        num_layers=1,
        expansion=4,
        dropout=0.1
    ):
        super().__init__(
            input_dim, text_input_dim, embedding_dim, temporal_embedding_dim,
            num_heads, K, num_samples, sigma, num_layers, expansion, dropout
        )

        self.social_input_dim = social_input_dim
        self.social_embed = SocialEmbedding(
            social_input_dim, text_input_dim, social_embedding_dim, embedding_dim - social_embedding_dim,
            temporal_embedding_dim, self.fin_embed.time_embed, dropout
        )
        self.social_selection = DynamicSelection(self.dim, K, num_samples, sigma)
        self.social_fusion_layer = CrossAttnTransformerLayers(
            self.dim, num_heads, num_layers, expansion, dropout
        )
        self.down_project = nn.Linear(self.dim * 2, self.dim)

    def social_fusion_transform(self, x, s, es, ts, mask, return_weights=False):
        social_embeddings = self.social_embed(es, s, ts)                                               # (B, Tn, En)
        indicators = self.social_selection(x, social_embeddings, mask)

        if return_weights:
            sft_out, sft_attn_weights = self.social_fusion_layer(x, social_embeddings, indicators, return_weights=True)          # (B, S, Es)
            return sft_out, sft_attn_weights, indicators
        else:
            sft_out = self.social_fusion_layer(x, social_embeddings, indicators)
            return sft_out
    
    def forward(self, t, tn, ts, x, s, en, es, mn, ms, return_weights=False):

        if return_weights:
            tst_out, tst_attn_weights = self.time_series_transform(x, t, return_weights=True)
            sft_out, sft_attn_weights, s_indicators = self.social_fusion_transform(tst_out, s, es, ts, ms, return_weights=True)
            nft_out, nft_attn_weights, n_indicators = self.news_fusion_transform(tst_out, en, tn, mn, return_weights=True)
            prj_out = tst_out + self.down_project(torch.cat([sft_out, nft_out], dim=-1))
            ist_out, ist_attn_weights = self.inter_stock_transform(prj_out, return_weights=True)
            return ist_out, tst_attn_weights, sft_attn_weights, nft_attn_weights, ist_attn_weights, s_indicators, n_indicators
        else:
            tst_out = self.time_series_transform(x, t)
            sft_out = self.social_fusion_transform(tst_out, s, es, ts, ms)
            nft_out = self.news_fusion_transform(tst_out, en, tn, mn)
            prj_out = tst_out + self.down_project(torch.cat([sft_out, nft_out], dim=-1))
            ist_out = self.inter_stock_transform(prj_out)
            return ist_out

class HiddenLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.hidden_layer(x)

class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=5,
        dropout=0.1
    ):
        super().__init__()
        self.input_layer = HiddenLayer(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            HiddenLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_dim, 2)
    
    def forward(self, x, return_weights=False):
        out_vects = []

        out = self.input_layer(x)
        out_vects.append(out.clone())
        for _, layer in enumerate(self.layers):
            out = layer(out)
            out_vects.append(out.clone())
        out = self.output_layer(out).unsqueeze(1)       # (B, 1, 2)

        if return_weights:
            return out, torch.stack(out_vects, dim=0)
        else:
            return out

class GroupSHAPWrapper(nn.Module):
    def __init__(self, model, args, group_to_indices, non_float_indices, mlp_group_slices=None):
        """
        mlp_group_slices: dict {group_name: slice} for MLP case where groups
                          are contiguous slices of a single flat feature tensor.
                          If None, uses the standard arg-gating approach.
        """
        super().__init__()
        self.model = model
        self.args = [a.detach() for a in args]
        self.group_to_indices = group_to_indices
        self.non_float_indices = non_float_indices
        self.group_names = list(group_to_indices.keys())
        self.mlp_group_slices = mlp_group_slices
        self.positive_class_idx = (
            1 if isinstance(model, MultiLayerPerceptron)
            else 0
        )

    def forward(self, gates):
        # gates: (B_shap, num_groups) — B_shap may differ from original batch size
        B_shap = gates.shape[0]

        # expand args to match B_shap
        full_args = []
        for a in self.args:
            if a.shape[0] != B_shap:
                repeats = [B_shap // a.shape[0]] + [1] * (a.dim() - 1)
                full_args.append(a.repeat(repeats))
            else:
                full_args.append(a.clone())

        if self.mlp_group_slices is not None:
            x = full_args[0].clone()
            mask = torch.ones_like(x)
            for g, (group_name, slc) in enumerate(self.mlp_group_slices.items()):
                mask[:, slc] = gates[:, g].unsqueeze(-1).expand_as(mask[:, slc])
            full_args[0] = x * mask
        else:
            for g, (group_name, arg_indices) in enumerate(self.group_to_indices.items()):
                gate = gates[:, g]
                for idx in arg_indices:
                    arg = full_args[idx]
                    scale = gate
                    for _ in range(arg.dim() - 1):
                        scale = scale.unsqueeze(-1)
                    full_args[idx] = arg * scale

        logits = self.model(*full_args)

        up_idx = self.positive_class_idx
        down_idx = 1 - up_idx

        score = logits[..., up_idx] - logits[..., down_idx]

        return score


def _build_group_map(self, sample_args):

    if not self.transformer:
        # MLP: single flat tensor, slice into groups by feature range
        # base: 100 stock features + 10 timestamp features = 110
        # news (if present): next 15 dims
        # social (if present): next 15 dims after news

        mlp_group_slices = {
            'stock_features':    slice(0, 100),
            'stock_timestamps':  slice(100, 110),
        }
        offset = 110
        if self.news:
            mlp_group_slices['news_features'] = slice(offset, offset + 15)
            offset += 15
        if self.social:
            mlp_group_slices['social_features'] = slice(offset, offset + 15)

        group_to_indices  = {name: [0] for name in mlp_group_slices}   # all map to arg 0
        non_float_indices = []

        return group_to_indices, non_float_indices, [0], mlp_group_slices

    mlp_group_slices = None

    if self.transformer and not self.news and not self.social:
        group_to_indices  = {
            'stock_timestamps': [0],
            'stock_features':   [1],
        }
        non_float_indices = []

    elif self.transformer and self.news and not self.social:
        group_to_indices  = {
            'stock_timestamps': [0],
            'stock_features':   [2],
            'news_embeddings':  [3],
        }
        non_float_indices = [4]

    elif self.transformer and self.social and not self.news:
        group_to_indices  = {
            'stock_features':    [2],
            'stock_timestamps':  [0],
            'social_embeddings': [4],
            'social_impact':     [3],
        }
        non_float_indices = [5]

    elif self.transformer and self.news and self.social:
        group_to_indices  = {
            'stock_features':    [3],
            'stock_timestamps':  [0],
            'news_embeddings':   [5],
            'social_embeddings': [6],
            'social_impact':     [4],
        }
        non_float_indices = [7, 8]

    float_indices = [i for i in range(len(sample_args)) if i not in non_float_indices]

    return group_to_indices, non_float_indices, float_indices, mlp_group_slices

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