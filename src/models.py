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

        self.dim = embedding_dim + 32

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
    def __init__(self, input_dim, embedding_dim):
        super().__init__()

        self.dim = embedding_dim

        self.linear = nn.Linear(input_dim, self.dim)
        self.time_embed = Time2Vec(32)
    
    def forward(self, x, t):
        
        stock_vector = self.linear(x)
        time_vector = self.time_embed(t)

        return torch.cat([stock_vector, time_vector], dim=-1)