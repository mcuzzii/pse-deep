import torch
import torch.nn as nn
import torch.nn.functional as F

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000, sigma: float = 0.05):
        super(PerturbedTopK, self).__init__()
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

class Time2VecEmbedding(nn.Module):
    """
    Implements the learnable Time2Vec embedding function highlighted 
    in Equation (2) of the paper.
    """
    def __init__(self, embedding_dim):
        super(Time2VecEmbedding, self).__init__()
        self.dim = embedding_dim
        # Linear terms parameter (l=1)
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))
        
        # Periodic terms parameters (l > 1)
        self.w = nn.Parameter(torch.randn(1, embedding_dim - 1))
        self.b = nn.Parameter(torch.randn(1, embedding_dim - 1))

    def forward(self, t):
        # t shape: [Batch, Tokens] or [Batch, Num_Observations]
        t = t.unsqueeze(-1) # [Batch, Tokens, 1]
        
        linear = t * self.w0 + self.b0 # [Batch, Tokens, 1]
        periodic = torch.sin(t * self.w + self.b) # [Batch, Tokens, dim-1]
        
        return torch.cat([linear, periodic], dim=-1) # [Batch, Tokens, dim]


class AdaptiveTimeEncoding(nn.Module):
    """
    Core ATE module utilizing learnable reference points as Queries 
    and actual multi-variable transaction timestamps as Keys/Values.
    """
    def __init__(self, num_variables, num_ref_points, embedding_dim, num_heads=2):
        super(AdaptiveTimeEncoding, self).__init__()
        self.V = num_variables
        self.K = num_ref_points
        self.L = embedding_dim
        self.H = num_heads
        
        # 1. Globally shared learnable reference time points (initialized uniform 0 to 1)
        # These act as our target anchors for structural market events.
        self.r = nn.Parameter(torch.linspace(0.0, 1.0, num_ref_points))
        
        # 2. Time2Vec Embedders for both actual timestamps and reference anchors
        self.time_embedders = nn.ModuleList([Time2VecEmbedding(embedding_dim) for _ in range(num_heads)])
        
        # 3. Scaling parameters for attention weights scaling factor
        self.epsilon = embedding_dim ** 0.5
        
        # 4. Out projection alignment matrix W_hvv' from Equation (5)
        self.W = nn.Parameter(torch.randn(num_heads, num_variables, num_variables))

    def forward(self, t_observed, x_observed, masking_ratio=None):
        """
        t_observed: Tensor [B, V, T_max] -> Timestamps for each batch element, asset, and observation step.
        x_observed: Tensor [B, V, T_max] -> Price/Volume tracking values matching the timestamps.
        masking_ratio: Float [0.1, 0.9] -> Used optionally to build a temporal consistency view during training.
        """
        B, V, T_max = t_observed.shape
        
        # Handle optional Temporal Consistency Regularization Masking if active
        if masking_ratio is not None and self.training:
            mask = torch.rand_like(x_observed) > masking_ratio
            x_observed = x_observed * mask
            t_observed = t_observed * mask

        # Prepare global reference tensor shape to match batch constraints
        # r shape: [K] -> expand to [B, K]
        r_expanded = self.r.unsqueeze(0).expand(B, -1) 
        
        # Collect univariate kernel-smoothing context structures across all attention heads
        psi_heads = []
        
        for h in range(self.H):
            embedder = self.time_embedders[h]
            
            # Embed global reference points: [B, K, L]
            phi_r = embedder(r_expanded) 
            
            # Embed actual transaction steps per asset variable:
            # Flatten to pass through embedder smoothly: [B*V, T_max] -> [B*V, T_max, L]
            phi_t = embedder(t_observed.view(B * V, T_max))
            phi_t = phi_t.view(B, V, T_max, self.L)
            
            # Calculate Scaled Cross-Attention (Equation 3 & 4)
            # Reshape tensors for targeted matrix multiplication matching reference anchors to observations
            # phi_r: [B, 1, K, L] vs phi_t: [B, V, T_max, L]
            phi_r_unsqueezed = phi_r.unsqueeze(1) # [B, 1, K, L]
            
            # Compute inner product alignment scores: [B, V, K, T_max]
            scores = torch.matmul(phi_r_unsqueezed, phi_t.transpose(-2, -1)) / self.epsilon
            attn_weights = F.softmax(scores, dim=-1) # Softmax normalize over the time steps index
            
            # Kernel smoothing lookup (Equation 4): Multiply weights by raw pricing input context
            # x_observed shape layout modified to target: [B, V, 1, T_max]
            # psi_h: [B, V, K]
            psi_h = torch.sum(attn_weights * x_observed.unsqueeze(2), dim=-1)
            psi_heads.append(psi_h)
            
        # Stack heads context array: [H, B, V, K]
        psi = torch.stack(psi_heads, dim=0)
        
        # Linear Fusion stage across multiple variables and heads (Equation 5)
        # Target representation matrix output tensor: z_n [B, V, K]
        z = torch.zeros(B, V, self.K, device=t_observed.device)
        for h in range(self.H):
            # Compute weight transformations: [B, V, K] matmul [V, V]
            z += torch.matmul(psi[h].transpose(-2, -1), self.W[h]).transpose(-2, -1)
            
        return z


class ATENetClassifier(nn.Module):
    """
    Implements the Simple Classifier backend engine (GRU + FC Blocks) 
    highlighted in Section 3.3.1 of the text.
    """
    def __init__(self, num_variables, hidden_dim, num_classes):
        super(ATENetClassifier, self).__init__()
        self.gru = nn.GRU(input_size=num_variables, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, z):
        # Input shape z: [B, V, K] -> Transpose tracking window to process sequentially: [B, K, V]
        z_seq = z.transpose(-2, -1)
        out, _ = self.gru(z_seq)
        
        # Pool across sequence window by picking the final sequential state output step
        out = out[:, -1, :] 
        
        # Classification blocks
        out = self.fc1(out)
        out = self.bn(out)
        out = F.gelu(out)
        return self.fc2(out)

# --- Auxiliary Custom Training Loss Components ---

def compute_intervariable_loss(x_raw, z_latent):
    """
    Calculates Intervariable Consistency Regularization from Equation (10) & (11).
    Ensures input structural asset relationships are retained in the latency maps.
    """
    B, V, _ = x_raw.shape
    
    # Generate correlation ground truth using the raw data sequence layer matrices
    x_mean = torch.mean(x_raw, dim=-1) # Collapse time window: [B, V]
    P = torch.sigmoid(torch.bmm(x_mean.unsqueeze(-1), x_mean.unsqueeze(1)))
    P = torch.round(P) # Binary target mapping matrix [B, V, V]
    
    # Map matching structural layouts within output latency layers
    z_mean = torch.mean(z_latent, dim=-1) # Collapse anchors window: [B, V]
    Q = torch.sigmoid(torch.bmm(z_mean.unsqueeze(-1), z_mean.unsqueeze(1))) # [B, V, V]
    
    # Binary Cross Entropy Loss between tracking arrays
    return F.binary_cross_entropy(Q, P)