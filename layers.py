""" Imports """
from functools import wraps

# general imports
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat # tensor manipulation
from torch_cluster import fps # further point sampling
from timm.layers import DropPath


""" Utility functions"""
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


""" Layers """
class PreNorm(nn.Module):
    '''
    Pre-norm layer that applies layer normalization and wraps the layer fn.
    
    Args:
        dim (int): Dimension of the input tensor.
        fn (callable): Function to apply after normalization.
        context_dim (int, optional): Dimension of the context tensor, if applicable.
    '''
    
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim) # normalizes last dimension of input tensor with dimension dim - used to normalize embedded point feature vector
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None # if there are context vectors, normalize them as well

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context): # if passed a context_dim retrieve and normalize the context vectors as well 
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)
  
  
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    ''' Feed-forward layer with GEGLU activation 
    
        Args:
            dim (int): Dimension of the input tensor.
            mult (int, optional): Multiplier for the hidden dimension. Defaults to 4.
            drop_path_rate (float, optional): Drop path rate for stochastic depth regularization. Defaults to 0.0.
    '''
    
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x)) # returns tensor of the same shape as input x after applying the feed-forward network and drop path regularization

class Attention(nn.Module):
    '''
    Multi-head attention layer.

    Args:
        query_dim (int): Dimension of the query vectors.
        context_dim (int, optional): Dimension of the context vectors. Defaults to None, which uses query_dim - if not provided, does self-attention.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        drop_path_rate (float): Drop path rate for stochastic depth regularization.
    '''
    
    
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False) # maps to query vectors of dimension inner_dim = dim_head * heads
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False) # maps to key and value vectors of dimension inner_dim = dim_head * heads
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    '''
    Point embedding layer 
    
    Performs Fourier positional encoding on 3D points and then maps them into a higher-dimensional embedding space
    '''
    def __init__(self, embedding_dim=48, out_dim=128):
        super().__init__()

        assert embedding_dim % 6 == 0
        self.embedding_dim = embedding_dim # positional embedding dimension (must be divisible by 6)
        Fdim = self.embedding_dim // 6 # number of frequencies for each axis (x, y, z), 3 axis with sin and cos for each = 6
        
        # creates exponentially spaced frequencies for Fourier encoding
        e = torch.pow(2, torch.arange(Fdim)).float() * np.pi # 1, 2, 4, 8, 16, ... (up to Fdim)
        
        # create the basis matrix for Fourier encoding with dimension (3, Fdim) - construct separate frequency bands for each axis
        e = torch.stack([
            # x axis encodings
            torch.cat([e, torch.zeros(Fdim),
                        torch.zeros(Fdim)]), # e, 0, 0 (where each 0 also has size Fdim)
            # y axis encodings
            torch.cat([torch.zeros(Fdim), e,
                        torch.zeros(Fdim)]), # 0, e, 0
            # z axis encodings
            torch.cat([torch.zeros(Fdim),
                        torch.zeros(Fdim), e]), # 0, 0, e
        ])
        self.register_buffer('basis', e)  # attaches the tensor e (under the name 'basis') to the module, so it is not a parameter but a buffer

        self.mlp = nn.Linear(self.embedding_dim+3, out_dim) 

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis) # input: B x N x 3, basis: 3 x Fdim -
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed # outputs B x N x dim - point embeddings for each point with dimension out_dim


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24, # depth of self-attention+feedforward blocks in decoder
        dim=512, # dimension of point embeddings (also referred to as out_dim) (each point is represented by a vector of this dimension)
        queries_dim = 512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        latent_dim = 64, # smaller latent dimension for the latent vector set
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False
    ):
        super().__init__()

        self.depth = depth # number of latent self-attention+feedforward blocks in decoder

        self.num_inputs = num_inputs # number of input points
        self.num_latents = num_latents # number of latent points - number of points in fps

        """ Encoder Layers """
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim), # takes input x (B, N, feature_dim), normalizes each feature (also for context), and applies attention 
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(out_dim=dim)
        
        """ Decoder Layers """
        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim) #(we learn how to project each latent vector in the vector set to a latent mean vector of dimension latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def encode(self, pc):
        # pc: B x N x 3 = point cloud with B batches, N points, and 3 coordinates (x, y, z)
        B, N, D = pc.shape
        assert N == self.num_inputs
        
        """ Further point sampling """
        all_points = pc.view(B*N, D) # flattens to one long vector of points

        batch = torch.arange(B).to(pc.device) # create batch indices/identifiers
        batch = torch.repeat_interleave(batch, N) # repeat each batch index N times - "batch" now labels each point in the all_points vector with its corresponding batch

        ratio = 1.0 * self.num_latents / self.num_inputs # downsample ratio

        idx = fps(all_points, batch, ratio=ratio) # fps in batch mode, returns indices of sampled points

        sampled_pc = all_points[idx]
        sampled_pc = sampled_pc.view(B, -1, 3) # go back to batches (B, N', 3) where N' is the number of sampled points
        """ """


        """ Point embedding """
        sampled_pc_embeddings = self.point_embed(sampled_pc) # embed (B x N') sampled points (B, N', 3) to (B, N', out_dim)
        pc_embeddings = self.point_embed(pc) # embed (B x N) original points (B, N, 3) to (B, N, out_dim)
        """ """
        
        
        """ Cross-attend from sampled points to original points + feedforward """
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings # cross attention + residual connection
        x = cross_ff(x) + x # feedforward + residual connection (B, N', out_dim)
        """ """


        """ Variational autoencoder encoding """
        mean = self.mean_fc(x) # project to latent mean (B, N', latent_dim) - here N' is the number of latent vectors in latent set
        logvar = self.logvar_fc(x) # project to latent log variance (B, N', latent_dim)

        posterior = DiagonalGaussianDistribution(mean, logvar) # wraps the parameters into a diagonal Gaussian distribution
        x = posterior.sample() # sample from the distribution to get the latent vector set (B, N', latent_dim) - i.e. sample form the latent set of vectors
        kl = posterior.kl() # get kl divergence of the posterior distribution
        """ """

        return kl, x # return: sampled latent vector set (B, N', latent_dim) from learned distribution and kl divergence of this distribution (from prior)


    def decode(self, x, queries):

        x = self.proj(x) # map from latent set of N' vectors to core set of N' latent vectors (B, N', out_dim) -> (B, N', dim)

        """ Self-attention + feedforward on latent vectors """
        for self_attn, self_ff in self.layers: # allow each of N' latent vectors to attend to each other
            x = self_attn(x) + x
            x = self_ff(x) + x
        """ """

        
        """ Cross-attend from decoder queries to latents """
        queries_embeddings = self.point_embed(queries) # embed queries (B, M, 3) to (B, M, out_dim) where M is the number of queries i.e. make the same point embeddings/tokens
        latents = self.decoder_cross_attn(queries_embeddings, context = x) # each query point attends to (gets info from) the latent set of points (B, M, out_dim) -> (B, M, out_dim)
        """ """
        
        # optional decoder feedforward
        # if exists(self.decoder_ff):
        #     latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents) # for each query point embedding (B, M, out_dim) map it to SDF value (B, M, out_dim) -> (B, M, 1)

    def forward(self, pc, queries):
        kl, x = self.encode(pc)

        o = self.decode(x, queries).squeeze(-1)

        # return o.squeeze(-1), kl
        return {'sdf': o, 'kl': kl}