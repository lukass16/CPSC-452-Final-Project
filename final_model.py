import torch
from torch_geometric.nn import MLP, Linear, PointNetConv, fps, global_max_pool, radius

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=128)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([4, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.5, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.encode = MLP([1024, 512], dropout=0.4, norm=None) # learn global features -> reduce to 512

        self.mlp_mu = MLP([512, 512], dropout=0.2, norm=None) # get mu
        self.mlp_logvar = MLP([512, 512], dropout=0.2, norm=None) # get logvar

        self.sdf1 = MLP([512 + 3, 256, 128], norm=None)
        self.sdf2 = MLP([128 + 3, 64, 1], norm=None)
        
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std) # std just gives dimension of tensor to give back
        return mu + epsilon * std


    def encoder(self, x, pos, batch):
        # encode shape
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        return sa3_out # x latent encoding from pointnet


    def vae(self, x):
        x = self.encode(x)
        mu = self.mlp_mu(x) # [1, 512]
        logvar = self.mlp_logvar(x) # [1, 512]
        z = self.reparametrize(mu, logvar) # [1, 512]
        return z, mu, logvar


    def decoder(self, x, query_pos):
        x = torch.cat((x.repeat(query_pos.shape[0], 1), query_pos), dim=-1) # concatenate encoded shape with query positions) # [B, np_q, 515]
        x = self.sdf1(x) # [B, np_q, 128]
        x = torch.cat((x, query_pos), dim=-1) # [B, np_q, 131]
        x = self.sdf2(x) # [B, np_q, 1]
        out = torch.tanh(x) # [B, np_q, 1]

        return out


    def forward(self, x, pos, batch, query_pos):
        # encode
        x, pos, batch = self.encoder(x, pos, batch)

        # vae
        x, mu, logvar = self.vae(x)
       
        # decoder
        out = self.decoder(x, query_pos)

        return out, mu, logvar