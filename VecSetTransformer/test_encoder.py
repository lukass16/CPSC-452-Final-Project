# %% [markdown]
# # **Test Encoder**

# %%
# general imports
import torch

# import custom modules
import dataset_utils 
import layers

# %% [markdown]
# **Load and prepare dataset**

# %%
""" Hyperparameters"""
# training and setup
train_percent = 0.8
batch_size = 10 #(B)
lr = 0.001
epochs = 10

# model parameters
num_inputs = 10000 # number of input points (N)
num_latents = 500 # number of latent points (N')
dim = 500 # dimension of point embeddings (D)
num_query_points = 1000 # number of query points (M)


# %%
# load dataset
dataset = dataset_utils.SDFDataset("./cars100")

# %%
from torch.utils.data import DataLoader, random_split

# get set sizes for train and validation splits
train_size = int(train_percent * len(dataset))
val_size = len(dataset) - train_size
print(f"Dataset size: {len(dataset)}, Train size: {train_size}, Validation size: {val_size}")

# split dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# create data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# %% [markdown]
# **Setup model**

# %%
# get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize model and optimizer
model = layers.KLAutoEncoder(num_inputs=num_inputs, num_latents=num_latents, dim = dim, queries_dim=dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %%
import trainer as t
# # from importlib import reload
# # reload(t)
t.train(model, train_loader, val_loader, optimizer, device, num_epochs=epochs, points_used = num_inputs, num_query_points=num_query_points)

# %%
# save model
torch.save(model.state_dict(), "klautoencoder.pth")

# load model
# model = layers.KLAutoEncoder(num_inputs=num_inputs, num_latents=num_latents, dim = dim).to(device)
# model.load_state_dict(torch.load("klautoencoder.pth", map_location=device))

# %%
def test_model(example, num_query = 10000):
    B, N, D = example.shape
    
    """ Sample from point clouds in the example """ # (each example contains multiple point clouds)
    ### input points ###
    sample_pos = torch.zeros((B, num_inputs, 3), device=device) # shape [B, num_inputs, 3] - surface points sampled from each shape in the example

    for i in range(B):
        shape = example[i]  # [N, 4]
        surface_pts = shape[shape[:, 3] == 0]  # sdf == 0 → surface points
        # Random sample
        sample_idx = torch.randperm(surface_pts.shape[0])[:num_inputs]
        sample_pos[i] = surface_pts[sample_idx, :3]  # only x,y,z
    
    ### query points ###
    querys_idx = torch.randint(0, N, (B, num_query), device=example.device) # dim = (B, num_query)
    query_idx = torch.arange(B, device=example.device).unsqueeze(1).expand(-1, num_query) # [B, num_query]
    # use advanced indexing to gather the sampled points
    query_points = example[query_idx, querys_idx] # [B, query_points, 4]
    query_pos = query_points[:, :, :3].to(device) # [B, num_inputs, 3]
    query_sdf = query_points[:, :, 3].to(device)   # shape [B, query_points]
    """ """

    outputs = model(sample_pos, query_pos)
    sdf_values = outputs['sdf'].unsqueeze(-1)

    combined = torch.cat([query_pos, sdf_values], dim=-1)[0]  # return only result for first shape
    return combined

def totally_random(example):
    shape0 = example[0]
    query_idx = torch.randperm(shape0.shape[0])[:10000]
    queries = shape0[query_idx].unsqueeze(0) # [1, 2048, 4]
    query_pos = queries[:, :, :3].to(device) # [B, num_inputs, 3]
    sdf_values = torch.rand(1, 10000, 1) - 0.15
    combined = torch.cat([query_pos.cpu(), sdf_values.cpu()], dim=-1)  # shape [1, 512, 4]
    return combined

# %%
import dataset_utils
model.eval()
example = next(iter(train_loader))
print(example.shape)
pred = test_model(example).squeeze(0).cpu()
rand = totally_random(example).squeeze(0)

# dataset_utils.visualize_sdf_2d(pred.detach().cpu())
print("TRUE")
example = example[0]
dataset_utils.visualize_sdf_3d(example)
dataset_utils.visualize_sdf_2d(example)
print("MODEL")
dataset_utils.visualize_sdf_3d(pred.detach().cpu())
dataset_utils.visualize_sdf_2d(pred.detach().cpu(), tolerance=0.1)
print("TOTALLY RANDOM")
dataset_utils.visualize_sdf_3d(rand)
dataset_utils.visualize_sdf_2d(rand)



