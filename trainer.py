import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(model, train_loader, val_loader, optimizer, device, num_epochs, kl_weight=1e-3, points_used=2048, num_query_points=512):
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, device, kl_weight, points_used, num_query_points)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.6f}")


def train_one_epoch(model, data_loader, optimizer, device, kl_weight, points_used = 2048, num_query_points=2048):
    model.train()

    for batch in tqdm(data_loader):
        B, N, D = batch.shape  # B=10, N=50000, D=4

        """ Sample from point clouds in the batch """ # (each batch contains multiple point clouds)
        ### input points ###
        sample_pos = torch.zeros((B, points_used, 3), device=device) # shape [B, points_used, 3] - surface points sampled from each shape in the batch

        for i in range(B):
            shape = batch[i]  # [N, 4]
            surface_pts = shape[shape[:, 3] == 0]  # sdf == 0 → surface points
            # Random sample
            sample_idx = torch.randperm(surface_pts.shape[0])[:points_used]
            sample_pos[i] = surface_pts[sample_idx, :3]  # only x,y,z
        
        ### query points ###
        querys_idx = torch.randint(0, N, (B, num_query_points), device=batch.device) # dim = (B, num_query_points)
        query_idx = torch.arange(B, device=batch.device).unsqueeze(1).expand(-1, num_query_points) # [B, num_query_points]
        # use advanced indexing to gather the sampled points
        query_points = batch[query_idx, querys_idx] # [B, query_points, 4]
        query_pos = query_points[:, :, :3].to(device) # [B, points_used, 3]
        query_sdf = query_points[:, :, 3].to(device)   # shape [B, query_points]
        """ """
        # Note: effectively every batch we use a different subset of points from each shape in the batch

        """ Perform step """
        optimizer.zero_grad()
        outputs = model(sample_pos, query_pos)
        predicted_sdfs = outputs['sdf']   # (B, N_query)
        kl_loss = outputs['kl']            # (B, )

        sdf_loss = F.mse_loss(predicted_sdfs, query_sdf)
        kl_loss = torch.mean(kl_loss)

        loss = sdf_loss + kl_weight * kl_loss

        loss.backward()
        optimizer.step()
        """ """


@torch.no_grad()
def evaluate(model, data_loader, device, points_used = 2048, num_query_points=512):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        B, N, D = batch.shape  # B=20, N=500, D=4

        """ Sample from point clouds in the batch """ # (each batch contains multiple point clouds)
        ### input points ###
        sample_pos = torch.zeros((B, points_used, 3), device=device) # shape [B, points_used, 3] - surface points sampled from each shape in the batch

        for i in range(B):
            shape = batch[i]  # [N, 4]
            surface_pts = shape[shape[:, 3] == 0]  # sdf == 0 → surface points
            # Random sample
            sample_idx = torch.randperm(surface_pts.shape[0])[:points_used]
            sample_pos[i] = surface_pts[sample_idx, :3]  # only x,y,z
        
        ### query points ###
        querys_idx = torch.randint(0, N, (B, num_query_points), device=batch.device) # dim = (B, num_query_points)
        query_idx = torch.arange(B, device=batch.device).unsqueeze(1).expand(-1, num_query_points) # [B, num_query_points]
        # use advanced indexing to gather the sampled points
        query_points = batch[query_idx, querys_idx] # [B, query_points, 4]
        query_pos = query_points[:, :, :3].to(device) # [B, points_used, 3]
        query_sdf = query_points[:, :, 3].to(device)   # shape [B, query_points]
        """ """

        outputs = model(sample_pos, query_pos)

        predicted_sdfs = outputs['sdf']
        loss = F.mse_loss(predicted_sdfs, query_sdf)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches
