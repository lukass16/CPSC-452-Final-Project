import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, device, num_epochs, kl_weight=1e-3):
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, device, kl_weight)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.6f}")

def train_one_epoch(model, data_loader, optimizer, device, kl_weight, points_used = 2048, num_query_points=512):
    model.train()

    for batch in tqdm(data_loader):
        B, N, D = batch.shape  # B=20, N=500, D=4

        # Random indices: shape [B, points_used]
        samples_idx = torch.randint(0, N, (B, points_used), device=batch.device)
        query_idx = torch.randint(0, N, (B, num_query_points), device=batch.device)

        # Use batch-wise indexing to gather the sampled points
        sample_idx = torch.arange(B, device=batch.device).unsqueeze(1).expand(-1, points_used)
        query_idx = torch.arange(B, device=batch.device).unsqueeze(1).expand(-1, num_query_points)

        sampled_points = batch[sample_idx, samples_idx] # [B, points_used, 4]
        query_points = batch[query_idx, query_idx] # [B, query_points, 4]

        sample_pos = sampled_points[:, :, :3].to(device) # [B, points_used, 3]
        query_pos = query_points[:, :, :3].to(device) # [B, points_used, 3]

        query_sdf = query_points[:, :, 3].to(device)   # shape [B, query_points]

        optimizer.zero_grad()

        outputs = model(sample_pos, query_pos)

        predicted_sdfs = outputs['sdf']   # (B, N_query)
        kl_loss = outputs['kl']            # (B, )

        sdf_loss = F.l1_loss(predicted_sdfs, query_sdf)

        kl_loss = torch.mean(kl_loss)

        loss = sdf_loss + kl_weight * kl_loss

        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, data_loader, device, points_used = 2048, num_query_points=512):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        B, N, D = batch.shape  # B=20, N=500, D=4

        # Random indices: shape [B, points_used]
        samples_idx = torch.randint(0, N, (B, points_used), device=batch.device)
        query_idx = torch.randint(0, N, (B, num_query_points), device=batch.device)

        # Use batch-wise indexing to gather the sampled points
        sample_idx = torch.arange(B, device=batch.device).unsqueeze(1).expand(-1, points_used)
        query_idx = torch.arange(B, device=batch.device).unsqueeze(1).expand(-1, num_query_points)

        sampled_points = batch[sample_idx, samples_idx] # [B, points_used, 4]
        query_points = batch[query_idx, query_idx] # [B, query_points, 4]

        sample_pos = sampled_points[:, :, :3].to(device) # [B, points_used, 3]
        query_pos = query_points[:, :, :3].to(device) # [B, points_used, 3]

        query_sdf = query_points[:, :, 3].to(device)   # shape [B, query_points]

        outputs = model(sample_pos, query_pos)

        predicted_sdfs = outputs['sdf']
        loss = F.l1_loss(predicted_sdfs, query_sdf)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches
