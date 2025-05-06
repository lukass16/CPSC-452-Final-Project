import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import os

class SDFDataset(Dataset):
    def __init__(self, path_to_data):
        self.path = path_to_data
        self.files = os.listdir(self.path)

    def __len__(self):
        return len(self.files)	

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.path, self.files[idx]))

    def get_name(self, idx):
        return self.files[idx]
    
def visualize_sdf_3d(sample):
    # Convert to NumPy
    coords = sample[:, :3].numpy()  # (x, y, z)
    sdf = sample[:, 3].numpy()      # scalar values for color

    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot colored by sdf values
    p = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=sdf, cmap='coolwarm', s=1)

    # Add colorbar
    fig.colorbar(p, ax=ax, label='Signed Distance')

    # Axes settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D SDF Visualization')

    plt.tight_layout()
    plt.show()

def visualize_sdf_surface_3d(sample, tolerance = 0.01):
    # Convert to NumPy
    coords = sample[:, :3].numpy()  # (x, y, z)
    sdf = sample[:, 3].numpy()      # scalar values for color

    mask = np.abs(sdf) <= tolerance
    coords_surface = coords[mask]
    sdf_surface = sdf[mask]

    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot colored by sdf values
    p = ax.scatter(coords_surface[:, 0], coords_surface[:, 1], coords_surface[:, 2], c=sdf_surface, cmap='coolwarm', s=1)

    # Add colorbar
    fig.colorbar(p, ax=ax, label='Signed Distance')

    # Axes settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D SDF Visualization')

    plt.tight_layout()
    plt.show()

def visualize_sdf_2d(sample, x_target = 0.0, tolerance = 0.01):
    data = sample.numpy()

    # Mask for points near x = x_target
    slice_mask = np.abs(data[:, 0] - x_target) < tolerance # creates boolean mask of points that are within our range
    yz_slice = data[slice_mask]

    # Unpack y, z, sdf values
    y, z, sdf = yz_slice[:, 1], yz_slice[:, 2], yz_slice[:, 3]

    # Create regular grid in YZ plane
    grid_y, grid_z = np.mgrid[y.min():y.max():200j, z.min():z.max():200j]

    # Interpolate sdf values onto the grid
    grid_sdf = griddata((y, z), sdf, (grid_y, grid_z), method='linear')

    plt.figure(figsize=(6, 5))
    cont = plt.contourf(grid_y, grid_z, grid_sdf, levels=100, cmap='coolwarm')
    plt.contour(grid_y, grid_z, grid_sdf, levels=[0], colors='black', linewidths=1.5)
    plt.colorbar(cont, label='Signed Distance')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(f'YZ Slice of SDF at x ≈ {x_target}')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def print_summary(sample):
    coords = sample[:, :3].numpy()  # (x, y, z)
    sdf = sample[:, 3].numpy()      # scalar values for color

    # print summary of SDF in nicely formatted table
    print(f"Shape: {sample.shape}")
    print(f"Min SDF: {sdf.min()}")
    print(f"Max SDF: {sdf.max()}")
    print(f"Mean SDF: {sdf.mean()}")
    print(f"Std SDF: {sdf.std()}")
    print(f"Min X: {coords[:,0].min()}")
    print(f"Max X: {coords[:,0].max()}")
    print(f"Min Y: {coords[:,1].min()}")
    print(f"Max Y: {coords[:,1].max()}")
    print(f"Min Z: {coords[:,2].min()}")
    print(f"Max Z: {coords[:,2].max()}")