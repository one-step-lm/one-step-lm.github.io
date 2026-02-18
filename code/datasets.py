import torch
from torch.utils.data import Dataset

class SimplexDataset(Dataset):
    """
    A simple dataset that creates pair of distributions p_0, p_1,
    where p_0 is gaussian source and p_1 is simplex-vertices represented as one-hot vectors.
    """
    def __init__(self, n_samples=10000, n_dims=2, noise_std=1.0,
        embed_matrix=None, seed=0):
        """
        Args:
            n_samples: Total number of points to generate.
            n_dims: Number of dimensions for the simplex.
            noise_std: Standard deviation of the Gaussian noise prior.
            seed: Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.n_dims = n_dims
        self.noise_std = noise_std
        
        torch.manual_seed(seed)

        bins = torch.randint(0, self.n_dims, (self.n_samples,))
        onehot_points = torch.zeros(self.n_samples, self.n_dims)
        onehot_points[torch.arange(self.n_samples), bins] = 1.0

        if embed_matrix is None:
            embed_matrix = torch.eye(self.n_dims)
        
        embedded_points = onehot_points @ embed_matrix
        
        noise_points = torch.randn_like(embedded_points) * self.noise_std
        
        self.onehot_points = onehot_points
        self.embedded_points = embedded_points
        self.noise_points = noise_points

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x1_emb = self.embedded_points[idx]
        x1_onehot = self.onehot_points[idx]
        x0 = self.noise_points[idx]
        return x1_emb, x1_onehot, x0