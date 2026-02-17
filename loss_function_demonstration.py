from datasets import SimplexDataset
from models import SimpleMLP
from train import training_run
from plotting_utils import *

import torch
from torch.utils.data import DataLoader
from torch import nn

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

angles = torch.linspace(0, 2 * torch.pi, 6)[:-1]
pentagon_mapping = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

N_SAMPLES = 1024 * 5000

dataset = SimplexDataset(n_samples=N_SAMPLES, n_dims=5, embed_matrix=pentagon_mapping)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)

def get_model(mode):
    return SimpleMLP(input_dim=pentagon_mapping.shape[1], hidden_dim=128, output_dim=pentagon_mapping.shape[0] if mode=="x_pred_ce" else pentagon_mapping.shape[1])

trained_models = {}
for mode in ["v_pred", "x_pred_mse", "x_pred_ce"]:
    model = get_model(mode)
    training_run(model, get_optimizer, dataloader, mode=mode)
    trained_models[mode] = model

x1_emb, x1_onehot, noise = next(iter(dataloader))

for mode in trained_models.keys():
    t = torch.zeros(x1_emb.shape[0], 1, device=device)
    xt = noise.to(device).clone()
    num_steps = 100
    xts = [xt.detach().clone().cpu().numpy()]
    model = get_model(mode).to(device)
    model.load_state_dict(trained_models[mode].state_dict())
    model.eval()
    for i in range(num_steps):
        with torch.no_grad():
            if mode == "v_pred":
                # v-prediction
                v_hat = model(xt, t)
            else:
                # x-prediction
                x1_hat = model(xt, t)
                if mode == "x_pred_ce":
                    x1_hat = nn.functional.softmax(x1_hat, dim=1)
                    x1_hat = x1_hat @ pentagon_mapping.to(device)
                v_hat = (x1_hat - xt) / (1 - t)
        xt = xt + v_hat / num_steps
        t += 1 / num_steps
        xts.append(xt.detach().clone().cpu().numpy())
    
    colors = torch.cdist(torch.tensor(xts[-1]), pentagon_mapping).argmin(dim=1)
    colors = colors.detach().cpu().numpy()
    plot_xts(xts, 5, colors, mode, s=10)
    plt.savefig(f"figures/inference_trajectory_{mode}.png")
    plt.close()