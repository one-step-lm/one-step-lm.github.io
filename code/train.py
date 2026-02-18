import torch
import torch.nn as nn

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

def training_run(model, get_optimizer, dataloader, mode):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        optimizer: The optimizer to use
        dataloader: The dataloader to use
        mode: The mode of the parametrization
            "v_pred": model predicts velocity, we use MSE loss with target = x1 - x0
            "x_pred_mse": model predicts x1, we use MSE loss with target = x1
            "x_pred_ce": model predicts x1, we use CE loss with target = x1
    """

    # init in/out dims and do sanity checks
    input_dim = model.net[0].in_features - 1 # subtract 1 for time dimension
    output_dim = model.net[-1].out_features

    if mode in ["v_pred", "x_pred_mse"]:
        assert input_dim == output_dim, f"Input and output dimensions must be equal for v_pred and x_pred_mse modes. Got {input_dim} and {output_dim}"
    
    
    # set model to train mode
    model = model.to(device)
    model.train()

    # create fresh optimizer
    optimizer = get_optimizer(model)

    # training loop
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training {mode}"):
        x1_emb, x1_onehot, x0 = batch
        x1_emb = x1_emb.to(device)
        x1_onehot = x1_onehot.to(device)
        x0 = x0.to(device)
        t = torch.rand(x1_emb.shape[0], 1, device=device)
        xt = t * x1_emb + (1 - t) * x0
        optimizer.zero_grad()
        out = model(xt, t)
        
        if mode == "v_pred":
            target = x1_emb - x0
            loss = mse_loss(out, target)
        elif mode == "x_pred_mse":
            target = x1_emb
            loss = mse_loss(out, target)
        elif mode == "x_pred_ce":
            target = x1_onehot
            loss = ce_loss(out, target)
        
        loss.backward()
        optimizer.step()
