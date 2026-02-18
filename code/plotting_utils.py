import matplotlib.pyplot as plt
import numpy as np

# for accessibility: Wong's color pallette: cf. https://davidmathlogic.com/colorblind
wong_black = [0/255, 0/255, 0/255]          # #000000
wong_amber = [230/255, 159/255, 0/255]      # #E69F00
wong_cyan = [86/255, 180/255, 233/255]      # #56B4E9
wong_green = [0/255, 158/255, 115/255]      # #009E73
wong_yellow = [240/255, 228/255, 66/255]    # #F0E442
wong_navy = [0/255, 114/255, 178/255]       # #0072B2
wong_red = [213/255, 94/255, 0/255]         # #D55E00
wong_pink = [204/255, 121/255, 167/255]     # #CC79A7
wong_cmap = [wong_amber, wong_cyan, wong_green, wong_yellow, wong_navy, wong_red, wong_pink]

source_color = wong_navy
target_color = wong_red
pred_color = wong_green
line_color = wong_yellow
bg_theme = 'dark' #  'black', 'white', 'dark', 'light'
if bg_theme in ['black','dark']:
    plt.style.use('dark_background')
else:
    plt.rcdefaults()

plt.rcParams.update({
    "text.usetex": False,            # Set to True if you have LaTeX installed
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "font.size": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

def interpolate_color(t, start='blue', end='red'):
    """Interpolate from start color to end color"""
    start_color = plt.cm.colors.to_rgb(start)
    end_color = plt.cm.colors.to_rgb(end)
    return (1-t) * np.array(start_color) + t * np.array(end_color)

def add_styling_to_scatter_ax(ax):
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False)
    ax.tick_params(bottom=False)
    ax.grid(True, linestyle='--', alpha=0.3, lw=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def plot_xts(xts, num_timesteps=5, labels=None, mode=None, s=3, alpha=0.6):
    """
    ICML-style plotting for 2D particle distributions.
    
    Args:
        xts: Tensor/List of shape [T, N, 2]
        num_timesteps: How many snapshots to show
        labels: Integer labels for each point [N] to color-code groups
        mode: Mode for figure suptitle
        s: Marker size
        alpha: Point transparency
    """

    fig, axes = plt.subplots(1, num_timesteps, figsize=(4 * num_timesteps, 4), 
                             constrained_layout=True, sharex=True, sharey=True)
    
    # Generate indices for snapshots
    T = len(xts)
    idxs = np.linspace(0, T - 1, num_timesteps, dtype=int)
    
    # Determine colors
    if labels is not None:
        # Map labels to Wong colors
        c_list = [wong_cmap[int(l) % len(wong_cmap)] for l in labels]
    else:
        # Default to Wong Navy if no labels provided
        c_list = wong_navy

    for i, t_idx in enumerate(idxs):
        ax = axes[i]
        curr_x = xts[t_idx]
        
        # Plotting
        ax.scatter(curr_x[:, 0], curr_x[:, 1], s=s, alpha=alpha, 
                   c=c_list, edgecolors='none', rasterized=True) # Rasterized for smaller PDF size
        
        # Title using LaTeX notation for time
        ax.set_title(rf"$t = {t_idx/(T-1):.2f}$", usetex=False)
        add_styling_to_scatter_ax(ax)

    if mode:
        if mode == "v_pred":
            title = r"$\mathcal{L}_{v, \text{MSE}}$"
        elif mode == "x_pred_mse":
            title = r"$\mathcal{L}_{x_1, \text{MSE}}$"
        elif mode == "x_pred_ce":
            title = r"$\mathcal{L}_{x_1, \text{CE}}$"
        fig.suptitle(title, fontweight='bold', fontsize=22)
    
    return fig