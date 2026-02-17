import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
})

# Data for all three animations
vocab = ["I", "live", "in", "New", "York", "queens", "colors", "cloud", "sea", "waves",
         "San", "Diego", "how", "who", "where", "zumba"]

# AR configuration
ar_clean_tokens = ["I", "live", "in", "New", "York"]
ar_n_rows = 5

# Mask Diffusion configuration
md_clean_tokens = ["I", "live", "in", "New", "Diego"]
md_unmask = [1, 2, 1, 3, 3]
md_n_rows = 4

# FLM configuration
flm_clean_tokens = ["I", "live", "in", "New", "York"]
flm_n_rows = 4
flm_layers_top = 30
flm_layers_bottom = 0
np.random.seed(1)

# Create figure with 3 subplots side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))

# Maximum number of frames across all animations
max_frames = max(ar_n_rows, md_n_rows, flm_n_rows)


def draw_ar(ax, frame):
    """Draw AR animation on given axis"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Add title
    ax.set_title("Autoregressive", fontweight='bold', pad=20, y=1.05)

    x_positions = np.linspace(0.04, 0.96, len(ar_clean_tokens))
    current_frame = min(frame, ar_n_rows - 1)

    for r in range(current_frame + 1):
        y = 0.98 - r * (0.98 / (ar_n_rows - 1))

        for idx, x_c in enumerate(x_positions[:r+1]):
            correct_word = ar_clean_tokens[idx]

            # Box
            ax.text(
                x_c, y, "           ",
                ha='center', va='center', fontsize=15,
                bbox=dict(
                    boxstyle="round,pad=0.22,rounding_size=0.22",
                    facecolor="#FFF2E8",
                    edgecolor="#C6A437",
                    linewidth=0.8,
                ),
                zorder=1
            )

            # Word (except last frame full reveal handled separately)
            if r < ar_n_rows - 1:
                ax.text(
                    x_c, y, correct_word,
                    fontsize=15, fontweight='medium',
                    ha='center', va='center',
                    color="#111111", zorder=2
                )

    # Final clean bottom row
    if current_frame == ar_n_rows - 1:
        y_bot = 0.98 - (ar_n_rows - 1) * (0.98 / (ar_n_rows - 1))
        for idx, x_c in enumerate(x_positions):
            ax.text(
                x_c, y_bot, ar_clean_tokens[idx],
                fontsize=15, fontweight='medium',
                ha='center', va='center',
                color="#000000", zorder=5
            )


def draw_mask_diffusion(ax, frame):
    """Draw Mask Diffusion animation on given axis"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Add title
    ax.set_title("Mask Diffusion", fontweight='bold', pad=20, y=1.05)

    x_positions = np.linspace(0.04, 0.96, len(md_clean_tokens))
    current_frame = min(frame, md_n_rows - 1)

    for r in range(current_frame + 1):
        y = 0.98 - r * (0.98 / (md_n_rows - 1))

        for idx, x_c in enumerate(x_positions):
            correct_word = md_clean_tokens[idx]

            # Box
            ax.text(
                x_c, y, "           ",
                ha='center', va='center', fontsize=15,
                bbox=dict(
                    boxstyle="round,pad=0.22,rounding_size=0.22",
                    facecolor="#FFF2E8",
                    edgecolor="#C6A437",
                    linewidth=0.8,
                ),
                zorder=1
            )

            # Mask / token logic
            if r < md_n_rows - 1:
                word = correct_word if r >= md_unmask[idx] else "[Mask]"
                ax.text(
                    x_c, y, word,
                    fontsize=15, fontweight='medium',
                    ha='center', va='center',
                    color="#111111", zorder=2
                )

    # Final clean bottom row
    if current_frame == md_n_rows - 1:
        y_bot = 0.98 - (md_n_rows - 1) * (0.98 / (md_n_rows - 1))
        for idx, x_c in enumerate(x_positions):
            ax.text(
                x_c, y_bot, md_clean_tokens[idx],
                fontsize=15, fontweight='medium',
                ha='center', va='center',
                color="#000000", zorder=5
            )


def layers_for_row(r):
    return int(np.round(np.interp(r, [0, flm_n_rows-1], [flm_layers_top, flm_layers_bottom])))


def draw_flm(ax, frame):
    """Draw FLM animation on given axis"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Add title
    ax.set_title("FLM", fontweight='bold', pad=20, y=1.05)

    x_positions = np.linspace(0.04, 0.96, len(flm_clean_tokens))
    current_frame = min(frame, flm_n_rows - 1)

    for r in range(current_frame + 1):
        y = 0.98 - r * (0.98 / (flm_n_rows - 1))
        p_clean = (r / (flm_n_rows - 1)) ** 3
        jitter_scale = 0.01 * (1 - p_clean)
        num_layers = layers_for_row(r)

        for idx, x_c in enumerate(x_positions):
            correct_word = flm_clean_tokens[idx]

            # Fixed RNG per (row, token)
            rng = np.random.default_rng(seed=1000 + r * 100 + idx)

            # Box
            ax.text(
                x_c, y, "           ",
                ha='center', va='center', fontsize=15,
                bbox=dict(
                    boxstyle="round,pad=0.22,rounding_size=0.22",
                    facecolor="#FFF2E8",
                    edgecolor="#C6A437",
                    linewidth=0.8,
                ),
                zorder=1
            )

            if r < flm_n_rows - 1:
                for layer in range(num_layers):
                    word = correct_word if rng.random() < p_clean else rng.choice(vocab)
                    dx = rng.normal(scale=jitter_scale)
                    alpha = 0.06 * (1 - p_clean) + 0.8 * p_clean

                    ax.text(
                        x_c + dx, y, word,
                        fontsize=15, fontweight='medium',
                        ha='center', va='center',
                        color="#111111", alpha=alpha, zorder=2
                    )

    # Final clean bottom row
    if current_frame == flm_n_rows - 1:
        y_bot = 0.98 - (flm_n_rows - 1) * (0.98 / (flm_n_rows - 1))
        for idx, x_c in enumerate(x_positions):
            ax.text(
                x_c, y_bot, flm_clean_tokens[idx],
                fontsize=15, fontweight='medium',
                ha='center', va='center',
                color="#000000", zorder=5
            )


def update_all(frame):
    """Update all three subplots"""
    draw_ar(ax1, frame)
    draw_mask_diffusion(ax2, frame)
    draw_flm(ax3, frame)


# Create animation
anim = FuncAnimation(
    fig,
    update_all,
    frames=max_frames,
    interval=800,
    repeat=False
)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.92], pad=2.0)
anim.save("overview.gif", writer=PillowWriter(fps=1.2))

