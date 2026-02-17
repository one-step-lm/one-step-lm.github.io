import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "axes.unicode_minus": False,
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

# Color scheme
BOX_FACE_COLOR = "#FFFFFF"
BOX_EDGE_COLOR = "#2E86AB"
TEXT_COLOR = "#1A1A1A"
FINAL_TEXT_COLOR = "#000000"
TITLE_COLOR = "#1A1A1A"

# Create figure with 3 subplots side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3.2))
fig.patch.set_facecolor('#FAFAFA')

# Animation settings - more frames for smoother transitions
frames_per_step = 8  # Interpolation frames between each step
max_steps = max(ar_n_rows, md_n_rows, flm_n_rows)
linger_frames = 30  # 5 seconds at 10 fps
max_frames = max_steps * frames_per_step + linger_frames


def ease_in_out(t):
    """Smooth easing function for transitions"""
    return t * t * (3.0 - 2.0 * t)


def draw_ar(ax, frame):
    """Draw AR animation on given axis"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#FAFAFA')
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Add title
    ax.set_title("Autoregressive", fontsize=16, fontweight='600', pad=20, y=1.05, color=TITLE_COLOR)

    x_positions = np.linspace(0.04, 0.96, len(ar_clean_tokens))

    # Calculate current step and interpolation factor
    current_step = min(frame // frames_per_step, ar_n_rows - 1)
    t = (frame % frames_per_step) / frames_per_step
    t_smooth = ease_in_out(t)

    for r in range(current_step + 1):
        y = 0.98 - r * (0.98 / (ar_n_rows - 1))

        # Row fade-in alpha
        row_alpha = 1.0
        if r == current_step and current_step < ar_n_rows - 1:
            row_alpha = min(1.0, t_smooth * 1.5)

        # Show all tokens up to r+1 for this row
        for idx in range(r + 1):
            x_c = x_positions[idx]
            correct_word = ar_clean_tokens[idx]

            # Box
            ax.text(
                x_c, y, "           ",
                ha='center', va='center', fontsize=14,
                bbox=dict(
                    boxstyle="round,pad=0.25,rounding_size=0.2",
                    facecolor=BOX_FACE_COLOR,
                    edgecolor=BOX_EDGE_COLOR,
                    linewidth=1.5,
                ),
                alpha=row_alpha,
                zorder=1
            )

            # Word (except last frame full reveal handled separately)
            if r < ar_n_rows - 1:
                ax.text(
                    x_c, y, correct_word,
                    fontsize=14, fontweight='500',
                    ha='center', va='center',
                    color=TEXT_COLOR, alpha=row_alpha, zorder=2
                )

    # Final clean bottom row
    if current_step == ar_n_rows - 1:
        y_bot = 0.98 - (ar_n_rows - 1) * (0.98 / (ar_n_rows - 1))
        # Only fade in during the actual transition to final step
        if frame < ar_n_rows * frames_per_step:
            final_alpha = min(1.0, t_smooth * 1.5)
        else:
            final_alpha = 1.0
        for idx, x_c in enumerate(x_positions[:ar_n_rows]):
            ax.text(
                x_c, y_bot, ar_clean_tokens[idx],
                fontsize=14, fontweight='normal',
                ha='center', va='center',
                color=FINAL_TEXT_COLOR, alpha=final_alpha, zorder=5
            )


def draw_mask_diffusion(ax, frame):
    """Draw Mask Diffusion animation on given axis"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#FAFAFA')
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Add title
    ax.set_title("Masked Diffusion", fontsize=16, fontweight='600', pad=20, y=1.05, color=TITLE_COLOR)

    x_positions = np.linspace(0.04, 0.96, len(md_clean_tokens))

    # Calculate current step and interpolation factor
    current_step = min(frame // frames_per_step, md_n_rows - 1)
    t = (frame % frames_per_step) / frames_per_step
    t_smooth = ease_in_out(t)

    for r in range(current_step + 1):
        y = 0.98 - r * (0.98 / (md_n_rows - 1))

        # Row fade-in alpha
        row_alpha = 1.0
        if r == current_step and current_step < md_n_rows - 1:
            row_alpha = min(1.0, t_smooth * 1.5)

        for idx, x_c in enumerate(x_positions):
            correct_word = md_clean_tokens[idx]

            # Box
            ax.text(
                x_c, y, "           ",
                ha='center', va='center', fontsize=14,
                bbox=dict(
                    boxstyle="round,pad=0.25,rounding_size=0.2",
                    facecolor=BOX_FACE_COLOR,
                    edgecolor=BOX_EDGE_COLOR,
                    linewidth=1.5,
                ),
                alpha=row_alpha,
                zorder=1
            )

            # Mask / token logic
            if r < md_n_rows - 1:
                word = correct_word if r >= md_unmask[idx] else "[Mask]"
                ax.text(
                    x_c, y, word,
                    fontsize=14, fontweight='500',
                    ha='center', va='center',
                    color=TEXT_COLOR, alpha=row_alpha, zorder=2
                )

    # Final clean bottom row
    if current_step == md_n_rows - 1:
        y_bot = 0.98 - (md_n_rows - 1) * (0.98 / (md_n_rows - 1))
        # Only fade in during the actual transition to final step
        if frame < md_n_rows * frames_per_step:
            final_alpha = min(1.0, t_smooth * 1.5)
        else:
            final_alpha = 1.0
        for idx, x_c in enumerate(x_positions):
            ax.text(
                x_c, y_bot, md_clean_tokens[idx],
                fontsize=14, fontweight='normal',
                ha='center', va='center',
                color=FINAL_TEXT_COLOR, alpha=final_alpha, zorder=5
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
    ax.set_facecolor('#FAFAFA')
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Add title
    ax.set_title("FLM (Ours)", fontsize=16, fontweight='600', pad=20, y=1.05, color=TITLE_COLOR)

    x_positions = np.linspace(0.04, 0.96, len(flm_clean_tokens))

    # Calculate current step and interpolation factor
    current_step = min(frame // frames_per_step, flm_n_rows - 1)
    t = (frame % frames_per_step) / frames_per_step
    t_smooth = ease_in_out(t)

    for r in range(current_step + 1):
        y = 0.98 - r * (0.98 / (flm_n_rows - 1))
        p_clean = (r / (flm_n_rows - 1)) ** 3
        jitter_scale = 0.01 * (1 - p_clean)
        num_layers = layers_for_row(r)

        # Row fade-in alpha
        row_alpha = 1.0
        if r == current_step and current_step < flm_n_rows - 1:
            row_alpha = min(1.0, t_smooth * 1.5)

        for idx, x_c in enumerate(x_positions):
            correct_word = flm_clean_tokens[idx]

            # Fixed RNG per (row, token)
            rng = np.random.default_rng(seed=1000 + r * 100 + idx)

            # Box
            ax.text(
                x_c, y, "           ",
                ha='center', va='center', fontsize=14,
                bbox=dict(
                    boxstyle="round,pad=0.25,rounding_size=0.2",
                    facecolor=BOX_FACE_COLOR,
                    edgecolor=BOX_EDGE_COLOR,
                    linewidth=1.5,
                ),
                alpha=row_alpha,
                zorder=1
            )

            if r < flm_n_rows - 1:
                for layer in range(num_layers):
                    word = correct_word if rng.random() < p_clean else rng.choice(vocab)
                    dx = rng.normal(scale=jitter_scale)
                    alpha = (0.06 * (1 - p_clean) + 0.85 * p_clean) * row_alpha

                    ax.text(
                        x_c + dx, y, word,
                        fontsize=14, fontweight='500',
                        ha='center', va='center',
                        color=TEXT_COLOR, alpha=alpha, zorder=2
                    )

    # Final clean bottom row
    if current_step == flm_n_rows - 1:
        y_bot = 0.98 - (flm_n_rows - 1) * (0.98 / (flm_n_rows - 1))
        # Only fade in during the actual transition to final step
        if frame < flm_n_rows * frames_per_step:
            final_alpha = min(1.0, t_smooth * 1.5)
        else:
            final_alpha = 1.0
        for idx, x_c in enumerate(x_positions):
            ax.text(
                x_c, y_bot, flm_clean_tokens[idx],
                fontsize=14, fontweight='normal',
                ha='center', va='center',
                color=FINAL_TEXT_COLOR, alpha=final_alpha, zorder=5
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
    interval=100,  # Reduced interval for smoother animation
    repeat=False
)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.92], pad=2.5)
anim.save("overview.gif", writer=PillowWriter(fps=10), dpi=150)

