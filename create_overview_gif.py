import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
})

############## AR ##############
clean_tokens = ["I", "live", "in", "New", "York"]
unmask = [1, 2, 1, 3, 3]

n_rows = 5
x_positions = np.linspace(0.04, 0.96, len(clean_tokens))

fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1.05)

ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)


def draw_until(frame):
    ax.clear()

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    for r in range(frame + 1):
        y = 0.98 - r * (0.98 / (n_rows - 1))

        for idx, x_c in enumerate(x_positions[:r+1]):
            correct_word = clean_tokens[idx]

            # Box
            ax.text(
                x_c,
                y,
                "           ",
                ha='center',
                va='center',
                fontsize=15,
                bbox=dict(
                    boxstyle="round,pad=0.22,rounding_size=0.22",
                    facecolor="#FFF2E8",
                    edgecolor="#C6A437",
                    linewidth=0.8,
                ),
                zorder=1
            )

            # Word (except last frame full reveal handled separately)
            if r < n_rows - 1:
                ax.text(
                    x_c,
                    y,
                    correct_word,
                    fontsize=15,
                    fontweight='medium',
                    ha='center',
                    va='center',
                    color="#111111",
                    zorder=2
                )

    # Final clean bottom row
    if frame == n_rows - 1:
        y_bot = 0.98 - (n_rows - 1) * (0.98 / (n_rows - 1))
        for idx, x_c in enumerate(x_positions):
            ax.text(
                x_c,
                y_bot,
                clean_tokens[idx],
                fontsize=15,
                fontweight='medium',
                ha='center',
                va='center',
                color="#000000",
                zorder=5
            )


anim = FuncAnimation(
    fig,
    draw_until,
    frames=n_rows,
    interval=700,  # milliseconds per frame
    repeat=False
)

# Save GIF
anim.save("ar.gif", writer=PillowWriter(fps=1.5))


############## Mask Diffusion ##############
clean_tokens = ["I", "live", "in", "New", "Diego"]
unmask = [1, 2, 1, 3, 3]

n_rows = 4
x_positions = np.linspace(0.04, 0.96, len(clean_tokens))
fig, ax = plt.subplots(figsize=(5, 3))

def draw_frame(frame):
    ax.clear()

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Draw rows progressively
    for r in range(frame + 1):
        y = 0.98 - r * (0.98 / (n_rows - 1))

        for idx, x_c in enumerate(x_positions):
            correct_word = clean_tokens[idx]

            # ---- Box ----
            ax.text(
                x_c,
                y,
                "           ",
                ha='center',
                va='center',
                fontsize=15,
                bbox=dict(
                    boxstyle="round,pad=0.22,rounding_size=0.22",
                    facecolor="#FFF2E8",
                    edgecolor="#C6A437",
                    linewidth=0.8,
                ),
                zorder=1
            )

            # ---- Mask / token logic ----
            if r < n_rows - 1:
                word = correct_word if r >= unmask[idx] else "[Mask]"

                ax.text(
                    x_c,
                    y,
                    word,
                    fontsize=15,
                    fontweight='medium',
                    ha='center',
                    va='center',
                    color="#111111",
                    zorder=2
                )

    # Final clean bottom row
    if frame == n_rows - 1:
        y_bot = 0.98 - (n_rows - 1) * (0.98 / (n_rows - 1))
        for idx, x_c in enumerate(x_positions):
            ax.text(
                x_c,
                y_bot,
                clean_tokens[idx],
                fontsize=15,
                fontweight='medium',
                ha='center',
                va='center',
                color="#000000",
                zorder=5
            )


anim = FuncAnimation(
    fig,
    draw_frame,
    frames=n_rows,
    interval=800,
    repeat=False
)

anim.save("mask_diffusion.gif", writer=PillowWriter(fps=1.2))



############## FLM ##############
np.random.seed(1)
clean_tokens = ["I", "live", "in", "New", "York"]

n_rows = 4
layers_top = 30
layers_bottom = 0
x_positions = np.linspace(0.04, 0.96, len(clean_tokens))

fig, ax = plt.subplots(figsize=(5, 3))


def layers_for_row(r):
    return int(np.round(np.interp(r, [0, n_rows-1], [layers_top, layers_bottom])))


def draw_frame(frame):
    ax.clear()

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    for r in range(frame + 1):
        y = 0.98 - r * (0.98 / (n_rows - 1))
        p_clean = (r / (n_rows - 1)) ** 3
        jitter_scale = 0.01 * (1 - p_clean)
        num_layers = layers_for_row(r)

        for idx, x_c in enumerate(x_positions):
            correct_word = clean_tokens[idx]

            # --------- FIXED RNG PER (row, token) ----------
            rng = np.random.default_rng(seed=1000 + r * 100 + idx)

            # ---- Box ----
            ax.text(
                x_c,
                y,
                "           ",
                ha='center',
                va='center',
                fontsize=15,
                bbox=dict(
                    boxstyle="round,pad=0.22,rounding_size=0.22",
                    facecolor="#FFF2E8",
                    edgecolor="#C6A437",
                    linewidth=0.8,
                ),
                zorder=1
            )

            if r < n_rows - 1:
                for layer in range(num_layers):
                    word = correct_word if rng.random() < p_clean else rng.choice(vocab)
                    dx = rng.normal(scale=jitter_scale)
                    alpha = 0.06 * (1 - p_clean) + 0.8 * p_clean

                    ax.text(
                        x_c + dx,
                        y,
                        word,
                        fontsize=15,
                        fontweight='medium',
                        ha='center',
                        va='center',
                        color="#111111",
                        alpha=alpha,
                        zorder=2
                    )

    # Final clean bottom row
    if frame == n_rows - 1:
        y_bot = 0.98 - (n_rows - 1) * (0.98 / (n_rows - 1))
        for idx, x_c in enumerate(x_positions):
            ax.text(
                x_c,
                y_bot,
                clean_tokens[idx],
                fontsize=15,
                fontweight='medium',
                ha='center',
                va='center',
                color="#000000",
                zorder=5
            )


anim = FuncAnimation(
    fig,
    draw_frame,
    frames=n_rows,
    interval=800,
    repeat=False
)

anim.save("flm.gif", writer=PillowWriter(fps=1.2))

