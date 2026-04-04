#!/usr/bin/env python3
"""
Two portrait animations (uniform vs reparameterized time) on the simplex.
8 particles with deliberately chosen critical times so that in tau-space
errors resolve at evenly-spaced intervals. P_e computed for |V|=100.
No side error plot; Pe shown as text below the progress bar.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d
import os

# ── Simplex vertices (equilateral triangle, scaled) ──────────────
SCALE = 5.0
V1 = np.array([0.0, 0.0]) * SCALE
V2 = np.array([1.0, 0.0]) * SCALE
V3 = np.array([0.5, np.sqrt(3) / 2]) * SCALE
VERTS = np.array([V1, V2, V3])
VERT_LABELS = ['flows', 'rock', 'totally']

# ── Colors ────────────────────────────────────────────────────────
BG = '#4d4d61'
CORRECT_C = '#4a9eff'    # blue
WRONG_C = '#ff6b6b'      # red
EDGE_C = '#ffffff'

# ── Compute P_e(t) analytically for |V|=100 ─────────────────────
V_SIZE = 100
t_grid = np.linspace(0, 1, 2000)
Pe_curve = np.zeros_like(t_grid)

z_quad = np.linspace(-6, 6, 1000)
dz = z_quad[1] - z_quad[0]
phi_z = norm.pdf(z_quad)

for i, t in enumerate(t_grid):
    if t >= 1.0:
        Pe_curve[i] = 0.0
    elif t <= 0.0:
        Pe_curve[i] = 1.0 - 1.0 / V_SIZE
    else:
        shift = t / (1.0 - t)
        log_Phi = np.log(np.clip(norm.cdf(z_quad + shift), 1e-300, 1.0))
        integrand = np.exp((V_SIZE - 1) * log_Phi) * phi_z
        p_correct = np.sum(integrand) * dz
        Pe_curve[i] = 1.0 - np.clip(p_correct, 0, 1)

Pe_curve = uniform_filter1d(Pe_curve, size=5)
Pe_curve = np.clip(Pe_curve, 0, 1)
Pe_curve[0] = 1.0 - 1.0 / V_SIZE
Pe_curve[-1] = 0.0

# ── tau(t) and inverse ───────────────────────────────────────────
tau_curve = 1.0 - (V_SIZE / (V_SIZE - 1.0)) * Pe_curve
tau_curve = np.clip(tau_curve, 0, 1)
tau_curve[0] = 0.0
tau_curve[-1] = 1.0

tau_mono = np.maximum.accumulate(tau_curve)
mask = np.concatenate([[True], np.diff(tau_mono) > 1e-12])
tau_unique = tau_mono[mask]
t_unique = t_grid[mask]
if tau_unique[0] > 0:
    tau_unique = np.concatenate([[0], tau_unique])
    t_unique = np.concatenate([[0], t_unique])
if tau_unique[-1] < 1:
    tau_unique = np.concatenate([tau_unique, [1]])
    t_unique = np.concatenate([t_unique, [1]])

t_of_tau = interp1d(tau_unique, t_unique, kind='linear',
                     bounds_error=False, fill_value=(0, 1))
Pe_of_t = interp1d(t_grid, Pe_curve, kind='linear',
                    bounds_error=False, fill_value=(Pe_curve[0], 0))
tau_of_t = interp1d(t_grid, tau_mono, kind='linear',
                     bounds_error=False, fill_value=(0, 1))

# ── Decision boundaries (Voronoi of 3 vertices) ─────────────────
centroid = VERTS.mean(axis=0)
boundary_lines = []
for i in range(3):
    j = (i + 1) % 3
    mid = (VERTS[i] + VERTS[j]) / 2.0
    direction = mid - centroid
    direction = direction / np.linalg.norm(direction)
    p1 = centroid - direction * 8.0
    p2 = centroid + direction * 8.0
    boundary_lines.append((p1, p2))


def nearest_vertex(pos):
    """Return index of nearest vertex for each point (N, 2)."""
    dists = np.linalg.norm(pos[:, None, :] - VERTS[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def nearest_vertex_single(pos):
    """Return index of nearest vertex for a single point (2,)."""
    dists = np.linalg.norm(VERTS - pos, axis=1)
    return np.argmin(dists)


def dist_to_edges(pt):
    """Min distance from pt to any simplex edge."""
    min_d = np.inf
    for i in range(3):
        j = (i + 1) % 3
        a, b = VERTS[i], VERTS[j]
        ab = b - a
        ap = pt - a
        t_proj = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        closest = a + t_proj * ab
        d = np.linalg.norm(pt - closest)
        min_d = min(min_d, d)
    return min_d


def compute_t_crit(x0, target_idx):
    """Find critical time when nearest vertex becomes the target."""
    target_v = VERTS[target_idx]
    # Check if already correct at t=0
    if nearest_vertex_single(x0) == target_idx:
        return 0.0
    # Binary search for the crossing time
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pos = (1 - mid) * x0 + mid * target_v
        if nearest_vertex_single(pos) == target_idx:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


# ── Select 8 particles with evenly-spaced tau_crit ──────────────
N_PTS = 8
# Desired tau_crit: evenly spaced from 1/9 to 8/9
desired_tau_crits = np.array([(k + 1) / (N_PTS + 1) for k in range(N_PTS)])

# Convert to desired t_crit via inverse
desired_t_crits = np.array([float(t_of_tau(tc)) for tc in desired_tau_crits])

print(f"Desired tau_crits: {desired_tau_crits}")
print(f"Desired t_crits:   {desired_t_crits}")

# Generate many candidates with varied spread to cover all tau_crit values
np.random.seed(777)
sigmas = [0.4, 0.55, 0.7, 0.85, 1.0]
all_cand_2d = []
all_cand_targets = []
for sig in sigmas:
    n = 30000
    c3d = np.random.randn(n, 3) * sig
    cs_ = c3d.sum(axis=1, keepdims=True)
    cp = c3d - (cs_ - 1) / 3
    c2d = np.column_stack([
        cp[:, 1] + cp[:, 2] / 2,
        cp[:, 2] * np.sqrt(3) / 2
    ]) * SCALE
    all_cand_2d.append(c2d)
    all_cand_targets.append(np.random.choice(3, n))

cand_2d = np.concatenate(all_cand_2d, axis=0)
cand_targets = np.concatenate(all_cand_targets)
N_CANDIDATES = len(cand_targets)

# Filter: must start in wrong region, not too close to edges,
# and not too far from centroid (keeps trails tidy)
MIN_EDGE_DIST = 0.15 * SCALE
MAX_CENTROID_DIST = 2.2 * SCALE  # keep starts within reasonable range
valid = []
for c in range(N_CANDIDATES):
    x0 = cand_2d[c]
    tgt = cand_targets[c]
    # Must start wrong
    if nearest_vertex_single(x0) == tgt:
        continue
    # Not too close to any edge
    if dist_to_edges(x0) < MIN_EDGE_DIST:
        continue
    # Not too far from centroid
    if np.linalg.norm(x0 - centroid) > MAX_CENTROID_DIST:
        continue
    # Compute critical time
    tc = compute_t_crit(x0, tgt)
    if tc < 0.05 or tc > 0.999:
        continue
    tau_c = float(tau_of_t(tc))
    valid.append((c, tc, tau_c))

print(f"Valid candidates: {len(valid)}")

# Greedily match to desired tau_crits, starting from hardest (highest tau)
selected = []
used = set()

# Sort desired targets by distance from median (hardest first = extremes)
order = np.argsort(-desired_tau_crits)  # match high tau first (rarest)

for idx_in_order in order:
    desired_tau = desired_tau_crits[idx_in_order]
    best_idx = -1
    best_dist = np.inf
    for vi, (cidx, tc, tauc) in enumerate(valid):
        if vi in used:
            continue
        d = abs(tauc - desired_tau)
        if d < best_dist:
            best_dist = d
            best_idx = vi
    if best_idx >= 0:
        used.add(best_idx)
        selected.append((idx_in_order, valid[best_idx]))

# Re-sort by original index
selected.sort(key=lambda x: x[0])
selected = [s[1] for s in selected]

print(f"Selected {len(selected)} particles:")
x0_2d = np.zeros((N_PTS, 2))
targets = np.zeros(N_PTS, dtype=int)
target_2d = np.zeros((N_PTS, 2))

for i, (cidx, tc, tauc) in enumerate(selected):
    cidx = int(cidx)
    x0_2d[i] = cand_2d[cidx]
    targets[i] = cand_targets[cidx]
    target_2d[i] = VERTS[targets[i]]
    print(f"  Particle {i}: target={targets[i]}, t_crit={tc:.4f}, "
          f"tau_crit={tauc:.4f} (desired {desired_tau_crits[i]:.4f})")


def smooth_step(x):
    return 3 * x**2 - 2 * x**3


# ── Animation parameters ─────────────────────────────────────────
FPS = 30
DURATION = 6.0
N_FRAMES = int(FPS * DURATION)
HOLD_START = int(0.3 * FPS)
HOLD_END = int(1.5 * FPS)
TRANS_FRAMES = N_FRAMES - HOLD_START - HOLD_END


def make_animation(mode='uniform'):
    fig, ax = plt.subplots(figsize=(8, 11), dpi=300, facecolor=BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.16)

    # Data limits
    pad = 0.8
    xmin = VERTS[:, 0].min() - pad * 2
    xmax = VERTS[:, 0].max() + pad * 2
    ymin = VERTS[:, 1].min() - pad * 1.5
    ymax = VERTS[:, 1].max() + pad * 1.5

    # Adjust for portrait aspect
    fig_w = fig.get_figwidth()
    fig_h = fig.get_figheight()
    left, right = fig.subplotpars.left, fig.subplotpars.right
    bot, top = fig.subplotpars.bottom, fig.subplotpars.top
    ax_w = (right - left) * fig_w
    ax_h = (top - bot) * fig_h
    fig_aspect = ax_w / ax_h
    data_xrange = xmax - xmin
    data_yrange = ymax - ymin
    current_aspect = data_xrange / data_yrange
    if current_aspect > fig_aspect:
        new_yrange = data_xrange / fig_aspect
        yc = (ymin + ymax) / 2
        ymin, ymax = yc - new_yrange / 2, yc + new_yrange / 2
    else:
        new_xrange = data_yrange * fig_aspect
        xc = (xmin + xmax) / 2
        xmin, xmax = xc - new_xrange / 2, xc + new_xrange / 2

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw simplex
    triangle = plt.Polygon(VERTS, fill=False, edgecolor=EDGE_C,
                            linewidth=1.5, alpha=0.4, zorder=2)
    ax.add_patch(triangle)
    ax.scatter(VERTS[:, 0], VERTS[:, 1], s=25, c='#ffffff',
               alpha=0.5, edgecolors='none', zorder=3)

    # Decision boundaries
    for p1, p2 in boundary_lines:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                ':', color='#ffffff', linewidth=0.8, alpha=0.25, zorder=1)

    # Vertex labels
    off = 0.35
    ax.text(V1[0] - 0.15, V1[1] - off, VERT_LABELS[0], fontsize=26,
            color='#ffffff', ha='center', va='top', fontfamily='monospace')
    ax.text(V2[0] + 0.15, V2[1] - off, VERT_LABELS[1], fontsize=26,
            color='#ffffff', ha='center', va='top', fontfamily='monospace')
    ax.text(V3[0], V3[1] + off, VERT_LABELS[2], fontsize=26,
            color='#ffffff', ha='center', va='bottom', fontfamily='monospace')

    # Trails
    trail_artists = []
    for i in range(N_PTS):
        trail, = ax.plot([], [], '-', linewidth=1.2, alpha=0.35, zorder=4,
                         solid_capstyle='round')
        trail_artists.append(trail)

    # Dots + dashed lines to nearest vertex
    dot_artists = []
    line_artists = []
    for i in range(N_PTS):
        dot, = ax.plot([], [], 'o', markersize=8, zorder=10)
        line, = ax.plot([], [], linestyle=(0, (4, 4)), linewidth=2.5,
                        alpha=0.7, zorder=5)
        dot_artists.append(dot)
        line_artists.append(line)

    # Trail history
    trail_history_x = [[] for _ in range(N_PTS)]
    trail_history_y = [[] for _ in range(N_PTS)]

    # ── Progress bar ─────────────────────────────────────────────
    BX0, BX1, BY = 0.08, 0.92, 0.065
    bar_bg = plt.Rectangle((BX0, BY), BX1 - BX0, 0.003,
                            transform=ax.transAxes, fc='#3a3a4d', ec='none',
                            clip_on=False, zorder=20)
    bar_fg = plt.Rectangle((BX0, BY), 0, 0.003,
                            transform=ax.transAxes, fc='#a8b4ff', ec='none',
                            clip_on=False, zorder=21)
    ax.add_patch(bar_bg)
    ax.add_patch(bar_fg)

    if mode == 'uniform':
        ax.text(BX0, BY - 0.015, '$t = 0$', fontsize=28,
                color='#ffffff', va='top', ha='left', transform=ax.transAxes)
        ax.text(BX1, BY - 0.015, '$t = 1$', fontsize=28,
                color='#ffffff', va='top', ha='right', transform=ax.transAxes)
    else:
        ax.text(BX0, BY - 0.015, r'$\tau = 0$', fontsize=28,
                color='#ffffff', va='top', ha='left', transform=ax.transAxes)
        ax.text(BX1, BY - 0.015, r'$\tau = 1$', fontsize=28,
                color='#ffffff', va='top', ha='right', transform=ax.transAxes)

    # P_e text below progress bar (centered)
    pe_label = ax.text(0.5, BY - 0.06, '', fontsize=24,
                       color='#a8b4ff', va='top', ha='center',
                       transform=ax.transAxes, fontfamily='sans-serif')

    # Error counter (both modes, below Pe label with spacing)
    error_label = ax.text(0.5, BY - 0.12, '', fontsize=20,
                          color='#ffffff', va='top', ha='center',
                          transform=ax.transAxes, fontfamily='sans-serif',
                          alpha=0.7)

    def update(frame):
        if frame < HOLD_START:
            progress = 0.0
        elif frame >= N_FRAMES - HOLD_END:
            progress = 1.0
        else:
            lin = (frame - HOLD_START) / TRANS_FRAMES
            progress = smooth_step(lin)

        if mode == 'uniform':
            t = progress
        else:
            tau = progress
            t = float(t_of_tau(tau))

        pos = (1 - t) * x0_2d + t * target_2d
        nn = nearest_vertex(pos)

        n_wrong = 0
        for i in range(N_PTS):
            correct = (nn[i] == targets[i])
            if not correct:
                n_wrong += 1
            color = CORRECT_C if correct else WRONG_C

            dot_artists[i].set_data([pos[i, 0]], [pos[i, 1]])
            dot_artists[i].set_color(color)

            nv = VERTS[nn[i]]
            line_artists[i].set_data([pos[i, 0], nv[0]], [pos[i, 1], nv[1]])
            line_artists[i].set_color(color)

            trail_history_x[i].append(pos[i, 0])
            trail_history_y[i].append(pos[i, 1])
            trail_artists[i].set_data(trail_history_x[i], trail_history_y[i])
            trail_artists[i].set_color(color)

        # Progress bar
        bar_fg.set_width((BX1 - BX0) * (frame / N_FRAMES))

        # P_e text
        current_pe = float(Pe_of_t(t))
        if mode == 'uniform':
            pe_label.set_text(f'$P_e(t) = {current_pe:.2f}$')
        else:
            pe_label.set_text(f'$P_e(\\tau) = {current_pe:.2f}$')

        # Error counter
        error_label.set_text(f'Errors: {n_wrong}/{N_PTS}')

        return (dot_artists + line_artists + trail_artists +
                [bar_fg, pe_label, error_label])

    anim = FuncAnimation(fig, update, frames=N_FRAMES,
                         interval=1000 / FPS, blit=False)

    suffix = 'uniform' if mode == 'uniform' else 'reparam'
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'time_reparam_{suffix}.mp4')
    writer = FFMpegWriter(fps=FPS, bitrate=15000, codec='h264',
                          extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(out, writer=writer)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    make_animation('uniform')
    make_animation('reparam')
