import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# 1. Step Definitions
# ----------------------------------------------------------------------
step = np.linspace(2, 200, 100)
step_height = np.linspace(16800, 41160000, 2450) / 20.0 / 10000.0

# ----------------------------------------------------------------------
# 2. Load Subplot 1 Baselines (DRL-VO, LNDNL, Tani-Nav, RGL)
# ----------------------------------------------------------------------
drl_vo_runs = np.array(
    [
        np.loadtxt("drl_vo.txt"),
        np.loadtxt("./data_multi_seeds/drl_vo_seed_2.txt"),
        np.loadtxt("./data_multi_seeds/drl_vo_seed_3.txt"),
        np.loadtxt("./data_multi_seeds/drl_vo_seed_4.txt"),
        np.loadtxt("./data_multi_seeds/drl_vo_seed_5.txt"),
    ]
)

lndnl_runs = np.array(
    [
        np.loadtxt("lndnl.txt"),
        np.loadtxt("./data_multi_seeds/lndnl_seed_2.txt"),
        np.loadtxt("./data_multi_seeds/lndnl_seed_3.txt"),
        np.loadtxt("./data_multi_seeds/lndnl_seed_4.txt"),
        np.loadtxt("./data_multi_seeds/lndnl_seed_5.txt"),
    ]
)

trance_nav_runs = np.array(
    [
        np.loadtxt("tani_nav.txt"),
        np.loadtxt("./data_multi_seeds/tani_nav_seed_2.txt"),
        np.loadtxt("./data_multi_seeds/tani_nav_seed_3.txt"),
        np.loadtxt("./data_multi_seeds/tani_nav_seed_4.txt"),
        np.loadtxt("./data_multi_seeds/tani_nav_seed_5.txt"),
    ]
)

rgl_raw = [
    np.loadtxt("rgl.txt"),
    np.loadtxt("./data_multi_seeds/rgl_seed_2.txt"),
    np.loadtxt("./data_multi_seeds/rgl_seed_3.txt"),
    np.loadtxt("./data_multi_seeds/rgl_seed_4.txt"),
    np.loadtxt("./data_multi_seeds/rgl_seed_5.txt"),
]

step_scale = 1000.0
rgl_runs = np.array(
    [np.interp(step, f[:, 0] / step_scale, f[:, 1]) for f in rgl_raw]
)

# ----------------------------------------------------------------------
# 3. Load Subplot 2 Baseline (HEIGHT - 2450 values per file)
# ----------------------------------------------------------------------
height_runs = np.array(
    [
        np.loadtxt("height.txt")[:, 0],
        np.loadtxt("./data_multi_seeds/height_seed_2.txt")[:, 0],
        np.loadtxt("./data_multi_seeds/height_seed_3.txt")[:, 0],
        np.loadtxt("./data_multi_seeds/height_seed_4.txt")[:, 0],
        np.loadtxt("./data_multi_seeds/height_seed_5.txt")[:, 0],
    ]
)

# ----------------------------------------------------------------------
# 4. Multi-Subplot Visualization
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=120)
font_type = 'Times New Roman'
plt.rcParams['font.family'] = font_type
# Set math font to match Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'
font_size = 20

# Helper function to plot mean and min/max shading on a target axis
def plot_baseline_with_range(ax, x_steps, data, label, color=None):
  mean = np.mean(data, axis=0)
  min_val = np.min(data, axis=0)
  max_val = np.max(data, axis=0)

  (line,) = ax.plot(x_steps, mean, label=label, linewidth=2, color=color)
  ax.fill_between(
      x_steps, min_val, max_val, color=line.get_color(), alpha=0.2, linewidth=0
  )

# --- Subfigure 1: Four Main Baselines ---
baselines_sub1 = {
    "DRL-VO": drl_vo_runs,
    "LNDNL": lndnl_runs,
    "RGL": rgl_runs,
    "TRAIN-Nav (Ours)": trance_nav_runs,
}

for label, data in baselines_sub1.items():
  plot_baseline_with_range(ax1, step, data, label)

ax1.set_xlabel(r'steps ($1 \times 10^4$)', fontproperties=font_type, fontsize=font_size)
ax1.set_ylabel("success rate", fontproperties=font_type, fontsize=font_size)
ax1.legend(loc="lower center", fontsize=font_size - 4, ncol=4)
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontproperties(font_type)
    label.set_fontsize(font_size)

ax1.grid(True, linestyle="--", alpha=0.6)

# --- Subfigure 2: HEIGHT Baseline ---
plot_baseline_with_range(
    ax2, step_height, height_runs, label="HEIGHT", color="tab:purple"
)

ax2.set_xlabel(r'steps ($2 \times 10^5$)', fontproperties=font_type, fontsize=font_size)
ax2.set_ylabel("success rate", fontproperties=font_type, fontsize=font_size)
ax2.legend(fontsize=font_size - 4)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontproperties(font_type)
    label.set_fontsize(font_size)
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()