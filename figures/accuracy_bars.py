import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from style import apply_style, C_BASE, C_DIRECT, C_SQGRPO, TEXT_WIDTH

apply_style()

labels = ["Base", "Direct+GRPO", "SQ+GRPO"]
colors = [C_BASE, C_DIRECT, C_SQGRPO]

aokvqa_vals = [46.8, 51.6, 52.2]
aokvqa_se   = [2.23, 2.23, 2.23]

clevr_vals = [97.4, 98.4, 98.6]
clevr_se   = [0.71, 0.56, 0.53]

x = np.arange(len(labels))
width = 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4))

bars1 = ax1.bar(x, aokvqa_vals, width, yerr=aokvqa_se, color=colors,
                edgecolor="white", linewidth=0.5,
                capsize=3, error_kw=dict(lw=1.0, capthick=1.0, color="#444"))
for bar, val, se in zip(bars1, aokvqa_vals, aokvqa_se):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.8,
             f"{val:.1f}", ha="center", va="bottom", fontsize=7, weight="bold")
ax1.axhline(y=46.8, color=C_BASE, linestyle="--", linewidth=0.6, alpha=0.5)
ax1.set_ylabel("Accuracy (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=6.5)
ax1.set_ylim(0, 62)
ax1.set_title("(a) A-OKVQA")

bars2 = ax2.bar(x, clevr_vals, width, yerr=clevr_se, color=colors,
                edgecolor="white", linewidth=0.5,
                capsize=3, error_kw=dict(lw=1.0, capthick=1.0, color="#444"))
for bar, val, se in zip(bars2, clevr_vals, clevr_se):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.3,
             f"{val:.1f}", ha="center", va="bottom", fontsize=7, weight="bold")
ax2.axhline(y=97.4, color=C_BASE, linestyle="--", linewidth=0.6, alpha=0.5)
ax2.set_ylabel("Accuracy (%)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=6.5)
ax2.set_ylim(94, 101)
ax2.set_title("(b) CLEVR")

plt.tight_layout(w_pad=1.5)
out = os.path.join(os.path.dirname(__file__), "output", "accuracy_bars.pdf")
fig.savefig(out)
print(f"Saved -> {out}")
plt.close()
