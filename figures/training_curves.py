import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from style import apply_style, C_SQGRPO, C_DIRECT, TEXT_WIDTH

apply_style()

aokvqa_sq_steps = np.array([
    399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999,
    4399, 4799, 5199, 5599, 5999, 6399, 6799, 6999,
])
aokvqa_sq_reward = np.array([
    0.3438, 0.3309, 0.3469, 0.3559, 0.3409, 0.3650, 0.4147, 0.3816,
    0.4197, 0.3478, 0.3872, 0.4034, 0.3234, 0.3953, 0.4181, 0.3928,
    0.4316, 0.4644,
])

aokvqa_direct_steps = np.array([
    399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999,
    4399, 4799, 5199, 5599, 5999, 6399, 6499,
])
aokvqa_direct_reward = np.array([
    0.3513, 0.3538, 0.3519, 0.3625, 0.3734, 0.4194, 0.4291, 0.4387,
    0.4416, 0.3741, 0.4119, 0.4428, 0.3609, 0.4453, 0.4397, 0.4516,
    0.4300,
])

clevr_sq_steps = np.array([
    399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999,
    4399, 4799, 5199, 5599, 5999, 6399, 6799, 7199, 7499,
])
clevr_sq_reward = np.array([
    0.6381, 0.8409, 0.9113, 0.9325, 0.9400, 0.9581, 0.9619, 0.9337,
    0.9475, 0.9463, 0.9706, 0.9700, 0.9806, 0.9725, 0.9663, 0.9475,
    0.9400, 0.9731, 0.9725,
])

clevr_direct_steps = np.array([
    399, 799, 1199, 1599, 1999, 2399, 2799, 3199, 3599, 3999,
    4399, 4799, 5199, 5599, 5999, 6399, 6799, 6999,
])
clevr_direct_reward = np.array([
    0.9691, 0.9650, 0.9788, 0.9806, 0.9750, 0.9831, 0.9838, 0.9681,
    0.9894, 0.9844, 0.9781, 0.9900, 0.9906, 0.9875, 0.9862, 0.9769,
    0.9806, 0.9988,
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.2))

ax1.plot(aokvqa_sq_steps, aokvqa_sq_reward, "-", color=C_SQGRPO,
         label="SQ+GRPO", linewidth=1.5)
ax1.plot(aokvqa_direct_steps, aokvqa_direct_reward, "-", color=C_DIRECT,
         label="Direct+GRPO", linewidth=1.5)
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Mean Reward")
ax1.set_title("(a) A-OKVQA")
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower right")

ax2.plot(clevr_sq_steps, clevr_sq_reward, "-", color=C_SQGRPO,
         label="SQ+GRPO", linewidth=1.5)
ax2.plot(clevr_direct_steps, clevr_direct_reward, "-", color=C_DIRECT,
         label="Direct+GRPO", linewidth=1.5)
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Mean Reward")
ax2.set_title("(b) CLEVR")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="lower right")

plt.tight_layout(w_pad=1.5)
out = os.path.join(os.path.dirname(__file__), "output", "training_curves.pdf")
fig.savefig(out)
print(f"Saved → {out}")
plt.close()
