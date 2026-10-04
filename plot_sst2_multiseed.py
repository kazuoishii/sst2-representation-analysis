import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sst2_multiseed_summary.csv")

# Exclude layer 0 from the main figure because the initial CLS
# representation is not yet contextualized.
df = df[df["layer"] >= 1].copy()

fig, axes = plt.subplots(
    3, 1,
    figsize=(6.5, 7.5),
    sharex=True
)

# ------------------------------------------------------------
# (a) Task-relevant information: mean +/- SD over five seeds
# ------------------------------------------------------------

axes[0].errorbar(
    df["layer"],
    df["Q_mean"],
    yerr=df["Q_sd"],
    marker="o",
    capsize=3
)

axes[0].set_ylabel("Linear-probe accuracy")
axes[0].set_title("(a) Task-relevant information (mean ± SD, 5 seeds)")
axes[0].grid(alpha=0.25)

best_idx = df["Q_mean"].idxmax()
best_layer = int(df.loc[best_idx, "layer"])
best_q = df.loc[best_idx, "Q_mean"]

axes[0].annotate(
    f"Peak: layer {best_layer}",
    xy=(best_layer, best_q),
    xytext=(best_layer - 3, best_q - 0.025),
    arrowprops=dict(arrowstyle="->")
)

# ------------------------------------------------------------
# (b) Effective rank
# ------------------------------------------------------------

axes[1].plot(
    df["layer"],
    df["effective_rank"],
    marker="o"
)

axes[1].set_ylabel("Effective rank")
axes[1].set_title("(b) Representational dimensionality")
axes[1].grid(alpha=0.25)

# ------------------------------------------------------------
# (c) Inter-layer geometric deformation
# Delta_CKA at layer l represents l -> l+1
# ------------------------------------------------------------

geom = df.dropna(subset=["Delta_CKA"])

axes[2].plot(
    geom["layer"],
    geom["Delta_CKA"],
    marker="o"
)

axes[2].set_ylabel(
    r"$1-\mathrm{CKA}(H_l,H_{l+1})$"
)
axes[2].set_xlabel("BERT layer")
axes[2].set_title("(c) Inter-layer geometric deformation")
axes[2].grid(alpha=0.25)

axes[2].set_xticks(range(1, 13))

plt.tight_layout()

plt.savefig(
    "figure1_representation_dynamics_multiseed.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "figure1_representation_dynamics_multiseed.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved:")
print("  figure1_representation_dynamics_multiseed.pdf")
print("  figure1_representation_dynamics_multiseed.png")
