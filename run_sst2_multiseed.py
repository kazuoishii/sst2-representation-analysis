import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MODEL_NAME = "bert-base-uncased"
TRAIN_FILE = "data/SST-2/train.tsv"
DEV_FILE = "data/SST-2/dev.tsv"

# First proof-of-concept run.
# Increase after confirming the pipeline works.
N_TRAIN = 5000
N_DEV = 872
BATCH_SIZE = 32
MAX_LENGTH = 128
import os

SEED = int(os.environ.get("SEED", "42"))

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("device:", device)

train = pd.read_csv(TRAIN_FILE, sep="\t")
dev = pd.read_csv(DEV_FILE, sep="\t")

train = train.sample(
    n=min(N_TRAIN, len(train)),
    random_state=SEED
).reset_index(drop=True)

dev = dev.iloc[:min(N_DEV, len(dev))].reset_index(drop=True)

print("train:", train.shape)
print("dev:", dev.shape)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
)
model.to(device)
model.eval()


def extract_hidden_states(df):
    all_layers = None

    sentences = df["sentence"].tolist()

    with torch.no_grad():
        for start in range(0, len(sentences), BATCH_SIZE):
            batch_sentences = sentences[start:start + BATCH_SIZE]

            encoded = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                k: v.to(device)
                for k, v in encoded.items()
            }

            outputs = model(**encoded)

            # 13 states for BERT-base:
            # embedding + 12 transformer layers
            hidden_states = outputs.hidden_states

            # Use CLS representation
            batch_layers = [
                h[:, 0, :].detach().cpu().numpy()
                for h in hidden_states
            ]

            if all_layers is None:
                all_layers = [[] for _ in batch_layers]

            for i, x in enumerate(batch_layers):
                all_layers[i].append(x)

            print(
                f"\rExtracted {min(start+BATCH_SIZE, len(sentences))}"
                f"/{len(sentences)}",
                end=""
            )

    print()

    return [
        np.concatenate(layer, axis=0).astype(np.float64)
        for layer in all_layers
    ]

def linear_cka(X, Y):
    """
    Linear CKA between two representation matrices.
    Rows = samples, columns = representation dimensions.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    # Center features across samples
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    cross = X.T @ Y

    hsic = np.sum(cross ** 2)
    norm_x = np.sqrt(np.sum((X.T @ X) ** 2))
    norm_y = np.sqrt(np.sum((Y.T @ Y) ** 2))

    return hsic / (norm_x * norm_y + 1e-12)

def effective_rank(X):
    """
    Effective rank based on the entropy of singular values.
    Rows = samples, columns = representation dimensions.
    """
    X = X.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)

    s = np.linalg.svd(X, compute_uv=False)

    # Normalize singular values to a probability distribution
    p = s / (s.sum() + 1e-12)
    p = p[p > 0]

    entropy = -np.sum(p * np.log(p))

    return np.exp(entropy)



print("Extracting train representations...")
X_train_layers = extract_hidden_states(train)

print("Extracting dev representations...")
X_dev_layers = extract_hidden_states(dev)


# Diagnostic check: dev representations
print("\n=== DEV REPRESENTATION CHECK ===")
for i, X in enumerate(X_dev_layers):
    print(
        i,
        "dtype=", X.dtype,
        "finite=", np.isfinite(X).all(),
        "min=", np.nanmin(X),
        "max=", np.nanmax(X),
        "absmax=", np.nanmax(np.abs(X))
    )

# Diagnostic check: train representations
print("\n=== TRAIN REPRESENTATION CHECK ===")
for i, X in enumerate(X_train_layers):
    print(
        "train", i,
        "finite=", np.isfinite(X).all(),
        "absmax=", np.nanmax(np.abs(X))
    )


y_train = train["label"].to_numpy()
y_dev = dev["label"].to_numpy()

print("\n=== EFFECTIVE RANK ===")

effective_ranks = []

for layer, X in enumerate(X_dev_layers):
    r_eff = effective_rank(X)
    effective_ranks.append(r_eff)

    print(
        f"Layer {layer:2d}: "
        f"effective_rank = {r_eff:.4f}"
    )

results = []

for layer in range(len(X_train_layers)):

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            random_state=SEED
        )
    )

    clf.fit(X_train_layers[layer], y_train)

    pred = clf.predict(X_dev_layers[layer])
    acc = accuracy_score(y_dev, pred)

    results.append({
        "layer": layer,
        "Q_accuracy": acc,
        "effective_rank": effective_ranks[layer]
    })

    print(f"Layer {layer:2d}: Q = {acc:.4f}")

# Representation-geometric deformation using linear CKA
for layer in range(len(X_dev_layers) - 1):

    cka = linear_cka(
        X_dev_layers[layer],
        X_dev_layers[layer + 1]
    )

    results[layer]["CKA"] = cka
    results[layer]["Delta_CKA"] = 1.0 - cka

# Geometric change:
# normalized Frobenius distance between adjacent representation matrices.
for layer in range(len(X_dev_layers) - 1):

    A = X_dev_layers[layer]
    B = X_dev_layers[layer + 1]

    delta_geom = (
        np.linalg.norm(B - A, ord="fro")
        /
        (np.linalg.norm(A, ord="fro") + 1e-12)
    )

    results[layer]["Delta"] = delta_geom

# Information degradation:
# positive decrease in linear-probe accuracy.
for layer in range(len(results) - 1):

    q0 = results[layer]["Q_accuracy"]
    q1 = results[layer + 1]["Q_accuracy"]

    results[layer]["delta"] = max(q0 - q1, 0.0)

results[-1]["Delta"] = np.nan
results[-1]["delta"] = np.nan
results[-1]["Delta_CKA"] = np.nan
results[-1]["delta"] = np.nan

results_df = pd.DataFrame(results)

print("\n=== RESULTS ===")
print(results_df)

results_df["seed"] = SEED

results_df.to_csv(
    f"sst2_representation_results_seed{SEED}.csv",
    index=False
)

print(f"\nSaved: sst2_representation_results_seed{SEED}.csv")
