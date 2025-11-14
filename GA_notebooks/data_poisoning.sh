#!/bin/bash

#Run from the root folder of repo

echo "===========Data Poisoning test========"

git checkout v1-version
dvc pull -r mygcs
dvc checkout
echo "Clean data loaded from DVC...."

DATA="data.csv"
total_samples=$(($(wc -l < data.csv) - 1))

echo "Total samples in data: $total_samples"

POISON_LEVELS=("0.05" "0.10" "0.50")

for p in "${POISON_LEVELS[@]}"; do
    echo "---- Applying poisoning: $p ----"

    poisoned_file="data_poison_${p}.csv"

    # 3. Python poisoning block
    python3 <<EOF
import pandas as pd
import numpy as np

df = pd.read_csv("$DATA")

p = float("$p")
n = int(len(df) * p)

# pick n random rows and flip their labels
poison_idx = np.random.choice(df.index, size=n, replace=False)

# simple poisoning: random wrong label
labels = df["species"].unique()

def corrupt_label(old):
    new = np.random.choice([l for l in labels if l != old])
    return new

df.loc[poison_idx, "species"] = df.loc[poison_idx, "species"].apply(corrupt_label)

df.to_csv("$poisoned_file", index=False)
print(f"Poisoned file written: $poisoned_file")
EOF

    # 4. Call train.py
    echo "Training on poisoned data ($p)..."
    python3 train.py "$poisoned_file"
done

echo "All poisoning experiments completed."
