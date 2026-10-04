#!/usr/bin/env python3
"""Aggregate five SST-2 probe runs; SD uses ddof=1. No model training."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path, default=Path('sst2_multiseed_summary_rebuilt.csv'))
    p.add_argument('--reference', type=Path)
    a = p.parse_args()
    frames = []
    geometry = ['CKA', 'Delta_CKA', 'Delta', 'effective_rank']
    for seed in range(42, 47):
        path = a.input_dir / f'sst2_representation_results_seed{seed}.csv'
        d = pd.read_csv(path)
        required = ['layer', 'Q_accuracy', 'delta', 'seed'] + geometry
        if not set(required).issubset(d.columns):
            raise ValueError(f'Missing columns: {path}')
        if len(d) != 13 or set(d.layer) != set(range(13)) or not d.seed.eq(seed).all():
            raise ValueError(f'Invalid layers or seed: {path}')
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    grouped = data.groupby('layer', sort=True)
    # The frozen encoder and validation set are shared across all runs.
    for name in geometry:
        if (grouped[name].nunique(dropna=False) > 1).any():
            raise ValueError(f'{name} differs across seeds; inspect before aggregating.')
    result = grouped.agg(Q_mean=('Q_accuracy', 'mean'), Q_sd=('Q_accuracy', 'std'),
                         delta_mean=('delta', 'mean'), delta_sd=('delta', 'std'))
    result = result.join(grouped[geometry].first()).reset_index()
    if a.reference:
        ref = pd.read_csv(a.reference).sort_values('layer').reset_index(drop=True)
        if list(ref.columns) != list(result.columns) or ref.shape != result.shape:
            raise ValueError('Reference schema does not match.')
        np.testing.assert_allclose(result.to_numpy(), ref.to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True)
        print('PASS: rebuilt summary matches reference.')
    if a.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {a.output}')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False)
    print(f'Saved: {a.output} (5 seeds, 13 layers; sample SD)')


if __name__ == '__main__':
    main()
