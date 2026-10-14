"""Validate generated sparse feature shapes and label alignment."""
import argparse
from pathlib import Path
import numpy as np
from scipy import sparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir', type=Path, default=Path('features'))
    args = parser.parse_args()
    shapes = {}
    for split in ['train', 'val', 'test']:
        matrix = sparse.load_npz(args.feature_dir / f'X_{split}.npz')
        if len(matrix.shape) != 2 or min(matrix.shape) < 1:
            raise ValueError(f'{split}: expected a non-empty feature matrix')
        if not np.isfinite(matrix.data).all():
            raise ValueError(f'{split}: features contain non-finite values')
        shapes[split] = matrix.shape
        if split != 'test':
            labels = np.load(args.feature_dir / f'y_{split}.npy', allow_pickle=False)
            if labels.ndim != 1 or labels.size != matrix.shape[0]:
                raise ValueError(f'{split}: labels do not match feature rows')
            if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [0, 1, 2, 3]).all():
                raise ValueError(f'{split}: invalid four-class label IDs')
    if len({shape[1] for shape in shapes.values()}) != 1:
        raise ValueError('Train, validation, and test feature dimensions differ')
    print('Feature integrity verified:', shapes)

if __name__ == '__main__':
    main()
