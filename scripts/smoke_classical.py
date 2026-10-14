"""Exercise classical training/reporting on synthetic fixtures, not benchmark data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scipy import sparse

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='misconception-smoke-') as directory:
    path = Path(directory)
    features = path / 'features'
    features.mkdir()
    rng = np.random.default_rng(42)
    for split, count in [('train', 64), ('val', 32)]:
        labels = np.tile(np.arange(4), count // 4)
        values = np.column_stack([np.eye(4)[labels], rng.random((count, 12)) * 0.01])
        sparse.save_npz(features / f'X_{split}.npz', sparse.csr_matrix(values))
        np.save(features / f'y_{split}.npy', labels)
    names = ['True_Correct', 'False_Misconception', 'False_Neither', 'True_Neither']
    (features / 'feature_metadata.json').write_text(json.dumps({'labels': names}))
    environment = dict(os.environ, MPLCONFIGDIR=str(path / 'matplotlib'))
    for folder, script in [('logistic_regression', 'train_logreg.py'), ('svm', 'train_svm.py')]:
        output = path / folder
        subprocess.run([sys.executable, str(root / 'experiments' / folder / script),
                        '--feature-dir', str(features), '--output-dir', str(output), '--no-save-model'],
                       env=environment, check=True, capture_output=True, text=True)
        for artifact in ['metrics.json', 'classification_report.txt', 'confusion_matrix.csv', 'val_predictions.csv']:
            if not (output / artifact).is_file():
                raise RuntimeError(f'{folder}: missing {artifact}')
        metrics = json.loads((output / 'metrics.json').read_text())
        if not 0 <= metrics['macro_f1'] <= 1:
            raise RuntimeError(f'{folder}: invalid metric')
        print(f'{folder}: synthetic training and reporting passed')
print('These checks do not measure performance on the original dataset.')
