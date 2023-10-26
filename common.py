import json
from pathlib import Path


def load_train_index(index_directory: str) -> list[dict[str, str]]:
    """
    Load train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other index
    """
    index_directory = Path(index_directory)
    assert index_directory.exists()

    # Load observations from train datasets
    datasets = []
    for path in index_directory.iterdir():
        if not path.name.endswith('_index.json'):
            continue  # Skip non-index files
        # Take only train and validation index
        if 'train' in path.name or 'dev' in path.name:
            with open(path, 'r') as f:
                datasets.append(json.load(f))

    assert len(datasets) == 5
    return datasets
