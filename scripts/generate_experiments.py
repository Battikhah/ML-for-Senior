"""Regenerate the versioned catalog from deterministic recipes."""
from pathlib import Path
from treevibes.catalog import generate_catalog

if __name__ == '__main__':
    paths = generate_catalog(Path('experiments'))
    print(f'Generated {len(paths)} configurations')
