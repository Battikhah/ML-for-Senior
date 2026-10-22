"""Build trusted on-disk completion fixtures for direct trainer/export unit tests."""
import json
import pytest


@pytest.fixture
def complete_training_run():
    """Supply real artifact checksums when a unit test intentionally bypasses MLflow."""
    def complete(directory):
        """Commit completion metadata around the actual trained checkpoint files."""
        from treevibes.data import write_json
        from treevibes.tracking import file_hash
        result=json.loads((directory/'training_result.json').read_text())
        result['artifact_hashes']={p.name:file_hash(p) for p in directory.iterdir() if p.is_file()
                                  and p.name not in {'status.json','result.json'}}
        write_json(directory/'result.json',result)
        write_json(directory/'status.json',{'state':'completed'})
    return complete
