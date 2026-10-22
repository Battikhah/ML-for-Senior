"""Keep the final-test operating point immutable during the PCA audit."""
import pytest
try:
 from scripts.pca_audit import freeze
except ImportError:
 freeze=None


def test_finalist_lock_rejects_model_or_threshold_changes(tmp_path):
 """A second test inspection must not silently use a newly selected model or threshold."""
 assert freeze is not None,'Immutable finalist lock not implemented'
 model=tmp_path/'model';model.write_bytes(b'fixed weights')
 lock=tmp_path/'locked.json';first=freeze(lock,{'model':model},.2873)
 assert freeze(lock,{'model':model},.2873)==first
 with pytest.raises(ValueError):freeze(lock,{'model':model},.1)
 model.write_bytes(b'changed weights')
 with pytest.raises(ValueError):freeze(lock,{'model':model},.2873)
