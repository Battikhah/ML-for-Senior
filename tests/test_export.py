"""Check real integer conversion and firmware fixture encoding."""
import json
import numpy as np
import pytest


def test_integer_encoding_clips_rounds_and_rejects_invalid_scale():
    """Firmware bytes must use the converter's scale and zero point."""
    from treevibes.export import encode_input, decode_output
    details=dict(dtype=np.int8, quantization=(.5,-3))
    np.testing.assert_array_equal(encode_input(np.array([-1000.,0.,1.,1000.]),details),[-128,-3,-1,127])
    np.testing.assert_allclose(decode_output(np.array([-3,-1]),details),[0.,1.])
    with pytest.raises(ValueError):
        encode_input(np.array([1]),dict(dtype=np.int8,quantization=(0.,0)))


@pytest.mark.parametrize('method',['float','ptq','qat'])
def test_real_export_predicts_and_keeps_board_status_pending(tmp_path,method,complete_training_run):
    """Successful host conversion must never certify ESP32 runtime memory."""
    import treevibes
    pytest.importorskip('tensorflow')
    from treevibes.config import DEFAULTS
    from treevibes.data import DEFAULT_FEATURES, write_json
    from treevibes.train import train_run
    from treevibes.synthetic import synthetic_dataset
    from treevibes.export import export_model, create_firmware_bundle
    data=synthetic_dataset(20)
    config=DEFAULTS | dict(id='export_test',family='mlp',width=4,dropout=0.,epochs=1,features=DEFAULT_FEATURES)
    parent=tmp_path/'parent';parent.mkdir()
    train_run(config,data,parent)
    complete_training_run(parent)
    meta=export_model(parent,data,method,tmp_path/method,qat_epochs=1)
    assert meta['state']=='exported' and not meta['board_verified']
    assert meta['validation']['count']==8
    assert meta['bytes']>0
    assert meta['calibration_partition']=='train' if method!='float' else meta['calibration_partition'] is None
    if method!='float':
        assert meta['input']['dtype']=='int8' and meta['output']['dtype']=='int8'
        bundle=create_firmware_bundle(tmp_path/method,tmp_path/'bundle')
        assert (bundle/'model_data.cc').exists()
        assert (bundle/'model_config.h').exists()
        assert json.loads((bundle/'bundle.json').read_text())['model_sha256']==meta['model_sha256']
