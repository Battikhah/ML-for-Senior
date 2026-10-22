"""Generate a portable C feature encoder and host parity fixture from a measured PCA profile."""
from pathlib import Path
import json
import subprocess
import numpy as np


def c_array(name, values):
    """Preserve float32 calibration constants in a compiler-readable array."""
    values=np.asarray(values,dtype='float32').ravel()
    return f'static const float {name}[{len(values)}] = {{'+','.join(float(v).hex()+'f' for v in values)+'};\n'


def build(profile, contract_path, raw_features, destination):
    """Check a real C encoder against held-out Python codes without claiming ESP32 timing."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    contract=json.loads(Path(contract_path).read_text())
    with np.load(profile,allow_pickle=False) as p:
        mean=p['mean'];components=p['components'];lo=p['lo'];hi=p['hi']
    rows,columns=components.shape
    coefficients=contract['features']['coefficients']
    stats=contract['statistics'] if contract['normalization']!='none' else dict(mean=[0]*coefficients,scale=[1]*coefficients)
    scale=np.maximum((hi-lo)/255,1e-8).astype('float32')
    source='#include <stdint.h>\n#include <math.h>\n#include "encoder.h"\n'
    for name,value in [('means',stats['mean']),('scales',stats['scale']),('center',mean),
                       ('weights',components),('lower',lo),('steps',scale)]:
        source+=c_array(name,value)
    source+=f'''/* Project normalized MFCCs into the frozen training-only PCA basis. */
void tv_encode(const float *raw, uint8_t *output) {{
    for (int k=0;k<{rows};++k) {{
        float value=0.f;
        for (int j=0;j<{columns};++j) {{
            float normalized=(raw[j]-means[j%{coefficients}])/scales[j%{coefficients}];
            value += weights[k*{columns}+j]*(normalized-center[j]);
        }}
        float code=nearbyintf((value-lower[k])/steps[k]);
        output[k]=(uint8_t)fmaxf(0.f,fminf(255.f,code));
    }}
}}
'''
    (destination/'encoder.c').write_text(source)
    (destination/'encoder.h').write_text('#pragma once\n#include <stdint.h>\n'
        f'#define TV_INPUT_VALUES {columns}\n#define TV_PACKET_VALUES {rows}\n'
        '/* Requires the exact exported MFCC contract; output bytes are unsigned quantized PCA scores. */\n'
        'void tv_encode(const float *raw, uint8_t *output);\n')
    raw=np.asarray(raw_features[:8],dtype='float32').reshape(-1,columns)
    normalized=(raw-np.tile(stats['mean'],columns//coefficients))/np.tile(stats['scale'],columns//coefficients)
    projected=(normalized.astype('float32')-mean)@components.T
    expected=np.clip(np.rint((projected-lo)/scale),0,255).astype('uint8')
    fixture='#include <stdio.h>\n#include "encoder.h"\n'+c_array('fixtures',raw)
    fixture+=f'''/* Emit eight independent held-out encoded packets for the Python parity check. */
int main(void) {{
    for(int i=0;i<{len(raw)};++i) {{
        uint8_t output[TV_PACKET_VALUES];
        tv_encode(fixtures+i*TV_INPUT_VALUES,output);
        fwrite(output,1,TV_PACKET_VALUES,stdout);
    }}
    return 0;
}}
'''
    (destination/'fixture.c').write_text(fixture)
    subprocess.run(['cc','-std=c99','-O2',str(destination/'encoder.c'),str(destination/'fixture.c'),
                    '-lm','-o',str(destination/'host_fixture')],check=True,capture_output=True)
    result=subprocess.run([str(destination/'host_fixture')],check=True,capture_output=True)
    actual=np.frombuffer(result.stdout,dtype='uint8').reshape(expected.shape)
    delta=int(np.abs(actual.astype('int16')-expected.astype('int16')).max())
    if delta>1:
        raise ValueError(f'C encoder differs from Python by {delta} quantization levels')
    np.savez_compressed(destination/'fixture.npz',raw=raw,expected=expected)
    meta=dict(scope='host_C_encoder_parity_only',fixtures=len(raw),encoded_bytes_per_measurement=rows,
              maximum_code_delta=delta,profile_constant_bytes=int((len(mean)+components.size+len(lo)*2+coefficients*2)*4),
              multiplies_per_measurement=int(components.size),board_verified=False)
    (destination/'verification.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta
