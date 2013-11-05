# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.mrtrix.tensors import FSL2MRTrix
def test_FSL2MRTrix_inputs():
    input_map = dict(invert_y=dict(usedefault=True,
    ),
    invert_x=dict(usedefault=True,
    ),
    invert_z=dict(usedefault=True,
    ),
    bvec_file=dict(mandatory=True,
    ),
    bval_file=dict(mandatory=True,
    ),
    out_encoding_file=dict(genfile=True,
    ),
    )
    inputs = FSL2MRTrix.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_FSL2MRTrix_outputs():
    output_map = dict(encoding_file=dict(),
    )
    outputs = FSL2MRTrix.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
