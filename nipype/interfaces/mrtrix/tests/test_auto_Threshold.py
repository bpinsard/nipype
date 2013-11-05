# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.mrtrix.preprocess import Threshold
def test_Threshold_inputs():
    input_map = dict(out_filename=dict(position=-1,
    genfile=True,
    argstr='%s',
    ),
    absolute_threshold_value=dict(argstr='-abs %s',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    invert=dict(position=1,
    argstr='-invert',
    ),
    args=dict(argstr='%s',
    ),
    percentage_threshold_value=dict(argstr='-percent %s',
    ),
    quiet=dict(position=1,
    argstr='-quiet',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(position=-2,
    mandatory=True,
    argstr='%s',
    ),
    debug=dict(position=1,
    argstr='-debug',
    ),
    replace_zeros_with_NaN=dict(position=1,
    argstr='-nan',
    ),
    )
    inputs = Threshold.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_Threshold_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Threshold.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
