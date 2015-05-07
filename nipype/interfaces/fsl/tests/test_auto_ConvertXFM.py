# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.fsl.utils import ConvertXFM

def test_ConvertXFM_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    concat_xfm=dict(argstr='-concat',
    position=-3,
    requires=['in_file2'],
    xor=['invert_xfm', 'concat_xfm', 'fix_scale_skew'],
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    fix_scale_skew=dict(argstr='-fixscaleskew',
    position=-3,
    requires=['in_file2'],
    xor=['invert_xfm', 'concat_xfm', 'fix_scale_skew'],
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    mandatory=True,
    position=-1,
    ),
    in_file2=dict(argstr='%s',
    position=-2,
    ),
    invert_xfm=dict(argstr='-inverse',
    position=-3,
    xor=['invert_xfm', 'concat_xfm', 'fix_scale_skew'],
    ),
    out_file=dict(argstr='-omat %s',
    genfile=True,
    hash_files=False,
    position=1,
    ),
    output_type=dict(),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = ConvertXFM.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_ConvertXFM_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = ConvertXFM.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

