# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.fsl.utils import ConvertXFM
def test_ConvertXFM_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    out_file=dict(hash_files=False,
    genfile=True,
    position=1,
    argstr='-omat %s',
    ),
    args=dict(argstr='%s',
    ),
    in_file2=dict(position=-2,
    argstr='%s',
    ),
    fix_scale_skew=dict(xor=['invert_xfm', 'concat_xfm', 'fix_scale_skew'],
    position=-3,
    requires=['in_file2'],
    argstr='-fixscaleskew',
    ),
    invert_xfm=dict(position=-3,
    xor=['invert_xfm', 'concat_xfm', 'fix_scale_skew'],
    argstr='-inverse',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    concat_xfm=dict(xor=['invert_xfm', 'concat_xfm', 'fix_scale_skew'],
    position=-3,
    requires=['in_file2'],
    argstr='-concat',
    ),
    in_file=dict(position=-1,
    mandatory=True,
    argstr='%s',
    ),
    output_type=dict(),
    environ=dict(nohash=True,
    usedefault=True,
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
