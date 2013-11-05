# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.freesurfer.utils import ApplyMask
def test_ApplyMask_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    xfm_file=dict(argstr='-xform %s',
    ),
    use_abs=dict(argstr='-abs',
    ),
    out_file=dict(position=-1,
    genfile=True,
    argstr='%s',
    ),
    args=dict(argstr='%s',
    ),
    xfm_target=dict(argstr='-lta_dst %s',
    ),
    invert_xfm=dict(argstr='-invert',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(position=-3,
    mandatory=True,
    argstr='%s',
    ),
    mask_thresh=dict(argstr='-T %.4f',
    ),
    subjects_dir=dict(),
    mask_file=dict(position=-2,
    mandatory=True,
    argstr='%s',
    ),
    xfm_source=dict(argstr='-lta_src %s',
    ),
    )
    inputs = ApplyMask.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_ApplyMask_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = ApplyMask.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
