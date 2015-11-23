# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..maths import Threshold


def test_Threshold_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    direction=dict(usedefault=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    mandatory=True,
    position=2,
    ),
    internal_datatype=dict(argstr='-dt %s',
    position=1,
    ),
    nan2zeros=dict(argstr='-nan',
    position=3,
    ),
    out_file=dict(argstr='%s',
    genfile=True,
    hash_files=False,
    position=-2,
    ),
    output_datatype=dict(argstr='-odt %s',
    position=-1,
    ),
    output_type=dict(),
    terminal_output=dict(nohash=True,
    ),
    thresh=dict(argstr='%s',
    mandatory=True,
    position=4,
    ),
    use_nonzero_voxels=dict(requires=['use_robust_range'],
    ),
    use_robust_range=dict(),
    )
    inputs = Threshold.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Threshold_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Threshold.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
