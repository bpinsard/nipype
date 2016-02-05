# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..vista import VtoMat


def test_VtoMat_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-in %s',
    mandatory=True,
    position=1,
    ),
    out_file=dict(argstr='-out %s',
    hash_files=False,
    keep_extension=False,
    name_source=['in_file'],
    name_template='%s.mat',
    position=-1,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = VtoMat.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_VtoMat_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = VtoMat.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
