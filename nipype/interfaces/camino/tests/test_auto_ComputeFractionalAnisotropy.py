# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..dti import ComputeFractionalAnisotropy


def test_ComputeFractionalAnisotropy_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='< %s',
    mandatory=True,
    position=1,
    ),
    inputdatatype=dict(argstr='-inputdatatype %s',
    ),
    inputmodel=dict(argstr='-inputmodel %s',
    ),
    out_file=dict(argstr='> %s',
    genfile=True,
    position=-1,
    ),
    outputdatatype=dict(argstr='-outputdatatype %s',
    ),
    scheme_file=dict(argstr='%s',
    position=2,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = ComputeFractionalAnisotropy.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_ComputeFractionalAnisotropy_outputs():
    output_map = dict(fa=dict(),
    )
    outputs = ComputeFractionalAnisotropy.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
