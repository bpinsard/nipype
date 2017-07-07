# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..utils import Edge3


def test_Edge3_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    datum=dict(argstr='-datum %s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    fscale=dict(argstr='-fscale',
    xor=['gscale', 'nscale', 'scale_floats'],
    ),
    gscale=dict(argstr='-gscale',
    xor=['fscale', 'nscale', 'scale_floats'],
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-input %s',
    copyfile=False,
    mandatory=True,
    position=0,
    ),
    nscale=dict(argstr='-nscale',
    xor=['fscale', 'gscale', 'scale_floats'],
    ),
    out_file=dict(argstr='-prefix %s',
    position=-1,
    ),
    outputtype=dict(),
    scale_floats=dict(argstr='-scale_floats %f',
    xor=['fscale', 'gscale', 'nscale'],
    ),
    terminal_output=dict(nohash=True,
    ),
    verbose=dict(argstr='-verbose',
    ),
    )
    inputs = Edge3.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_Edge3_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Edge3.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
