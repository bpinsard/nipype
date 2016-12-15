# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..surface import GrayscaleModelMaker


def test_GrayscaleModelMaker_inputs():
    input_map = dict(InputVolume=dict(argstr='%s',
    position=-2,
    ),
    OutputGeometry=dict(argstr='%s',
    hash_files=False,
    position=-1,
    ),
    args=dict(argstr='%s',
    ),
    decimate=dict(argstr='--decimate %f',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    name=dict(argstr='--name %s',
    ),
    pointnormals=dict(argstr='--pointnormals ',
    ),
    smooth=dict(argstr='--smooth %d',
    ),
    splitnormals=dict(argstr='--splitnormals ',
    ),
    terminal_output=dict(nohash=True,
    ),
    threshold=dict(argstr='--threshold %f',
    ),
    )
    inputs = GrayscaleModelMaker.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_GrayscaleModelMaker_outputs():
    output_map = dict(OutputGeometry=dict(position=-1,
    ),
    )
    outputs = GrayscaleModelMaker.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
