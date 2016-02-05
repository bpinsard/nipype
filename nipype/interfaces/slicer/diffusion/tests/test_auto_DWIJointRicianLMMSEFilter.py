# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from .....testing import assert_equal
from ..diffusion import DWIJointRicianLMMSEFilter


def test_DWIJointRicianLMMSEFilter_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    compressOutput=dict(argstr='--compressOutput ',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputVolume=dict(argstr='%s',
    position=-2,
    ),
    ng=dict(argstr='--ng %d',
    ),
    outputVolume=dict(argstr='%s',
    hash_files=False,
    position=-1,
    ),
    re=dict(argstr='--re %s',
    sep=',',
    ),
    rf=dict(argstr='--rf %s',
    sep=',',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = DWIJointRicianLMMSEFilter.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_DWIJointRicianLMMSEFilter_outputs():
    output_map = dict(outputVolume=dict(position=-1,
    ),
    )
    outputs = DWIJointRicianLMMSEFilter.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
