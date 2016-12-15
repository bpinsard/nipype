# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..utils import Smooth


def test_Smooth_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    fwhm=dict(argstr='-kernel gauss %.03f -fmean',
    mandatory=True,
    position=1,
    xor=[u'sigma'],
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    mandatory=True,
    position=0,
    ),
    output_type=dict(),
    sigma=dict(argstr='-kernel gauss %.03f -fmean',
    mandatory=True,
    position=1,
    xor=[u'fwhm'],
    ),
    smoothed_file=dict(argstr='%s',
    hash_files=False,
    name_source=[u'in_file'],
    name_template='%s_smooth',
    position=2,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = Smooth.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_Smooth_outputs():
    output_map = dict(smoothed_file=dict(),
    )
    outputs = Smooth.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
