# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..preprocess import Normalize


def test_Normalize_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    gradient=dict(argstr='-g %d',
    usedefault=False,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    mandatory=True,
    position=-2,
    ),
    mask=dict(argstr='-mask %s',
    ),
    out_file=dict(argstr='%s',
    hash_files=False,
    keep_extension=True,
    name_source=['in_file'],
    name_template='%s_norm',
    position=-1,
    ),
    segmentation=dict(argstr='-aseg %s',
    ),
    subjects_dir=dict(),
    terminal_output=dict(nohash=True,
    ),
    transform=dict(),
    )
    inputs = Normalize.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_Normalize_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Normalize.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
