# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..mesh import WarpPoints


def test_WarpPoints_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    interp=dict(mandatory=True,
    usedefault=True,
    ),
    out_points=dict(keep_extension=True,
    name_source='points',
    name_template='%s_warped',
    output_name='out_points',
    ),
    points=dict(mandatory=True,
    ),
    warp=dict(mandatory=True,
    ),
    )
    inputs = WarpPoints.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_WarpPoints_outputs():
    output_map = dict(out_points=dict(),
    )
    outputs = WarpPoints.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
