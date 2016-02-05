# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from .....testing import assert_equal
from ..brains import landmarksConstellationAligner


def test_landmarksConstellationAligner_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputLandmarksPaired=dict(argstr='--inputLandmarksPaired %s',
    ),
    outputLandmarksPaired=dict(argstr='--outputLandmarksPaired %s',
    hash_files=False,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = landmarksConstellationAligner.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_landmarksConstellationAligner_outputs():
    output_map = dict(outputLandmarksPaired=dict(),
    )
    outputs = landmarksConstellationAligner.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
