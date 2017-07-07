# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..utils import RelabelHypointensities


def test_RelabelHypointensities_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    aseg=dict(argstr='%s',
    mandatory=True,
    position=-3,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    lh_white=dict(copyfile=True,
    mandatory=True,
    ),
    out_file=dict(argstr='%s',
    hash_files=False,
    keep_extension=False,
    name_source=['aseg'],
    name_template='%s.hypos.mgz',
    position=-1,
    ),
    rh_white=dict(copyfile=True,
    mandatory=True,
    ),
    subjects_dir=dict(),
    surf_directory=dict(argstr='%s',
    position=-2,
    usedefault=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = RelabelHypointensities.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_RelabelHypointensities_outputs():
    output_map = dict(out_file=dict(argstr='%s',
    ),
    )
    outputs = RelabelHypointensities.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
