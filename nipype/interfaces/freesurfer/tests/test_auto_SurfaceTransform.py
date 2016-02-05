# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..utils import SurfaceTransform


def test_SurfaceTransform_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    hemi=dict(argstr='--hemi %s',
    mandatory=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    out_file=dict(argstr='--tval %s',
    genfile=True,
    ),
    reshape=dict(argstr='--reshape',
    ),
    reshape_factor=dict(argstr='--reshape-factor',
    ),
    source_annot_file=dict(argstr='--sval-annot %s',
    mandatory=True,
    xor=['source_file'],
    ),
    source_file=dict(argstr='--sval %s',
    mandatory=True,
    xor=['source_annot_file'],
    ),
    source_subject=dict(argstr='--srcsubject %s',
    mandatory=True,
    ),
    source_type=dict(argstr='--sfmt %s',
    requires=['source_file'],
    ),
    subjects_dir=dict(),
    target_ico_order=dict(argstr='--trgicoorder %d',
    ),
    target_subject=dict(argstr='--trgsubject %s',
    mandatory=True,
    ),
    target_type=dict(argstr='--tfmt %s',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = SurfaceTransform.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_SurfaceTransform_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = SurfaceTransform.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
