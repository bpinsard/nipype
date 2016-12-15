# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..utils import RemoveIntersection


def test_RemoveIntersection_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    copyfile=True,
    mandatory=True,
    position=-2,
    ),
    out_file=dict(argstr='%s',
    hash_files=False,
    keep_extension=True,
    name_source=[u'in_file'],
    name_template='%s',
    position=-1,
    ),
    subjects_dir=dict(),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = RemoveIntersection.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_RemoveIntersection_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = RemoveIntersection.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
