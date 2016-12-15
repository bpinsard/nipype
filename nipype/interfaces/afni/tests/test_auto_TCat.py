# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..utils import TCat


def test_TCat_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_files=dict(argstr=' %s',
    copyfile=False,
    mandatory=True,
    position=-1,
    ),
    out_file=dict(argstr='-prefix %s',
    name_source='in_files',
    name_template='%s_tcat',
    ),
    outputtype=dict(),
    rlt=dict(argstr='-rlt%s',
    position=1,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = TCat.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_TCat_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = TCat.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
