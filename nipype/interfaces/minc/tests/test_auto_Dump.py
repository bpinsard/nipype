# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..minc import Dump


def test_Dump_inputs():
    input_map = dict(annotations_brief=dict(argstr='-b %s',
    xor=(u'annotations_brief', u'annotations_full'),
    ),
    annotations_full=dict(argstr='-f %s',
    xor=(u'annotations_brief', u'annotations_full'),
    ),
    args=dict(argstr='%s',
    ),
    coordinate_data=dict(argstr='-c',
    xor=(u'coordinate_data', u'header_data'),
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    header_data=dict(argstr='-h',
    xor=(u'coordinate_data', u'header_data'),
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    input_file=dict(argstr='%s',
    mandatory=True,
    position=-2,
    ),
    line_length=dict(argstr='-l %d',
    usedefault=False,
    ),
    netcdf_name=dict(argstr='-n %s',
    ),
    out_file=dict(argstr='> %s',
    genfile=True,
    position=-1,
    ),
    output_file=dict(hash_files=False,
    keep_extension=False,
    name_source=[u'input_file'],
    name_template='%s_dump.txt',
    position=-1,
    ),
    precision=dict(argstr='%s',
    ),
    terminal_output=dict(nohash=True,
    ),
    variables=dict(argstr='-v %s',
    sep=',',
    ),
    )
    inputs = Dump.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Dump_outputs():
    output_map = dict(output_file=dict(),
    )
    outputs = Dump.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
