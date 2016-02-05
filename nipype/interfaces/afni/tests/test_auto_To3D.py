# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..preprocess import To3D


def test_To3D_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    assumemosaic=dict(argstr='-assume_dicom_mosaic',
    ),
    datatype=dict(argstr='-datum %s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    filetype=dict(argstr='-%s',
    ),
    funcparams=dict(argstr='-time:zt %s alt+z2',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_folder=dict(argstr='%s/*.dcm',
    mandatory=True,
    position=-1,
    ),
    out_file=dict(argstr='-prefix %s',
    name_source=['in_folder'],
    name_template='%s',
    ),
    outputtype=dict(),
    skipoutliers=dict(argstr='-skip_outliers',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = To3D.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_To3D_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = To3D.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
