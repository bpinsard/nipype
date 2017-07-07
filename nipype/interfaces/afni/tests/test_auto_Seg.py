# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..preprocess import Seg


def test_Seg_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    bias_classes=dict(argstr='-bias_classes %s',
    ),
    bias_fwhm=dict(argstr='-bias_fwhm %f',
    ),
    blur_meth=dict(argstr='-blur_meth %s',
    ),
    bmrf=dict(argstr='-bmrf %f',
    ),
    classes=dict(argstr='-classes %s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-anat %s',
    copyfile=True,
    mandatory=True,
    position=-1,
    ),
    main_N=dict(argstr='-main_N %d',
    ),
    mask=dict(argstr='-mask %s',
    mandatory=True,
    position=-2,
    ),
    mixfloor=dict(argstr='-mixfloor %f',
    ),
    mixfrac=dict(argstr='-mixfrac %s',
    ),
    prefix=dict(argstr='-prefix %s',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = Seg.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_Seg_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Seg.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
