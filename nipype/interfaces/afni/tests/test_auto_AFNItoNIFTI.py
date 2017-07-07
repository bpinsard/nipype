# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..utils import AFNItoNIFTI


def test_AFNItoNIFTI_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    denote=dict(argstr='-denote',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    copyfile=False,
    mandatory=True,
    position=-1,
    ),
    newid=dict(argstr='-newid',
    xor=['oldid'],
    ),
    oldid=dict(argstr='-oldid',
    xor=['newid'],
    ),
    out_file=dict(argstr='-prefix %s',
    hash_files=False,
    name_source='in_file',
    name_template='%s.nii',
    ),
    outputtype=dict(),
    pure=dict(argstr='-pure',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = AFNItoNIFTI.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_AFNItoNIFTI_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = AFNItoNIFTI.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
