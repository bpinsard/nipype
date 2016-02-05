# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..epi import ApplyTOPUP


def test_ApplyTOPUP_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    datatype=dict(argstr='-d=%s',
    ),
    encoding_file=dict(argstr='--datain=%s',
    mandatory=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_files=dict(argstr='--imain=%s',
    mandatory=True,
    sep=',',
    ),
    in_index=dict(argstr='--inindex=%s',
    mandatory=True,
    sep=',',
    ),
    in_topup_fieldcoef=dict(argstr='--topup=%s',
    copyfile=False,
    requires=['in_topup_movpar'],
    ),
    in_topup_movpar=dict(copyfile=False,
    requires=['in_topup_fieldcoef'],
    ),
    interp=dict(argstr='--interp=%s',
    ),
    method=dict(argstr='--method=%s',
    ),
    out_corrected=dict(argstr='--out=%s',
    name_source=['in_files'],
    name_template='%s_corrected',
    ),
    output_type=dict(),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = ApplyTOPUP.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_ApplyTOPUP_outputs():
    output_map = dict(out_corrected=dict(),
    )
    outputs = ApplyTOPUP.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
