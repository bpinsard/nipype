# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.afni.preprocess import ZCutUp
def test_ZCutUp_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    out_file=dict(name_source='in_file',
    name_template='%s_zcupup',
    argstr='-prefix %s',
    ),
    args=dict(argstr='%s',
    ),
    outputtype=dict(),
    keep=dict(argstr='-keep %s',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(position=-1,
    mandatory=True,
    argstr='%s',
    ),
    )
    inputs = ZCutUp.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_ZCutUp_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = ZCutUp.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
