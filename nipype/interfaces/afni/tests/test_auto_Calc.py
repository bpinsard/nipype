# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.afni.preprocess import Calc
def test_Calc_inputs():
    input_map = dict(stop_idx=dict(requires=['start_idx'],
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    out_file=dict(name_source='in_file_a',
    name_template='%s_calc',
    argstr='-prefix %s',
    ),
    expr=dict(position=2,
    mandatory=True,
    argstr='-expr "%s"',
    ),
    args=dict(argstr='%s',
    ),
    outputtype=dict(),
    in_file_b=dict(position=1,
    argstr=' -b %s',
    ),
    other=dict(argstr='',
    ),
    in_file_a=dict(position=0,
    mandatory=True,
    argstr='-a %s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    single_idx=dict(),
    start_idx=dict(requires=['stop_idx'],
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    )
    inputs = Calc.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_Calc_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Calc.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
