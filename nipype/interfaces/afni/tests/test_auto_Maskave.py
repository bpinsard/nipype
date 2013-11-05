# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.afni.preprocess import Maskave
def test_Maskave_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    out_file=dict(name_source='in_file',
    name_template='%s_maskave.1D',
    position=-1,
    keep_extension=True,
    argstr='> %s',
    ),
    args=dict(argstr='%s',
    ),
    mask=dict(position=1,
    argstr='-mask %s',
    ),
    outputtype=dict(),
    quiet=dict(position=2,
    argstr='-quiet',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(position=-2,
    mandatory=True,
    argstr='%s',
    ),
    )
    inputs = Maskave.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_Maskave_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Maskave.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
