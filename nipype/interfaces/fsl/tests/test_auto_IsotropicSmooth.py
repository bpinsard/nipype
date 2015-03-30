# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.fsl.maths import IsotropicSmooth

def test_IsotropicSmooth_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    fwhm=dict(argstr='-s %.5f',
    mandatory=True,
    position=4,
    xor=['sigma'],
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    mandatory=True,
    position=2,
    ),
    internal_datatype=dict(argstr='-dt %s',
    position=1,
    ),
    nan2zeros=dict(argstr='-nan',
    position=3,
    ),
    out_file=dict(argstr='%s',
    genfile=True,
    hash_files=False,
    position=-2,
    ),
    output_datatype=dict(argstr='-odt %s',
    position=-1,
    ),
    output_type=dict(),
    sigma=dict(argstr='-s %.5f',
    mandatory=True,
    position=4,
    xor=['fwhm'],
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = IsotropicSmooth.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_IsotropicSmooth_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = IsotropicSmooth.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

