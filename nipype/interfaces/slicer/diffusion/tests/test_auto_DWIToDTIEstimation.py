# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.slicer.diffusion.diffusion import DWIToDTIEstimation

def test_DWIToDTIEstimation_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    enumeration=dict(argstr='--enumeration %s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputVolume=dict(argstr='%s',
    position=-3,
    ),
    mask=dict(argstr='--mask %s',
    ),
    outputBaseline=dict(argstr='%s',
    hash_files=False,
    position=-1,
    ),
    outputTensor=dict(argstr='%s',
    hash_files=False,
    position=-2,
    ),
    shiftNeg=dict(argstr='--shiftNeg ',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    )
    inputs = DWIToDTIEstimation.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_DWIToDTIEstimation_outputs():
    output_map = dict(outputBaseline=dict(position=-1,
    ),
    outputTensor=dict(position=-2,
    ),
    )
    outputs = DWIToDTIEstimation.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

