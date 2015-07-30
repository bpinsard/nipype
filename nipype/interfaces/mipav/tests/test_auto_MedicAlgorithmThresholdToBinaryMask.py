# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.mipav.developer import MedicAlgorithmThresholdToBinaryMask

def test_MedicAlgorithmThresholdToBinaryMask_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inLabel=dict(argstr='--inLabel %s',
    sep=';',
    ),
    inMaximum=dict(argstr='--inMaximum %f',
    ),
    inMinimum=dict(argstr='--inMinimum %f',
    ),
    inUse=dict(argstr='--inUse %s',
    ),
    null=dict(argstr='--null %s',
    ),
    outBinary=dict(argstr='--outBinary %s',
    sep=';',
    ),
    terminal_output=dict(nohash=True,
    ),
    xDefaultMem=dict(argstr='-xDefaultMem %d',
    ),
    xMaxProcess=dict(argstr='-xMaxProcess %d',
    usedefault=True,
    ),
    xPrefExt=dict(argstr='--xPrefExt %s',
    ),
    )
    inputs = MedicAlgorithmThresholdToBinaryMask.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_MedicAlgorithmThresholdToBinaryMask_outputs():
    output_map = dict()
    outputs = MedicAlgorithmThresholdToBinaryMask.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

