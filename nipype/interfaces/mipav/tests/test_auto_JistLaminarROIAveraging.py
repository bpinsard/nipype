# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.mipav.developer import JistLaminarROIAveraging

def test_JistLaminarROIAveraging_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inIntensity=dict(argstr='--inIntensity %s',
    ),
    inMask=dict(argstr='--inMask %s',
    ),
    inROI=dict(argstr='--inROI %s',
    ),
    inROI2=dict(argstr='--inROI2 %s',
    ),
    null=dict(argstr='--null %s',
    ),
    outROI3=dict(argstr='--outROI3 %s',
    hash_files=False,
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    xDefaultMem=dict(argstr='-xDefaultMem %d',
    ),
    xMaxProcess=dict(argstr='-xMaxProcess %d',
    usedefault=True,
    ),
    xPrefExt=dict(argstr='--xPrefExt %s',
    ),
    )
    inputs = JistLaminarROIAveraging.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_JistLaminarROIAveraging_outputs():
    output_map = dict(outROI3=dict(),
    )
    outputs = JistLaminarROIAveraging.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

