# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.slicer.diffusion.diffusion import DiffusionWeightedVolumeMasking
def test_DiffusionWeightedVolumeMasking_inputs():
    input_map = dict(outputBaseline=dict(position=-2,
    hash_files=False,
    argstr='%s',
    ),
    thresholdMask=dict(position=-1,
    hash_files=False,
    argstr='%s',
    ),
    args=dict(argstr='%s',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    removeislands=dict(argstr='--removeislands ',
    ),
    otsuomegathreshold=dict(argstr='--otsuomegathreshold %f',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    inputVolume=dict(position=-4,
    argstr='%s',
    ),
    )
    inputs = DiffusionWeightedVolumeMasking.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_DiffusionWeightedVolumeMasking_outputs():
    output_map = dict(outputBaseline=dict(position=-2,
    ),
    thresholdMask=dict(position=-1,
    ),
    )
    outputs = DiffusionWeightedVolumeMasking.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
