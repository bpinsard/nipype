# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from .....testing import assert_equal
from ..gtract import gtractTransformToDisplacementField


def test_gtractTransformToDisplacementField_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputReferenceVolume=dict(argstr='--inputReferenceVolume %s',
    ),
    inputTransform=dict(argstr='--inputTransform %s',
    ),
    numberOfThreads=dict(argstr='--numberOfThreads %d',
    ),
    outputDeformationFieldVolume=dict(argstr='--outputDeformationFieldVolume %s',
    hash_files=False,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = gtractTransformToDisplacementField.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_gtractTransformToDisplacementField_outputs():
    output_map = dict(outputDeformationFieldVolume=dict(),
    )
    outputs = gtractTransformToDisplacementField.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
