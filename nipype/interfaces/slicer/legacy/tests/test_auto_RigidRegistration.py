# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..registration import RigidRegistration


def test_RigidRegistration_inputs():
    input_map = dict(FixedImageFileName=dict(argstr='%s',
    position=-2,
    ),
    MovingImageFileName=dict(argstr='%s',
    position=-1,
    ),
    args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    fixedsmoothingfactor=dict(argstr='--fixedsmoothingfactor %d',
    ),
    histogrambins=dict(argstr='--histogrambins %d',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    initialtransform=dict(argstr='--initialtransform %s',
    ),
    iterations=dict(argstr='--iterations %s',
    sep=',',
    ),
    learningrate=dict(argstr='--learningrate %s',
    sep=',',
    ),
    movingsmoothingfactor=dict(argstr='--movingsmoothingfactor %d',
    ),
    outputtransform=dict(argstr='--outputtransform %s',
    hash_files=False,
    ),
    resampledmovingfilename=dict(argstr='--resampledmovingfilename %s',
    hash_files=False,
    ),
    spatialsamples=dict(argstr='--spatialsamples %d',
    ),
    terminal_output=dict(nohash=True,
    ),
    testingmode=dict(argstr='--testingmode ',
    ),
    translationscale=dict(argstr='--translationscale %f',
    ),
    )
    inputs = RigidRegistration.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_RigidRegistration_outputs():
    output_map = dict(outputtransform=dict(),
    resampledmovingfilename=dict(),
    )
    outputs = RigidRegistration.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
