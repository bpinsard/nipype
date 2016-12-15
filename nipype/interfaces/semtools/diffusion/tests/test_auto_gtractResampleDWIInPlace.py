# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..gtract import gtractResampleDWIInPlace


def test_gtractResampleDWIInPlace_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    debugLevel=dict(argstr='--debugLevel %d',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    imageOutputSize=dict(argstr='--imageOutputSize %s',
    sep=',',
    ),
    inputTransform=dict(argstr='--inputTransform %s',
    ),
    inputVolume=dict(argstr='--inputVolume %s',
    ),
    numberOfThreads=dict(argstr='--numberOfThreads %d',
    ),
    outputResampledB0=dict(argstr='--outputResampledB0 %s',
    hash_files=False,
    ),
    outputVolume=dict(argstr='--outputVolume %s',
    hash_files=False,
    ),
    referenceVolume=dict(argstr='--referenceVolume %s',
    ),
    terminal_output=dict(nohash=True,
    ),
    warpDWITransform=dict(argstr='--warpDWITransform %s',
    ),
    )
    inputs = gtractResampleDWIInPlace.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_gtractResampleDWIInPlace_outputs():
    output_map = dict(outputResampledB0=dict(),
    outputVolume=dict(),
    )
    outputs = gtractResampleDWIInPlace.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
