# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.slicer.filtering.arithmetic import SubtractScalarVolumes
def test_SubtractScalarVolumes_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    outputVolume=dict(position=-1,
    hash_files=False,
    argstr='%s',
    ),
    args=dict(argstr='%s',
    ),
    inputVolume2=dict(position=-2,
    argstr='%s',
    ),
    inputVolume1=dict(position=-3,
    argstr='%s',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    order=dict(argstr='--order %s',
    ),
    )
    inputs = SubtractScalarVolumes.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_SubtractScalarVolumes_outputs():
    output_map = dict(outputVolume=dict(position=-1,
    ),
    )
    outputs = SubtractScalarVolumes.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
