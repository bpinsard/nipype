# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.spm.preprocess import CreateWarped
def test_CreateWarped_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    paths=dict(),
    mfile=dict(usedefault=True,
    ),
    image_files=dict(copyfile=False,
    mandatory=True,
    field='crt_warped.images',
    ),
    use_v8struct=dict(min_ver='8',
    usedefault=True,
    ),
    flowfield_files=dict(copyfile=False,
    mandatory=True,
    field='crt_warped.flowfields',
    ),
    interp=dict(field='crt_warped.interp',
    ),
    use_mcr=dict(),
    matlab_cmd=dict(),
    iterations=dict(field='crt_warped.K',
    ),
    )
    inputs = CreateWarped.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_CreateWarped_outputs():
    output_map = dict(warped_files=dict(),
    )
    outputs = CreateWarped.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
