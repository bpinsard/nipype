# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..dti import TractSkeleton


def test_TractSkeleton_inputs():
    input_map = dict(alt_data_file=dict(argstr='-a %s',
    ),
    alt_skeleton=dict(argstr='-s %s',
    ),
    args=dict(argstr='%s',
    ),
    data_file=dict(),
    distance_map=dict(),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-i %s',
    mandatory=True,
    ),
    output_type=dict(),
    project_data=dict(argstr='-p %.3f %s %s %s %s',
    requires=[u'threshold', u'distance_map', u'data_file'],
    ),
    projected_data=dict(),
    search_mask_file=dict(xor=[u'use_cingulum_mask'],
    ),
    skeleton_file=dict(argstr='-o %s',
    ),
    terminal_output=dict(nohash=True,
    ),
    threshold=dict(),
    use_cingulum_mask=dict(usedefault=True,
    xor=[u'search_mask_file'],
    ),
    )
    inputs = TractSkeleton.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_TractSkeleton_outputs():
    output_map = dict(projected_data=dict(),
    skeleton_file=dict(),
    )
    outputs = TractSkeleton.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
