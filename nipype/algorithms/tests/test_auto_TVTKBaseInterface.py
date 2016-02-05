# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ...testing import assert_equal
from ..mesh import TVTKBaseInterface


def test_TVTKBaseInterface_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    )
    inputs = TVTKBaseInterface.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

