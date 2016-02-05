# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ...testing import assert_equal
from ..io import JSONFileSink


def test_JSONFileSink_inputs():
    input_map = dict(_outputs=dict(usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_dict=dict(usedefault=True,
    ),
    out_file=dict(),
    )
    inputs = JSONFileSink.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_JSONFileSink_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = JSONFileSink.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
