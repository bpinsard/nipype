# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.algorithms.misc import AddCSVColumn
def test_AddCSVColumn_inputs():
    input_map = dict(extra_column_heading=dict(),
    extra_field=dict(),
    out_file=dict(usedefault=True,
    ),
    in_file=dict(mandatory=True,
    ),
    )
    inputs = AddCSVColumn.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_AddCSVColumn_outputs():
    output_map = dict(csv_file=dict(),
    )
    outputs = AddCSVColumn.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
