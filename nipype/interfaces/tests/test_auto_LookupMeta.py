# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.dcmstack import LookupMeta
def test_LookupMeta_inputs():
    input_map = dict(in_file=dict(mandatory=True,
    ),
    meta_keys=dict(mandatory=True,
    ),
    )
    inputs = LookupMeta.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_LookupMeta_outputs():
    output_map = dict()
    outputs = LookupMeta.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
