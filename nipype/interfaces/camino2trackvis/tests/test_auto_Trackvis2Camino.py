# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.camino2trackvis.convert import Trackvis2Camino

def test_Trackvis2Camino_inputs():
    input_map = dict(append_file=dict(argstr='-a %s',
    position=2,
    ),
    args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-i %s',
    mandatory=True,
    position=1,
    ),
    out_file=dict(argstr='-o %s',
    genfile=True,
    position=2,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = Trackvis2Camino.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_Trackvis2Camino_outputs():
    output_map = dict(camino=dict(),
    )
    outputs = Trackvis2Camino.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

