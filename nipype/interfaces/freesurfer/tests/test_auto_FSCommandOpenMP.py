# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..base import FSCommandOpenMP


def test_FSCommandOpenMP_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    num_threads=dict(),
    subjects_dir=dict(),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = FSCommandOpenMP.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

