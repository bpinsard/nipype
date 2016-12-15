# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..preprocess import Fim


def test_Fim_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    fim_thr=dict(argstr='-fim_thr %f',
    position=3,
    ),
    ideal_file=dict(argstr='-ideal_file %s',
    mandatory=True,
    position=2,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-input %s',
    copyfile=False,
    mandatory=True,
    position=1,
    ),
    out=dict(argstr='-out %s',
    position=4,
    ),
    out_file=dict(argstr='-bucket %s',
    name_source='in_file',
    name_template='%s_fim',
    ),
    outputtype=dict(),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = Fim.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_Fim_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Fim.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
