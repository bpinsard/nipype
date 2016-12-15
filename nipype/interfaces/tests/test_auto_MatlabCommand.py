# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..matlab import MatlabCommand


def test_MatlabCommand_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    logfile=dict(argstr='-logfile %s',
    ),
    mfile=dict(usedefault=True,
    ),
    nodesktop=dict(argstr='-nodesktop',
    nohash=True,
    usedefault=True,
    ),
    nosplash=dict(argstr='-nosplash',
    nohash=True,
    usedefault=True,
    ),
    paths=dict(),
    postscript=dict(usedefault=True,
    ),
    prescript=dict(usedefault=True,
    ),
    script=dict(argstr='-r "%s;exit"',
    mandatory=True,
    position=-1,
    ),
    script_file=dict(usedefault=True,
    ),
    single_comp_thread=dict(argstr='-singleCompThread',
    nohash=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    uses_mcr=dict(nohash=True,
    xor=[u'nodesktop', u'nosplash', u'single_comp_thread'],
    ),
    )
    inputs = MatlabCommand.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value

