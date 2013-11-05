# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.fsl.model import ContrastMgr
def test_ContrastMgr_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    suffix=dict(argstr='-suffix %s',
    ),
    fcon_file=dict(argstr='-f %s',
    ),
    tcon_file=dict(position=-1,
    mandatory=True,
    argstr='%s',
    ),
    param_estimates=dict(copyfile=False,
    mandatory=True,
    argstr='',
    ),
    args=dict(argstr='%s',
    ),
    dof_file=dict(copyfile=False,
    mandatory=True,
    argstr='',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    corrections=dict(copyfile=False,
    mandatory=True,
    ),
    output_type=dict(),
    contrast_num=dict(argstr='-cope',
    ),
    sigmasquareds=dict(copyfile=False,
    mandatory=True,
    position=-2,
    argstr='',
    ),
    )
    inputs = ContrastMgr.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_ContrastMgr_outputs():
    output_map = dict(neffs=dict(),
    varcopes=dict(),
    tstats=dict(),
    zstats=dict(),
    fstats=dict(),
    zfstats=dict(),
    copes=dict(),
    )
    outputs = ContrastMgr.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
