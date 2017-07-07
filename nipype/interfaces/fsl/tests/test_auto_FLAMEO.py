# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..model import FLAMEO


def test_FLAMEO_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    burnin=dict(argstr='--burnin=%d',
    ),
    cope_file=dict(argstr='--copefile=%s',
    mandatory=True,
    ),
    cov_split_file=dict(argstr='--covsplitfile=%s',
    mandatory=True,
    ),
    design_file=dict(argstr='--designfile=%s',
    mandatory=True,
    ),
    dof_var_cope_file=dict(argstr='--dofvarcopefile=%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    f_con_file=dict(argstr='--fcontrastsfile=%s',
    ),
    fix_mean=dict(argstr='--fixmean',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    infer_outliers=dict(argstr='--inferoutliers',
    ),
    log_dir=dict(argstr='--ld=%s',
    usedefault=True,
    ),
    mask_file=dict(argstr='--maskfile=%s',
    mandatory=True,
    ),
    n_jumps=dict(argstr='--njumps=%d',
    ),
    no_pe_outputs=dict(argstr='--nopeoutput',
    ),
    outlier_iter=dict(argstr='--ioni=%d',
    ),
    output_type=dict(),
    run_mode=dict(argstr='--runmode=%s',
    mandatory=True,
    ),
    sample_every=dict(argstr='--sampleevery=%d',
    ),
    sigma_dofs=dict(argstr='--sigma_dofs=%d',
    ),
    t_con_file=dict(argstr='--tcontrastsfile=%s',
    mandatory=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    var_cope_file=dict(argstr='--varcopefile=%s',
    ),
    )
    inputs = FLAMEO.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_FLAMEO_outputs():
    output_map = dict(copes=dict(),
    fstats=dict(),
    mrefvars=dict(),
    pes=dict(),
    res4d=dict(),
    stats_dir=dict(),
    tdof=dict(),
    tstats=dict(),
    var_copes=dict(),
    weights=dict(),
    zfstats=dict(),
    zstats=dict(),
    )
    outputs = FLAMEO.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
