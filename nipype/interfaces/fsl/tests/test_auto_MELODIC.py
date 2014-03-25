# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.fsl.model import MELODIC

def test_MELODIC_inputs():
    input_map = dict(ICs=dict(argstr='--ICs=%s',
    ),
    approach=dict(argstr='-a %s',
    ),
    args=dict(argstr='%s',
    ),
    bg_image=dict(argstr='--bgimage=%s',
    ),
    bg_threshold=dict(argstr='--bgthreshold=%f',
    ),
    cov_weight=dict(argstr='--covarweight=%f',
    ),
    dim=dict(argstr='-d %d',
    ),
    dim_est=dict(argstr='--dimest=%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    epsilon=dict(argstr='--eps=%f',
    ),
    epsilonS=dict(argstr='--epsS=%f',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_files=dict(argstr='-i %s',
    mandatory=True,
    position=0,
    sep=',',
    ),
    log_power=dict(argstr='--logPower',
    ),
    mask=dict(argstr='-m %s',
    ),
    max_restart=dict(argstr='--maxrestart=%d',
    ),
    maxit=dict(argstr='--maxit=%d',
    ),
    mix=dict(argstr='--mix=%s',
    ),
    mm_thresh=dict(argstr='--mmthresh=%f',
    ),
    no_bet=dict(argstr='--nobet',
    ),
    no_mask=dict(argstr='--nomask',
    ),
    no_mm=dict(argstr='--no_mm',
    ),
    non_linearity=dict(argstr='--nl=%s',
    ),
    num_ICs=dict(argstr='-n %d',
    ),
    out_all=dict(argstr='--Oall',
    ),
    out_dir=dict(argstr='-o %s',
    genfile=True,
    ),
    out_mean=dict(argstr='--Omean',
    ),
    out_orig=dict(argstr='--Oorig',
    ),
    out_pca=dict(argstr='--Opca',
    ),
    out_stats=dict(argstr='--Ostats',
    ),
    out_unmix=dict(argstr='--Ounmix',
    ),
    out_white=dict(argstr='--Owhite',
    ),
    output_type=dict(),
    pbsc=dict(argstr='--pbsc',
    ),
    rem_cmp=dict(argstr='-f %d',
    ),
    remove_deriv=dict(argstr='--remove_deriv',
    ),
    report=dict(argstr='--report',
    ),
    report_maps=dict(argstr='--report_maps=%s',
    ),
    s_con=dict(argstr='--Scon=%s',
    ),
    s_des=dict(argstr='--Sdes=%s',
    ),
    sep_vn=dict(argstr='--sep_vn',
    ),
    sep_whiten=dict(argstr='--sep_whiten',
    ),
    smode=dict(argstr='--smode=%s',
    ),
    t_con=dict(argstr='--Tcon=%s',
    ),
    t_des=dict(argstr='--Tdes=%s',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    tr_sec=dict(argstr='--tr=%f',
    ),
    update_mask=dict(argstr='--update_mask',
    ),
    var_norm=dict(argstr='--vn',
    ),
    )
    inputs = MELODIC.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_MELODIC_outputs():
    output_map = dict(out_dir=dict(),
    report_dir=dict(),
    )
    outputs = MELODIC.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

