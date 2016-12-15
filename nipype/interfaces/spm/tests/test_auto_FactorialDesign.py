# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..model import FactorialDesign


def test_FactorialDesign_inputs():
    input_map = dict(covariates=dict(field='cov',
    ),
    explicit_mask_file=dict(field='masking.em',
    ),
    global_calc_mean=dict(field='globalc.g_mean',
    xor=[u'global_calc_omit', u'global_calc_values'],
    ),
    global_calc_omit=dict(field='globalc.g_omit',
    xor=[u'global_calc_mean', u'global_calc_values'],
    ),
    global_calc_values=dict(field='globalc.g_user.global_uval',
    xor=[u'global_calc_mean', u'global_calc_omit'],
    ),
    global_normalization=dict(field='globalm.glonorm',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    matlab_cmd=dict(),
    mfile=dict(usedefault=True,
    ),
    no_grand_mean_scaling=dict(field='globalm.gmsca.gmsca_no',
    ),
    paths=dict(),
    spm_mat_dir=dict(field='dir',
    ),
    threshold_mask_absolute=dict(field='masking.tm.tma.athresh',
    xor=[u'threshold_mask_none', u'threshold_mask_relative'],
    ),
    threshold_mask_none=dict(field='masking.tm.tm_none',
    xor=[u'threshold_mask_absolute', u'threshold_mask_relative'],
    ),
    threshold_mask_relative=dict(field='masking.tm.tmr.rthresh',
    xor=[u'threshold_mask_absolute', u'threshold_mask_none'],
    ),
    use_implicit_threshold=dict(field='masking.im',
    ),
    use_mcr=dict(),
    use_v8struct=dict(min_ver='8',
    usedefault=True,
    ),
    )
    inputs = FactorialDesign.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_FactorialDesign_outputs():
    output_map = dict(spm_mat_file=dict(),
    )
    outputs = FactorialDesign.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
