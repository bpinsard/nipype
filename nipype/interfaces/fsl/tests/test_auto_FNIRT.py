# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..preprocess import FNIRT


def test_FNIRT_inputs():
    input_map = dict(affine_file=dict(argstr='--aff=%s',
    ),
    apply_inmask=dict(argstr='--applyinmask=%s',
    sep=',',
    xor=[u'skip_inmask'],
    ),
    apply_intensity_mapping=dict(argstr='--estint=%s',
    sep=',',
    xor=[u'skip_intensity_mapping'],
    ),
    apply_refmask=dict(argstr='--applyrefmask=%s',
    sep=',',
    xor=[u'skip_refmask'],
    ),
    args=dict(argstr='%s',
    ),
    bias_regularization_lambda=dict(argstr='--biaslambda=%f',
    ),
    biasfield_resolution=dict(argstr='--biasres=%d,%d,%d',
    ),
    config_file=dict(argstr='--config=%s',
    ),
    derive_from_ref=dict(argstr='--refderiv',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    field_file=dict(argstr='--fout=%s',
    hash_files=False,
    ),
    fieldcoeff_file=dict(argstr='--cout=%s',
    ),
    hessian_precision=dict(argstr='--numprec=%s',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='--in=%s',
    mandatory=True,
    ),
    in_fwhm=dict(argstr='--infwhm=%s',
    sep=',',
    ),
    in_intensitymap_file=dict(argstr='--intin=%s',
    ),
    inmask_file=dict(argstr='--inmask=%s',
    ),
    inmask_val=dict(argstr='--impinval=%f',
    ),
    intensity_mapping_model=dict(argstr='--intmod=%s',
    ),
    intensity_mapping_order=dict(argstr='--intorder=%d',
    ),
    inwarp_file=dict(argstr='--inwarp=%s',
    ),
    jacobian_file=dict(argstr='--jout=%s',
    hash_files=False,
    ),
    jacobian_range=dict(argstr='--jacrange=%f,%f',
    ),
    log_file=dict(argstr='--logout=%s',
    genfile=True,
    hash_files=False,
    ),
    max_nonlin_iter=dict(argstr='--miter=%s',
    sep=',',
    ),
    modulatedref_file=dict(argstr='--refout=%s',
    hash_files=False,
    ),
    out_intensitymap_file=dict(argstr='--intout=%s',
    hash_files=False,
    ),
    output_type=dict(),
    ref_file=dict(argstr='--ref=%s',
    mandatory=True,
    ),
    ref_fwhm=dict(argstr='--reffwhm=%s',
    sep=',',
    ),
    refmask_file=dict(argstr='--refmask=%s',
    ),
    refmask_val=dict(argstr='--imprefval=%f',
    ),
    regularization_lambda=dict(argstr='--lambda=%s',
    sep=',',
    ),
    regularization_model=dict(argstr='--regmod=%s',
    ),
    skip_implicit_in_masking=dict(argstr='--impinm=0',
    ),
    skip_implicit_ref_masking=dict(argstr='--imprefm=0',
    ),
    skip_inmask=dict(argstr='--applyinmask=0',
    xor=[u'apply_inmask'],
    ),
    skip_intensity_mapping=dict(argstr='--estint=0',
    xor=[u'apply_intensity_mapping'],
    ),
    skip_lambda_ssq=dict(argstr='--ssqlambda=0',
    ),
    skip_refmask=dict(argstr='--applyrefmask=0',
    xor=[u'apply_refmask'],
    ),
    spline_order=dict(argstr='--splineorder=%d',
    ),
    subsampling_scheme=dict(argstr='--subsamp=%s',
    sep=',',
    ),
    terminal_output=dict(nohash=True,
    ),
    warp_resolution=dict(argstr='--warpres=%d,%d,%d',
    ),
    warped_file=dict(argstr='--iout=%s',
    genfile=True,
    hash_files=False,
    ),
    )
    inputs = FNIRT.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_FNIRT_outputs():
    output_map = dict(field_file=dict(),
    fieldcoeff_file=dict(),
    jacobian_file=dict(),
    log_file=dict(),
    modulatedref_file=dict(),
    out_intensitymap_file=dict(),
    warped_file=dict(),
    )
    outputs = FNIRT.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
