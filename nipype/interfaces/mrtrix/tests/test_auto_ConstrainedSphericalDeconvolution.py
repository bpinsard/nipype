# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.mrtrix.tensors import ConstrainedSphericalDeconvolution
def test_ConstrainedSphericalDeconvolution_inputs():
    input_map = dict(out_filename=dict(position=-1,
    genfile=True,
    argstr='%s',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    args=dict(argstr='%s',
    ),
    mask_image=dict(position=2,
    argstr='-mask %s',
    ),
    maximum_harmonic_order=dict(argstr='-lmax %s',
    ),
    filter_file=dict(position=-2,
    argstr='-filter %s',
    ),
    normalise=dict(position=3,
    argstr='-normalise',
    ),
    directions_file=dict(position=-2,
    argstr='-directions %s',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(position=-3,
    mandatory=True,
    argstr='%s',
    ),
    iterations=dict(argstr='-niter %s',
    ),
    debug=dict(argstr='-debug',
    ),
    threshold_value=dict(argstr='-threshold %s',
    ),
    lambda_value=dict(argstr='-lambda %s',
    ),
    encoding_file=dict(position=1,
    argstr='-grad %s',
    ),
    response_file=dict(position=-2,
    mandatory=True,
    argstr='%s',
    ),
    )
    inputs = ConstrainedSphericalDeconvolution.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_ConstrainedSphericalDeconvolution_outputs():
    output_map = dict(spherical_harmonics_image=dict(),
    )
    outputs = ConstrainedSphericalDeconvolution.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
