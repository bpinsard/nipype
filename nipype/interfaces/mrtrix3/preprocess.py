from nipype.interfaces.base import CommandLineInputSpec, CommandLine, traits, TraitedSpec, File, InputMultiPath, isdefined
from .base import MRtrixCommandInputSpec, MRtrixCommandOutputSpec, MRtrixCommand, MRtrixDwiCommandInputSpec

class Dwi2ResponseInputSpec(MRtrixDwiCommandInputSpec):
    
    dwi = File(
        argstr="%s",
        position=-2,
        mandatory=True,
        desc="the input diffusion-weighted images")

    response = File(
        argstr="%s",
        position=-1,
        name_source="dwi",
        name_template='%s_response',
        desc="the output rotational harmonic coefficients")

    # DW Shell selection options

    mask = File(
        argstr="-mask %s",
        desc="provide an initial mask image")

    sf = File(
        argstr="-sf %s",
        desc="output a mask highlighting the final selection of single-fibre voxels")

    test_all = traits.Bool(
        argstr = "-test_all",
        desc = """
     by default, only those voxels selected as single-fibre in the previous
     iteration are evaluated. Set this option to re-test all voxels at every
     iteration (slower).""")

    #Options for terminating the optimisation algorithm

    max_iters = traits.Int(
        argstr = "-max_iters %d",
        desc = "maximum number of iterations per pass (set to zero to disable)")

    max_change = traits.Float(
        argstr = "-max_change %f",
        desc = """
     maximum percentile change in any response function coefficient; if no
     individual coefficient changes by more than this fraction, the algorithm
     is terminated.""")

    # Thresholds for single-fibre voxel selection


    volume_ratio = traits.Float(
        argstr = "-volume_ratio %f",
        desc="""
     required volume ratio between largest FOD lobe and the sum of all other
     positive lobes in the voxel""")

    dispersion_multiplier = traits.Float(
        argstr="-dispersion_multiplier %f",
        desc="""
     dispersion of FOD lobe must not exceed some threshold as determined by
     this factor and FOD dispersion statistics. The threshold is: (mean +
     (multiplier * (mean - min))). Criterion is only applied in second pass of
     RF estimation.""")

    integral_multiplier = traits.Float(
        argstr = "-integral_multiplier %f",
        desc = """
     integral of FOD lobe must not be outside some range as determined by this
     factor and FOD lobe integral statistics. The range is: (mean +-
     (multiplier * stdev)). Criterion is only applied in second pass of RF
     estimation. """)

class Dwi2ResponseOutputSpec(MRtrixCommandOutputSpec):
    response = File()
    sf = File()

class Dwi2Response(MRtrixCommand):
    """
    generate an appropriate response function from the image data for
    spherical deconvolution

    >>> from nipype.interfaces.mrtrix3 import Dwi2Response
    >>> dwi2resp = Dwi2Response(dwi='dwi.nii',response='response.txt')
    >>> res = dwi2resp.run()   # doctest: +SKIP
    """

    _cmd = 'dwi2response'
    
    input_spec = Dwi2ResponseInputSpec
    output_spec = Dwi2ResponseOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name=='shells':
            value = value.join(',')
        return super(Dwi2Response,self)._format_arg(name, trait_spec,value)


class Dwi2FodInputSpec(MRtrixDwiCommandInputSpec):
    dwi = File(
        argstr='%s',
        mandatory = True,
        position = -3,
        desc='the input diffusion-weighted image.')

    response = File(
        argstr='%s',
        position=-2,
        mandatory = True,
        desc=""" the diffusion-weighted signal response function for a
                     single fibre population, either as a comma-separated
                     vector of floating-point values, or a text file containing
                     the coefficients.""")
    sh_out_file = File(
        argstr='%s',
        mandatory = True,
        position = -1,
        name_source = 'dwi',
        name_template='%s_fod',
        desc = 'the output spherical harmonics coefficients image')
    
    
    """
    bvalue_scaling = traits.Bool(
            -bvalue_scaling mode
     specifies whether the b-values should be scaled by the square of the
     corresponding DW gradient norm, as often required for multi-shell or DSI
     DW acquisition schemes. The default action can also be set in the MRtrix
     config file, under the BValueScaling entry. Valid choices are yes/no,
     true/false, 0/1."""


    mask = File(
        argstr='-mask %s',
        desc = 'only perform computation within the specified binary brain mask image')

    directions = File(
        argstr='-directions %s',
        desc="""
     specify the directions over which to apply the non-negativity constraint
     (by default, the built-in 300 direction set is used). These should be
     supplied as a text file containing the [ az el ] pairs
     for the directions.""")

    filter = File(
        argstr='-filter %s',
        desc="""
     the linear frequency filtering parameters used for the initial linear
     spherical deconvolution step (default = [ 1 1 1 0 0 ]). These should be 
     supplied as a text file containing the filtering coefficients for each
     even harmonic order.""")

    neg_lambda = traits.Float(
        argstr='-neg_lambda %f',
        desc="""
     the regularisation parameter lambda that controls the strength of the
     non-negativity constraint (default = 1.0).""")

    norm_lambda = traits.Float(
        argstr='-norm_%f',
        desc="""
     the regularisation parameter lambda that controls the strength of the
     constraint on the norm of the solution (default = 1.0).""")

    threshold = traits.Float(
        argstr='-threshold %f',
        desc="""
     the threshold below which the amplitude of the FOD is assumed to be zero,
     expressed as an absolute amplitude (default = 0.0).""")

    niter = traits.Int(
        argstr="-niter %d",
        desc="""the maximum number of iterations to perform for each voxel
                (default = 50).""")

    stride = File(
        argstr='-stride %s',
        desc="""
     specify the strides of the output data in memory, as a comma-separated
     list. The actual strides produced will depend on whether the output image
     format can support it.""")

class Dwi2FodOutputSpec(MRtrixCommandOutputSpec):
    sh_out_file = File()

class Dwi2Fod(MRtrixCommand):
    """
         perform non-negativity constrained spherical deconvolution.

     Note that this program makes use of implied symmetries in the diffusion
     profile. First, the fact the signal attenuation profile is real implies
     that it has conjugate symmetry, i.e. Y(l,-m) = Y(l,m)* (where * denotes
     the complex conjugate). Second, the diffusion profile should be
     antipodally symmetric (i.e. S(x) = S(-x)), implying that all odd l
     components should be zero. Therefore, this program only computes the even
     elements.

     Note that the spherical harmonics equations used here differ slightly from
     those conventionally used, in that the (-1)^m factor has been omitted.
     This should be taken into account in all subsequent calculations.

     The spherical harmonic coefficients are stored as follows. First, since
     the signal attenuation profile is real, it has conjugate symmetry, i.e.
     Y(l,-m) = Y(l,m)* (where * denotes the complex conjugate). Second, the
     diffusion profile should be antipodally symmetric (i.e. S(x) = S(-x)),
     implying that all odd l components should be zero. Therefore, only the
     even elements are computed.
     Note that the spherical harmonics equations used here differ slightly from
     those conventionally used, in that the (-1)^m factor has been omitted.
     This should be taken into account in all subsequent calculations.
     Each volume in the output image corresponds to a different spherical
     harmonic component. Each volume will correspond to the following:
     volume 0: l = 0, m = 0
     volume 1: l = 2, m = -2 (imaginary part of m=2 SH)
     volume 2: l = 2, m = -1 (imaginary part of m=1 SH)
     volume 3: l = 2, m = 0
     volume 4: l = 2, m = 1 (real part of m=1 SH)
     volume 5: l = 2, m = 2 (real part of m=2 SH)
     etc...
     
     Examples
     ========

    >>> from nipype.interfaces.mrtrix3 import Dwi2Fod
    >>> dwi2fod = Dwi2Fod(dwi='dwi.nii',response='response.txt',mask='mask.nii')
    >>> res = dwi2fod.run()   # doctest: +SKIP
"""
    _cmd = 'dwi2fod'
    input_spec = Dwi2FodInputSpec
    output_spec = Dwi2FodOutputSpec

