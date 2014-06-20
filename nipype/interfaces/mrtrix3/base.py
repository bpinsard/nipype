from nipype.interfaces.base import CommandLineInputSpec, CommandLine, traits, TraitedSpec, File, InputMultiPath, isdefined
import os.path as op


class MRtrixCommandInputSpec(CommandLineInputSpec):

    quiet = traits.Bool(
        argstr='-quiet',
        desc='do not display information messages or progress status.')
    debug = traits.Bool(
        argstr='-debug',
        desc='display debugging messages.')
    failonwarn = traits.Bool(
        argstr='-failonwarn',
        desc='terminate program if a warning is produced')

    force = traits.Bool(
        argstr='-force',
        desc='force overwrite of output files.')

    nthreads = traits.Int(
        argstr='-nthreads %d',
        desc='use this number of threads in multi-threaded applications')

class MRtrixCommandOutputSpec(TraitedSpec):
    pass

class MRtrixCommand(CommandLine):
    pass    

class MRtrixDwiCommandInputSpec(MRtrixCommandInputSpec):
    shell =  traits.List(
        traits.Int,
        argstr='-shell %s',
        desc="""
     specify one or more diffusion-weighted gradient shells to use during
     processing, as a comma-separated list of the desired approximate b-values.
     Note that some commands are incompatible with multiple shells, and will
     throw an error if more than one b-value are provided.""")

    lmax = traits.Int(
        argstr='-lmax %s',
        desc="""
     set the maximum harmonic order for the output series. By default, the
     program will use the highest possible lmax given the number of
     diffusion-weighted images.""")

    # DW gradient encoding options

    gradient = File(
        argstr = '-grad %s',
        desc="""
     specify the diffusion-weighted gradient scheme used in the acquisition.
     The program will normally attempt to use the encoding stored in the image
     header. This should be supplied as a 4xN text file with each line is in
     the format [ X Y Z b ], where [ X Y Z ] describe the direction of the
     applied gradient, and b gives the b-value in units of s/mm^2.""")

    fslgradient = traits.Tuple(
        File,File,
        argstr = "-fslgrad %s %s",
        desc="""bvecs bvals
     specify the diffusion-weighted gradient scheme used in the acquisition in
     FSL bvecs/bvals format.""")
