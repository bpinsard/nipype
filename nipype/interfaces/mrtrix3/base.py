from nipype.interfaces.base import CommandLineInputSpec, CommandLine, traits, TraitedSpec, File, InputMultiPath, isdefined
import os
from ...utils.filemanip import split_filename


class Info(object):

    _outputtype = 'NIFTI_GZ'
    ftypes = {'NIFTI': '.nii',
              'NIFTI_GZ': '.nii.gz'}

    @classmethod
    def outputtype_to_ext(cls, outputtype):
        """Get the file extension for the given output type.

        Parameters
        ----------
        outputtype : {'NIFTI', 'NIFTI_GZ'}
            String specifying the output type.

        Returns
        -------
        extension : str
            The file extension for the output type.
        """

        try:
            return cls.ftypes[outputtype]
        except KeyError:
            msg = 'Invalid MRTRIXOUTPUTTYPE: ', outputtype
            raise KeyError(msg)

class MRtrixCommandInputSpec(CommandLineInputSpec):

    quiet = traits.Bool(
        argstr='-quiet',
        nohash=True,
        desc='do not display information messages or progress status.')
    debug = traits.Bool(
        argstr='-debug',
        nohash=True,
        desc='display debugging messages.')
    failonwarn = traits.Bool(
        argstr='-failonwarn',
        nohash=True,
        desc='terminate program if a warning is produced')

    force = traits.Bool(
        argstr='-force',
        nohash=True,
        desc='force overwrite of output files.')

    nthreads = traits.Int(
        argstr='-nthreads %d',
        nohash=True,
        desc='use this number of threads in multi-threaded applications')

    outputtype = traits.Enum('NIFTI_GZ', Info.ftypes.keys(),
                             desc='MRtrix output filetype')

class MRtrixCommandOutputSpec(TraitedSpec):
    pass

class MRtrixCommand(CommandLine):
    _outputtype = None

    def __init__(self, **inputs):
        super(MRtrixCommand, self).__init__(**inputs)

        if self._outputtype is None:
            self._outputtype = Info._outputtype

        if not isdefined(self.inputs.outputtype):
            self.inputs.outputtype = self._outputtype
        else:
            self._output_update()

    @classmethod
    def set_default_output_type(cls, outputtype):
        """Set the default output type for MRtrix classes.

        This method is used to set the default output type for all MRtrix
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.outputtype.
        """

        if outputtype in Info.ftypes:
            cls._outputtype = outputtype
        else:
            raise AttributeError('Invalid MRTRIX outputtype: %s' % outputtype)


    def _overload_extension(self, value, name=None):
        path, base, _ = split_filename(value)
        return os.path.join(path, base + Info.outputtype_to_ext(self.inputs.outputtype))


class MRtrixDwiCommandInputSpec(MRtrixCommandInputSpec):
    shell =  traits.List(
        traits.Int,
        argstr='-shell %s',
        sep=',',
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
