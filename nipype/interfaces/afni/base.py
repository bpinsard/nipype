# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Provide interface to AFNI commands."""


import os
import warnings

from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
    CommandLine, traits, CommandLineInputSpec, isdefined, File, TraitedSpec,
    BaseInterface)

warn = warnings.warn
warnings.filterwarnings('always', category=UserWarning)

###################################
#
# NEW_AFNI base class
#
###################################


class Info(object):
    """Handle afni output type and version information.
    """
    __outputtype = 'AFNI'
    ftypes = {'NIFTI': '.nii',
              'AFNI': '+orig.BRIK',
              'NIFTI_GZ': '.nii.gz'}

    @staticmethod
    def version():
        """Check for afni version on system

        Parameters
        ----------
        None

        Returns
        -------
        version : str
           Version number as string or None if AFNI not found

        """
        clout = CommandLine(command='afni_vcheck').run()
        out = clout.runtime.stdout
        return out.split('\n')[1]

    @classmethod
    def outputtype_to_ext(cls, outputtype):
        """Get the file extension for the given output type.

        Parameters
        ----------
        outputtype : {'NIFTI', 'NIFTI_GZ', 'AFNI'}
            String specifying the output type.

        Returns
        -------
        extension : str
            The file extension for the output type.
        """

        try:
            return cls.ftypes[outputtype]
        except KeyError:
            msg = 'Invalid AFNIOUTPUTTYPE: ', outputtype
            raise KeyError(msg)

    @classmethod
    def outputtype(cls):
        """AFNI has no environment variables,
        Output filetypes get set in command line calls
        Nipype uses AFNI as default


        Returns
        -------
        None
        """
        #warn(('AFNI has no environment variable that sets filetype '
        #      'Nipype uses NIFTI_GZ as default'))
        return 'AFNI'

    @staticmethod
    def standard_image(img_name):
        '''Grab an image from the standard location.

        Could be made more fancy to allow for more relocatability'''
        clout = CommandLine('which afni').run()
        if clout.runtime.returncode is not 0:
            return None

        out = clout.runtime.stdout
        basedir = os.path.split(out)[0]
        return os.path.join(basedir, img_name)


class AFNIBaseCommandInputSpec(CommandLineInputSpec):
    outputtype = traits.Enum('AFNI', Info.ftypes.keys(),
                             desc='AFNI output filetype')
    
class AFNITraitedSpec(AFNIBaseCommandInputSpec):
    pass


class AFNIBaseCommand(CommandLine):
    """General support for AFNI commands. Every AFNI command accepts 'outputtype' input. For example:
    afni.Threedsetup(outputtype='NIFTI_GZ')
    """

    input_spec = AFNIBaseCommandInputSpec
    _outputtype = None
    

    def __init__(self, **inputs):
        super(AFNIBaseCommand, self).__init__(**inputs)
        self.inputs.on_trait_change(self._output_update, 'outputtype')
        
        if self._outputtype is None:
            self._outputtype = Info.outputtype()

        if not isdefined(self.inputs.outputtype):
            self.inputs.outputtype = self._outputtype
        else:
            self._output_update()

    def _output_update(self):
        """ i think? updates class private attribute based on instance input
         in fsl also updates ENVIRON variable....not valid in afni
         as it uses no environment variables
        """
        self._outputtype = self.inputs.outputtype

    # name change for standardisation with others interfaces 
    @classmethod 
    def set_default_output_type(cls, outputtype):
        """Set the default output type for AFNI classes.

        This method is used to set the default output type for all afni
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.outputtype.
        """

        if outputtype in Info.ftypes:
            cls._outputtype = outputtype
        else:
            raise AttributeError('Invalid AFNI outputtype: %s' % outputtype)

    def _gen_fname(self, basename, cwd=None, suffix='_afni', change_ext=True, prefix=''):
        """Generate a filename based on the given parameters.

        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extensions specified in
        <instance>inputs.outputtype.

        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (default is '_fsl')
        change_ext : bool
            Flag to change the filename extension to the FSL output type.
            (default True)

        Returns
        -------
        fname : str
            New filename based on given parameters.

        """
        if isinstance(basename,tuple):
            basename = basename[0]

        if basename == '':
            msg = 'Unable to generate filename for command %s. ' % self.cmd
            msg += 'basename is not set!'
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        ext = Info.outputtype_to_ext(self.inputs.outputtype)
        if change_ext:
            if suffix:
                suffix = ''.join((suffix, ext))
            else:
                suffix = ext
        fname = fname_presuffix(basename, suffix=suffix,
                                use_ext=False, newpath=cwd, prefix=prefix)
        return fname


    def _format_arg(self, name, trait_spec, value):
        # dirty checking for an AFNIFile
        if trait_spec.is_trait_type(traits.TraitCompound) and\
                isinstance(value,tuple) and\
                isinstance(value[0],str) and\
                os.path.exists(value[0]):
            if isinstance(value[1],list):
                bricks = ','.join(value[1])
            elif isinstance(value[1],tuple):
                bricks = '%d..%s\(%d\)'%(
                    value[1][0],
                    (str(value[1][1]) if value[1][1]>=0 else '$'),
                    value[1][2])
            elif isinstance(value[1],int):
                bricks = str(value[1])
            
            fname = '%s[%s]'%(value[0],bricks)
            arg = trait_spec.argstr % fname
            return arg
        return super(AFNICommand, self)._format_arg(name, trait_spec, value)



class AFNIFile(traits.Either):
    """
    traits describing a sub-brick of a file
    This is a file with either a list of indices of images in the files, 
    or a tuple with (start,stop,step), with let say -1 (pythonic) for end of file
    """

    def __init__(self, value = '', filter = None, auto_set = False,
                 entries = 0, exists = False, **metadata):
        super( AFNIFile, self ).__init__(
            File(value=value,
                 filter=filter,
                 auto_set=auto_set,
                 entries=entries,
                 exists=exists), 
            traits.Tuple(File(value=value,
                              filter=filter,
                              auto_set=auto_set,
                              entries=entries,
                              exists=exists),
                         traits.Either(traits.Tuple(traits.Int(0),
                                                    traits.Int(-1),
                                                    traits.Int(1)),
                                       traits.Int(),
                                       traits.List(traits.Int()),
                                       usedefault=True ) ),
            **metadata)
        
def brick_selector(f,bricks):
    # simple interface function to add static brick selection to a pipeline
    return (f,bricks)

def afni_file_path(af):
    if isinstance(af,tuple):
        return af[0]
    return af

class BrickSelectorInputSpec(TraitedSpec):
    
    in_file = traits.File(
        mandatory=True,
        exists=True,
        desc='Regular file from which to select sub-bricks.')
    bricks = traits.Either(
        traits.Tuple(traits.Int(0),traits.Int(-1),traits.Int(1)),
        traits.Int(),
        traits.ListInt(),
        desc='Sub bricks as a list of indices or a tuple (start,stop,step) or single index')

class BrickSelectorOutputSpec(TraitedSpec):

    out_file = AFNIFile()

class BrickSelector(BaseInterface):
    """
    Simple interface to perform brick selection with bricks being a dynamic input.
    We could add a brick selection validation with file dimensions etc...
    """

    input_spec  = BrickSelectorInputSpec
    output_spec = BrickSelectorOutputSpec

    def _run_interface(self,runtime): 
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = (self.inputs.in_file, self.inputs.bricks)
        return outputs

class AFNICommandInputSpec(AFNIBaseCommandInputSpec):
    out_file = File("%s_afni", desc='output image file name',
                    argstr='-prefix %s', xor=['out_file', 'prefix', 'suffix'], name_source="in_file", usedefault=True)
    prefix = traits.Str(
        desc='output image prefix', deprecated='0.8', new_name="out_file")
    suffix = traits.Str(
        desc='output image suffix', deprecated='0.8', new_name="out_file")


class AFNICommand(AFNIBaseCommand):
    input_spec = AFNICommandInputSpec

    def _gen_filename(self, name):
        trait_spec = self.inputs.trait(name)
        if name == "out_file" and (isdefined(self.inputs.prefix) or isdefined(self.inputs.suffix)):
            suffix = ''
            prefix = ''
            if isdefined(self.inputs.prefix):
                prefix = self.inputs.prefix
            if isdefined(self.inputs.suffix):
                suffix = self.inputs.suffix

            _, base, _ = split_filename(
                getattr(self.inputs, trait_spec.name_source))
            return self._gen_fname(basename=base, prefix=prefix, suffix=suffix, cwd='')
        else:
            return super(AFNICommand, self)._gen_filename(name)

    def _overload_extension(self, value):
        path, base, _ = split_filename(value)
        return os.path.join(path, base + Info.outputtype_to_ext(self.inputs.outputtype))

    def _list_outputs(self):
        metadata = dict(name_source=lambda t: t is not None)
        out_names = self.inputs.traits(**metadata).keys()
        if out_names:
            outputs = self.output_spec().get()
            for name in out_names:
                out = self._gen_filename(name)
                outputs[name] = os.path.abspath(out)
            return outputs


class AFNICommandOutputSpec(TraitedSpec):
    out_file = File(desc='output file',
                    exists=True)
