
import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
    CommandLine, traits, CommandLineInputSpec, isdefined, File, TraitedSpec,
    BaseInterface, BaseInterfaceInputSpec)

class Info(object):
    """Handle afni output type and version information.
    """
    __outputtype = 'AFNI'
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
            msg = 'Invalid NIPYOUTPUTTYPE: ', outputtype
            raise KeyError(msg)

class NipyBaseInterfaceInputSpec(BaseInterfaceInputSpec):
    outputtype = traits.Enum('NIFTI', Info.ftypes.keys(),
                             desc='Nipy/nibabel output filetype')

class NipyBaseInterface(BaseInterface):


    def __init__(self, **inputs):
        super(NipyBaseInterface, self).__init__(**inputs)
        self.inputs.on_trait_change(self._output_update, 'outputtype')
        
        if self._outputtype is None:
            self._outputtype = Info.outputtype()

        if not isdefined(self.inputs.outputtype):
            self.inputs.outputtype = self._outputtype
        else:
            self._output_update()

    def _output_update(self):
        self._outputtype = self.inputs.outputtype

    @classmethod 
    def set_default_output_type(cls, outputtype):
        """Set the default output type for nipy classes.
        """

        if outputtype in Info.ftypes:
            cls._outputtype = outputtype
        else:
            raise AttributeError('Invalid nipy outputtype: %s' % outputtype)

    
    def _gen_filename(self, name):
        trait_spec = self.inputs.trait(name)
        value = getattr(self.inputs, name)
        if isdefined(value):
            if "%s" in value:
                if isinstance(trait_spec.name_source, list):
                    for ns in trait_spec.name_source:
                        if isdefined(getattr(self.inputs, ns)):
                            name_source = ns
                            break
                else:
                    name_source = trait_spec.name_source
                if name_source.endswith(os.path.sep):
                    name_source = name_source[:-len(os.path.sep)]
                _, base, _ = split_filename(getattr(self.inputs, name_source))
                    
                retval = value%base
            else:
                retval = value
        else:
            raise NotImplementedError
        _,_,ext = split_filename(retval)
        if trait_spec.overload_extension or not ext:
            return self._overload_extension(retval)
        else:
            return retval
            
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
                if isdefined(out):
                    outputs[name] = os.path.abspath(out)
            return outputs
