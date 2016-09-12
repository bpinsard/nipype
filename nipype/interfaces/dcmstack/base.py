import os,re
import warnings
from glob import glob
import numpy as np
from ..base import TraitedSpec, InputMultiPath, File, \
    Directory, traits, BaseInterface, BaseInterfaceInputSpec, isdefined
from ...utils.filemanip import filename_to_list, split_filename

from ...utils.misc import package_check

try:
    package_check('dcmstack')
    package_check('dicom')
except Exception as e:
    warnings.warn('dcmstack/pydicom not installed')
else:
    import dcmstack, dcmstack.extract, dicom

class Info(object):
    """Handle dcmstack output type and version information.
    """
    __outputtype = 'NIFTI'
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
            msg = 'Invalid DCMSTACKOUTPUTTYPE: ', outputtype
            raise KeyError(msg)

    @classmethod
    def outputtype(cls):
        return cls.__outputtype


class DCMStackBaseInputSpec(BaseInterfaceInputSpec):
    dicom_files = traits.Either(
        InputMultiPath(File(exists=True)),
        InputMultiPath(Directory(exists=True)),
        InputMultiPath(traits.Str()), #allow glob file pattern
        mandatory=True)
    out_file = File("sequence.nii", hash_files=False)
    out_file_format = traits.Str(
        mandatory=True,
        desc='format with placeholder for output filename')

    outputtype = traits.Enum('NIFTI', Info.ftypes.keys(),
                             desc='DCMStack output filetype')

    meta_force_add = traits.List(
        traits.Str(),
        desc='force add meta tags in dicom parsing')

    voxel_order = traits.String(
        '', use_default=True,
        desc='Voxel order of nifti output, any combination of {R,L} {A,P} {I,S}')
#   output_dtype = traits.DType( ???
#       desc = 'choose output type ')
#    output_units = traits.Str(desc='choose RealWorldValueMapping for specific rescale to apply to data')
#    custom_rescale_slope = traits.Float()
#    custom_rescale_intercept = traits.Float()
    custom_time_order = traits.String()
    custom_vector_order = traits.String()
    dicom_read_force = traits.Bool(
        False, use_default=True,
        desc = 'use dicom.read_file force parameter for wrong dicom files')
    
class DCMStackBaseOutputSpec(TraitedSpec):
    nifti_file = File(exists=True)
    #sequence parameters
    tr = traits.Float()
    te = traits.Float()
    is3d = traits.Bool()
    phase_encoding_direction = traits.Int()
    
    #TODO: list here the most common useful parameters
#    slice_order = traits.List(traits.Int())
    
class DCMStackBase(BaseInterface):
    input_spec = DCMStackBaseInputSpec
    output_spec = DCMStackBaseOutputSpec

    _outputtype = None

    @classmethod 
    def set_default_output_type(cls, outputtype):
        """Set the default output type for DCMStack classes.

        This method is used to set the default output type for all dcmstack
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.outputtype.
        """

        if outputtype in Info.ftypes:
            cls._outputtype = outputtype
        else:
            raise AttributeError('Invalid DCMStack outputtype: %s'%outputtype)

    def __init__(self,*args,**kwargs):
        super(DCMStackBase,self).__init__(*args,**kwargs)
        self.n_ommited=0
        self.n_repeat=0

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


    @classmethod
    def _get_manufacturer(self):
        return self._manufacturer

    def _save_nifti(self):
       self.nii_wrp.to_filename(self._list_outputs()['nifti_file'])

    @property
    def _meta_filter(self):
        if isdefined(self.inputs.meta_force_add):
            return dcmstack.make_key_regex_filter(
                dcmstack.default_key_excl_res,
                dcmstack.default_key_incl_res+self.inputs.meta_force_add)

    
    @property
    def _meta_extractor(self):
        return dcmstack.extract.default_extractor

    def _add_dirty_dcm_to_stack(self,stack,dcm):
        try:
            meta = self._meta_extractor(dcm)
            stack.add_dcm(dcm,meta=meta)
        except dcmstack.IncongruentImageError:
            self.n_ommited += 1
        except dcmstack.ImageCollisionError:
            self.n_repeat += 1
        
    
    def _stack_dicom_files(self):
        dicom_stack = dcmstack.DicomStack(
            time_order = self.inputs.custom_time_order or None,
            vector_order = self.inputs.custom_vector_order or None,
            meta_filter=self._meta_filter)
        for src_path in self.dicom_files:
            try:
                src_dcm = dicom.read_file(src_path, force=self.inputs.dicom_read_force)
                self._add_dirty_dcm_to_stack(dicom_stack,src_dcm)
            except dicom.filereader.InvalidDicomError: # not a dicom file
                self.n_ommited += 1
            except:
                self.n_ommited += 1
            finally:
                del src_dcm
        try:
            self.nii_wrp=dicom_stack.to_nifti_wrapper(self.inputs.voxel_order)
        finally:
            del dicom_stack

    def _list_files(self):
        # list files depending on input type
        df = filename_to_list(self.inputs.dicom_files)
        self.dicom_files = []
        for p in df:
            if os.path.isfile(p):
                self.dicom_files.append(p)
            elif os.path.isdir(p):
                self.dicom_files.extend(sorted(glob(
                        os.path.join(p,'*.dcm'))))
            elif isinstance(p,str):
                self.dicom_files.extend(sorted(glob(p)))

    def _mutate(self):
        sample_dcm = dicom.read_file(self.dicom_files[0], force=self.inputs.dicom_read_force)
        manufacturer = sample_dcm.Manufacturer.lower()
        for subc in self.__class__.__subclasses__():
            if subc._get_manufacturer() in manufacturer:
                self.__class__ = subc
                return

    def _run_interface(self, runtime):
        try:
            self._list_files()
            self._mutate()
            
            if not hasattr(self,'dicom_files'):
                self._list_files()
            self._stack_dicom_files()
            self._extract_parameters()
        
            self._save_nifti()
        finally:
            self._post_run_cleanup()
        return runtime

    def _extract_parameters(self):
        self._tr = self.nii_wrp.get_meta('RepetitionTime',default=0)*1e-3
        self._te = self.nii_wrp.get_meta('EchoTime',default=0)
        if isinstance(self._te,list):
            self._te = min(self._te)
        self._te *= 1e-3

    def _post_run_cleanup(self):
        if hasattr(self,'nii_wrp'):
            del self.nii_wrp

    def _list_outputs(self):
        outputs = self._outputs().get()
        if not isdefined(self.inputs.out_file):
            outputs['nifti_file'] = self._gen_fname()
        else:
            outputs['nifti_file'] = os.path.abspath(self.inputs.out_file)
        outputs["tr"] = self._tr
        return outputs

    def _overload_extension(self, value):
        path, base, _ = split_filename(value)
        return os.path.join(path, base + Info.outputtype_to_ext(self.inputs.outputtype))

    def _gen_fname(self):
        if hasattr(self,'_fname'):
            return self._fname
        keys = re.findall('\%\((\w*)\)',self.inputs.out_file_format)
        if hasattr(self,'nii_wrp'):
            values = dict([(k,self.nii_wrp.meta_ext.get_values(k).strip()) for k in keys])
            fname_base = str(self.inputs.out_file_format % values)
            self._fname=self._overload_extension(os.path.abspath(fname_base))
            return self._fname
        return
        


def extract_from_wrapper(in_file,key,index=None,default=None,
                         scaling_factor=None):
    import dcmstack
    nii=dcmstack.NiftiWrapper.from_filename(in_file)
    val = nii.get_meta(key,index,default)
    if scaling_factor != None:
        val *= scaling_factor
    return val

