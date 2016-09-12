from ..base import isdefined
import nibabel as nb
from .base import DCMStackBase, DCMStackBaseInputSpec, DCMStackBaseOutputSpec, dcmstack
from nipype.interfaces.base import TraitedSpec, File, Directory, traits
from nipype.utils.filemanip import fname_presuffix

# sequences base interfaces ###################################################

class DCMStackAnatomical(DCMStackBase):

    _subclasses = dict()

class DCMStackDTIOutputSpec(DCMStackBaseOutputSpec):
    bvec = File(desc='fsl style gradient vectors')
    bval = File(desc='fsl style gradient values')
    gradient_table = File(desc='mrtrix style gradient table')

class DCMStackDTI(DCMStackBase):
    output_spec = DCMStackDTIOutputSpec

    _subclasses = dict()

    def _get_bvecval(self): #TODO: I do not know a thingalingaling about DTI
        self.nii_wrp.meta_ext.get_values('')
        raise NotImplementedError()
    
    def _list_outputs(self):
        outputs = super(DCMStackDTI,self)._list_outputs()
        bvecval = self._get_bvecval()
        outputs['bvec'] = bvecval['bvec']
        outputs['bval'] = bvecval['bval']
        return outputs

class DCMStackFieldmapOutputSpec(DCMStackBaseOutputSpec):
    # there could be difference between precomputed phase difference/complex data/... , TODO: deal with the mess to provide standardized output, this might require multiple 'stack' inputs for machine outputting separately the 2 echoes
    magnitude_file = File(desc='fieldmap magnitude image for coregister')
    phases_file = File(desc='phase file ordered by decreasing echo time (for fugue) if available')
    phase_diff_file = File('phase difference file')
    fieldmap_file = File(desc='computed fieldmap in radians/sec')
    complex_file = File(desc='complex file if provided as such in dicom')
    
    te_difference = traits.Float(desc='echo time difference in sec')
    short_te = traits.Float(desc='shorter echo time')
    long_te = traits.Float(desc='longer echo time')

class DCMStackFieldmap(DCMStackBase):
    output_spec = DCMStackFieldmapOutputSpec

    def __init__(self,**kwargs):
        super(DCMStackFieldmap,self).__init__(**kwargs)
        self._complex_file, self._phases_file = False, False

    def _save_nifti(self):
        outputs = self._list_outputs()
        if hasattr(self,'magnitude_nii'):
            self.magnitude_nii.to_filename(outputs['magnitude_file'])
        self.fieldmap_nii.to_filename(outputs['fieldmap_file'])
        if hasattr(self,'complex_nii'):
            self.complex_nii.to_filename(outputs['complex_file'])
        if hasattr(self,'phases_nii'):
            self.phases_nii.to_filename(outputs['phases_file'])

    def _post_run_cleanup(self):
        if hasattr(self,'magnitude_nii'):
            del self.magnitude_nii
        if hasattr(self,'fieldmap_nii'):
            del self.fieldmap_nii
        if hasattr(self,'complex_nii'):
            del self.complex_nii
        if hasattr(self,'phases_nii'):
            del self.phases_nii

    def _get_TEs(self):
        return self.short_te, self.long_te

    def _list_outputs(self):
        outputs = super(DCMStackFieldmap,self)._list_outputs()
        outputs['magnitude_file'] = fname_presuffix(outputs['nifti_file'],
                                                    suffix='_mag')
        outputs['fieldmap_file'] = fname_presuffix(outputs['nifti_file'],
                                                   suffix='_field')
        if self._complex_file:
            outputs['complex_file'] = fname_presuffix(outputs['nifti_file'],
                                                      suffix='_cplx')
        if self._phases_file:
            outputs['phases_file'] = fname_presuffix(outputs['nifti_file'],
                                                     suffix='_phases')

        outputs['nifti_file'] = outputs['fieldmap_file']
        outputs['short_te'], outputs['long_te'] = self._get_TEs()
        outputs['te_difference'] = outputs['long_te']-outputs['short_te']
        return outputs

class DCMStackfMRIInputSpec(DCMStackBaseInputSpec):
    volume_range = traits.Tuple(
        *([traits.Trait(None,None,traits.Int())]*2),
        desc='use this to trim volumes in fMRI output nifti data')

class DCMStackfMRIOutputSpec(DCMStackBaseOutputSpec):
    slice_trigger_time = traits.List(traits.Float())
    effective_echo_spacing = traits.Float(desc='effective echo spacing in sec')
    slice_order = traits.List(traits.Int())

class DCMStackfMRI(DCMStackBase):
    input_spec = DCMStackfMRIInputSpec
    output_spec = DCMStackfMRIOutputSpec


    def _save_nifti(self):
        # TODO trim start/end volumes before saving
        # must adapt other parameters according to volumes removed
        if isdefined(self.inputs.volume_range):
            tdim = 3
            vr = slice(*self.inputs.volume_range)
            header = self.nii_wrp.nii_img.get_header().copy()
            header['dim'][tdim+1]=len(range(self.nii_wrp.nii_img.shape[tdim])[vr])
            new_meta_ext = dcmstack.DcmMeta.from_sequence(
                [self.nii_wrp.meta_ext.get_subset(tdim,t) for t in xrange(self.nii_wrp.nii_img.shape[tdim])][vr],
                tdim,
                self.nii_wrp.nii_img.get_affine(),
                self.nii_wrp.meta_ext['meta']['slice_dim'])

            nii_img = nb.Nifti1Image(
                self.nii_wrp.nii_img.get_data()[...,vr],
                self.nii_wrp.nii_img.get_affine(),
                header)
            self.nii_wrp = dcmstack.NiftiWrapper(nii_img)
            self.nii_wrp.replace_extension(new_meta_ext)
        self.nii_wrp.to_filename(self._list_outputs()['nifti_file'])
            

    def _list_outputs(self):
        outputs = super(DCMStackfMRI,self)._list_outputs()
        outputs['slice_order'] = self._get_slice_order()
        return outputs
    
    
