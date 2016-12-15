import numpy as np
import nibabel as nb
import dicom
from nipype.interfaces.base import TraitedSpec, File, Directory, traits
from .base import DCMStackBase, DCMStackBaseInputSpec, DCMStackBaseOutputSpec, dcmstack
from .sequences import DCMStackfMRI, DCMStackDTI, DCMStackFieldmap

# ordering class for complex data
"""
class ComplexDataOrdering(dcmstack.dcmstack.DicomOrdering):
    
    def __init__(self):
        pass

    def get_ordinate(self, ds):
        if ds['ImageType'][3] == 'I':
            return 1
        return 0

class GEComplexDataOrdering(dcmstack.dcmstack.DicomOrdering):
    
    def __init__(self):
        pass

    def get_ordinate(self, ds):
        return ds[0x0043,0x0102f].value

# ordering class for decreasing TE in phase fieldmap for fugue
class DecreasingEchoTimeOrdering(dcmstack.dcmstack.DicomOrdering):
    def __init__(self):
        pass
    def get_ordinate(self,ds):
        return -float(ds['EchoTime'])
"""
# constructor base interfaces #################################################


class DCMStackSiemens(DCMStackBase):

    _manufacturer = 'siemens' 

    def _extract_parameters(self):
        super(DCMStackSiemens, self)._extract_parameters()
        acqtimes=self.nii_wrp.meta_ext.get_values('CsaImage.MosaicRefAcqTimes')
        if isinstance(acqtimes,list):
            if isinstance(acqtimes[0],list):
                self._acqtimes = acqtimes[0]
                self._slice_order = np.argsort(acqtimes[0]).tolist()
            else: #single volume
                self._acqtimes = acqtimes
                self._slice_order = np.argsort(acqtimes).tolist()
        else:
            ucMode = self.nii_wrp.meta_ext.get_values(
                'CsaSeries.MrPhoenixProtocol.sSliceArray.ucMode')
            if not ucMode is None:
                if ucMode & 1 == 1:
                    self.slicing_direction = 'ascending'
                elif ucMode & 2 == 2:
                    self.slicing_direction = 'descending'
                self.slice_acquisition_interleaved = 0
                if ucMode & 4 == 4: # supposed this is a 2 slabs interleaving
                    self.slicing_interleaved = 2 

    def _get_slice_order(self):
        if hasattr(self,'_slice_order'):
            return self._slice_order
        return []

    def _get_time_order(self):
        return dcmstack.DicomOrdering('AcquisitionTime')

    def _get_effective_echo_spacing(self):
        ees=1./self.nii_wrp.get_meta('CsaImage.BandwidthPerPixelPhaseEncode')/\
            nw.get_meta('AcquisitionMatrix')[0]
        return ees

class DCMStackPhilips(DCMStackBase):

    _manufacturer = 'philips' 

    # a few sequences where philips hides interesting info at least in my data
    _philips_private_sequences = [
        (0x2005,0x140f),
        (0x2001,0x105f),
        (0x2001,0x1022),]

    @property
    def _meta_extractor(self):
        def ignore_private_philips(tag):
            ignore = tag.tag.group % 2 == 1
            for group,elem in DCMStackPhilips._philips_private_sequences:
                ignore &= not(tag.tag.group==group and tag.tag.elem==elem)
            return ignore
        return dcmstack.extract.MetaExtractor(
            ignore_rules=[ignore_private_philips, 
                          dcmstack.extract.ignore_non_ascii_bytes])

    def _get_slice_order(self):
        return []

#    def _get_time_order(self):
#        return dcmstack.DicomOrdering('AcquisitionTime')

class DCMStackGE(DCMStackBase):
    _manufacturer = 'ge'

    _ge_private_sequences = [
        (0x0021,0x105e),]

    @property
    def _meta_extractor(self):
        def ignore_private_ge(tag):
            ignore = tag.tag.group % 2 == 1
            for group,elem in DCMStackGE._ge_private_sequences:
                ignore &= not(tag.tag.group==group and tag.tag.elem==elem)
            return ignore
        return dcmstack.extract.MetaExtractor(
            ignore_rules=[ignore_private_ge,
                          dcmstack.extract.ignore_non_ascii_bytes])

    def _get_effective_echo_spacing(self):
        return self.nii_wrp.get_meta('EffectiveEchoSpacing')*1e-6

    def _extract_parameters(self):
        super(DCMStackGE, self)._extract_parameters()
        self._acqtimes = [self.nii_wrp.get_meta('RTIA_timer',(0,0,i,0)) for i in range(self.nii_wrp.nii_img.shape[2])]
        if np.all(np.diff(self._acqtimes,1,0)==0):
            self._acqtimes = [self.nii_wrp.get_meta('TriggerTime',(0,0,i,0)) for i in range(self.nii_wrp.nii_img.shape[2])]

    def _get_slice_order(self):
        return np.argsort(self._acqtimes).tolist()

    def _add_dirty_dcm_to_stack(self,stack,dcm):        
        if not stack._ref_input is None:
            ref_iop = stack._ref_input['ImageOrientationPatient']
            if not ref_iop == dcm.ImageOrientationPatient and \
                    np.allclose(ref_iop, dcm.ImageOrientationPatient, 0, 1e-5):
                dcm.ImageOrientationPatient = ref_iop
        super(DCMStackGE, self)._add_dirty_dcm_to_stack(stack,dcm)

# end user interfaces #######

class DCMStackfMRISiemens(DCMStackfMRI,DCMStackSiemens):
    pass

class DCMStackfMRIPhilips(DCMStackfMRI,DCMStackPhilips):
    pass

class DCMStackfMRIGE(DCMStackfMRI,DCMStackGE):
    pass

class DCMStackFieldmapSiemens(DCMStackFieldmap, DCMStackSiemens):

    def _stack_dicom_files(self):
        dcms=[]
        for f in self.dicom_files:
            try:
                dcms.append(dicom.read_file(f))
            except dicom.filereader.InvalidDicomError:
                self.n_ommited += 1
        mag_dcms = [d for d in dcms if hasattr(d,'ImageType') and\
                        d.ImageType[2]=='M']
        phase_dcms = [d for d in dcms if hasattr(d,'ImageType') and\
                          d.ImageType[2]=='P']
        parsing_opts = dict(
            meta_filter = self._meta_filter,)
#            time_order=DecreasingEchoTimeOrdering())
        phase_stack = dcmstack.DicomStack(**parsing_opts)
        for d in phase_dcms:
            self._add_dirty_dcm_to_stack(phase_stack,d)

        vo = self.inputs.voxel_order
        dt = np.float32
        try:
            self.phases_nii = phase_stack.to_nifti_wrapper(vo)
            self._phases_file = True
        finally:
            del phase_dcms, phase_stack
        if len(mag_dcms)>0:
            mag_stack = dcmstack.DicomStack(**parsing_opts)        
            for d in mag_dcms:
                self._add_dirty_dcm_to_stack(mag_stack,d)
            try:
                self.magnitude_nii = mag_stack.to_nifti_wrapper(vo)
            finally:
                del mag_dcms, mag_stack
            self.nii_wrp = self.magnitude_nii # alias for parent methods
        else:
            self.nii_wrp = self.phases_nii

        self.short_te, self.long_te = 0,0
        if hasattr(self,'nii_wrp'):
            if len(self.nii_wrp.nii_img.shape)>3 :
                self.short_te = self.nii_wrp.get_meta('EchoTime',[0,0,0,1])
                self.long_te = self.nii_wrp.get_meta('EchoTime',[0,0,0,0])
            else:
                self.short_te = self.nii_wrp.get_meta(
                    'CsaSeries.MrPhoenixProtocol.alTE[0]')
                self.long_te = self.nii_wrp.get_meta(
                    'CsaSeries.MrPhoenixProtocol.alTE[1]')
            if self.short_te is None or self.long_te is None:
                # dirty fix for the case when dicoms contained mix TE values
                all_tes = [self.nii_wrp.get_meta('EchoTime',[0,0,s])\
                              for s in range(self.nii_wrp.nii_img.shape[2])]
                self.short_te = min(all_tes)
                self.long_te = max(all_tes)
            self.short_te *= 1e-3
            self.long_te *= 1e-3

        if len(self.phases_nii.nii_img.shape)<4:
            newdata=np.concatenate((
               self.phases_nii.nii_img.get_data()[...,np.newaxis]*(np.pi/4096),
               np.zeros(self.phases_nii.nii_img.shape+(1,))),3)
            header = self.phases_nii.nii_img.get_header()
            header.set_data_dtype(dt)
            header.has_data_intercept,header.has_data_slope = False,False
            self.phases_nii = dcmstack.NiftiWrapper(nb.Nifti1Image(
                    newdata.astype(dt),
                    self.phases_nii.nii_img.get_affine(),
                    header))
            self.fieldmap_nii = nb.Nifti1Image(
                (newdata[...,0]/(self.long_te-self.short_te)).astype(dt),
                self.phases_nii.nii_img.get_affine())
        else:
            self.phases_nii = dcmstack.NiftiWrapper(nb.Nifti1Image(
                    self.phases_nii.nii_img.get_data()*(np.pi/4096),
                    self.phases_nii.nii_img.get_affine(),
                    self.phases_nii.nii_img.get_header()))

def complex_to_magphase(re,im):
    cplx = re+1j*im
    return np.abs(cplx),np.angle(cplx)

def phase_diff(ph1,ph2):
    return np.mod(ph1-ph2+np.pi*2,np.pi*2)

class DCMStackFieldmapPhilips(DCMStackFieldmap, DCMStackPhilips):
    
    def _stack_dicom_files(self):
        dcms=[]
        for f in self.dicom_files:
            try:
                dcms.append(dicom.read_file(f))
            except dicom.filereader.InvalidDicomError:
                self.n_ommited += 1
        mag_dcms = [d for d in dcms if hasattr(d,'ImageType') and\
                        d.ImageType[3]=='M']
        phase_dcms = [d for d in dcms if hasattr(d,'ImageType') and\
                          d.ImageType[3]=='P']
        real_dcms = [d for d in dcms if hasattr(d,'ImageType') and\
                         d.ImageType[3]=='R']
        imag_dcms = [d for d in dcms if hasattr(d,'ImageType') and\
                         d.ImageType[3]=='I']

        vo = self.inputs.voxel_order
        dt = np.float32
        parsing_opts = dict(
            meta_filter = self._meta_filter,)
#            time_order=DecreasingEchoTimeOrdering())

        if len(mag_dcms)>0 and len(phase_dcms)>0:
            mag_stack = dcmstack.DicomStack(**parsing_opts)
            phase_stack =  dcmstack.DicomStack(**parsing_opts)
            for d in mag_dcms:
                self._add_dirty_dcm_to_stack(mag_stack,d)
            for d in phase_dcms:
                self._add_dirty_dcm_to_stack(phase_stack,d)
            try: 
                self.magnitude_nii = mag_stack.to_nifti_wrapper(vo)
                self.phases_nii = phase_stack.to_nifti_wrapper(vo)
                self._phases_file = True
                rescale_intercept=self.phases_nii.get_meta('RescaleIntercept')
                rescale_slope=self.phases_nii.get_meta('RescaleSlope')
                unit = phase_dcms[0].get((0x2005,0x140b))
                if unit != None and \
                        (rescale_intercept==None or rescale_slope==None):
                    if unit.value == 'milliradials':
                        rescale_intercept = 0
                        rescale_slope = 1e3 #requires pydicom real world value patch
                    elif unit.value == 'cm/sec':
                        rescale_intercept = phase_dcms[0][0x2005,0x1409].value
                        rescale_slope = phase_dcms[0][0x2005,0x140a].value
            finally:
                del phase_stack, mag_stack, mag_dcms, phase_dcms
            self.nii_wrp = self.magnitude_nii # alias for base

            rescale_intercept = np.abs(float(rescale_intercept))
            rescale_slope = np.abs(float(rescale_slope))
            radians = np.pi*(self.phases_nii.nii_img.get_data()/rescale_intercept)
            if radians.ndim < 4:
                # fake zeros phase image in case it is already substracted
                radians=np.concatenate((np.zeros(radians.shape+(1,)),
                                        radians[...,np.newaxis]),3)
            header = self.phases_nii.nii_img.get_header()
            header.set_data_dtype(dt)
            header.has_data_intercept,header.has_data_slope = False,False

            self.phases_nii = dcmstack.NiftiWrapper(
                nb.Nifti1Image(
                    radians.astype(dt),
                    self.phases_nii.nii_img.get_affine(), 
                    header))

            self.short_te, self.long_te = 0,0
            if hasattr(self,'nii_wrp'):
                self.short_te=self.nii_wrp.get_meta('EchoTime',[0,0,0,1])/1e3
                self.long_te=self.nii_wrp.get_meta('EchoTime',[0,0,0,0])/1e3

            radiansec = np.diff(radians,1,-1) / (self.long_te-self.short_te)
            self.fieldmap_nii = nb.Nifti1Image(
                radiansec.astype(dt), self.phases_nii.nii_img.get_affine())
        elif len(real_dcms)>0 and len(imag_dcms)>0:
            real_stack = dcmstack.DicomStack(**parsing_opts)
            imag_stack =  dcmstack.DicomStack(**parsing_opts)
            for d in real_dcms:
                self._add_dirty_dcm_to_stack(real_stack,d)
            for d in imag_dcms:
                self._add_dirty_dcm_to_stack(imag_stack,d)
            try:
                real_nii = real_stack.to_nifti_wrapper(vo)
                imag_nii = imag_stack.to_nifti_wrapper(vo)
            finally:
                del real_dcms, imag_dcms, real_stack, imag_stack
            header = real_nii.nii_img.get_header().copy()
            affine = real_nii.nii_img.get_affine().copy()
            header.set_data_dtype(np.complex64)
            header.has_data_intercept, header.has_data_slope = False, False
            self.complex_nii = dcmstack.NiftiWrapper(nb.Nifti1Image(
                    real_nii.nii_img.get_data()+1j*imag_nii.nii_img.get_data(),
                    affine, header))
            self._complex_file = True
            self.nii_wrp = self.complex_nii # alias for base
            self.magnitude_nii = nb.Nifti1Image(
                np.abs(self.complex_nii.nii_img.get_data()).astype(dt), affine)
            phases = np.angle(self.complex_nii.nii_img.get_data())
            self.phases_nii = nb.Nifti1Image(phases.astype(dt),affine)
            self._phases_file = True
            phasediff = np.mod(np.diff(phases,1,-1)+np.pi*2,np.pi*2)
            self.fieldmap_nii = nb.Nifti1Image(phasediff.astype(dt),affine)

            self.short_te, self.long_te = 0,0
            if hasattr(self,'nii_wrp'):
                self.short_te=self.nii_wrp.get_meta('EchoTime',[0,0,0,1])/1e3
                self.long_te=self.nii_wrp.get_meta('EchoTime',[0,0,0,0])/1e3

class DCMStackFieldmapGE(DCMStackFieldmap, DCMStackGE):

    def _stack_dicom_files(self):
        dcms=[]
        data_type_tag = (0x0043,0x102f)
        for f in self.dicom_files:
            try:
                dcms.append(dicom.read_file(f))
            except dicom.filereader.InvalidDicomError:
                self.n_ommited += 1
        real_dcms = [d for d in dcms if d.get(data_type_tag) and\
                        d[data_type_tag].value==2]
        imag_dcms = [d for d in dcms if d.get(data_type_tag) and\
                        d[data_type_tag].value==3]
        mag_dcms = [d for d in dcms if d.get(data_type_tag) and\
                        d[data_type_tag].value==0]
        phase_dcms = [d for d in dcms if d.get(data_type_tag) and\
                        d[data_type_tag].value==1]


        parsing_opts = dict(
            meta_filter = self._meta_filter,)
#            time_order=DecreasingEchoTimeOrdering())
        vo = self.inputs.voxel_order
        dt = np.float32        
        
        if len(real_dcms)>0 and len(imag_dcms)>0:
             real_stack = dcmstack.DicomStack(**parsing_opts)
             imag_stack =  dcmstack.DicomStack(**parsing_opts)
             for d in real_dcms:
                 self._add_dirty_dcm_to_stack(real_stack,d)
             for d in imag_dcms:
                 self._add_dirty_dcm_to_stack(imag_stack,d)
             try:
                 real_nii = real_stack.to_nifti_wrapper(vo)
                 imag_nii = imag_stack.to_nifti_wrapper(vo)
             finally:
                 del real_dcms, imag_dcms, real_stack, imag_stack
             header = real_nii.nii_img.get_header().copy()
             affine = real_nii.nii_img.get_affine().copy()
             header.set_data_dtype(np.complex64)
             header.has_data_intercept, header.has_data_slope = False, False
             self.complex_nii = dcmstack.NiftiWrapper(nb.Nifti1Image(
                     real_nii.nii_img.get_data()+1j*imag_nii.nii_img.get_data(),
                     affine, header))
             self._complex_file = True
             self.nii_wrp = self.complex_nii # alias for base
             self.magnitude_nii = nb.Nifti1Image(
                 np.abs(self.complex_nii.nii_img.get_data()).astype(dt), affine)
             phases = np.angle(self.complex_nii.nii_img.get_data())
             self.phases_nii = nb.Nifti1Image(phases.astype(dt),affine)
             self._phases_file = True
             phasediff = np.mod(np.diff(phases,1,-1)+np.pi*2,np.pi*2)
             self.fieldmap_nii = nb.Nifti1Image(phasediff.astype(dt),affine)

        elif len(mag_dcms)>0 and len(phase_dcms)>0:
             #TODO rescale of phases
             mag_stack = dcmstack.DicomStack(**parsing_opts)
             phase_stack =  dcmstack.DicomStack(**parsing_opts)
             for d in mag_dcms:
                 self._add_dirty_dcm_to_stack(mag_stack,d)
             for d in phase_dcms:
                 self._add_dirty_dcm_to_stack(phase_stack,d)
             try:
                 self.magnitude_nii = mag_stack.to_nifti_wrapper(vo)
                 self.phases_nii = phase_stack.to_nifti_wrapper(vo)
                 self._phases_file = True
             finally:
                 del mag_dcms, imag_dcms, mag_stack, phase_stack
             self.nii_wrp = self.magnitude_nii # alias for base
            
        self.short_te, self.long_te = 0,0
        if hasattr(self,'nii_wrp'):
            self.short_te=self.nii_wrp.get_meta('EchoTime',[0,0,0,1]) / 1000.0
            self.long_te=self.nii_wrp.get_meta('EchoTime',[0,0,0,0]) / 1000.0
