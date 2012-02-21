import os
import warnings

import nibabel as nb
import numpy as np

from nipype.utils.misc import package_check

try:
    package_check('nipy')
except Exception, e:
    warnings.warn('nipy not installed')
else:
    from nipy.labs.mask import compute_mask
    import nipy.algorithms.utils.preprocess as preproc

from nipype.interfaces.base import (TraitedSpec, BaseInterface, traits,
                                    BaseInterfaceInputSpec, isdefined, File,
                                    InputMultiPath)
from nipype.utils.filemanip import (fname_presuffix, filename_to_list,
                                    list_to_filename, split_filename,
                                    savepkl, loadpkl)


class ComputeMaskInputSpec(BaseInterfaceInputSpec):
    mean_volume = File(exists=True, mandatory=True, desc="mean EPI image, used to compute the threshold for the mask")
    reference_volume = File(exists=True, desc="reference volume used to compute the mask. If none is give, the \
        mean volume is used.")
    m = traits.Float(desc="lower fraction of the histogram to be discarded")
    M = traits.Float(desc="upper fraction of the histogram to be discarded")
    cc = traits.Bool(desc="if True, only the largest connect component is kept")


class ComputeMaskOutputSpec(TraitedSpec):
    brain_mask = File(exists=True)


class ComputeMask(BaseInterface):
    input_spec = ComputeMaskInputSpec
    output_spec = ComputeMaskOutputSpec

    def _run_interface(self, runtime):

        args = {}
        for key in [k for k, _ in self.inputs.items() if k not in BaseInterfaceInputSpec().trait_names()]:
            value = getattr(self.inputs, key)
            if isdefined(value):
                if key in ['mean_volume', 'reference_volume']:
                    nii = nb.load(value)
                    value = nii.get_data()
                args[key] = value

        brain_mask = compute_mask(**args)

        self._brain_mask_path = os.path.abspath("brain_mask.nii")
        nb.save(nb.Nifti1Image(brain_mask.astype(np.uint8), nii.get_affine()), self._brain_mask_path)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["brain_mask"] = self._brain_mask_path
        return outputs

class RegressOutMotionInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        desc='Run file to process')
    mask = File(
        exists=True,
        desc='Mask file to select voxel to regress out.')
    motion = File(
        exists=True,
        desc='Motion parameters files')
    motion_source = traits.Enum(
        'spm','fsl','afni',
        desc = 'software used to estimate motion',
        usedefault = True)
    regressors_type = traits.Enum(
        'global',
        'voxelwise_drms', 'voxelwise_translation','voxelwise_outplane',
        desc='Which motion parameters to use as regressors',
        usedefault = True)
    global_signal = traits.Bool(
        False, usedefault = True,
        desc = 'Regress out global signal (estimated with mean in mask) from data.')
    regressors_transform = traits.Enum(
        'raw', 'bw_derivatives', 'fw_derivatives',
        desc = 'Which transform to apply to motion parameters'
        )

    slicing_axis = traits.Int(
        2, usedefault = True, desc = 'Axis for outplane motion measure')

    prefix = traits.String('m', usedefault=True)

class RegressOutMotionOutputSpec(TraitedSpec):
    out_file = File(exists=True,
                    desc = 'File with regressed out motion parameters')
    beta_maps = File(exists=True,
                     desc = 'File containing betas maps for each regressor.')
    
class RegressOutMotion(BaseInterface):
    input_spec = RegressOutMotionInputSpec
    output_spec = RegressOutMotionOutputSpec


    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.in_file)
        motion = np.loadtxt(self.inputs.motion)
        motion = preproc.motion_parameter_standardize(motion,self.inputs.motion_source)
        mask = nb.load(self.inputs.mask).get_data()>0
        cdata, _, betamaps = preproc.regress_out_motion_parameters(
            nii,motion,mask,
            regressors_type = self.inputs.regressors_type,
            regressors_transform = self.inputs.regressors_transform,
            slicing_axis = self.inputs.slicing_axis,
            global_signal = self.inputs.global_signal
            )
        outnii = nb.Nifti1Image(cdata,nii.get_affine())
        outnii.set_data_dtype(np.float32)
        nb.save(outnii, self._list_outputs()['out_file'])
        
        betanii = nb.Nifti1Image(betamaps,nii.get_affine())
        nb.save(betanii, self._list_outputs()['beta_maps'])
        del nii, motion, cdata, outnii
        return runtime
        
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = fname_presuffix(
            self.inputs.in_file,
            prefix = self.inputs.prefix,
            newpath = os.getcwd())
        outputs["beta_maps"] = fname_presuffix(
            self.inputs.in_file,
            suffix = '_betas',
            newpath = os.getcwd())

        return outputs


class RegressOutMaskSignalInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        desc='Run file to process')
    mask = File(
        exists=True,
        desc = 'Mask to select the voxels to be regressed.')
    signal_masks = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = "Masks file to compute the regressed out signals.")

    signal_masks_threshold = traits.Float(
        0, usedefault=True,
        desc = 'Value to threshold mask images. Default is >0.')
    signal_estimate_function = traits.Function(
        (lambda x: np.mean(x,0)),
        usedefault = True,
        desc = """Function to estimate the signal from voxel timeseries.
Default is mean along the voxel dimension.""")
    prefix = traits.String('g', usedefault=True)

class RegressOutMaskSignalOutputSpec(TraitedSpec):
    out_file = File(exists=True,
                    desc = 'File with regressed out global signal')

class RegressOutMaskSignal(BaseInterface):
    input_spec  = RegressOutMaskSignalInputSpec
    output_spec = RegressOutMaskSignalOutputSpec

    
    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.in_file)
        data = nii.get_data()
        mask = nb.load(self.inputs.mask).get_data()>0
        signal_masks_nii = [nb.load(m) for m in self.inputs.signal_masks]
        thr = self.inputs.signal_masks_threshold
        signal_masks = [m.get_data()>thr for m in signal_masks_nii]

        m = np.isnan(data).sum(1)
        #correct for isolated nan values in mask timeseries due to realign
        # linearly interpolate in ts and extrapolate at ends of ts
        # TODO:optimize
        y = lambda z: z.nonzero()[0]
        for i in m.nonzero()[0]:
            nans = np.isnan(data[i])
            data[i] = np.interp(y(nans),y(~nans),data[i,~nans])
        
        signals = np.array([self.inputs.signal_estimate_function(data[m]) for m in signal_masks])
        #normalize
        for sig in signals:
            sig[...] = (sig-sig.mean())/sig.var()
        signals = signals.T
        data = data[mask]
        nt=data.shape[-1]

        reg_pinv = np.linalg.pinv(np.concatenate((signals,np.ones((nt,1))),
                                                 axis=1))
        for ts in data:
            beta = reg_pinv.dot(ts)
            ts -= signals.dot(beta[:-1])
        
        cdata = np.zeros(nii.shape)
        cdata[mask] = data

        outnii = nb.Nifti1Image(cdata,nii.get_affine())
        outnii.set_data_dtype(np.float32)
        out_fname = self._list_outputs()['out_file']

        nb.save(outnii, out_fname)
        del nii, data, cdata, outnii
        
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath(
            self.inputs.prefix + os.path.basename(self.inputs.in_file))
        return outputs
    


class ScrubbingInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        desc='The 4D run to be processed')
    mask = File(
        exists=True,
        desc='The brain mask used to compute the framewise derivatives')

    motion = File(
        exists=True,
        desc='Motion parameters files')
    motion_source = traits.Enum(
        'spm','fsl','afni',
        desc = 'software used to estimate motion',
        usedefault = True)
    head_radius = traits.Float(
        50,usedefault=True,
        desc='The head radius in mm to be used for framewise displacement computation.')
    fd_threshold = traits.Float(
        -1, usedefault=True,
        desc="Threshold applied to framewise displacement. Default is half voxel size.")

    drms_threshold = traits.Float(
        -1, usedefault = True,
        desc = """The threshold applied to derivative root mean square.
defaults is an automatic threshold based on otsu.""")
    
    extend_scrubbing = traits.Bool(
        True, usedefault = True,
        desc = 'Extend scrubbing mask to 1 back and 2 forward as in Power et al.')

    
class ScrubbingOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc = 'Scrubbed data')
    motion = File(
        exists=True,
        desc='Motion file with volume scrubbed out parameters removed')
    volume_count = traits.Int(desc='The number of remaining volumes.')
    scrubbed_out_volumes=traits.List(traits.Int(),
                                     desc = 'Index of scrubbed volumes.')
    
class Scrubbing(BaseInterface):
    """ Implement Power et al. scrubbing method,
    suppress volumes with criterion on framewise displacement (FD) and
    bw derivative RMS over voxels
    
    Automatic threshold is half voxel size for FD and 
    is determined by Otsu's algorithm for DRMS
    """
    input_spec = ScrubbingInputSpec
    output_spec = ScrubbingOutputSpec

    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.in_file)
        mask = nb.load(self.inputs.mask).get_data()>0
        motion = np.loadtxt(self.inputs.motion)
        motion = preproc.motion_parameter_standardize(
            motion, self.inputs.motion_source)
        self.scrubbed, self.scrub_mask, _,_ = preproc.scrub_data(
            nii, motion, mask, 
            head_radius = self.inputs.head_radius,
            fd_threshold = self.inputs.fd_threshold,
            drms_threshold = self.inputs.drms_threshold,
            extend_mask = self.inputs.extend_scrubbing)
        
        out_nii = nb.Nifti1Image(self.scrubbed, nii.get_affine())
        nb.save(out_nii, self._list_outputs()['out_file'])
        np.savetxt(self._list_outputs()['motion'], motion[self.scrub_mask])
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = fname_presuffix(
                self.inputs.in_file, newpath=os.getcwd(),
                prefix = 't')
        outputs["motion"] = fname_presuffix(
                self.inputs.motion, newpath=os.getcwd(),
                prefix = 't')

        outputs['volume_count'] = self.scrubbed.shape[-1]
        outputs['scrubbed_out_volumes'] = np.where(self.scrub_mask==0)[0].tolist()
        return outputs
    
