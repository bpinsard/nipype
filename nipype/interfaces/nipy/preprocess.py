# -*- coding: utf-8 -*-
"""
    Change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
    >>> datadir = os.path.realpath(os.path.join(filepath, '../../testing/data'))
    >>> os.chdir(datadir)

"""
from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import open

import os

import nibabel as nb
import numpy as np

from ...utils.misc import package_check
from ...utils.filemanip import (fname_presuffix, filename_to_list,
                                list_to_filename, split_filename,
                                savepkl, loadpkl)
from ..base import (TraitedSpec, BaseInterface, traits,
                    BaseInterfaceInputSpec, isdefined, File,
                    InputMultiPath, OutputMultiPath)

have_nipy = True
try:
    package_check('nipy')
except Exception as e:
    have_nipy = False
else:
    from nipy.labs.mask import compute_mask
    import nipy.algorithms.utils.preprocess as preproc
    from nipy.algorithms.registration import FmriRealign4d as FR4d
    import nipy
    from nipy import save_image, load_image
    nipy_version = nipy.__version__


class ComputeMaskInputSpec(BaseInterfaceInputSpec):
    mean_volume = File(exists=True, mandatory=True,
                       desc="mean EPI image, used to compute the threshold for the mask")
    reference_volume = File(exists=True,
                            desc=("reference volume used to compute the mask. "
                                  "If none is give, the mean volume is used."))
    m = traits.Float(desc="lower fraction of the histogram to be discarded")
    M = traits.Float(desc="upper fraction of the histogram to be discarded")
    cc = traits.Bool(desc="Keep only the largest connected component")


class ComputeMaskOutputSpec(TraitedSpec):
    brain_mask = File(exists=True)


class ComputeMask(BaseInterface):
    input_spec = ComputeMaskInputSpec
    output_spec = ComputeMaskOutputSpec

    def _run_interface(self, runtime):
        from nipy.labs.mask import compute_mask
        args = {}
        for key in [k for k, _ in list(self.inputs.items())
                    if k not in BaseInterfaceInputSpec().trait_names()]:
            value = getattr(self.inputs, key)
            if isdefined(value):
                if key in ['mean_volume', 'reference_volume']:
                    nii = nb.load(value)
                    value = nii.get_data()
                args[key] = value

        brain_mask = compute_mask(**args)
        _, name, ext = split_filename(self.inputs.mean_volume)
        self._brain_mask_path = os.path.abspath("%s_mask.%s" % (name, ext))
        nb.save(nb.Nifti1Image(brain_mask.astype(np.uint8), nii.affine),
                self._brain_mask_path)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["brain_mask"] = self._brain_mask_path
        return outputs

        
class FmriRealign4dInputSpec(BaseInterfaceInputSpec):

    in_file = InputMultiPath(File(exists=True),
                             mandatory=True,
                             desc="File to realign")
    tr = traits.Float(desc="TR in seconds",
                      mandatory=True)
    slice_order = traits.List(traits.Int(),
                              desc=('0 based slice order. This would be equivalent to entering'
                                    'np.argsort(spm_slice_order) for this field. This effects'
                                    'interleaved acquisition. This field will be deprecated in'
                                    'future Nipy releases and be replaced by actual slice'
                                    'acquisition times.'),
                              requires=["time_interp"])
    tr_slices = traits.Float(desc="TR slices", requires=['time_interp'])
    start = traits.Float(0.0, usedefault=True,
                         desc="time offset into TR to align slices to")
    time_interp = traits.Enum(True, requires=["slice_order"],
                              desc="Assume smooth changes across time e.g.,\
                     fmri series. If you don't want slice timing \
                     correction set this to undefined")
    loops = InputMultiPath([5], traits.Int, usedefault=True,
                           desc="loops within each run")
    between_loops = InputMultiPath([5], traits.Int,
                                   usedefault=True, desc="loops used to \
                                                          realign different \
                                                          runs")
    speedup = InputMultiPath([5], traits.Int,
                             usedefault=True,
                             desc="successive image \
                                  sub-sampling factors \
                                  for acceleration")


class FmriRealign4dOutputSpec(TraitedSpec):

    out_file = OutputMultiPath(File(exists=True),
                               desc="Realigned files")
    par_file = OutputMultiPath(File(exists=True),
                               desc="Motion parameter files")


class FmriRealign4d(BaseInterface):
    """Simultaneous motion and slice timing correction algorithm

    This interface wraps nipy's FmriRealign4d algorithm [1]_.

    Examples
    --------
    >>> from nipype.interfaces.nipy.preprocess import FmriRealign4d
    >>> realigner = FmriRealign4d()
    >>> realigner.inputs.in_file = ['functional.nii']
    >>> realigner.inputs.tr = 2
    >>> realigner.inputs.slice_order = list(range(0,67))
    >>> res = realigner.run() # doctest: +SKIP

    References
    ----------
    .. [1] Roche A. A four-dimensional registration algorithm with \
       application to joint correction of motion and slice timing \
       in fMRI. IEEE Trans Med Imaging. 2011 Aug;30(8):1546-54. DOI_.

    .. _DOI: http://dx.doi.org/10.1109/TMI.2011.2131152

    """

    input_spec = FmriRealign4dInputSpec
    output_spec = FmriRealign4dOutputSpec
    keywords = ['slice timing', 'motion correction']

    def __init__(self, **inputs):
        DeprecationWarning(('Will be deprecated in release 0.13. Please use'
                            'SpaceTimeRealigner'))
        BaseInterface.__init__(self, **inputs)

    def _run_interface(self, runtime):
        from nipy.algorithms.registration import FmriRealign4d as FR4d
        all_ims = [load_image(fname) for fname in self.inputs.in_file]

        if not isdefined(self.inputs.tr_slices):
            TR_slices = None
        else:
            TR_slices = self.inputs.tr_slices

        R = FR4d(all_ims, tr=self.inputs.tr,
                 slice_order=self.inputs.slice_order,
                 tr_slices=TR_slices,
                 time_interp=self.inputs.time_interp,
                 start=self.inputs.start)

        R.estimate(loops=list(self.inputs.loops),
                   between_loops=list(self.inputs.between_loops),
                   speedup=list(self.inputs.speedup))

        corr_run = R.resample()
        self._out_file_path = []
        self._par_file_path = []

        for j, corr in enumerate(corr_run):
            self._out_file_path.append(os.path.abspath('corr_%s.nii.gz' %
                                                       (split_filename(self.inputs.in_file[j])[1])))
            save_image(corr, self._out_file_path[j])

            self._par_file_path.append(os.path.abspath('%s.par' %
                                                       (os.path.split(self.inputs.in_file[j])[1])))
            mfile = open(self._par_file_path[j], 'w')
            motion = R._transforms[j]
            # nipy does not encode euler angles. return in original form of
            # translation followed by rotation vector see:
            # http://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
            for i, mo in enumerate(motion):
                params = ['%.10f' % item for item in np.hstack((mo.translation,
                                                                mo.rotation))]
                string = ' '.join(params) + '\n'
                mfile.write(string)
            mfile.close()


    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._out_file_path
        outputs['par_file'] = self._par_file_path
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
    regressors_transform = traits.Str(
        desc = 'Which transform to apply to motion parameters')

    slicing_axis = traits.Int(
        2, usedefault = True, desc = 'Axis for outplane motion measure')

    prefix = traits.String('m', usedefault=True)

class RegressOutMotionOutputSpec(TraitedSpec):
    out_file = File(exists=True,
                    desc = 'File with regressed out motion parameters')
    beta_maps = File(#exists=True,
                     desc = 'File containing betas maps for each regressor.')
    
class RegressOutMotion(BaseInterface):
    input_spec = RegressOutMotionInputSpec
    output_spec = RegressOutMotionOutputSpec


    def _run_interface(self, runtime):
        try:
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
            outnii = nb.Nifti1Image(cdata,nii.get_affine(),nii.get_header().copy())
            outnii.set_data_dtype(np.float32)
            nb.save(outnii, self._list_outputs()['out_file'])
            
            betanii = nb.Nifti1Image(betamaps,nii.get_affine())
            nb.save(betanii, self._list_outputs()['beta_maps'])
        except Exception as e:
            print "RegressOutMotion failed", e
        finally:
            del nii, motion, cdata, betamaps, outnii, betanii, _
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

def mean_ts(tss):
    return tss.mean(axis=0)

def pca_ts(tss,numcomp=5):
    import scipy as sp
    u,s,v = sp.linalg.svd(tss, full_matrices=False)
    return v[:numcomp, :].T

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

    signal_masks_threshold = traits.Either(
        traits.Float(), traits.List(traits.Float()),
        0, usedefault=True,
        desc = 'Value to threshold mask images. Default is >0.')
    signal_estimate_function = traits.Function(
        mean_ts,
        usedefault = True,
        desc = """Function to estimate the signal from voxel timeseries.
Default is mean along the voxel dimension.""")
    prefix = traits.String('g', usedefault=True)

class RegressOutMaskSignalOutputSpec(TraitedSpec):
    out_file = File(exists=True,
                    desc = 'File with regressed out global signal')

    beta_maps = File(exists=True,
                     desc = 'Maps of beta values from regression.')
    regressors = File(exists=True,
                      desc = 'Timecourses used as regressors')
class RegressOutMaskSignal(BaseInterface):
    input_spec  = RegressOutMaskSignalInputSpec
    output_spec = RegressOutMaskSignalOutputSpec

    
    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.in_file)
        data = nii.get_data()
        mask = nb.load(self.inputs.mask).get_data()>0
        signal_masks_nii = [nb.load(m) for m in self.inputs.signal_masks]
        thr = self.inputs.signal_masks_threshold
        if isinstance(thr,list):
            signal_masks=[(m.get_data()>t)*mask for m,t in zip(signal_masks_nii,thr)]
        else:
            signal_masks = [(m.get_data()>thr)*mask for m in signal_masks_nii]

        m = np.isnan(data).sum(-1)*mask
        #correct for isolated nan values in mask timeseries due to realign
        # linearly interpolate in ts and extrapolate at ends of ts
        # TODO:optimize
        y = lambda z: z.nonzero()[0]
        if np.count_nonzero(m):
            for x,y,z in zip(m.nonzero()):
                nans = np.isnan(data[x,y,z])
                data[x,y,z] = np.interp(y(nans),y(~nans),data[x,y,z,~nans])
        nt=data.shape[-1]

        signals = np.squeeze(np.concatenate([self.inputs.signal_estimate_function(data[m])[...,np.newaxis] for m in signal_masks],1))
        #normalize
        signals = (signals-signals.mean(0))/signals.std(0)[np.newaxis]
        data = data[mask]

        reg_pinv = np.linalg.pinv(np.concatenate((signals,np.ones((nt,1))),
                                                 axis=1))
        betas = np.empty((data.shape[0],signals.shape[1]))
        for ti,ts in enumerate(data):
            betas[ti] = reg_pinv.dot(ts)[:-1]
            ts -= signals.dot(betas[ti])
        
        cdata = np.zeros(nii.shape)
        cdata[mask] = data

        outnii = nb.Nifti1Image(cdata,nii.get_affine(),nii.get_header().copy())
        outnii.set_data_dtype(np.float32)
        out_fname = self._list_outputs()['out_file']

        nb.save(outnii, out_fname)

        betamaps = np.empty(mask.shape+(betas.shape[1],),np.float32)
        betamaps.fill(np.nan)
        betamaps[mask] = betas
        betanii = nb.Nifti1Image(betamaps,nii.get_affine())
        nb.save(betanii, self._list_outputs()['beta_maps'])
        np.savetxt(self._list_outputs()['regressors'], signals)
        del nii, data, cdata, outnii, betas, betamaps, betanii
        return runtime
        
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = fname_presuffix(
            self.inputs.in_file,
            prefix = self.inputs.prefix,
            newpath = os.getcwd())
        outputs["beta_maps"] = fname_presuffix(
            self.inputs.in_file,
            prefix = self.inputs.prefix,
            suffix = '_betas',
            newpath = os.getcwd())
        outputs["regressors"] = fname_presuffix(
            self.inputs.in_file,
            prefix = self.inputs.prefix,
            suffix = '_regs.txt',
            newpath = os.getcwd(),
            use_ext=False)
        return outputs
    


class ScrubbingInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='The 4D run to be processed')
    mask = File(
        exists=True,
        mandatory=True,
        desc='The brain mask used to compute the framewise derivatives')

    motion = File(
        exists=True,
        mandatory=True,
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
        
        out_nii = nb.Nifti1Image(self.scrubbed, nii.get_affine(),
                                 nii.get_header().copy())
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

class SpaceTimeRealignerInputSpec(BaseInterfaceInputSpec):

    in_file = InputMultiPath(File(exists=True),
                             mandatory=True, min_ver='0.4.0.dev',
                             desc="File to realign")
    tr = traits.Float(desc="TR in seconds", requires=['slice_times'])
    slice_times = traits.Either(traits.List(traits.Float()),
                                traits.Enum('asc_alt_2', 'asc_alt_2_1',
                                            'asc_alt_half', 'asc_alt_siemens',
                                            'ascending', 'desc_alt_2',
                                            'desc_alt_half', 'descending'),
                                desc=('Actual slice acquisition times.'))
    slice_info = traits.Either(traits.Int,
                               traits.List(min_len=2, max_len=2),
                               desc=('Single integer or length 2 sequence '
                                     'If int, the axis in `images` that is the '
                                     'slice axis.  In a 4D image, this will '
                                     'often be axis = 2.  If a 2 sequence, then'
                                     ' elements are ``(slice_axis, '
                                     'slice_direction)``, where ``slice_axis`` '
                                     'is the slice axis in the image as above, '
                                     'and ``slice_direction`` is 1 if the '
                                     'slices were acquired slice 0 first, slice'
                                     ' -1 last, or -1 if acquired slice -1 '
                                     'first, slice 0 last.  If `slice_info` is '
                                     'an int, assume '
                                     '``slice_direction`` == 1.'),
                               requires=['slice_times'],
                               )


class SpaceTimeRealignerOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True),
                               desc="Realigned files")
    par_file = OutputMultiPath(File(exists=True),
                               desc=("Motion parameter files. Angles are not "
                                     "euler angles"))


class SpaceTimeRealigner(BaseInterface):
    """Simultaneous motion and slice timing correction algorithm

    If slice_times is not specified, this algorithm performs spatial motion
    correction

    This interface wraps nipy's SpaceTimeRealign algorithm [Roche2011]_ or simply the
    SpatialRealign algorithm when timing info is not provided.

    Examples
    --------
    >>> from nipype.interfaces.nipy import SpaceTimeRealigner
    >>> #Run spatial realignment only
    >>> realigner = SpaceTimeRealigner()
    >>> realigner.inputs.in_file = ['functional.nii']
    >>> res = realigner.run() # doctest: +SKIP

    >>> realigner = SpaceTimeRealigner()
    >>> realigner.inputs.in_file = ['functional.nii']
    >>> realigner.inputs.tr = 2
    >>> realigner.inputs.slice_times = list(range(0, 3, 67))
    >>> realigner.inputs.slice_info = 2
    >>> res = realigner.run() # doctest: +SKIP


    References
    ----------
    .. [Roche2011] Roche A. A four-dimensional registration algorithm with \
       application to joint correction of motion and slice timing \
       in fMRI. IEEE Trans Med Imaging. 2011 Aug;30(8):1546-54. DOI_.

    .. _DOI: http://dx.doi.org/10.1109/TMI.2011.2131152

    """

    input_spec = SpaceTimeRealignerInputSpec
    output_spec = SpaceTimeRealignerOutputSpec
    keywords = ['slice timing', 'motion correction']

    @property
    def version(self):
        return nipy_version

    def _run_interface(self, runtime):
        all_ims = [load_image(fname) for fname in self.inputs.in_file]

        if not isdefined(self.inputs.slice_times):
            from nipy.algorithms.registration.groupwise_registration import \
                SpaceRealign
            R = SpaceRealign(all_ims)
        else:
            from nipy.algorithms.registration import SpaceTimeRealign
            R = SpaceTimeRealign(all_ims,
                                 tr=self.inputs.tr,
                                 slice_times=self.inputs.slice_times,
                                 slice_info=self.inputs.slice_info,
                                 )

        R.estimate(refscan=None)

        corr_run = R.resample()
        self._out_file_path = []
        self._par_file_path = []

        for j, corr in enumerate(corr_run):
            self._out_file_path.append(os.path.abspath('corr_%s.nii.gz' %
                                                       (split_filename(self.inputs.in_file[j])[1])))
            save_image(corr, self._out_file_path[j])

            self._par_file_path.append(os.path.abspath('%s.par' %
                                                       (os.path.split(self.inputs.in_file[j])[1])))
            mfile = open(self._par_file_path[j], 'w')
            motion = R._transforms[j]
            # nipy does not encode euler angles. return in original form of
            # translation followed by rotation vector see:
            # http://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
            for i, mo in enumerate(motion):
                params = ['%.10f' % item for item in np.hstack((mo.translation,
                                                                mo.rotation))]
                string = ' '.join(params) + '\n'
                mfile.write(string)
            mfile.close()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._out_file_path
        outputs['par_file'] = self._par_file_path
        return outputs

class TrimInputSpec(NipyBaseInterfaceInputSpec):
    in_file = File(
        exists=True, mandatory=True,
        desc="EPI image to trim")
    begin_index = traits.Int(
        0, usedefault=True,
        desc='first volume')
    end_index = traits.Int(
        0, usedefault=True,
        desc='last volume indexed as in python (and 0 for last)')
    out_file = File('%s_trim', desc='output filename',
                    overload_extension=True,
                    name_source='in_file')

class TrimOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class Trim(NipyBaseInterface):
    """ Simple interface to trim a few volumes from a 4d fmri nifti file

    Examples
    --------
    >>> from nipype.interfaces.nipy.preprocess import Trim
    >>> trim = Trim()
    >>> trim.inputs.in_file = 'functional.nii'
    >>> trim.inputs.begin_index = 3 # remove 3 first volumes
    >>> res = trim.run() # doctest: +SKIP

    """

    input_spec = TrimInputSpec
    output_spec = TrimOutputSpec

    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.in_file)
        if self.inputs.end_index == 0:
            s = slice(self.inputs.begin_index, nii.shape[3])
        else:
            s = slice(self.inputs.begin_index, self.inputs.end_index)
        newdata = nii.get_data()[..., s]
        if len(newdata.shape)>3 and newdata.shape[-1]==1:
            newdata = newdata[...,0]
        nii2 = nb.Nifti1Image(
            newdata,
            nii.get_affine(),
            nii.get_header())
        out_file = self._list_outputs()['out_file']
        nb.save(nii2, out_file)
        return runtime


class CropInputSpec(NipyBaseInterfaceInputSpec):
    in_file = File(desc='input file', exists=True, mandatory=True,)
    out_file = File('%s_crop', desc='output file',
                    overload_extension=True,
                    name_source='in_file')

    x_min = traits.Int(0, usedefault=True)
    x_max = traits.Either(None,traits.Int, usedefault=True)
    y_min = traits.Int(0, usedefault=True)
    y_max = traits.Either(None,traits.Int, usedefault=True)
    z_min = traits.Int(0, usedefault=True)
    z_max = traits.Either(None,traits.Int, usedefault=True)

    padding = traits.Int(0,usedefault=True,
                         desc='add n voxels of padding in each direction',)

class CropOutputSpec(TraitedSpec):
    out_file = File(desc='output file')

class Crop(NipyBaseInterface):
    """ Simple interface to crop a volume using voxel space
    contrary to afni.Autobox or afni.Resample this keep the oblique matrices

    Examples
    --------
    >>> from nipype.interfaces.nipy.preprocess import Crop
    >>> crop = Crop(x_min=10,x_max=-10)
    >>> crop.inputs.in_file = 'anatomical.nii'
    >>> res = crop.run() # doctest: +SKIP

    """
    
    input_spec = CropInputSpec
    output_spec = CropOutputSpec
    
    def _run_interface(self, runtime):
        in_file = nb.load(self.inputs.in_file)
        mat = in_file.get_affine().copy()
        pad = self.inputs.padding
        x_max,y_max,z_max=self.inputs.x_max,self.inputs.y_max,self.inputs.z_max
        if pad!=0:
            if x_max > 0: x_max = min(x_max+pad,in_file.shape[0]-1)
            else : x_max = min(x_max+pad,0)
            if x_max == 0 : x_max = None
            if y_max > 0: y_max = min(y_max+pad,in_file.shape[1]-1)
            else : y_max = min(y_max+pad,0)
            if y_max == 0 : y_max = None
            if z_max > 0: z_max = min(z_max+pad,in_file.shape[2]-1)
            else : z_max = min(z_max+pad,0)
            if z_max == 0 : z_max = None
        slices = [slice(max(self.inputs.x_min-pad,0),x_max),
                  slice(max(self.inputs.y_min-pad,0),y_max),
                  slice(max(self.inputs.z_min-pad,0),z_max),]
        data = in_file.get_data()[slices]
        orig_coords = mat.dot([s.start for s in slices]+[1]).ravel()[:3]
        mat[:3,3] = orig_coords
        out_file = nb.Nifti1Image(data,mat)
        out_file.set_data_dtype(in_file.get_data_dtype())
        out_filename = self._list_outputs()['out_file']
        nb.save(out_file,out_filename)
        return runtime    
