"""
    Change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
    >>> datadir = os.path.realpath(os.path.join(filepath, '../../testing/data'))
    >>> os.chdir(datadir)

"""
import warnings

import nibabel as nb
import numpy as np
from ...utils.misc import package_check

try:
    package_check('nipy')
except Exception, e:
    warnings.warn('nipy not installed')
else:
    from nipy.algorithms.registration.histogram_registration import HistogramRegistration
    import nipy.algorithms.utils.preprocess as preproc
    from nipy.algorithms.registration.affine import Affine, to_matrix44

from ..base import (TraitedSpec, BaseInterface, traits,
                    BaseInterfaceInputSpec, File, isdefined)


class SimilarityInputSpec(BaseInterfaceInputSpec):

    volume1 = File(exists=True, desc="3D volume", mandatory=True)
    volume2 = File(exists=True, desc="3D volume", mandatory=True)
    mask1 = File(exists=True, desc="3D volume", mandatory=True)
    mask2 = File(exists=True, desc="3D volume", mandatory=True)
    metric = traits.Either(traits.Enum('cc', 'cr', 'crl1', 'mi', 'nmi', 'slr'),
                          traits.Callable(),
                         desc="""str or callable
Cost-function for assessing image similarity. If a string,
one of 'cc': correlation coefficient, 'cr': correlation
ratio, 'crl1': L1-norm based correlation ratio, 'mi': mutual
information, 'nmi': normalized mutual information, 'slr':
supervised log-likelihood ratio. If a callable, it should
take a two-dimensional array representing the image joint
histogram as an input and return a float.""", usedefault=True)


class SimilarityOutputSpec(TraitedSpec):

    similarity = traits.Float(desc="Similarity between volume 1 and 2")


class Similarity(BaseInterface):
    """Calculates similarity between two 3D volumes. Both volumes have to be in
    the same coordinate system, same space within that coordinate system and
    with the same voxel dimensions.

    Example
    -------
    >>> from nipype.interfaces.nipy.utils import Similarity
    >>> similarity = Similarity()
    >>> similarity.inputs.volume1 = 'rc1s1.nii'
    >>> similarity.inputs.volume2 = 'rc1s2.nii'
    >>> similarity.inputs.mask1 = 'mask.nii'
    >>> similarity.inputs.mask2 = 'mask.nii'
    >>> similarity.inputs.metric = 'cr'
    >>> res = similarity.run() # doctest: +SKIP
    """

    input_spec = SimilarityInputSpec
    output_spec = SimilarityOutputSpec

    def _run_interface(self, runtime):

        vol1_nii = nb.load(self.inputs.volume1)
        vol2_nii = nb.load(self.inputs.volume2)

        if isdefined(self.inputs.mask1):
            mask1_nii = nb.load(self.inputs.mask1)
            mask1_nii = nb.Nifti1Image(nb.load(self.inputs.mask1).get_data() == 1, mask1_nii.get_affine(),
                                       mask1_nii.get_header())
        else:
            mask1_nii = None

        if isdefined(self.inputs.mask2):
            mask2_nii = nb.load(self.inputs.mask2)
            mask2_nii = nb.Nifti1Image(nb.load(self.inputs.mask2).get_data() == 1, mask2_nii.get_affine(),
                                       mask2_nii.get_header())
        else:
            mask2_nii = None

        histreg = HistogramRegistration(from_img = vol1_nii,
                                        to_img = vol2_nii,
                                        similarity=self.inputs.metric,
                                        from_mask = mask1_nii,
                                        to_mask = mask2_nii)
        self._similarity = histreg.eval(Affine())

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['similarity'] = self._similarity
        return outputs



class MotionCorrectionEvaluationInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(
        mandatory = True,
        desc = 'motion corrected file')
    mask = traits.File(desc='brain mask')
    motion = traits.File(
        mandatory = True,
        desc = 'motion parameters file')
    motion_source = traits.Enum(
        'spm', 'fsl', 'slice_motion', 'afni',
        mandatory = True,
        desc = 'software that produced motion estimates')

class MotionCorrectionEvaluationOutputSpec(TraitedSpec):
    pass 

class MotionCorrectionEvaluation(BaseInterface):

    
    input_spec = MotionCorrectionEvaluationInputSpec
    output_spec = MotionCorrectionEvaluationOutputSpec

    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.in_file)
        affine = nii.get_affine()
        if isdefined(self.inputs.mask):
            mask = nb.load(self.inputs.mask).get_data() > 0
        else:
            mask = np.ones(nii.shape[:3],dtype=np.bool)
        motion = np.loadtxt(self.inputs.motion)
        if self.inputs.motion_source == 'slice_motion':
            motion[:,3:6] *= 0.01
        motion[:] = preproc.motion_parameter_standardize(
            motion, self.inputs.motion_source)
        indices = np.empty((3,np.count_nonzero(mask)),dtype=np.int32)
        indices[:] = np.nonzero(mask)
        motion_mats = np.array([to_matrix44(m) for m in motion])
        voxels_motion=np.empty((nii.shape[-1],indices.shape[-1],3),np.float32)
        for t,mat in enumerate(motion_mats):
            voxels_motion[t]=nb.affines.apply_affine(mat.dot(affine),indices.T)
        
        voxels_motion_diff = np.diff(voxels_motion, axis=0)
        voxels_motion_diff_drms = np.sqrt((voxels_motion_diff**2).sum(-1))
        data_diff = np.diff(nii.get_data()[mask].reshape(-1,nii.shape[-1]),1,-1)
        self.voxels_motion = voxels_motion
        self.voxels_motion_diff=voxels_motion_diff
        self.data_diff=data_diff
        
        
        

        #del voxels_motion, voxels_motion_diff, motion, mask, nii, data_diff
        return runtime


    
