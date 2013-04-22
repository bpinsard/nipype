"""
    Change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
    >>> datadir = os.path.realpath(os.path.join(filepath, '../../testing/data'))
    >>> os.chdir(datadir)

"""
import warnings

import nibabel as nb
import os

from ...utils.misc import package_check
from ...utils.filemanip import split_filename, fname_presuffix

try:
    package_check('nipy')
except Exception, e:
    warnings.warn('nipy not installed')
else:
    from nipy.algorithms.registration.histogram_registration import HistogramRegistration
    from nipy.algorithms.registration.affine import Affine

from ..base import (TraitedSpec, BaseInterface, traits,
                    BaseInterfaceInputSpec, File, isdefined)


class SimilarityInputSpec(BaseInterfaceInputSpec):

    volume1 = File(exists=True, desc="3D volume", mandatory=True)
    volume2 = File(exists=True, desc="3D volume", mandatory=True)
    mask1 = File(exists=True, desc="3D volume")
    mask2 = File(exists=True, desc="3D volume")
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
            mask1 = nb.load(self.inputs.mask1).get_data() == 1
        else:
            mask1 = None

        if isdefined(self.inputs.mask2):
            mask2 = nb.load(self.inputs.mask2).get_data() == 1
        else:
            mask2 = None

        histreg = HistogramRegistration(from_img = vol1_nii,
                                        to_img = vol2_nii,
                                        similarity=self.inputs.metric,
                                        from_mask = mask1,
                                        to_mask = mask2)
        self._similarity = histreg.eval(Affine())

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['similarity'] = self._similarity
        return outputs


class ROIsSlicerInputSpec(BaseInterfaceInputSpec):

    in_file = traits.File(desc='background file',exists=True,mandatory=True)
    
    overlays = traits.List(
        traits.Tuple(traits.File(exists=True),traits.File(exists=True)),
        desc='list of tuple of ROIs and label files')
    
    slices = traits.Tuple(
        traits.Either(None,traits.Int),
        traits.Either(None,traits.Int),
        traits.Either(None,traits.Int),
        desc = 'slices to extract')
    
class ROIsSlicerOutputSpec(TraitedSpec):

    out_file = traits.File(desc='slices image file')


class ROIsSlicer(BaseInterface):

    input_spec = ROIsSlicerInputSpec
    output_spec = ROIsSlicerOutputSpec

    
    def _run_interface(self, runtime):
        import matplotlib as mpl
        import numpy as np
        back = nb.load(self.inputs.in_file)
        sl = slice(*self.inputs.slices[:3])
        bg = back.get_data()[:,:,sl]

        nslices = bg.shape[-1]
        per_row = int(np.ceil(np.sqrt(nslices)))
        n_rows = nslices/per_row
        shape = bg.shape
        bg_flat = np.zeros((n_rows*shape[1],per_row*shape[0]))
        for r in range(n_rows):
            for c in range(per_row):
                if r*per_row+c < nslices:
                    bg_flat[r*shape[1]:(r+1)*shape[1],
                            c*shape[0]:(c+1)*shape[0]]=bg[:,::-1,r*per_row+c].T
        fig = mpl.figure.Figure(frameon=False)
        ax = fig.add_subplot(111,frame_on=False)
        ax.set_axis_off()
        ax.imshow(bg_flat, interpolation='nearest', cmap=mpl.cm.gray)
        rois_flat = np.empty(bg_flat.shape)
        rois_flat.fill(np.nan)
        idx = 0
        rois = np.empty(bg.shape)
        rois.fill(np.nan)
        for niif,labels in self.inputs.overlays:
            rois_nii = nb.load(niif)
            nrois = np.nanmax(rois_nii.get_data())
            mask = rois_nii.get_data()[:,:,sl]>0
            rois[mask] = rois_nii.get_data()[:,:,sl][mask] + idx
            idx += nrois
        for r in range(n_rows):
            for c in range(per_row):
                if r*per_row+c < nslices:
                    rois_flat[
                        r*shape[1]:(r+1)*shape[1],
                        c*shape[0]:(c+1)*shape[0]] = rois[:,::-1,r*per_row+c].T
        ax.imshow(rois_flat,interpolation='nearest',cmap=mpl.cm.hsv,alpha=.8)
        canvas = mpl.backends.backend_agg.FigureCanvasAgg(fig)
        extent = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted())
        canvas.print_figure(self._list_outputs()['out_file'],
                            dpi=80,bbox_inches=extent)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = fname_presuffix(
            self.inputs.in_file,
            newpath=os.getcwd(),
            suffix='.png',
            use_ext=False)
        return outputs
