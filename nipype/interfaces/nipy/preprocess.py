"""
    Change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
    >>> datadir = os.path.realpath(os.path.join(filepath, '../../testing/data'))
    >>> os.chdir(datadir)

"""
import os
import warnings

import nibabel as nb
import numpy as np

from ...utils.misc import package_check
from ...utils.filemanip import split_filename, fname_presuffix


try:
    package_check('nipy')
except Exception, e:
    warnings.warn('nipy not installed')
else:
    from nipy.labs.mask import compute_mask
    from nipy.algorithms.registration import FmriRealign4d as FR4d
    from nipy.algorithms.registration.affine import to_matrix44
    import nipy.algorithms.registration.slice_motion as sm
    from nipy import save_image, load_image

from ..base import (TraitedSpec, BaseInterface, traits,
                    BaseInterfaceInputSpec, isdefined, File,
                    InputMultiPath, OutputMultiPath)


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

        args = {}
        for key in [k for k, _ in self.inputs.items()
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
        nb.save(nb.Nifti1Image(brain_mask.astype(np.uint8),
                nii.get_affine()), self._brain_mask_path)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["brain_mask"] = self._brain_mask_path
        return outputs


class FmriRealign4dInputSpec(BaseInterfaceInputSpec):

    in_file = InputMultiPath(exists=True,
                             mandatory=True,
                             desc="File to realign")
    tr = traits.Float(desc="TR in seconds",
                      mandatory=True)
    slice_order = traits.List(traits.Int(), maxver=0.3,
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
    >>> realigner.inputs.slice_order = range(0,67)
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

    def _run_interface(self, runtime):

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

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._out_file_path
        outputs['par_file'] = self._par_file_path
        return outputs


class TrimInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True, mandatory=True,
        desc="EPI image to trim")
    begin_index = traits.Int(
        0, usedefault=True,
        desc='first volume')
    end_index = traits.Int(
        0, usedefault=True,
        desc='last volume indexed as in python (and 0 for last)')
    out_file = File(desc='output filename')
    suffix = traits.Str(
        '_trim', usedefault=True,
        desc='suffix for out_file to use if no out_file provided')


class TrimOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class Trim(BaseInterface):
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
        out_file = self._list_outputs()['out_file']
        nii = nb.load(self.inputs.in_file)
        if self.inputs.end_index == 0:
            s = slice(self.inputs.begin_index, nii.shape[3])
        else:
            s = slice(self.inputs.begin_index, self.inputs.end_index)
        nii2 = nb.Nifti1Image(
            nii.get_data()[..., s],
            nii.get_affine(),
            nii.get_header())
        nb.save(nii2, out_file)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']):
            outputs['out_file'] = fname_presuffix(
                self.inputs.in_file,
                newpath=os.getcwd(),
                suffix=self.inputs.suffix)
        outputs['out_file'] = os.path.abspath(outputs['out_file'])
        return outputs

class SliceInterpolationBaseInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='fmri run file')
    fieldmap_file = File(
        exists=True,
        desc='fieldmap file coregistered to t1 space for concurrent unwarping')
    mask_file = File(
        exists=True,
        desc='brain mask')    

    # epi acquisition parameters
    epi_echo_spacing = traits.Float(
        0.0006, usedefault=True,
        desc='effective echo spacing or dwelling time of the epi acquisition in sec, TODO: find it in in_file if dcmstack...')

    epi_echo_time = traits.Float(
        0.03, usedefault=True,
        desc='echo time of the epi acquisition in sec')

    epi_tr = traits.Float(desc='Repetition time')

    epi_slicing_axis = traits.Range(
        0, 2, 2,
        usedefault=True, mandatory=True,
        desc='slicing axis in epi file space')
    epi_phase_direction = traits.Range(-3,3,1,usedefault=True,
        desc='specifies direction of warping (default 1)')

    epi_slice_order = traits.Either(
        traits.List(traits.Int()), traits.Enum('ascending','descending'),
        desc='slice acquisition order')


class SliceMotionCorrectionInputSpec(SliceInterpolationBaseInputSpec):
    
    white_matter_file = File(
        exists=True,
        mandatory=True,
        desc='white matter segmentation surface or volume file( will be thresholded at 0.5)')
    exclude_points_mask  = File(
        exists=True,
        desc='mask to exclude region to be sampled, as for trunk due to pulsatility')
    surface_ref = traits.File(
        desc='volume defining surface space (eg. freesurfer volume)')

    #algorithm parameters
    strategy = traits.Enum(
        'volume','intensity_heuristic',
        usedefault=True,
        desc = 'how to split the run for motion estimate')
    nsamples_first_frame = traits.Int(
         20000,
         usedefault=True,
         desc='number of sample per slice for first frame realignement')
    nsamples_per_slicegroup = traits.Int(
         5000,
         usedefault = True,
         desc='number of samples per slice for whole run realignement')
    subsampling = traits.Int(
        1, usedefault=True,
        desc='subsample points on the surface or volume (stupid way)')
    surface_sample_distance = traits.Float(
        1.5, usedefault=True,
        desc='distance from surface to sample points')
    ftol = traits.Float(1e-4, usedefault=True, desc='optimizer tolerance')

    # options for excluding sample points using estimated sigloss
    sigloss_threshold = traits.Float(
        0,usedefault=True,
        desc="""exclude sampling points with signal lower than threshold""")

    #resampled output parameters
    output_voxel_size = traits.Tuple(*([traits.Float()]*3),
        desc = 'resample realigned file at a specific voxel size')
    out_file=File(
        '%s_smc', usedefault=True, name_source='in_file',
        desc='motion corrected resampled output filename')
    motions_parameters=File(
        '%s_params.txt', mandatory=True, usedefault=True,name_source='in_file',
        desc='motion parameters output filename')
    matrices=File(
        '%s_mats.npy', usedefault=True, name_source='in_file',
        desc='motion parameters output filename')
    resampling_ref = traits.File(
        desc='volume defining resampling space')

class SliceMotionCorrectionOutputSpec(TraitedSpec):
    out_file = File(exists=True)
    motion_parameters = File(exists=True)
    matrices = File(exists=True)
    slice_groups = traits.List(
        traits.Tuple(*([traits.Tuple(*([traits.Int(min=0)]*2))]*2)),
        desc='slice groups : list of ((#frame,#slice),(#frame,#slice))')

    all_data = File(exists=True)
    coords = File(exists=True)
class SliceMotionCorrection(BaseInterface):
    
    input_spec = SliceMotionCorrectionInputSpec
    output_spec = SliceMotionCorrectionOutputSpec

    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.in_file)
        data = nii.get_data()
        mask, surf_ref, fmap = None, None, None
        if isdefined(self.inputs.mask_file):
            mask = nb.load(self.inputs.mask_file)
            surf_ref = mask
        if isdefined(self.inputs.surface_ref):
            surf_ref = nb.load(self.inputs.surface_ref)
        if isdefined(self.inputs.fieldmap_file):
            fmap = nb.load(self.inputs.fieldmap_file)

        try:
            wm = nb.load(self.inputs.white_matter_file)
            exclude_mask=None
            if isdefined(self.inputs.exclude_points_mask):
                exclude_mask = nb.load(self.inputs.exclude_points_mask)
            bnd_coords,class_coords = sm.extract_boundaries(
                wm, self.inputs.surface_sample_distance,
                subsample=self.inputs.subsampling,exclude=exclude_mask,
                threshold=.5,margin=.25)
        except:
            from nibabel.freesurfer import read_geometry
            wm_coords,wm_faces = read_geometry(self.inputs.white_matter_file)
            ras2vox=np.array([[-1,0,0,128],[0,0,-1,128],[0,1,0,128],[0,0,0,1]])
            wm_coords = nb.affines.apply_affine(
                surf_ref.get_affine().dot(ras2vox),wm_coords)
            class_coords = sm.surface_to_samples(
                wm_coords,wm_faces,
                self.inputs.surface_sample_distance)
            if self.inputs.subsampling > 1:
                wm_coords = wm_coords[::self.inputs.subsampling]
                class_coords = class_coords[:,::self.inputs.subsampling]
            
        tr = self.inputs.epi_tr
        echo_time = self.inputs.epi_echo_time
        if not isdefined(tr):
            tr = nii.get_header().get_zooms()[3]
            if nii.get_header().get_xyzt_units()[-1] == 'msec':
                tr *= 1e-3

        self.first_frame_alg = sm.RealignSliceAlgorithm(
            nii,wm_coords,class_coords,
            [((0,0),(0,nii.shape[self.inputs.slicing_axis]))],None,fmap,mask,
            phase_encoding_dir = self.inputs.epi_phase_direction,
            repetition_time=tr, 
            slice_order = self.inputs.epi_slice_order,
            echo_spacing = self.inputs.epi_echo_spacing,
            echo_time = echo_time, ftol=self.inputs.ftol,
            nsamples_per_slicegroup=self.inputs.nsamples_first_frame)
        self.first_frame_alg.estimate_motion()


        if self.inputs.sigloss_threshold>0:
            reg = self.first_frame_alg.transforms[0].as_affine()
            sigloss = sm.compute_sigloss(
                fmap, mask,reg, nii.get_affine(), wm_coords,
                self.inputs.epi_echo_time, self.inputs.epi_slicing_axis)
            sigloss_subset = sigloss > self.inputs.sigloss_threshold
            print '%f percent of voxels kept after sigloss'%(np.count_nonzero(sigloss_subset)/float(sigloss_subset.size))
            wm_coords = wm_coords[sigloss_subset]
            class_coords = class_coords[:,sigloss_subset]

        if self.inputs.strategy=='volume':
            self.whole_run_alg = sm.RealignSliceAlgorithm(
                nii,wm_coords,class_coords,
                None, self.first_frame_alg.transforms,fmap,mask,
                phase_encoding_dir = self.inputs.epi_phase_direction,
                repetition_time=tr, 
                slice_order = self.inputs.epi_slice_order,
                echo_spacing = self.inputs.epi_echo_spacing,
                echo_time = echo_time, ftol=self.inputs.ftol,
                nsamples_per_slicegroup=self.inputs.nsamples_per_slicegroup)
#        else:
            #sm.
        del self.first_frame_alg
        self.whole_run_alg.estimate_motion()
        self._slice_groups = self.whole_run_alg.slice_groups

        outputs = self._list_outputs()

        params = np.array([t.param*t.precond[:6] for t in self.whole_run_alg.transforms])
        np.savetxt(outputs['motion_parameters'],params)
        if isdefined(outputs['matrices']):
            matrices = np.array([t.as_affine() \
                                     for t in self.whole_run_alg.transforms])
            np.save(outputs['matrices'],matrices)

        
        if isdefined(outputs['out_file']):
            output_voxel_size = self.inputs.output_voxel_size
            resamp_ref = None
            if isdefined(self.inputs.resampling_ref):
                resamp_ref = nb.load(self.inputs.resampling_ref)
            realigned = self.whole_run_alg.resample_full_data(
                voxsize=self.inputs.output_voxel_size,reference=resamp_ref)
#        realigned = self.first_frame_alg.resample_full_data(
#            voxsize=self.inputs.output_voxel_size,reference=resamp_ref)

            realigned.to_filename(outputs['out_file'])
        np.save(outputs['coords'],np.concatenate(
                (self.whole_run_alg.bnd_coords[np.newaxis],
                 self.whole_run_alg.class_coords),0))
        np.save(outputs['all_data'],self.whole_run_alg._all_data)
#        params = np.array([t.param*t.precond[:6] for t in self.first_frame_alg.transforms])
        del realigned
        del self.whole_run_alg
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']):
            outputs['out_file'] = fname_presuffix(
                self.inputs.in_file,
                newpath=os.getcwd(),
                suffix=self.inputs.suffix)
        outputs['out_file'] = os.path.abspath(outputs['out_file'])
        outputs['motion_parameters'] = os.path.abspath(self.inputs.out_parameters_file)
        if not isdefined(outputs['motion_parameters']):
            outputs['motion_parameters'] = fname_presuffix(
                self.inputs.in_file,
                newpath=os.getcwd(),
                use_ext = False,
                suffix=self.inputs.suffix + '.txt')
        outputs['coords'] = fname_presuffix(
            self.inputs.in_file,
            newpath=os.getcwd(),
            use_ext = False,
            suffix='_coords.npy')
        outputs['all_data'] = fname_presuffix(
            self.inputs.in_file,
            newpath=os.getcwd(),
            use_ext = False,
            suffix='_costs.npy')
        if hasattr(self,'_slice_groups'):
            outputs['slice_groups'] = self._slice_groups
        return outputs


class SiglossInputSpec(BaseInterfaceInputSpec):
    
    fieldmap_file = File(
        desc='preprocessed fieldmap file',
        exists=True, mandatory=True)
    mask_file = File(
        exists=True, mandatory=True,
        desc='mask for fieldmap')
    echo_time = traits.Float(desc='echo time in sec', mandatory=True)
    epi_file = File(
        exists=True,
        desc='file defining epi acquisition space')
    epi_registration = File(
        exists=True,
        desc='a registration matrix between epi and fieldmap')
    epi_slicing_axis = traits.Range(
        0, 2, 2,
        usedefault=True, mandatory=True,
        desc='slicing axis in epi file space')
    out_file = File(
        '%s_sigloss', usedefault=True, name_source='fieldmap_file',
        desc='output file name')

class SiglossOutputSpec(TraitedSpec):
    out_file = File(desc='signal loss file')

class Sigloss(BaseInterface):

    """
    compute EPI signal loss using a fieldmap which can be in another sampling
    and orientation than the original EPI file.
    There is still some border problem, sigloss not being estimated on the 
    border of the mask, should extrapolate
    """

    input_spec = SiglossInputSpec
    output_spec = SiglossOutputSpec


    def _run_interface(self,runtime):
        fmap = nb.load(self.inputs.fieldmap_file)
        mask = nb.load(self.inputs.mask_file).get_data()>0
        reg, epimat = np.eye(4),np.eye(4)
        if isdefined(self.inputs.epi_registration):
            rparams = np.loadtxt(self.inputs.epi_registration)
            if rparams.shape == (4,4): #regular flirt matrix
                regp[:] = rparams
            if rparams.shape[-1]==6: #motion parameters
                from nipy.algorithms.registration.affine import to_matrix44
                reg = to_matrix44(rparams[0])
        if isdefined(self.inputs.epi_file):
            epimat = nb.load(self.inputs.epi_file).get_affine()
        pts = nb.apply_affine(fmap.get_affine(),np.array(np.where(mask)).T)
        sigloss = np.zeros(fmap.shape, fmap.get_data_dtype())
        sigloss[mask] = compute_sigloss(
            fmap, mask, reg, epimat, pts,
            self.inputs.echo_time, self.inputs.slicing_axis, order=0)
        out_fname = self._list_outputs()['out_file']
        nb.save(nb.Nifti1Image(sigloss,fmap.get_affine()),out_fname)
        return runtime


class ExtractCiftiInputSpec(SliceInterpolationBaseInputSpec):
    slice_groups = traits.List(
        traits.Tuple(*([traits.Tuple(*([traits.Int(min=0)]*2))]*2)),
        desc="""slice groups : list of ((#frame,#slice),(#frame,#slice)),
                if not provided volumes will be used or 
                if single registration provided same transform will be applied
                for all""")
    motion_parameters = File(
        exists=True,
        desc='motion parameters in world space as text file')
    registration = File(
        exists=True,
        desc="""single registration matrix in text file""")
    surfaces = traits.List(
        traits.Tuple(traits.Str,traits.File(exists=True),
                     traits.File(exists=True)), 
        desc='label,surface,thickness(opt) tuples')
    rois = traits.List(
        traits.Tuple(File(),File(),traits.List(traits.Int)), exists=True,
        desc="""triplets of volume/label/ids rois from which to extract signal
                if ids is [], all rois will be sampled""")
    surface_ref = traits.File(
        desc='volume defining surface space (eg. freesurfer volume)')
    projection_fraction = traits.Float(
        .5,usedefault=True,
         desc='sampling at fraction of cortical sheet using thickness')
    interpolation = traits.Range(
        0,5,3, usedefault=True,
        desc='level of interpolation: see scipy map_coordinate')
    out_file = File(
        '%s_ts', name_source='in_file',
        desc='sampled signal output file, cifti? hdf5?')
    volume_voxel_size = traits.Tuple(*([traits.Float()]*3),
        desc = 'voxel size of volume rois')

class ExtractCiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True)

class ExtractCifti(BaseInterface):

    input_spec = ExtractCiftiInputSpec
    output_spec = ExtractCiftiOutputSpec
    
    def _run_interface(self,runtime):
        epi = nb.load(self.inputs.in_file)        
        
        transforms=[]
        reg = np.eye(4) #identity
        if isdefined(self.inputs.registration):
            reg = np.loadtxt(self.inputs.registration)
        if isdefined(self.inputs.motion_parameters):
            motion = np.loadtxt(self.inputs.motion_parameters)
            for p in motion:
                transforms.append(sm.Rigid(reg.dot(to_matrix44(t))))
        else:
            transforms.append(sm.Rigid(reg))

        slice_groups = self.inputs.slice_groups
        if not isdefined(slice_groups) or len(slice_groups)==0:
            nslices = epi.shape[self.inputs.epi_slicing_axis]
            if len(transforms)==epi.shape[-1]:
                slice_groups=[((t,0),(t,)) for t in epi.shape[-1]]
            else:
                slice_groups=[((0,0),(epi.shape[-1],nslices))]

        if isdefined(self.inputs.mask_file):
            mask = nb.load(self.inputs.mask_file)
        if isdefined(self.inputs.fieldmap_file):
            fieldmap = nb.load(self.inputs.fieldmap_file)

        tr = self.inputs.epi_tr
        if not isdefined(tr):
            tr = epi.get_header().get_zooms()[3]
            if epi.get_header().get_xyzt_units()[-1] == 'msec':
                tr *= 1e-3

        itpl = sm.EPIInterpolation(
            epi, slice_groups, transforms, fieldmap, mask,
            phase_encoding_dir = self.inputs.epi_phase_direction,
            repetition_time = tr,
            echo_time = self.inputs.epi_echo_time, 
            echo_spacing = self.inputs.epi_echo_spacing,
            slice_order = self.inputs.epi_slice_order,
            slice_axis = self.inputs.epi_slicing_axis)

        surfaces,rois = self._extract_coords()

        outputs = self._list_outputs()
        nvols = epi.shape[-1]
        import h5py
        f = h5py.File(outputs['out_file'], "w")
        surf_group = f.create_group('surfaces')
        proj_frac = self.inputs.projection_fraction
        for slabel, verts, tri, thick in surfaces.items():
            dset = surf_group.create_dataset(
                slabel, itpl.resample_surface(verts, tri, thick, proj_frac))
        rois_group = f.create_group('rois')
        for roi_lbl, coords in rois:
            rois_group.create_dataset(
                roi_lbl, itpl.resample_coords(coords))
        f.close()
        return runtime


    def _extract_coords(self):
        surfaces, rois = dict(), dict()
        ras2vox = np.array([[-1,0,0,128],[0,0,-1,128],[0,1,0,128],[0,0,0,1]])
        surf_ref = nb.load(self.inputs.surface_ref)
        surf2wld = surf_ref.get_affine().dot(ras2vox)
        del surf_ref
        for surf_lab,surf_file,thick_file in self.inputs.surfaces:
            verts, tri = nb.freesurfer.read_geometry(surf_file)
            thick = nb.freesurfer.read_morph_data(thick_file)
            verts[:] = nb.affines.apply_affine(surf2wld,verts)
            surfaces[surf_lab] = (verts,tri,thick)
        if isdefined(self.inputs.rois):
            for vf,lf,ids in self.inputs.rois:
                rois_file = nb.load(vf)
                # suppose freesurfer labels file
                clut = np.loadtxt(
                    lf,np.object,comments='#',
                    converters={0:int,1:str,2:int,3:int,4:int,5:int})
                clut = dict([(c[0],c[1:]) for c in clut])
                founds_ids = np.unique(rois_file.get_data())
                founds_ids = founds_ids[founds_ids > 0]
                for i in ids:
                    if i in founds_ids and i in clut.keys():
                        coords = nb.apply_affine(
                            rois_file.get_affine(),
                            np.array(np.where(rois_file.get_data()==i)).T)
                        rois[clut[i]] = coords
                del rois_file, clut, founds_ids
        return surfaces, rois

    
