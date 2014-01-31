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
import dicom
import numpy as np

import glob

from ...utils.misc import package_check
from ...utils.filemanip import (
    split_filename, fname_presuffix, filename_to_list)


have_nipy = True
try:
    package_check('nipy')
except Exception, e:
    have_nipy = False
else:
    import nipy
    from nipy import save_image, load_image
    nipy_version = nipy.__version__
    from nipy.algorithms.registration.online_preproc import (
        EPIOnlineRealign, EPIOnlineRealignFilter,
        surface_to_samples, NiftiIterator)
    

try:
    package_check('h5py')
except Exception, e:
    warnings.warn('h5py not installed')
else:
    import h5py

try:
    package_check('dcmstack')
except Exception, e:
    warnings.warn('dcmstack not installed')
else:
    from dcmstack.dcmstack import DicomStackOnline

from ..base import (TraitedSpec, BaseInterface, traits,
                    BaseInterfaceInputSpec, isdefined, File, Directory,
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
        from nipy.labs.mask import compute_mask
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

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._out_file_path
        outputs['par_file'] = self._par_file_path
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

    This interface wraps nipy's SpaceTimeRealign algorithm [1]_ or simply the
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
    >>> realigner.inputs.slice_times = range(0, 3, 67)
    >>> realigner.inputs.slice_info = 2
    >>> res = realigner.run() # doctest: +SKIP


    References
    ----------
    .. [1] Roche A. A four-dimensional registration algorithm with \
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

class OnlinePreprocInputSpecBase(BaseInterfaceInputSpec):
    dicom_files = traits.Either(
        InputMultiPath(File(exists=True)),
        InputMultiPath(Directory(exists=True)),
        InputMultiPath(traits.Str()), #allow glob file pattern
        mandatory=True,
        xor=['nifti_file'])
    
    nifti_file = File(exists=True,
                       mandatory=True,
                       xor=['dicom_files'])
    
    # Fieldmap parameters
    fieldmap = File(
        desc = 'precomputed fieldmap coregistered with reference space')
    fieldmap_reg = File(
        desc = 'fieldmap coregistration matrix')
    
    # space definition
    mask = File()
    surfaces_volume_reference = traits.File(
        mandatory = True,
        exists = True,
        desc='a volume defining space of surfaces')
    
    #EPI parameters
    phase_encoding_dir = traits.Range(-3,3, desc='phase encoding direction')
    repetition_time = traits.Float(desc='TR in secs.')
#    slice_repetition_time = traits.Float(desc='slice TR in secs'
    echo_time = traits.Float(desc='TE in secs.')
    echo_spacing = traits.Float(desc='effective echo spacing in secs.')
    slice_order = traits.List(traits.Int(), desc='slice order'),
    interleaved = traits.Int(),
    slice_trigger_times = traits.List(traits.Float()),
    slice_thickness = traits.Float(),
    slice_axis = traits.Range(0,2),

    # resampling objects
    resample_surfaces = traits.List(
        traits.Tuple(traits.Str, traits.Either(File,traits.Tuple(File,File))),
        desc='freesurfer surface files from which signal to be extracted')
    resample_rois = traits.List(
        traits.Tuple(traits.Str, File, File),
        desc = 'list of rois NIFTI files from which to extract signal and labels file')
    
    store_coords = traits.Bool(
        True, usedefault=True,
        desc='store surface and ROIs coordinates in output')

class OnlinePreprocBase(BaseInterface):
    def _list_files(self):
        # list files depending on input type
        df = filename_to_list(self.inputs.dicom_files)
        self.dicom_files = []
        for p in df:
            if os.path.isfile(p):
                self.dicom_files.append(p)
            elif os.path.isdir(p):
                self.dicom_files.extend(sorted(glob.glob(
                        os.path.join(p,'*.dcm'))))
            elif isinstance(p,str):
                self.dicom_files.extend(sorted(glob.glob(p)))

    ras2vox = np.array([[-1,0,0,128],[0,0,-1,128],[0,1,0,128],[0,0,0,1]])

    def _init_ts_file(self):
        out_file = h5py.File(self._list_outputs()['out_file'])
        
        surfaces = []
        rois = []        
        surf_ref = nb.load(self.inputs.surfaces_volume_reference)
        surf2world = surf_ref.get_affine().dot(OnlinePreprocBase.ras2vox)

        structs = out_file.create_group('STRUCTURES')
        coords = out_file.create_dataset('COORDINATES',
                                         (0,3),maxshape = (None,3),
                                         dtype = np.float)
        for surf_name, surf_file in self.inputs.resample_surfaces:
            surf_group = structs.create_group(surf_name)
            surf_group.attrs['ModelType'] = 'SURFACE'
            surf_group.attrs['SurfaceFile'] = surf_file
            if isinstance(surf_file, tuple):
                verts, tris = nb.freesurfer.read_geometry(surf_file[0])
                verts2, _ =  nb.freesurfer.read_geometry(surf_file[1])
                verts += verts2
                verts /= 2.
                del verts2
            else:
                verts, tris = nb.freesurfer.read_geometry(surf_file)
            verts[:] = nb.affines.apply_affine(surf2world, verts)
            ofst = coords.shape[0]
            count = verts.shape[0]
            coords.resize((ofst+count,3))
            coords[ofst:ofst+count] = verts
            surf_group.attrs['IndexOffset'] = ofst
            surf_group.attrs['IndexCount'] = count
            surf_group.attrs['COORDINATES'] = coords.regionref[ofst:ofst+count]
            surf_group.create_dataset('TRIANGLES', data=tris)
            del verts, tris
            
        for roiset_name,roiset_file,roiset_labels in self.inputs.resample_rois:
            rois_nii = nb.load(roiset_file)
            rois_data = rois_nii.get_data()
            roiset_labels = dict((k,l) for k,l in np.loadtxt(
                roiset_labels, dtype=np.object,
                converters = {0:int,1:str}))
            rois_group = structs.create_group(roiset_name)
            rois_group.attrs['ModelType'] = 'VOXELS'
            rois_group.attrs['ROIsFile'] = roiset_file

            rois_subset_data = np.zeros(rois_data.shape,dtype=np.int)
            for k in roiset_labels.keys():
                rois_subset_data[rois_data==k]=k
            rois_mask = rois_subset_data > 0
            order = np.argsort(rois_subset_data[rois_mask])
            # this allows using ROIs in different sampling
            rois_coords = nb.affines.apply_affine(
                rois_nii.get_affine(),
                np.c_[np.where(rois_subset_data)][order])
            ofst = coords.shape[0]
            count = rois_coords.shape[0]
            coords.resize((ofst+count,3))
            coords[ofst:ofst+count] = rois_coords
            counts = dict([(c,np.count_nonzero(rois_subset_data[rois_mask]==c)) for c in roiset_labels.keys()])
            rois = rois_group.create_dataset(
                'ROIS',(len(counts),),dtype=np.dtype(
                    [('name', 'S200'),('label',np.int),
                     ('IndexOffset', np.int),('IndexCount', np.int),
                     ('ref', h5py.special_dtype(ref=h5py.RegionReference))]))
            i=0
            for roi_idx, roi_count in counts.items():
                label = roiset_labels[roi_idx]
                rois[i] = (label[:200],roi_idx,ofst,roi_count,
                           coords.regionref[ofst:ofst+roi_count])
                ofst += roi_count
                i+=1
            
            del rois_nii, rois_data, rois_subset_data, rois_mask, order
        return out_file

    def _init_stack(self):
        if isdefined(self.inputs.dicom_files):
            self._list_files()
            stack = DicomStackOnline()
            stack.set_source(filenames_to_dicoms(self.dicom_files))
            stack._init_dataset()
        elif isdefined(self.inputs.nifti_file):
            stack = NiftiIterator(nb.load(self.inputs.nifti_file))
        return stack

class OnlinePreprocessingInputSpec(OnlinePreprocInputSpecBase):

    out_file_format = traits.Str(
        mandatory=True,
        desc='format with placeholder for output filename based on dicom')
    
    #realign parameters
    reference_boundary = traits.File(
        mandatory = True,
        exists = True,
        desc='the surface used for realignment')
    boundary_sampling_distance = traits.Float(
        1.5, usedefault = True,
        desc='distance from reference boundary in mm.')
    nsamples_per_slab = traits.Int(
        10000, usedefault=True,
        desc='number of samples to use during optimization of a slab')
    min_nsamples_per_slab = traits.Int(
        1000, usedefault=True,
        desc='min number of samples to perform a slab realignment')
    # optimizer
    optimization_ftol = traits.Float(
        1e-5, usedefault=True,
        desc='tolerance of optimizer for convergence')
    init_reg = File(
        desc = 'coarse init epi to t1 registration matrix')

    # resampling options
    resampled_first_frame = traits.File(
        desc = 'output first frame resampled and undistorted in reference space for visual registration check')


class OnlinePreprocessingOutputSpec(TraitedSpec):
    out_file = File(desc='hdf5 file containing the timeseries')
    first_frame = File(desc='resampled first frame in reference space')
    motion = File()
    
class OnlinePreprocessing(OnlinePreprocBase):

    input_spec = OnlinePreprocessingInputSpec
    output_spec = OnlinePreprocessingOutputSpec


    """
    TODO : 
    - encode surfaces and ROIs differently
    - store dicom metadata (TR,TE,...)
    """

    def _run_interface(self,runtime):

        out_file = self._init_ts_file()
        coords = out_file['COORDINATES']
        
        nsamples = coords.shape[0]
        
        fmri_group = out_file.create_group('FMRI')
#        fmri_group.attrs['RepetitionTime'] = self.inputs.tr
        stack = self._init_stack()

        if stack._nframes_per_dicom == 1:
            nvols = len(self.dicom_files)
        elif stack._nframes_per_dicom == 0:
            nvols = int(len(self.dicom_files)/stack.nslices)
        else:
            nvols = 1
            
        data = fmri_group.create_dataset(
            'DATA', dtype=np.float32,
            shape=(nsamples,nvols), maxshape=(nsamples,None))

        surf_ref = nb.load(self.inputs.surfaces_volume_reference)
        surf2world = surf_ref.get_affine().dot(OnlinePreprocessing.ras2vox)
        boundary_surf = nb.freesurfer.read_geometry(
            self.inputs.reference_boundary)
        boundary_surf[0][:] = nb.affines.apply_affine(surf2world,
                                                      boundary_surf[0])
        sampling_coords = surface_to_samples(
            boundary_surf[0], boundary_surf[1], 
            self.inputs.boundary_sampling_distance)
        fmap, fmap_reg = None, None
        if isdefined(self.inputs.fieldmap):
            fmap = nb.load(self.inputs.fieldmap)
            fmap_reg = np.eye(4)
            if isdefined(self.inputs.fieldmap_reg):
                fmap_reg[:] = np.loadtxt(self.inputs.fieldmap_reg)
        mask = nb.load(self.inputs.mask)
        init_reg = None
        if isdefined(self.inputs.init_reg):
            init_reg = np.loadtxt(self.inputs.init_reg)

        algo = EPIOnlineRealign(
            boundary_surf[0], sampling_coords,
            fieldmap = fmap, fieldmap_reg=fmap_reg,
            mask = mask,
            phase_encoding_dir = self.inputs.phase_encoding_dir,
            echo_time = self.inputs.echo_time,
            echo_spacing = self.inputs.echo_spacing,
            nsamples_per_slab = self.inputs.nsamples_per_slab,
            ftol = self.inputs.optimization_ftol,
            slice_thickness = self.inputs.slice_thickness,
            min_nsamples_per_slab = self.inputs.min_nsamples_per_slab,
            init_reg = init_reg)

        self.algo=algo
        tmp = np.empty(nsamples)
        
            
        t = 0
        self.slabs = []
        self.mats = []
        for slab, reg, vol in algo.process(stack, yield_raw=True):
            print 'frame %d'% t
            self.slabs.append(slab)
            self.mats.append(reg)
            algo.resample_coords(vol, [(slab,reg)], coords, tmp)
            if data.shape[-1] <= t:
                data.resize((nsamples,t))
            data[:,t] = tmp
            if t==0 and isdefined(self.inputs.resampled_first_frame):
                f1 = np.empty(surf_ref.shape)
                vol_coords = nb.affines.apply_affine(
                    surf_ref.get_affine(),
                    np.rollaxis(np.mgrid[[slice(0,d) for d in f1.shape]],0,4))
                algo.resample_coords(vol, [(slab,reg)], vol_coords, f1)
                nb.save(nb.Nifti1Image(f1, surf_ref.get_affine()),
                        self._list_outputs()['first_frame'])
                del vol_coords, f1
            t += 1

        out_file.close()
        motion = np.array([t.param*t.precond[:6] for t in algo.transforms])
        np.savetxt(self._list_outputs()['motion'], motion)
            
        del stack, sampling_coords, tmp, algo, surf_ref
        
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath('./ts.h5')
        outputs['motion'] = os.path.abspath('./motion.txt')
        if isdefined(self.inputs.resampled_first_frame):
            outputs['first_frame'] = os.path.abspath(
                self.inputs.resampled_first_frame)
        return outputs

def filenames_to_dicoms(fnames):
    for f in fnames:
        yield dicom.read_file(f)


class OnlineFilterInputSpec(OnlinePreprocInputSpecBase):
    motion = File(
        exists=True,
        mandatory=True,
        desc='the estimated motion')
    partial_volume_maps = InputMultiPath(
        File(exists=True),
        desc='partial volumes maps to regress out')
    poly_order = traits.Range(
        0,3,2, usedefault=True,
        desc="""the order of the 2d poly to regress out of each slices
for intensity inhomogeneity bias field correction""")
    
class OnlineFilterOutputSpec(TraitedSpec):

    timeseries = File(desc='resampled filtered timeseries')
    
class OnlineFilter(OnlinePreprocBase):

    input_spec = OnlineFilterInputSpec
    output_spec = OnlineFilterOutputSpec
    
    def _run_interface(self,runtime):

        out_file = self._init_ts_file()
        coords = out_file['COORDINATES']
        
        nsamples = coords.shape[0]
        
        fmri_group = out_file.create_group('FMRI')

        from nipy.algorithms.registration.affine import to_matrix44
        motion = np.loadtxt(self.inputs.motion)
        mats = [to_matrix44(m) for m in motion]
        
        def iter_precomp_realign(stack, affines):
            for frame, affine, data in stack.iter_frame():
                slab = ((frame,0),(frame,stack.nslices))
                yield slab, affines[frame].dot(affine),data

        stack = self._init_stack()

        if stack._nframes_per_dicom == 1:
            nvols = len(self.dicom_files)
        elif stack._nframes_per_dicom == 0:
            nvols = int(len(self.dicom_files)/stack.nslices)
        else:
            nvols = 1
        
        stack = iter_precomp_realign(stack, mats)

        data = fmri_group.create_dataset(
            'DATA_FILTERED', dtype=np.float32,
            shape=(nsamples,nvols), maxshape=(nsamples,None))
             
        fmap, fmap_reg = None, None
        if isdefined(self.inputs.fieldmap):
            fmap = nb.load(self.inputs.fieldmap)
            fmap_reg = np.eye(4)
            if isdefined(self.inputs.fieldmap_reg):
                fmap_reg[:] = np.loadtxt(self.inputs.fieldmap_reg)
        mask = nb.load(self.inputs.mask)


        if isdefined(self.inputs.partial_volume_maps):
            pvmaps = [nb.load(f) for f in self.inputs.partial_volume_maps]
            pvmaps = nb.Nifti1Image(
                np.concatenate([m.get_data().reshape((m.shape+(1,)[:4])) for m in pvmaps],3),
                pvmaps[0].get_affine())
        

        algo = EPIOnlineRealignFilter(
            fieldmap = fmap, fieldmap_reg = fmap_reg,
            mask = mask,
            phase_encoding_dir = self.inputs.phase_encoding_dir,
            echo_time = self.inputs.echo_time,
            echo_spacing = self.inputs.echo_spacing,
            slice_thickness = self.inputs.slice_thickness)

        self.algo = algo
        tmp = np.empty(nsamples)
            
        t = 0
        for slab, reg, cdata in algo.correct(
            stack,
            pvmaps = pvmaps,
            poly_order=self.inputs.poly_order):
            print 'frame %d'% t
            algo.resample_coords(cdata, [(slab,reg)], coords, tmp)
            if data.shape[-1] <= t:
                data.resize((nsamples,t))
            data[:,t] = tmp
            t += 1

        out_file.close()
            
        del stack, tmp, algo, surf_ref
        
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath('./ts.h5')
        return outputs
