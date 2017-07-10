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
import nibabel.gifti as gii
import dicom
import numpy as np

import glob
import re

from ...utils.misc import package_check
from ...utils import NUMPY_MMAP

from ...utils.filemanip import split_filename, fname_presuffix, filename_to_list
from ..base import (TraitedSpec, BaseInterface, traits,
                    BaseInterfaceInputSpec, isdefined, File, Directory,
                    InputMultiPath, OutputMultiPath)
from .base import Info, NipyBaseInterface, NipyBaseInterfaceInputSpec

from scipy.ndimage.interpolation import map_coordinates

have_nipy = True
try:
    package_check('nipy')
except Exception as e:
    have_nipy = False
else:
    import nipy
    from nipy import save_image, load_image
    nipy_version = nipy.__version__
    from nipy.algorithms.registration.online_preproc import (
        OnlineRealignBiasCorrection, NiftiIterator, Rigid, Affine, 
        resample_mat_shape, filenames_to_dicoms)
    from nipy.algorithms.registration.online_dcmstack import DicomStackOnline
    

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
    from .online_stack import DicomStackOnline, filenames_to_dicoms


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
                    nii = nb.load(value, mmap=NUMPY_MMAP)
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
        nii2 = nb.Nifti1Image(nii.get_data()[..., s], nii.affine, nii.header)
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
        mat = in_file.affine.copy()
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

class OnlinePreprocInputSpecBase(NipyBaseInterfaceInputSpec):
    dicom_files = traits.Either(
        InputMultiPath(File(exists=True)),
        InputMultiPath(Directory(exists=True)),
        InputMultiPath(traits.Str()), #allow glob file pattern
        mandatory=True,
        xor=['nifti_file'])
    
    nifti_file = File(exists=True,
                       mandatory=True,
                       xor=['dicom_files'])
    multiband_factor = traits.Int(
        1, usedefault=True,
        hash_files=False,
        desc='set the multiband factor for nifti input')

    # Fieldmap parameters
    fieldmap = File(
        desc = 'precomputed fieldmap coregistered with reference space')
    fieldmap_reg = File(
        desc = 'fieldmap coregistration matrix')
    fieldmap_recenter_values = traits.Bool(
        False, usedefault=True,
        desc = 'substract the mean of fieldmap values in brain mask to minimize shift')
    fieldmap_unmask = traits.Bool(
        False, usedefault=True,
        desc = 'unmask (extrapolate out of mask) the fieldmap to reduce border effect')


    # space definition
    mask = File(
        mandatory = True,
        exists = True,
        desc='a mask in reference space')
    
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


class OnlinePreprocBase(NipyBaseInterface):
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

    def _init_stack(self):
        if isdefined(self.inputs.dicom_files):
            self._list_files()
            stack = DicomStackOnline()
            stack.set_source(filenames_to_dicoms(self.dicom_files))
            stack._init_dataset()
        elif isdefined(self.inputs.nifti_file):
            stack = NiftiIterator(nb.load(self.inputs.nifti_file), mb=self.inputs.multiband_factor)
        return stack

    def _load_fieldmap(self):
        fmap, fmap_reg = None, None
        if isdefined(self.inputs.fieldmap):
            fmap = nb.load(self.inputs.fieldmap)
            fmap_reg = np.eye(4)
            if isdefined(self.inputs.fieldmap_reg):
                fmap_reg[:] = np.loadtxt(self.inputs.fieldmap_reg)        
        return fmap, fmap_reg

    def _gen_fname(self):
        if hasattr(self,'_fname'):
            return self._fname
        if isdefined(self.inputs.nifti_file):
            fname = fname_presuffix(self.inputs.nifti_file, suffix='_mc.h5',
                                    newpath=os.getcwd())
            self._fname = self._overload_extension(fname)
            return self._fname
        if hasattr(self,'dicom_files') and\
                isdefined(self.inputs.out_file_format):
            keys = re.findall('\%\((\w*)\)', self.inputs.out_file_format)
            dic = dicom.read_file(self.dicom_files[0])
            values = dict([(k,dic.get(k,'')) for k in keys])
            fname_base = str(self.inputs.out_file_format % values)
            
            self._fname = self._overload_extension(os.path.abspath(fname_base))
            del dic
            return self._fname
        return

    def _overload_extension(self, value):
        path, base, ext = split_filename(value)
        if ext not in Info.ftypes.values():
            return value
        return os.path.join(path, base + Info.outputtype_to_ext(
                self.inputs.outputtype))

class SurfaceResamplingBaseInputSpec(BaseInterfaceInputSpec):

    out_file_format = traits.Str(
        mandatory=True,
        desc='format with placeholder for output filename based on dicom')

    # resampling objects
    resample_surfaces = traits.List(
        traits.Tuple(
            traits.Str,
            traits.Either(
                File(exists=True),
                traits.Tuple(
                    File(exists=True),
                    File(exists=True)))),
        desc='freesurfer surface files from which signal to be extracted')
    middle_surface_position = traits.Float(
        .5, usedefault=True,
        desc='distance from inner to outer surface in ratio of thickness')

    gm_pve = File(
        exists=True,
        desc='gray matter pve map to weight voxels in resampling')
    interp_mask = traits.Bool(
        True, usedefault=True,
        desc='remove voxel out of brain for interpolation')
    interp_rbf_sigma = traits.Float(
        3, usedefault=True,
        desc='the sigma parameter for gaussian rbf interpolation')
    interp_cortical_anisotropic_kernel = traits.Bool(
        False, usedefault=True,
        desc='use surface constrained anisotropic kernel for cortical interpolation')

    resample_rois = traits.List(
        traits.Tuple(
            traits.Str, 
            File(exists=True), 
            File(exists=True)),
        desc = 'list of rois NIFTI files from which to extract signal and labels file')
    
    store_coords = traits.Bool(
        True, usedefault=True,
        desc='store surface and ROIs coordinates in output')

    # space definition
    surfaces_volume_reference = traits.File(
        mandatory = True,
        exists = True,
        desc='a volume defining space of surfaces')

    # output
    resampled_first_frame = traits.File(
        desc = 'output first frame resampled and undistorted in reference space for visual registration check')
    resampled_frame_index = traits.Int(
        0,usedefault=True,
        desc='index of the frame to resample')

class SurfaceResamplingBaseOutputSpec(TraitedSpec):
    out_file = File(desc='resampled filtered timeseries')
    resampled_first_frame = File(desc='resampled first frame in reference space')
    mask = File(desc='resampled mask in the same space as resampled_first_frame')

class SurfaceResamplingBase(NipyBaseInterface):

    ras2vox = np.array([[-1,0,0,128],[0,0,-1,128],[0,1,0,128],[0,0,0,1]])

    def load_gii_fs(self,sfilename):
        if split_filename(sfilename)[-1] == '.gii':
            sfile = gii.read(sfilename)
            return sfile.darrays[0].data, sfile.darrays[1].data
        else:
            surf_ref = nb.load(self.inputs.surfaces_volume_reference)
            surf2world = surf_ref.affine.dot(SurfaceResamplingBase.ras2vox)
            verts, tris = nb.freesurfer.read_geometry(sfilename)
            verts[:] = nb.affines.apply_affine(surf2world, verts)
        return verts, tris

    def _init_ts_file(self):
        out_file = h5py.File(self._list_outputs()['out_file'])
        
        surfaces = []
        rois = []        

        structs = out_file.create_group('STRUCTURES')
        coords = out_file.create_dataset('COORDINATES',
                                         (0,3),maxshape = (None,3),
                                         dtype = np.float)
        normals = out_file.create_dataset('NORMALS',
                                          (0,3),maxshape = (None,3),
                                          dtype = np.float)
        out_mask = out_file.create_dataset('MASK',
                                           (0,1),maxshape = (None,1),
                                           dtype = np.bool)
        
        if isdefined(self.inputs.resample_surfaces):
            for surf_name, surf_file in self.inputs.resample_surfaces:
                surf_group = structs.create_group(surf_name)
                surf_group.attrs['ModelType'] = 'SURFACE'
                surf_group.attrs['SurfaceFile'] = [f.encode('utf8') for f in surf_file]
                normal_tmp = None
                if isinstance(surf_file, tuple):
                    verts1, tris = self.load_gii_fs(surf_file[0])
                    verts = verts1*(1-self.inputs.middle_surface_position)
                    verts2, _ =  self.load_gii_fs(surf_file[1])
                    verts += verts2*self.inputs.middle_surface_position
                    mask = np.abs(verts-verts2).sum(1)>1e-2
                    normal_tmp = verts2 - verts1
                    del verts2
                else:
                    verts, tris = self.load_gii_fs(surf_file)
                    mask = True
                ofst = coords.shape[0]
                count = verts.shape[0]
                coords.resize((ofst+count,3))
                coords[ofst:ofst+count] = verts
                out_mask.resize((ofst+count,1))
                out_mask[ofst:ofst+count,0] = mask
                if not normal_tmp is None:
                    normals.resize((ofst+count,3))
                    normals[ofst:ofst+count] = normal_tmp

                surf_group.attrs['IndexOffset'] = ofst
                surf_group.attrs['IndexCount'] = count
                surf_group.attrs['COORDINATES'] = coords.regionref[ofst:ofst+count]
                surf_group.create_dataset('TRIANGLES', data=tris)
                del verts, tris


        if isdefined(self.inputs.resample_rois):
            for roiset_name,roiset_file,roiset_labels in self.inputs.resample_rois:
                roiset = np.loadtxt(
                    roiset_labels, dtype=np.object,
                    converters = {0:int,1:str})
                roiset_labels = dict((k,l) for k,l in roiset)
                rois_group = structs.create_group(roiset_name)
                rois_group.attrs['ModelType'] = 'VOXELS'
                rois_group.attrs['ROIsFile'] = roiset_file

                nvoxs = 0
                counts = dict()

                if nb.filename_parser.splitext_addext(roiset_file, ('.gz', '.bz2'))[1] in nb.ext_map:
                    voxs = []
                    rois_nii = nb.load(roiset_file)
                    rois_data = rois_nii.get_data()
                    for k in roiset[:,0]:
                        roi_mask = rois_data==k
                        counts[k] = np.count_nonzero(roi_mask)
                        nvoxs += counts[k]
                        voxs.append(np.argwhere(roi_mask))
                    voxs = np.vstack(voxs)
                    crds = nb.affines.apply_affine(rois_nii.affine, voxs)
                    del rois_nii, rois_data
                else:
                    rois_txt = np.loadtxt(roiset_file,delimiter=',',skiprows=1)
                    crds = []
                    voxs = []
                    for k in roiset[:,0]:
                        roi_mask = rois_txt[:,-1] == k
                        counts[k] = np.count_nonzero(roi_mask)
                        nvoxs += counts[k]
                        crds.append(rois_txt[roi_mask,:3])
                        voxs.append(rois_txt[roi_mask,3:6].astype(np.int))
                    del rois_txt
                    voxs = np.vstack(voxs)
                    crds = np.vstack(crds)
                    
                # this allows using ROIs in different sampling
                ofst = coords.shape[0]
                coords.resize((ofst+nvoxs,3))
                coords[ofst:ofst+nvoxs] = crds
                out_mask.resize((ofst+nvoxs,1))
                out_mask[ofst:ofst+nvoxs] = True
                voxel_indices = rois_group.create_dataset('INDICES',data=voxs)
                rois = rois_group.create_dataset(
                    'ROIS', (len(counts),),
                    dtype=np.dtype([(b'name', 'S200'),(b'label',np.int),(b'IndexOffset', np.int),(b'IndexCount', np.int)]))
                for i,roi in enumerate(counts.items()):
                    roi_idx, roi_count = roi
                    label = roiset_labels[roi_idx]
                    rois[i] = (label[:200],roi_idx,ofst,roi_count,)
                    ofst += roi_count
                del voxs, crds
        return out_file

 
    def resampler(self, iterator, out_file, dataset_path='FMRI/DATA'):
        out_mask = np.asarray(out_file['MASK']).ravel()
        coords = np.asarray(out_file['COORDINATES'])[out_mask]
        normals = None
        if self.inputs.interp_cortical_anisotropic_kernel:
            normals = np.asarray(out_file['NORMALS'])[out_mask[:len(out_file['NORMALS'])]]
            normals = np.vstack([normals, np.zeros((len(coords)-len(normals),3))])
        nsamples = out_mask.shape[0]

        nslabs = len(self.stack._slabs)
        if isinstance(self.stack, NiftiIterator):

            nvols = self.stack.nframes
        elif self.stack._nframes_per_dicom == 1:
            nvols = len(self.dicom_files)
        elif self.stack._nframes_per_dicom == 0:
            nvols = int(len(self.dicom_files)/self.stack.nslices)
        else:
            nvols = 1

        if nsamples>0:
            rdata = out_file.create_dataset(
                dataset_path, dtype=np.float32,
                shape=(nsamples,nvols), maxshape=(nsamples,None))

        gm_pve = None
        if self.inputs.gm_pve:
            gm_pve = nb.load(self.inputs.gm_pve)

        kneigh_dens_2mm = int((self.inputs.interp_rbf_sigma*3)**3) # 3 std *2/2
        self.slabs = []
        self.slabs_data = []
        tmp = np.empty(np.count_nonzero(out_mask))
        resampled_first_frame_exported = False
        for fr, slab, reg, data in iterator:
            self.slabs.append((fr,slab,reg))
            self.slabs_data.append(data)

            if len(self.slabs)%nslabs is 0:
                tmp_slabs = [s for s in self.slabs if s[0]==fr]
                if fr==self.inputs.resampled_frame_index and \
                   isdefined(self.inputs.resampled_first_frame) and \
                   not resampled_first_frame_exported:
                    print('resampling first frame')
                    mask = nb.load(self.inputs.mask)
                    mask_data = mask.get_data()>0
                    f1 = np.zeros(mask.shape)
                    tmp_f1 = np.empty(np.count_nonzero(mask_data))
                    vol_coords = nb.affines.apply_affine(
                        mask.affine,
                        np.rollaxis(np.mgrid[[slice(0,d) for d in f1.shape]],0,4)[mask_data])
                    self.algo.scatter_resample_rbf(
                        self.slabs_data, tmp_f1,
                        [s[1] for s in tmp_slabs],
                        [s[2] for s in tmp_slabs],
                        vol_coords, mask=False,
                        rbf_sigma=self.inputs.interp_rbf_sigma)
                    f1[mask_data] = tmp_f1
                    outputs = self._list_outputs()
                    nb.save(nb.Nifti1Image(f1.astype(np.float32),
                                           mask.affine),
                            outputs['resampled_first_frame'])
                    del vol_coords, f1
                    ornt_trsfrm = nb.orientations.ornt_transform(
                        nb.orientations.io_orientation(self.stack._affine),
                        nb.orientations.io_orientation(mask.affine)
                        ).astype(np.int)
                    voxel_size = self.stack._voxel_size[ornt_trsfrm[:,0]]
                    mat, shape = resample_mat_shape(
                        mask.affine, mask.shape, voxel_size)
                    vol_coords = nb.affines.apply_affine(
                        np.linalg.inv(mask.affine).dot(mat),
                        np.rollaxis(np.mgrid[[slice(0,d) for d in shape]],0,4))
                    resam_mask = map_coordinates(
                        mask.get_data(), vol_coords.reshape(-1,3).T,
                        order=0).reshape(shape)
                    nb.save(nb.Nifti1Image(resam_mask.astype(np.uint8), mat),
                            outputs['mask'])
                    resampled_first_frame_exported = True
                    del vol_coords, resam_mask

                if nsamples > 0:
                    self.algo.scatter_resample_rbf(
                        self.slabs_data, tmp,
                        [s[1] for s in tmp_slabs],
                        [s[2] for s in tmp_slabs],
                        coords, normals,
                        mask=self.inputs.interp_mask,
                        pve_map=gm_pve,
                        rbf_sigma=self.inputs.interp_rbf_sigma,
                        kneigh_dens=kneigh_dens_2mm)
                    rdata[:,fr] = np.nan
                    rdata[out_mask,fr] = tmp
                    if rdata.shape[-1] < fr:
                        rdata.resize((nsamples,fr))
                del self.slabs_data
                self.slabs_data = []        
            yield fr, slab, reg, data

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_fname())
        if isdefined(self.inputs.resampled_first_frame):
            outputs['resampled_first_frame'] = os.path.abspath(
                self.inputs.resampled_first_frame)
            outputs['mask'] = os.path.abspath(
                fname_presuffix(self.inputs.resampled_first_frame,
                                suffix='_mask'))
        return outputs

class SurfaceResamplingInputSpec(SurfaceResamplingBaseInputSpec,
                                 OnlinePreprocInputSpecBase):
    motion = File(
        exists=True,
        mandatory=True,
        desc='the estimated motion')

class SurfaceResamplingOutputSpec(SurfaceResamplingBaseOutputSpec):
    pass

class SurfaceResampling(SurfaceResamplingBase, 
                        OnlinePreprocBase):
    input_spec = SurfaceResamplingInputSpec
    output_spec = SurfaceResamplingOutputSpec
    
    def _run_interface(self,runtime):

        self.stack = self._init_stack()
        try:
            out_file =  self._init_ts_file()
            coords = out_file['COORDINATES']
        
            fmri_group = out_file.create_group('FMRI')

            from nipy.algorithms.registration.affine import to_matrix44
            motion = np.load(self.inputs.motion)
            fmap, fmap_reg = self._load_fieldmap()
            mask = nb.load(self.inputs.mask)
            mask_data = mask.get_data()>0
            
            from itertools import izip
            def iter_affreg(it, motion):
                for n, m in izip(it, motion):
                    fr, slab, aff, tt, data = n
                    yield fr, slab, m, data
                                   
            stack_it = iter_affreg(self.stack.iter_slabs(), motion)


            noise_filter = EPIOnlineRealignFilter(
                fieldmap = fmap, fieldmap_reg = fmap_reg,
                mask = mask,
                phase_encoding_dir = self.inputs.phase_encoding_dir,
                echo_time = self.inputs.echo_time,
                echo_spacing = self.inputs.echo_spacing,
                slice_thickness = self.inputs.slice_thickness)

            self.algo = noise_filter

            for fr, slab, reg, data in self.resampler(stack_it, out_file, 'FMRI/DATA'):
                print(fr, slab)

            if isdefined(self.inputs.dicom_files):
                dcm = dicom.read_file(self.dicom_files[0])
                out_file['FMRI/DATA'].attrs['scan_time'] = dcm.AcquisitionTime
                out_file['FMRI/DATA'].attrs['scan_date'] = dcm.AcquisitionDate
                del dcm
            
        finally:
            print('closing file')
            if 'out_file' in locals():
                out_file.close()

        del self.stack, noise_filter        
        return runtime


class OnlineRealignInputSpec(
        OnlinePreprocInputSpecBase,
        SurfaceResamplingBaseInputSpec):

    init_reg = File(
        desc = 'coarse init epi to t1 registration matrix')
    bias_correction = traits.Bool(
        True, usedefault=True,
        desc='perform bias correction')
    register_gradient = traits.Bool(
        False, usedefault=True,
        desc='register slices using images 2D gradient (discrete laplacian)')
    bias_sigma = traits.Float(
        8, usedefault=True,
        desc='the width of smoothing for bias correction')
    wm_pve = File(
        desc='partial volume map of white matter to perform bias correction')

    # iekf parameters
    iekf_min_nsamples_per_slab = traits.Int(
        200, usedefault=True,
        desc='minimum number of samples within current slab to perform iekf')
    iekf_jacobian_epsilon = traits.Float(
        1e-3, usedefault=True,
        desc = 'the delta to use to compute jacobian')
    iekf_convergence = traits.Float(
        1e-2, usedefault=True,
        desc = 'convergence threshold for iekf')
    iekf_max_iter = traits.Int(
        8, usedefault=True,
        desc = 'maximum number of iteration per slab for iekf')
    iekf_observation_var = traits.Float(
        1e5, usedefault=True,
        desc = 'iekf observation variance, covariance omitted for white noise')
    iekf_transition_cov = traits.Float(
        1e-3, usedefault=True,
        desc = 'iekf transition (co)variance, initialized with 0 covariance (ie. independence)')
    iekf_init_state_cov = traits.Float(
        1e-2, usedefault=True,
        desc = 'iekf initial state (co)variance')


class OnlineRealignOutputSpec(SurfaceResamplingBaseOutputSpec):
    motion = File()
    motion_params = File()
    slabs = File()


class OnlineRealign(
        OnlinePreprocBase,
        SurfaceResamplingBase):

    input_spec = OnlineRealignInputSpec
    output_spec = OnlineRealignOutputSpec

    def _run_interface(self,runtime):

        self.stack = self._init_stack()

        
        try:
            out_file = self._init_ts_file()
        
            fmri_group = out_file.create_group('FMRI')

            fmap, fmap_reg = self._load_fieldmap()

            mask = nb.load(self.inputs.mask)
            mask_data = mask.get_data()>0
            init_reg = None
            if isdefined(self.inputs.init_reg):
                init_reg = np.loadtxt(self.inputs.init_reg)
            elif self.inputs.init_center_of_mass:
                init_reg = 'auto'
            wm_pve = None
            if isdefined(self.inputs.wm_pve):
                wm_pve = nb.load(self.inputs.wm_pve)

            realigner = OnlineRealignBiasCorrection(
                anat_reg = init_reg,
                mask = mask,
                bias_correction = self.inputs.bias_correction,
                bias_sigma = self.inputs.bias_sigma,
                register_gradient = self.inputs.register_gradient,
                wm_weight = wm_pve,
                fieldmap = fmap, 
                fieldmap_reg = fmap_reg,
                recenter_fmap_data=self.inputs.fieldmap_recenter_values,
                unmask_fmap=self.inputs.fieldmap_unmask,
                phase_encoding_dir = self.inputs.phase_encoding_dir,
                echo_time = self.inputs.echo_time,
                echo_spacing = self.inputs.echo_spacing,
                slice_thickness = self.inputs.slice_thickness,
                iekf_min_nsamples_per_slab= self.inputs.iekf_min_nsamples_per_slab,
                iekf_jacobian_epsilon = self.inputs.iekf_jacobian_epsilon,
                iekf_convergence = self.inputs.iekf_convergence,
                iekf_max_iter = self.inputs.iekf_max_iter,
                iekf_observation_var = self.inputs.iekf_observation_var,
                iekf_transition_cov = self.inputs.iekf_transition_cov,
                iekf_init_state_cov = self.inputs.iekf_init_state_cov)
        
            self.algo=realigner
            
            for fr, slab, reg, data in self.resampler(
                    realigner.process(self.stack, yield_raw=True),out_file, 'FMRI/DATA'):
                print('frame %d, slab %s'% (fr,slab))

            if isdefined(self.inputs.dicom_files):
                dcm = dicom.read_file(self.dicom_files[0])
                out_file['FMRI/DATA'].attrs['scan_time'] = dcm.AcquisitionTime
                out_file['FMRI/DATA'].attrs['scan_date'] = dcm.AcquisitionDate
                del dcm

        finally:
            if 'out_file' in locals():
                out_file.close()

        outputs = self._list_outputs()
        motion = np.asarray([s[2] for s in self.slabs])
        slabs = np.array([[s[0]]+s[1] for s in self.slabs])
        np.savetxt(outputs['slabs'], slabs, bytes('%d'))
        np.save(outputs['motion'], motion)
        motion_params = np.array([Rigid(m)._vec12[:6] for m in realigner.matrices])
        np.savetxt(outputs['motion_params'], motion_params)
        
        del self.stack, realigner
        
        return runtime

    def _list_outputs(self):
        outputs = super(OnlineRealign,self)._list_outputs()
        outputs['slabs'] = os.path.abspath(bytes('./slabs.txt'))
        outputs['motion'] = os.path.abspath(bytes('./motion.npy'))
        outputs['motion_params'] = os.path.abspath(bytes('./motion_pars.txt'))
        return outputs
