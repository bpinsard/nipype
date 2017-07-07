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
from ...utils.filemanip import split_filename, fname_presuffix, filename_to_list
from ...utils import NUMPY_MMAP

from ..base import (TraitedSpec, BaseInterface, traits,
                    BaseInterfaceInputSpec, isdefined, File, Directory,
                    InputMultiPath, OutputMultiPath)
from .base import *

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
        EPIOnlineRealign, EPIOnlineRealignFilter, resample_mat_shape,
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
    from .online_stack import DicomStackOnline, filenames_to_dicoms



class Info(object):
    """Handle nibabel output type and version information.
"""
    __outputtype = 'NIFTI'
    ftypes = {'NIFTI': '.nii',
              'NIFTI_GZ': '.nii.gz',
              'MGZ':'.mgz'}

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
            msg = 'Invalid NIBABELOUTPUTTYPE: ', outputtype
            raise KeyError(msg)

    @classmethod
    def outputtype(cls):
        return cls.__outputtype


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

    # Fieldmap parameters
    fieldmap = File(
        desc = 'precomputed fieldmap coregistered with reference space')
    fieldmap_reg = File(
        desc = 'fieldmap coregistration matrix')
    
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

class SurfaceResamplingInputSpec(NipyBaseInterfaceInputSpec):

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

    resampled_first_frame = traits.File(
        hash_files = False,
        desc = 'output first frame resampled and undistorted in reference space for visual registration check')


class SurfaceResamplingOutputSpec(TraitedSpec):
    out_file = File(desc='resampled filtered timeseries')
    first_frame = File(desc='resampled first frame in reference space')
    mask = File(desc='resampled mask in the same space as first_frame')

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
            stack = NiftiIterator(nb.load(self.inputs.nifti_file))
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
            fname = fname_presuffix(self.inputs.nifti_file, suffix='_mc',
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
        if ext in Info.ftypes.values():
            return os.path.join(path, base + Info.outputtype_to_ext(
                    self.inputs.outputtype))
        return value


class SurfaceResamplingBase(NipyBaseInterface):

    ras2vox = np.array([[-1,0,0,128],[0,0,-1,128],[0,1,0,128],[0,0,0,1]])


    def load_gii_fs(self,sfilename):
        if split_filename(sfilename)[-1] == '.gii':
            sfile = gii.read(sfilename)
            return sfile.darrays[0].data, sfile.darrays[1].data
        else:
            surf_ref = nb.load(self.inputs.surfaces_volume_reference)
            surf2world = surf_ref.get_affine().dot(SurfaceResamplingBase.ras2vox)
            verts, tris = nb.freesurfer.read_geometry(sfilename)
            verts[:] = nb.affines.apply_affine(surf2world, verts)
        return verts, tris

    def _init_ts_file(self):
        out_file = h5py.File(self._list_outputs()['out_file'])
        
        surfaces = []
        rois = []        
        surf_ref = nb.load(self.inputs.surfaces_volume_reference)
        surf2world = surf_ref.get_affine().dot(SurfaceResamplingBase.ras2vox)

        structs = out_file.create_group('STRUCTURES')
        coords = out_file.create_dataset('COORDINATES',
                                         (0,3),maxshape = (None,3),
                                         dtype = np.float)
        
        if isdefined(self.inputs.resample_surfaces):
            for surf_name, surf_file in self.inputs.resample_surfaces:
                surf_group = structs.create_group(surf_name)
                surf_group.attrs['ModelType'] = 'SURFACE'
                surf_group.attrs['SurfaceFile'] = surf_file
                if isinstance(surf_file, tuple):
                    verts, tris = self.load_gii_fs(surf_file[0])
                    verts2, _ =  self.load_gii_fs(surf_file[1])
                    verts *=(1-self.inputs.middle_surface_position)
                    verts += verts2*self.inputs.middle_surface_position
                    del verts2
                else:
                    verts, tris = self.load_gii_fs(surf_file)
                ofst = coords.shape[0]
                count = verts.shape[0]
                coords.resize((ofst+count,3))
                coords[ofst:ofst+count] = verts
                surf_group.attrs['IndexOffset'] = ofst
                surf_group.attrs['IndexCount'] = count
                surf_group.attrs['COORDINATES'] = coords.regionref[ofst:ofst+count]
                surf_group.create_dataset('TRIANGLES', data=tris)
                del verts, tris

        if isdefined(self.inputs.resample_rois):
            for roiset_name,roiset_file,roiset_labels in self.inputs.resample_rois:
                roiset_labels = dict((k,l) for k,l in np.loadtxt(
                        roiset_labels, dtype=np.object,
                        converters = {0:int,1:str}))
                rois_group = structs.create_group(roiset_name)
                rois_group.attrs['ModelType'] = 'VOXELS'
                rois_group.attrs['ROIsFile'] = roiset_file

                nvoxs = 0
                counts = dict()

                if nb.filename_parser.splitext_addext(roiset_file, ('.gz', '.bz2'))[1] in nb.ext_map:
                    voxs = []
                    rois_nii = nb.load(roiset_file)
                    rois_data = rois_nii.get_data()
                    for k in roiset_labels.keys():
                        roi_mask = rois_data==k
                        counts[k] = np.count_nonzero(roi_mask)
                        nvoxs += counts[k]
                        voxs.append(np.argwhere(roi_mask))
                    voxs = np.vstack(voxs)
                    crds = nb.affines.apply_affine(rois_nii.get_affine(), voxs)
                    del rois_nii, rois_data
                else:
                    rois_txt = np.loadtxt(roiset_file,delimiter=',',skiprows=1)
                    crds = []
                    voxs = []
                    for k in roiset_labels.keys():
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
                voxel_indices = rois_group.create_dataset('INDICES',data=voxs)
                rois = rois_group.create_dataset(
                    'ROIS',(len(counts),),dtype=np.dtype(
                        [('name', 'S200'),('label',np.int),
                         ('IndexOffset', np.int),('IndexCount', np.int),
                         #('ref', h5py.special_dtype(ref=h5py.RegionReference))
                         ]))
                for i,roi in enumerate(counts.items()):
                    roi_idx, roi_count = roi
                    label = roiset_labels[roi_idx]
                    rois[i] = (label[:200],roi_idx,ofst,roi_count,)
                    #                           coords.regionref[ofst:ofst+roi_count])
                    ofst += roi_count
                del voxs, crds
        return out_file

    def resampler(self, iterator, out_file, dataset_path='FMRI/DATA'):
        coords = np.asarray(out_file['COORDINATES'])
        nsamples = coords.shape[0]

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

        self.slabs = []
        self.slabs_data = []
        tmp = np.empty(nsamples)
        first_frame_exported = False
        for fr, slab, reg, data in iterator:
            print 'frame %d, slab %s'% (fr,slab)
            self.slabs.append((fr,slab,reg))
            self.slabs_data.append(data)

            if len(self.slabs)%nslabs is 0:
                tmp_slabs = [s for s in self.slabs if s[0]==fr]
                if nsamples > 0:
                    self.algo.scatter_resample(
                        self.slabs_data, tmp,
                        [s[1] for s in tmp_slabs] ,
                        [s[2] for s in tmp_slabs],
                        coords, mask=True)
                    rdata[:,fr] = tmp
                    if rdata.shape[-1] < fr:
                        rdata.resize((nsamples,fr))
                if fr<1 and isdefined(self.inputs.resampled_first_frame) and not first_frame_exported:
                    print 'resampling first frame'
                    mask = nb.load(self.inputs.mask)
                    mask_data = mask.get_data()>0
                    f1 = np.zeros(mask.shape)
                    tmp_f1 = np.empty(np.count_nonzero(mask_data))
                    vol_coords = nb.affines.apply_affine(
                        mask.get_affine(),
                        np.rollaxis(np.mgrid[[slice(0,d) for d in f1.shape]],0,4)[mask_data])
                    self.algo.scatter_resample(
                        self.slabs_data, tmp_f1,
                        [s[1] for s in tmp_slabs],
                        [s[2] for s in tmp_slabs],
                        vol_coords, mask=True)
                    f1[mask_data] = tmp_f1
                    outputs = self._list_outputs()
                    nb.save(nb.Nifti1Image(f1.astype(np.float32),
                                           mask.get_affine()),
                            outputs['first_frame'])
                    del vol_coords, f1
                    ornt_trsfrm = nb.orientations.ornt_transform(
                        nb.orientations.io_orientation(self.stack._affine),
                        nb.orientations.io_orientation(mask.get_affine())
                        ).astype(np.int)
                    voxel_size = self.stack._voxel_size[ornt_trsfrm[:,0]]
                    mat, shape = resample_mat_shape(
                        mask.get_affine(), mask.shape, voxel_size)
                    vol_coords = nb.affines.apply_affine(
                        np.linalg.inv(mask.get_affine()),
                        np.rollaxis(np.mgrid[[slice(0,d) for d in shape]],0,4))
                    resam_mask = map_coordinates(
                        mask.get_data(), vol_coords.reshape(-1,3).T,
                        order=0).reshape(shape)
                    nb.save(nb.Nifti1Image(resam_mask.astype(np.uint8), mat),
                            outputs['mask'])
                    first_frame_exported = True
                    del vol_coords, resam_mask
                del self.slabs_data
                self.slabs_data = []        
            yield fr, slab, reg, data

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_fname())
        if isdefined(self.inputs.resampled_first_frame):
            outputs['first_frame'] = os.path.abspath(
                self.inputs.resampled_first_frame)
            outputs['mask'] = os.path.abspath(
                fname_presuffix(self.inputs.resampled_first_frame,
                                suffix='_mask'))
        return outputs


class OnlinePreprocessingInputSpec(OnlinePreprocInputSpecBase,
                                   SurfaceResamplingInputSpec):

    #realign parameters
    init_center_of_mass = traits.Bool(
        desc='initialize aligning center of mass of ref mask and first frame')
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
    optimization_xtol = traits.Float(
        1e-5, usedefault=True,
        desc='tolerance of optimizer for convergence')
    optimization_gtol = traits.Float(
        1e-5, usedefault=True,
        desc='tolerance of optimizer for convergence')

    init_reg = File(
        desc = 'coarse init epi to t1 registration matrix')

    # resampling options
    resampled_first_frame = traits.File(
        desc = 'output first frame resampled and undistorted in reference space for visual registration check')

class OnlinePreprocessingOutputSpec(SurfaceResamplingOutputSpec):
    motion = File()
    motion_params = File()
    slabs = File()
    
class OnlinePreprocessing(OnlinePreprocBase, SurfaceResamplingBase):

    input_spec = OnlinePreprocessingInputSpec
    output_spec = OnlinePreprocessingOutputSpec


    """
    TODO : 
    - encode surfaces and ROIs differently
    - store dicom metadata (TR,TE,...)
    """

    def _run_interface(self,runtime):

        self.stack = self._init_stack()

        
        try:
            out_file = self._init_ts_file()
        
            fmri_group = out_file.create_group('FMRI')
#        fmri_group.attrs['RepetitionTime'] = self.inputs.tr
            
            surf_ref = nb.load(self.inputs.surfaces_volume_reference)
            surf2world = surf_ref.get_affine().dot(SurfaceResamplingBase.ras2vox)
            boundary_surf = nb.freesurfer.read_geometry(self.inputs.reference_boundary)
            boundary_surf[0][:] = nb.affines.apply_affine(surf2world,boundary_surf[0])
            
            sampling_coords = surface_to_samples(
                boundary_surf[0], boundary_surf[1], 
                self.inputs.boundary_sampling_distance)
            fmap, fmap_reg = self._load_fieldmap()

            mask = nb.load(self.inputs.mask)
            mask_data = mask.get_data()>0
            init_reg = None
            if isdefined(self.inputs.init_reg):
                init_reg = np.loadtxt(self.inputs.init_reg)
            elif self.inputs.init_center_of_mass:
                init_reg = 'auto'

            realigner = EPIOnlineRealign(
                boundary_surf[0], sampling_coords,
                fieldmap = fmap, 
                fieldmap_reg=fmap_reg,
                init_reg = init_reg,
                mask = mask,
                phase_encoding_dir = self.inputs.phase_encoding_dir,
                echo_time = self.inputs.echo_time,
                echo_spacing = self.inputs.echo_spacing,
                nsamples_per_slab = self.inputs.nsamples_per_slab,
                ftol = self.inputs.optimization_ftol,
                xtol = self.inputs.optimization_xtol,
                gtol = self.inputs.optimization_gtol,
                slice_thickness = self.inputs.slice_thickness)

            self.algo=realigner

            for fr, slab, reg, data in self.resampler(
                realigner.process(self.stack, yield_raw=True),out_file, 'FMRI/DATA'):
                print 'frame %d, slab %s'% (fr,slab)

        finally:
            if 'out_file' in locals():
                out_file.close()

        outputs = self._list_outputs()
#        out_file.close()
        motion = np.asarray([s[2] for s in self.slabs])
        slabs = np.array([[s[0]]+s[1] for s in self.slabs])
        np.savetxt(outputs['slabs'], slabs, '%d')
        np.save(outputs['motion'], motion)
        motion_params = np.array([realigner.affine_class(m)._vec12[:6] for m in motion])
        np.savetxt(outputs['motion_params'], motion_params)
        
        del self.stack, sampling_coords, realigner, surf_ref
        
        return runtime

    def _list_outputs(self):
        outputs = super(self.__class__, self)._list_outputs()
        outputs['slabs'] = os.path.abspath('./slabs.txt')
        outputs['motion'] = os.path.abspath('./motion.npy')
        outputs['motion_params'] = os.path.abspath('./motion.txt')
        return outputs

class OnlineFilterInputSpec(OnlinePreprocInputSpecBase,
                            SurfaceResamplingInputSpec):
    motion = File(
        exists=True,
        mandatory=True,
        desc='the estimated motion')
    partial_volume_maps = InputMultiPath(
        File(exists=True),
        desc='partial volumes maps to regress out')

    resampled_first_frame = traits.File(
        desc = 'output first frame resampled and undistorted in reference space for visual registration check')
    
class OnlineFilterOutputSpec(SurfaceResamplingOutputSpec):    
    nifti_out = File(exists=True,
                     desc='at some point we might outputs also 4d nifti corrected for checking')
    
class OnlineFilter(OnlinePreprocBase, SurfaceResamplingBase):

    input_spec = OnlineFilterInputSpec
    output_spec = OnlineFilterOutputSpec
    
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

            if isdefined(self.inputs.partial_volume_maps):
                pvmaps = [nb.load(f) for f in self.inputs.partial_volume_maps]
                pvmaps = nb.Nifti1Image(
                    np.concatenate([m.get_data().reshape((m.shape+(1,)[:4])) for m in pvmaps],3),
                    pvmaps[0].get_affine())
        
            noise_filter = EPIOnlineRealignFilter(
                fieldmap = fmap, fieldmap_reg = fmap_reg,
                mask = mask,
                phase_encoding_dir = self.inputs.phase_encoding_dir,
                echo_time = self.inputs.echo_time,
                echo_spacing = self.inputs.echo_spacing,
                slice_thickness = self.inputs.slice_thickness)

            self.algo = noise_filter

            for fr, slab, reg, data in self.resampler(
                noise_filter.correct(stack_it, pvmaps, self.stack._shape[:3]), out_file, 'FMRI/DATA'):
                print 'frame %d, slab %s'% (fr,slab)
        finally:
            print 'closing file'
            if 'out_file' in locals():
                out_file.close()
        
        del self.stack, noise_filter        
        return runtime

class OnlineResample4DInputSpec(OnlinePreprocInputSpecBase):

    slabs = File(
        exists=True,
        mandatory=True,
        desc='slabs on which motion was estimated')

    motion = File(
        exists=True,
        mandatory=True,
        desc='the estimated motion')

    reference = File(
        exists=True,
        mandatory=True,
        desc='volume describing the space in which to resample')

    voxel_size = traits.Tuple(
        *([traits.Float()]*3),
        desc='size of the output voxels')
    
    outputtype = traits.Enum('NIFTI', Info.ftypes.keys(), usedefault=True,
                             desc='DCMStack output filetype')

class OnlineResample4DOutputSpec(TraitedSpec):
    out_file = File(desc='big nifti 4D native space file of timeseries')
    mask = File(desc='brain mask in resample space')
class OnlineResample4D(OnlinePreprocBase):

    input_spec = OnlineResample4DInputSpec
    output_spec = OnlineResample4DOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        base_fname = self._gen_fname()
        outputs['out_file'] = os.path.abspath(fname_presuffix(base_fname))
        outputs['mask'] = os.path.abspath(fname_presuffix(base_fname,
                                                          suffix='_mask'))
        return outputs

    def _run_interface(self,runtime):

        stack = self._init_stack()
        if isinstance(stack, DicomStackOnline):
            stack._init_dataset()

        slabs_array = np.loadtxt(self.inputs.slabs,np.int)
        # change to continuous slabs
        slabs_array[1:,1] = np.mod(slabs_array[:-1,3]+1, stack.nslices)
        slabs = [((s[0],s[1]),(s[2],s[3])) for s in slabs_array]

        from nipy.algorithms.registration.affine import to_matrix44
        motion_array = np.loadtxt(self.inputs.motion)
        motion_mats = [to_matrix44(m) for m in motion_array]

        fmap, fmap_reg = None, None
        if isdefined(self.inputs.fieldmap):
            fmap = nb.load(self.inputs.fieldmap)
            fmap_reg = np.eye(4)
            if isdefined(self.inputs.fieldmap_reg):
                fmap_reg[:] = np.loadtxt(self.inputs.fieldmap_reg)
        mask = nb.load(self.inputs.mask)

        algo = EPIOnlineRealignFilter(
            fieldmap = fmap,
            fieldmap_reg = fmap_reg,
            mask = mask,
            slice_order = stack._slice_order,
            phase_encoding_dir = self.inputs.phase_encoding_dir,
            echo_time = self.inputs.echo_time,
            echo_spacing = self.inputs.echo_spacing,
            slice_thickness = self.inputs.slice_thickness)

        voxel_size = self.inputs.voxel_size
        if not isdefined(voxel_size):
            voxel_size = stack._voxel_size
            ornt_trsfrm = nb.orientations.ornt_transform(
                nb.orientations.io_orientation(stack._affine),
                nb.orientations.io_orientation(mask.get_affine())
                ).astype(np.int)
            voxel_size = voxel_size[ornt_trsfrm[:,0]]

        ref = nb.load(self.inputs.reference)
        mat, shape = resample_mat_shape(ref.get_affine(),ref.shape,voxel_size)
        grid = np.rollaxis(np.mgrid[[slice(0,n) for n in shape]],0,4)
        coords = nb.affines.apply_affine(mat, grid)
        del grid
        out = np.empty(shape+(stack.nframes,),dtype=np.float32)
        tmp = np.empty(shape)

        for fr, affine, data in stack.iter_frame():
            print 'resampling frame %d'%fr
            slab_regs = [(slab,m.dot(affine)) \
                            for slab,m in zip(slabs,motion_mats)\
                            if slab[0][0]<=fr and slab[1][0]>=fr]
            print slab_regs
            algo.resample_coords(data, slab_regs, coords, tmp)
            out[...,fr] = tmp
        del tmp
        outputs = self._list_outputs()
        
        nb.save(nb.Nifti1Image(out, mat),outputs['out_file'])
        del out
        resam_mask = map_coordinates(
            mask.get_data(),
            nb.affines.apply_affine(
                np.linalg.inv(mask.get_affine()),coords).reshape(-1,3).T,
            order=0).reshape(shape)
        nb.save(nb.Nifti1Image(resam_mask.astype(np.uint8), mat),
                outputs['mask'])

        del resam_mask, coords
        return runtime

