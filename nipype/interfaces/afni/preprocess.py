# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft = python sts = 4 ts = 4 sw = 4 et:
"""Afni preprocessing interfaces

    Change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
    >>> datadir = os.path.realpath(os.path.join(filepath, '../testing/data'))
    >>> os.chdir(datadir)
"""
import warnings
import os, re
from .base import AFNITraitedSpec, AFNICommand
from ..base import (Directory, CommandLineInputSpec, CommandLine, TraitedSpec,
                    traits, isdefined, File, InputMultiPath, Undefined)
from ...utils.filemanip import (load_json, save_json, split_filename)
from nipype.utils.filemanip import fname_presuffix

warn = warnings.warn
warnings.filterwarnings('always', category=UserWarning)


class To3DInputSpec(AFNITraitedSpec):
    infolder = Directory(desc='folder with DICOM images to convert',
        argstr='%s/*.dcm',
        position=-1,
        mandatory=True,
        exists=True)

    out_file = File(desc='converted image file',
        argstr='-prefix %s',
        position=-2,
        mandatory=True)

    filetype = traits.Enum('spgr', 'fse', 'epan', 'anat', 'ct', 'spct',
        'pet', 'mra', 'bmap', 'diff',
        'omri', 'abuc', 'fim', 'fith', 'fico', 'fitt', 'fift',
        'fizt', 'fict', 'fibt',
        'fibn', 'figt', 'fipt',
        'fbuc', argstr='-%s', desc='type of datafile being converted')

    skipoutliers = traits.Bool(desc='skip the outliers check',
        argstr='-skip_outliers')

    assumemosaic = traits.Bool(desc='assume that Siemens image is mosaic',
        argstr='-assume_dicom_mosaic')

    datatype = traits.Enum('short', 'float', 'byte', 'complex',
        desc='set output file datatype', argstr='-datum %s')

    funcparams = traits.Str(desc='parameters for functional data',
        argstr='-time:zt %s alt+z2')


class To3DOutputSpec(TraitedSpec):
    out_file = File(desc='converted file',
        exists=True)


class To3D(AFNICommand):
    """Create a 3D dataset from 2D image files using AFNI to3d command

    For complete details, see the `to3d Documentation
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/to3d.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni
    >>> To3D = afni.To3D()
    >>> To3D.inputs.datatype = 'float'
    >>> To3D.inputs.infolder = 'dicomdir'
    >>> To3D.inputs.filetype = "anat"
    >>> res = To3D.run() #doctest: +SKIP

   """

    _cmd = 'to3d'
    input_spec = To3DInputSpec
    output_spec = To3DOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs


class TShiftInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dTShift',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)
    out_file = File(desc='output file from 3dTshift',
        argstr='-prefix %s',
        position=0,
        genfile=True,
        hash_files=False)

    tr = traits.Str(desc='manually set the TR' +
        'You can attach suffix "s" for seconds or "ms" for milliseconds.',
        argstr='-TR %s')

    tzero = traits.Float(desc='align each slice to given time offset',
        argstr='-tzero %s',
        xor=['tslice'])

    tslice = traits.Int(desc='align each slice to time offset of given slice',
        argstr='-slice %s',
        xor=['tzero'])

    ignore = traits.Int(desc='ignore the first set of points specified',
        argstr='-ignore %s')

    interp = traits.Enum(('Fourier', 'linear', 'cubic', 'quintic', 'heptic'),
        desc='different interpolation methods (see 3dTShift for details)' +
        ' default = Fourier', argstr='-%s')

    tpattern = traits.Enum(('alt+z', 'alt+z2', 'alt-z',
        'alt-z2', 'seq+z', 'seq-z'),
        desc='use specified slice time pattern rather than one in header',
        argstr='-tpattern %s')

    rlt = traits.Bool(desc='Before shifting, remove the mean and linear trend',
        argstr="-rlt")

    rltplus = traits.Bool(desc='Before shifting,' +
        ' remove the mean and linear trend and ' +
        'later put back the mean',
        argstr="-rlt+")

    suffix = traits.Str('_tshift',desc="out_file suffix",usedefault=True)


class TShiftOutputSpec(TraitedSpec):
    out_file = File(desc='post slice time shifted 4D image', exists=True)


class TShift(AFNICommand):
    """Shifts voxel time series from input
    so that seperate slices are aligned to the same
    temporal origin

    For complete details, see the `3dTshift Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTshift.html>

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> tshift = afni.TShift()
    >>> tshift.inputs.in_file = example_data('functional.nii')
    >>> tshift.inputs.out_file = 'functional_tshift.nii'
    >>> tshift.inputs.tpattern = 'alt+z'
    >>> tshift.inputs.tzero = 0.0
    >>> res = tshift.run()   # doctest: +SKIP

    """

    _cmd = '3dTshift'
    input_spec = TShiftInputSpec
    output_spec = TShiftOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs


class RefitInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3drefit',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True,
        copyfile=True)

    deoblique = traits.Bool(desc='replace current transformation' +
        ' matrix with cardinal matrix',
        argstr='-deoblique')

    xorigin = traits.Str(desc='x distance for edge voxel offset',
        argstr='-xorigin %s')

    yorigin = traits.Str(desc='y distance for edge voxel offset',
        argstr='-yorigin %s')
    zorigin = traits.Str(desc='z distance for edge voxel offset',
        argstr='-zorigin %s')

    suffix = traits.Str('_refit',desc="out_file suffix",usedefault=True)


class RefitOutputSpec(TraitedSpec):
    out_file = File(desc='Same file as original in_file with modified matrix',
        exists=True)


class Refit(AFNICommand):
    """Changes some of the information inside a 3D dataset's header

    For complete details, see the `3drefit Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3drefit.html>

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> refit = afni.Refit()
    >>> refit.inputs.in_file = example_data('structural.nii')
    >>> refit.inputs.deoblique=True
    >>> res = refit.run() # doctest: +SKIP

    """

    _cmd = '3drefit'
    input_spec = RefitInputSpec
    output_spec = RefitOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_fname(
            self.inputs.in_file, suffix=self.inputs.suffix)
        return outputs


class WarpInputSpec(AFNITraitedSpec):

    in_file = File(desc='input file to 3dWarp',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)

    out_file = File(desc='output file from 3dWarp',
        argstr='-prefix %s',
        position=0,
        genfile=True,
        hash_files=False)

    tta2mni = traits.Bool(desc='transform dataset from Talairach to MNI152',
        argstr='-tta2mni')

    mni2tta = traits.Bool(desc='transform dataset from MNI152 to Talaraich',
        argstr='-mni2tta')

    matparent = File(desc="apply transformation from 3dWarpDrive",
        argstr="-matparent %s",
        exists=True)

    deoblique = traits.Bool(desc='transform dataset from oblique to cardinal',
        argstr='-deoblique')

    interp = traits.Enum(('linear', 'cubic', 'NN', 'quintic'),
        desc='spatial interpolation methods [default = linear]',
        argstr='-%s')

    gridset = File(desc="copy grid of specified dataset",
        argstr="-gridset %s",
        exists=True)

    zpad = traits.Int(desc="pad input dataset with N planes" +
        " of zero on all sides.",
        argstr="-zpad %d")

    suffix = traits.Str('_warp',desc="out_file suffix",usedefault=True)

class WarpOutputSpec(TraitedSpec):
    out_file = File(desc='spatially transformed input image', exists=True)


class Warp(AFNICommand):
    """Use 3dWarp for spatially transforming a dataset

    For complete details, see the `3dWarp Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dWarp.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> warp = afni.Warp()
    >>> warp.inputs.in_file = example_data('structural.nii')
    >>> warp.inputs.deoblique = True
    >>> res = warp.run() # doctest: +SKIP

    """

    _cmd = '3dWarp'
    input_spec = WarpInputSpec
    output_spec = WarpOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs


class ResampleInputSpec(AFNITraitedSpec):

    in_file = File(desc='input file to 3dresample',
        argstr='-inset %s',
        position=-1,
        mandatory=True,
        exists=True)

    out_file = File(desc='output file from 3dresample',
        argstr='-prefix %s',
        position=-2,
        genfile=True,
        hash_files=False)

    orientation = traits.Str(desc='new orientation code',
        argstr='-orient %s')

    suffix = traits.Str('_resample', desc="out_file suffix",usedefault=True)


class ResampleOutputSpec(TraitedSpec):
    out_file = File(desc='reoriented or resampled file',
        exists=True)


class Resample(AFNICommand):
    """Resample or reorient an image using AFNI 3dresample command

    For complete details, see the `3dresample Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dresample.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> resample = afni.Resample()
    >>> resample.inputs.in_file = example_data('functional.nii')
    >>> resample.inputs.orientation= 'RPI'
    >>> res = resample.run() # doctest: +SKIP

    """

    _cmd = '3dresample'
    input_spec = ResampleInputSpec
    output_spec = ResampleOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs


class TStatInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dTstat',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)

    out_file = File(desc='output file from 3dTstat',
        argstr='-prefix %s',
        position=-2,
        genfile=True,
        hash_files=False)

    mask = File(desc='mask file',
        argstr='-mask %s',
        exists=True)

    options = traits.Str(desc='selected statistical output',
        argstr='%s')

    suffix = traits.Str('_tstat', desc="out_file suffix", usedefault=True)


class TStatOutputSpec(TraitedSpec):
    out_file = File(desc='statistical file',
        exists=True)


class TStat(AFNICommand):
    """Compute voxel-wise statistics using AFNI 3dTstat command

    For complete details, see the `3dTstat Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTstat.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> tstat = afni.TStat()
    >>> tstat.inputs.in_file = example_data('functional.nii')
    >>> tstat.inputs.args= '-mean'
    >>> res = tstat.run() # doctest: +SKIP

    """

    _cmd = '3dTstat'
    input_spec = TStatInputSpec
    output_spec = TStatOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs


class DetrendInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dDetrend',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)
    out_file = File(desc='output file from 3dDetrend',
         argstr='-prefix %s',
         position=-2,
         genfile=True,
         hash_files=False)
    suffix = traits.Str('_detrend', desc="out_file suffix", usedefault=True)


class DetrendOutputSpec(TraitedSpec):
    out_file = File(desc='statistical file',
        exists=True)


class Detrend(AFNICommand):
    """This program removes components from voxel time series using
    linear least squares

    For complete details, see the `3dDetrend Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDetrend.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> detrend = afni.Detrend()
    >>> detrend.inputs.in_file = example_data('functional.nii')
    >>> detrend.inputs.args = '-polort 2'
    >>> res = detrend.run() # doctest: +SKIP

    """

    _cmd = '3dDetrend'
    input_spec = DetrendInputSpec
    output_spec = DetrendOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs


class DespikeInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dDespike',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)

    out_file = File(desc='output file from 3dDespike',
         argstr='-prefix %s',
         position=-2,
         genfile=True,
         hash_files=False)

    suffix = traits.Str('_despike', desc="out_file suffix", usedefault=True)


class DespikeOutputSpec(TraitedSpec):
    out_file = File(desc='despiked img',
               exists=True)


class Despike(AFNICommand):
    """Removes 'spikes' from the 3D+time input dataset

    For complete details, see the `3dDespike Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDespike.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> despike = afni.Despike()
    >>> despike.inputs.in_file = example_data('functional.nii')
    >>> res = despike.run() # doctest: +SKIP

    """

    _cmd = '3dDespike'
    input_spec = DespikeInputSpec
    output_spec = DespikeOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs


class AutomaskInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dAutomask',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)

    out_file = File(desc='output file from 3dAutomask (a brain mask)',
        argstr='-prefix %s',
        position=-2,
        genfile=True,
        hash_files=False)

    apply_mask = File(desc="output file from 3dAutomask",
        argstr='-apply_prefix %s')

    clfrac = traits.Float(desc='sets the clip level fraction' +
        ' (must be 0.1-0.9). ' +
        'A small value will tend to make the mask larger [default = 0.5].',
        argstr="-dilate %s")

    dilate = traits.Int(desc='dilate the mask outwards',
        argstr="-dilate %s")

    erode = traits.Int(desc='erode the mask inwards',
        argstr="-erode %s")

    mask_suffix = traits.Str('_mask',desc="out_file suffix",usedefault=True)
    apply_suffix = traits.Str('_masked',desc="out_file suffix",usedefault=True)



class AutomaskOutputSpec(TraitedSpec):
    out_file = File(desc='mask file',
        exists=True)

    brain_file = File(desc='brain file (skull stripped)')


class Automask(AFNICommand):
    """Create a brain-only mask of the image using AFNI 3dAutomask command

    For complete details, see the `3dAutomask Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dAutomask.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> automask = afni.Automask()
    >>> automask.inputs.in_file = example_data('functional.nii')
    >>> automask.inputs.dilate = 1
    >>> res = automask.run() # doctest: +SKIP

    """

    _cmd = '3dAutomask'
    input_spec = AutomaskInputSpec
    output_spec = AutomaskOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file' or name == 'brain_file':
            return self._list_outputs()[name]
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(
                self.inputs.in_file, suffix=self.inputs.mask_suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)

        if not isdefined(self.inputs.apply_mask):
            outputs['brain_file'] = self._gen_fname(
                self.inputs.in_file, suffix=self.inputs.apply_suffix)
        else:
            outputs['brain_file'] = os.path.abspath(self.inputs.apply_mask)
        return outputs


class VolregInputSpec(AFNITraitedSpec):

    in_file = File(desc='input file to 3dvolreg',
       argstr='%s',
       position=-1,
       mandatory=True,
       exists=True)
    out_file = File(desc='output file from 3dvolreg',
       argstr='-prefix %s',
       position=-2,
       genfile=True,
       hash_files=False)
    basefile = File(desc='base file for registration',
        argstr='-base %s',
        position=-6,
        exists=True)
    zpad = traits.Int(desc='Zeropad around the edges' +
        ' by \'n\' voxels during rotations',
        argstr='-zpad %d',
        position=-5)
    md1dfile = File(desc='max displacement output file',
        argstr='-maxdisp1D %s',
        position=-4)
    oned_file = File(desc='1D movement parameters output file',
        argstr='-1Dfile %s',
        position=-3,
        genfile=True,
        hash_files=False)
    verbose = traits.Bool(desc='more detailed description of the process',
        argstr='-verbose')
    timeshift = traits.Bool(desc='time shift to mean slice time offset',
        argstr='-tshift 0')
    copyorigin = traits.Bool(desc='copy base file origin coords to output',
        argstr='-twodup')
    suffix = traits.Str('_volreg', desc="out_file suffix", usedefault=True)


class VolregOutputSpec(TraitedSpec):
    out_file = File(desc='registered file', exists=True)
    md1d_file = File(desc='max displacement info file', exists=True)
    oned_file = File(desc='movement parameters info file', exists=True)


class Volreg(AFNICommand):
    """Register input volumes to a base volume using AFNI 3dvolreg command

    For complete details, see the `3dvolreg Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dvolreg.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> volreg = afni.Volreg()
    >>> volreg.inputs.in_file = example_data('functional.nii')
    >>> volreg.inputs.other = '-Fourier -twopass'
    >>> volreg.inputs.zpad = '4'
    >>> res = volreg.run() # doctest: +SKIP

    """

    _cmd = '3dvolreg'
    input_spec = VolregInputSpec
    output_spec = VolregOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file' or name == 'oned_file':
            return self._list_outputs()[name]

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                                                     suffix=self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)

        if not isdefined(self.inputs.oned_file):
            outputs['oned_file'] = self._gen_fname(self.inputs.in_file,
                                            suffix = '%s.1D'%self.inputs.suffix)
        else:
            outputs['oned_file'] = os.path.abspath(self.inputs.oned_file)
        return outputs


class MergeInputSpec(AFNITraitedSpec):
    in_files = InputMultiPath(
        File(desc='input file to 3dmerge', exists=True),
        argstr='%s',
        position=-1,
        mandatory=True)
    out_file = File(desc='output file from 3dmerge',
         argstr='-prefix %s',
         position=-2,
         genfile=True,
         hash_files=False)
    doall = traits.Bool(desc='apply options to all sub-bricks in dataset',
        argstr='-doall')
    blurfwhm = traits.Int(desc='FWHM blur value (mm)',
          argstr='-1blur_fwhm %d',
          units='mm')
    suffix = traits.Str('_merge', desc="out_file suffix", usedefault=True)


class MergeOutputSpec(TraitedSpec):
    out_file = File(desc='smoothed file',
        exists=True)


class Merge(AFNICommand):
    """Merge or edit volumes using AFNI 3dmerge command

    For complete details, see the `3dmerge Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dmerge.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> merge = afni.Merge()
    >>> merge.inputs.in_files = example_data('functional.nii')
    >>> merge.inputs.blurfwhm = 4.0
    >>> merge.inputs.doall = True
    >>> merge.inputs.outfile = 'e7.nii'
    >>> res = merge.run() # doctest: +SKIP

    """

    _cmd = '3dmerge'
    input_spec = MergeInputSpec
    output_spec = MergeOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_files[0],
                                                  suffix=self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]


class CopyInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dcopy',
        argstr='%s',
        position=-2,
        mandatory=True,
        exists=True)
    out_file = File(desc='output file from 3dcopy',
        argstr='%s',
        position=-1,
        genfile=True,
        hash_files=False)
    suffix = traits.Str('_copy', desc="out_file suffix", usedefault=True)


class CopyOutputSpec(TraitedSpec):
    out_file = File(desc='copied file', exists=True)


class Copy(AFNICommand):
    """Copies an image of one type to an image of the same
    or different type using 3dcopy command

    For complete details, see the `3dcopy Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dcopy.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> copy = afni.Copy()
    >>> copy.inputs.in_file = example_data('functional.nii')
    >>> copy.inputs.out_file = 'new_func.nii'
    >>> res = copy.run() # doctest: +SKIP

    """

    _cmd = '3dcopy'
    input_spec = CopyInputSpec
    output_spec = CopyOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                                                  suffix=self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]


class FourierInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dFourier',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)
    out_file = File(desc='output file from 3dFourier',
         argstr='-prefix %s',
         position=-2,
         genfile=True,
         hash_files=False)
    lowpass = traits.Float(desc='lowpass',
        argstr='-lowpass %f',
        position=0,
        mandatory=True)
    highpass = traits.Float(desc='highpass',
        argstr='-highpass %f',
        position=1,
        mandatory=True)
    suffix = traits.Str('_fourier', desc="out_file suffix", usedefault=True)


class FourierOutputSpec(TraitedSpec):
    out_file = File(desc='band-pass filtered file', exists=True)


class Fourier(AFNICommand):
    """Program to lowpass and/or highpass each voxel time series in a
    dataset, via the FFT

    For complete details, see the `3dFourier Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dfourier.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> fourier = afni.Fourier()
    >>> fourier.inputs.in_file = example_data('functional.nii')
    >>> fourier.inputs.args = '-retrend'
    >>> fourier.inputs.highpass = 0.005
    >>> fourier.inputs.lowpass = 0.1
    >>> res = fourier.run() # doctest: +SKIP

    """

    _cmd = '3dFourier'
    input_spec = FourierInputSpec
    output_spec = FourierOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]

class BandpassInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dBandpass',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)
    out_file = File(desc='output file from 3dBandpass',
         argstr='-prefix %s',
         position=1 ,
         genfile=True)
    lowpass = traits.Float(desc='lowpass',
        argstr='%f',
        position=-2,
        mandatory=True)
    highpass = traits.Float(desc='highpass',
        argstr='%f',
        position=-3,
        mandatory=True)
    other = traits.Str(desc='other options',
        argstr='%s')
    mask = File(desc='mask file',
        position=2,
        argstr='-mask %s',
        exists=True)
    suffix = traits.Str('_bandpass', desc="out_file suffix", usedefault=True)
    
class BandpassOutputSpec(TraitedSpec):
    out_file = File(desc='band-pass filtered file',
        exists=True)


class Bandpass(AFNICommand):
    """Program to lowpass and/or highpass each voxel time series in a
    dataset, offering more/different options than Fourier

    For complete details, see the `3dBandpass Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dbandpass.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> bandpass = afni.Bandpass()
    >>> bandpass.inputs.in_file = example_data('functional.nii')
    >>> bandpass.inputs.highpass = 0.005
    >>> bandpass.inputs.lowpass = 0.1
    >>> res = bandpass.run() # doctest: +SKIP

    """

    _cmd = '3dBandpass'
    input_spec = BandpassInputSpec
    output_spec = BandpassOutputSpec


    def _list_outputs(self):
        outputs = self.output_spec().get()

        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                                                  suffix=self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]

class ZCutUpInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dZcutup',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)
    out_file = File(desc='output file from 3dZcutup',
         argstr='-prefix %s',
         position=-2,
         mandatory=True,
         hash_files=False)
    keep = traits.Str(desc='slice range to keep in output',
            argstr='-keep %s')
    suffix = traits.Str('_zcutup', desc="out_file suffix", usedefault=True)


class ZCutUpOutputSpec(TraitedSpec):
    out_file = File(desc='cut file', exists=True)


class ZCutUp(AFNICommand):
    """Cut z-slices from a volume using AFNI 3dZcutup command

    For complete details, see the `3dZcutup Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dZcutup.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> zcutup = afni.Zcutup()
    >>> zcutup.inputs.in_file = example_data('functional.nii')
    >>> zcutup.inputs.outfile= 'functional_zcutup.nii'
    >>> zcutup.inputs.keep= '0 10'
    >>> res = zcutup.run() # doctest: +SKIP

    """

    _cmd = '3dZcutup'
    input_spec = ZCutUpInputSpec
    output_spec = ZCutUpOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]


class AllineateInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dAllineate',
        argstr='-source %s',
        position=-1,
        mandatory=True,
        exists=True)

    reference = File(
        exists=True,
        argstr='-base %s',
        desc="""file to be used as reference, the first volume will be used
if not given the reference will be the first volume of in_file.""")

    out_file = File(desc='output file from 3dAllineate',
         argstr='-prefix %s',
         position=-2,
         genfile=True,
         hash_files=False)

    suffix = traits.Str('_allineate', desc="out_file suffix", usedefault=True)

    out_param_file = File(
        argstr = '-1Dparam_save %s',
        desc = 'Save the warp parameters in ASCII (.1D) format.')
    in_param_file = File(
        exists = True,
        argstr = '-1Dparam_apply %s',
        desc = """Read warp parameters from file and apply them to 
                  the source dataset, and produce a new dataset""")
    out_matrix = File(
        argstr='-1Dmatrix_save %s',
        desc='Save the transformation matrix for each volume.')
    in_matrix = File(desc='matrix to align input file',
        argstr='-1Dmatrix_apply %s',
        position=-3)

    _cost_funcs = [
        'leastsq', 'ls',
        'mutualinfo', 'mi',
        'corratio_mul', 'crM',
        'norm_mutualinfo', 'nmi',
        'hellinger', 'hel',
        'corratio_add', 'crA',
        'corratio_uns', 'crU']

    cost = traits.Enum(
        *_cost_funcs, argstr='-cost %s',
        desc="""Defines the 'cost' function that defines the matching
                between the source and the base""")
    _interp_funcs= ['nearestneighbour','linear','cubic','quintic','wsinc5']
    interpolation = traits.Enum(
        *_interp_funcs[:-1], argstr='-interp %s',
        desc='Defines interpolation method to use during matching')
    final_interpolation = traits.Enum(
        *_interp_funcs, argstr='-final %s',
        desc='Defines interpolation method used to create the output dataset')
    
    #   TECHNICAL OPTIONS (used for fine control of the program):
    nmatch = traits.Int(
        argstr='-nmatch %d',
        desc='Use at most n scattered points to match the datasets.')
    no_pad = traits.Bool(
        argstr='-nopad',
        desc='Do not use zero-padding on the base image.')
    zclip = traits.Bool(
        argstr='-zclip',
        desc='Replace negative values in the input datasets (source & base) with zero.')
    convergence = traits.Float(
        argstr='-conv %f',
        desc='Convergence test in millimeters (default 0.05mm).')
    usetemp = traits.Bool(argstr='-usetemp', desc='temporary file use')
    check = traits.List(
        traits.Enum(*_cost_funcs), argstr='-check %s',
        desc="""After cost functional optimization is done, start at the
                final parameters and RE-optimize using this new cost functions.
                If the results are too different, a warning message will be 
                printed.  However, the final parameters from the original
                optimization will be used to create the output dataset.""")

    #      ** PARAMETERS THAT AFFECT THE COST OPTIMIZATION STRATEGY **
    one_pass = traits.Bool(
        argstr='-onepass', 
        desc = """Use only the refining pass -- do not try a coarse
                resolution pass first.  Useful if you know that only
                small amounts of image alignment are needed.""")
    two_pass = traits.Bool(
        argstr='-twopass',
        desc="""Use a two pass alignment strategy for all volumes, searching
              for a large rotation+shift and then refining the alignment.""")
    two_blur = traits.Float(
        argstr='-twoblur',
        desc='Set the blurring radius for the first pass in mm.')
    two_first = traits.Bool(
        argstr='-twofirst',
        desc = """Use -twopass on the first image to be registered, and
               then on all subsequent images from the source dataset,
               use results from the first image's coarse pass to start
               the fine pass.""")
    two_best = traits.Int(
        argstr='-twobest %d',
        desc="""In the coarse pass, use the best 'bb' set of initial
               points to search for the starting point for the fine
               pass.  If bb==0, then no search is made for the best
               starting point, and the identity transformation is
               used as the starting point.  [Default=5; min=0 max=11]""")
    fine_blur = traits.Float(
        argstr='-fineblur %f',
        desc="""Set the blurring radius to use in the fine resolution
               pass to 'x' mm.  A small amount (1-2 mm?) of blurring at
               the fine step may help with convergence, if there is
               some problem, especially if the base volume is very noisy.
               [Default == 0 mm = no blurring at the final alignment pass]""")

    center_of_mass = traits.Str(
        argstr = '-cmass%s',
        desc='Use the center-of-mass calculation to bracket the shifts.')
    autoweight = traits.Str(
        argstr='-autoweight%s',
        desc="""Compute a weight function using the 3dAutomask
               algorithm plus some blurring of the base image.""")
    automask = traits.Int(
        argstr='-automask+%d',
        desc="""Compute a mask function, set a value for dilation or 0.""")
    autobox = traits.Bool(
        argstr='-autobox',
        desc="""Expand the -automask function to enclose a rectangular
                box that holds the irregular mask.""")
    nomask = traits.Bool(
        argstr='-nomask',
        desc="""Don't compute the autoweight/mask; if -weight is not
                also used, then every voxel will be counted equally.""")
    weight_file = File(
        argstr='-weight %s',exists=True,
        desc="""Set the weighting for each voxel in the base dataset;
                larger weights mean that voxel count more in the cost function.
                Must be defined on the same grid as the base dataset""")
    out_weight_file = traits.File(
        argstr='-wtprefix %s',
        desc="""Write the weight volume to disk as a dataset""")
    
    source_mask = File(
        exists=True, argstr='-source_mask %s',
        desc='mask the input dataset')
    source_automask = traits.Int(
        argstr='-source_automask+%d',
        desc='Automatically mask the source dataset with dilation or 0.')
    warp_type = traits.Enum(
        'shift_only','shift_rotate','shift_rotate_scale','affine_general',
        argstr='-warp %s',
        desc='Set the warp type.')
    warpfreeze = traits.Bool(
        argstr='-warpfreeze',
        desc='Freeze the non-rigid body parameters after first volume.')
    replacebase = traits.Bool(
        argstr='-replacebase',
        desc="""If the source has more than one volume, then after the first 
                volume is aligned to the base""")
    replacemeth = traits.Enum(
        *_cost_funcs,
        argstr='-replacemeth %s',
        desc="""After first volume is aligned, switch method for later volumes.
                For use with '-replacebase'.""")
    epi = traits.Bool(
        argstr='-EPI',
        desc="""Treat the source dataset as being composed of warped
                EPI slices, and the base as comprising anatomically
                'true' images.  Only phase-encoding direction image
                shearing and scaling will be allowed with this option.""")
    master = File(
        exists=True, argstr='-master %s',
        desc='Write the output dataset on the same grid as this file')
    newgrid = traits.Float(
        argstr='-newgrid %f',
        desc='Write the output dataset using isotropic grid spacing in mm')
    
    
    #Non-linear experimental
    _nwarp_types=['bilinear',
                  'cubic', 'quintic', 'heptic', 'nonic',
                  'poly3', 'poly5', 'poly7',  'poly9'] # same non-hellenistic
    nwarp = traits.Enum(
        *_nwarp_types, argstr='-nwarp %s',
         desc='Experimental nonlinear warping: bilinear or legendre poly.')
    _dirs = ['X','Y','Z','I','J','K']
    nwarp_fixmot = traits.List(
        traits.Enum(*_dirs),
        argstr='-nwarp_fixmot%s',
        desc='To fix motion along directions.')
    nwarp_fixdep = traits.List(
        traits.Enum(*_dirs),
        argstr='-nwarp_fixdep%s',
        desc='To fix non-linear warp dependency along directions.')

class AllineateOutputSpec(TraitedSpec):
    out_file = File(desc='out file',exists=True)

class Allineate(AFNICommand):
    """Program to align one dataset (the 'source') to a base dataset

    For complete details, see the `3dAllineate Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dAllineate.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> allineate = afni.Allineate()
    >>> allineate.inputs.in_file = example_data('functional.nii')
    >>> allineate.inputs.outfile= 'functional_allineate.nii'
    >>> allineate.inputs.matrix= example_data('cmatrix.mat')
    >>> res = allineate.run() # doctest: +SKIP

    """

    _cmd = '3dAllineate'
    input_spec = AllineateInputSpec
    output_spec = AllineateOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'nwarp_fixmot' or name == 'nwarp_fixdep':
            arg = ' '.join([trait_spec.argstr % v for v in value])
            return arg
        return super(Allineate, self)._format_arg(name, trait_spec, value)


    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]


class MaskaveInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dmaskave',
        argstr='%s',
        position=-2,
        mandatory=True,
        exists=True)
    out_file = File(desc='output to the file',
         argstr='> %s',
         position=-1,
         genfile=True,
         hash_files=False)
    mask = File(desc='matrix to align input file',
        argstr='-mask %s',
        position=1,
        exists=True)

    quiet = traits.Bool(desc='matrix to align input file',
        argstr='-quiet',
        position=2)
    suffix = traits.Str('_maskave', desc="out_file suffix", usedefault=True)


class MaskaveOutputSpec(TraitedSpec):
    out_file = File(desc='outfile',
          exists=True)


class Maskave(AFNICommand):
    """Computes average of all voxels in the input dataset
    which satisfy the criterion in the options list

    For complete details, see the `3dmaskave Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dmaskave.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> maskave = afni.Maskave()
    >>> maskave.inputs.in_file = example_data('functional.nii')
    >>> maskave.inputs.mask= example_data('seed_mask.nii')
    >>> maskave.inputs.quiet= True
    >>> maskave.inputs.out_file= 'maskave.1D'
    >>> res = maskave.run() # doctest: +SKIP

    """

    _cmd = '3dmaskave'
    input_spec = MaskaveInputSpec
    output_spec = MaskaveOutputSpec


    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]

class SkullStripInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dSkullStrip',
        argstr='-input %s',
        position=1,
        mandatory=True,
        exists=True)
    out_file = File(desc='output to the file',
         argstr='%s',
         position=-1,
        genfile=True,
        hash_files=False)
    suffix = traits.Str('_skullstrip', desc="out_file suffix", usedefault=True)


class SkullStripOutputSpec(TraitedSpec):
    out_file = File(desc='outfile',
        exists=True)


class SkullStrip(AFNICommand):
    """A program to extract the brain from surrounding
    tissue from MRI T1-weighted images

    For complete details, see the `3dSkullStrip Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dSkullStrip.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> skullstrip = afni.Skullstrip()
    >>> skullstrip.inputs.in_file = example_data('functional.nii')
    >>> skullstrip.inputs.args = '-o_ply'
    >>> res = skullstrip.run() # doctest: +SKIP

    """
    _cmd = '3dSkullStrip'
    input_spec = SkullStripInputSpec
    output_spec = SkullStripOutputSpec


    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]


class TCatInputSpec(AFNITraitedSpec):
    in_files = InputMultiPath(
        File(exists=True),
        desc='input file to 3dTcat',
        argstr=' %s',
        position=-1,
        mandatory=True)
    out_file = File(desc='output to the file',
         argstr='-prefix %s',
         position=-2,
         genfile=True,
         hash_files=False)
    rlt = traits.Str(desc='options', argstr='-rlt%s', position=1)
    suffix = traits.Str('_tcat', desc="out_file suffix", usedefault=True)


class TCatOutputSpec(TraitedSpec):
    out_file = File(desc='outfile',
        exists=True)


class TCat(AFNICommand):
    """Concatenate sub-bricks from input datasets into
    one big 3D+time dataset

    For complete details, see the `3dTcat Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTcat.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> tcat = afni.TCat()
    >>> tcat.inputs.in_file = example_data('functional.nii')
    >>> tcat.inputs.out_file= 'functional_tcat.nii'
    >>> tcat.inputs.rlt = '+'
    >>> res = tcat.run() # doctest: +SKIP

    """

    _cmd = '3dTcat'
    input_spec = TCatInputSpec
    output_spec = TCatOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_files[0],
                                                  suffix=self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]


class FimInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dfim+',
        argstr=' -input %s',
        position=1,
        mandatory=True,
        exists=True)
    ideal_file = File(desc='ideal time series file name',
        argstr='-ideal_file %s',
        position=2,
        mandatory=True,
        exists=True)
    fim_thr = traits.Float(desc='fim internal mask threshold value',
        argstr='-fim_thr %f', position=3)

    out = traits.Str(desc='Flag to output the specified parameter',
        argstr='-out %s', position=4)

    out_file = File(desc='output file from 3dfim+', argstr='-bucket %s',
        position=-1, genfile=True, hash_files=False)
    suffix = traits.Str('_fim', desc="out_file suffix", usedefault=True)


class FimOutputSpec(TraitedSpec):
    out_file = File(desc='outfile',
        exists=True)


class Fim(AFNICommand):
    """Program to calculate the cross-correlation of
    an ideal reference waveform with the measured FMRI
    time series for each voxel

    For complete details, see the `3dfim+ Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dfim+.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> fim = afni.Fim()
    >>> fim.inputs.in_file = example_data('functional.nii')
    >>> fim.inputs.ideal_file= example_data('seed.1D')
    >>> fim.inputs.out_file = 'functional_corr.nii'
    >>> fim.inputs.out = 'Correlation'
    >>> fim.inputs.fim_thr = 0.0009
    >>> res = fim.run() # doctest: +SKIP

    """

    _cmd = '3dfim+'
    input_spec = FimInputSpec
    output_spec = FimOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]


class TCorrelateInputSpec(AFNITraitedSpec):
    xset = File(desc='input xset',
        argstr=' %s',
        position=-2,
        mandatory=True,
        exists=True)
    yset = File(desc='input yset',
        argstr=' %s',
        position=-1,
        mandatory=True,
        exists=True)
    pearson = traits.Bool(desc='Correlation is the normal' +
        ' Pearson correlation coefficient',
        argstr='-pearson',
        position=1)
    polort = traits.Int(desc='Remove polynomical trend of order m',
        argstr='-polort %d', position=2)

    out_file = File(desc='Save output into dataset with prefix ',
        argstr='-prefix %s',
        position=3, genfile=True, hash_files=False)

    suffix = traits.Str('_tcor', desc="out_file suffix", usedefault=True)


class TCorrelateOutputSpec(TraitedSpec):
    out_file = File(desc='outfile',
        exists=True)


class TCorrelate(AFNICommand):
    """Computes the correlation coefficient between corresponding voxel
    time series in two input 3D+time datasets 'xset' and 'yset'

    For complete details, see the `3dTcorrelate Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTcorrelate.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> tcorrelate = afni.TCorrelate()
    >>> tcorrelate.inputs.in_file = example_data('functional.nii')
    >>> tcorrelate.inputs.xset= example_data('u_rc1s1_Template.nii')
    >>> tcorrelate.inputs.yset = example_data('u_rc1s2_Template.nii')
    >>> tcorrelate.inputs.out_file = 'functional_tcorrelate.nii.gz'
    >>> tcorrelate.inputs.polort = -1
    >>> tcorrelate.inputs.pearson = True
    >>> res = tcarrelate.run() # doctest: +SKIP

    """

    _cmd = '3dTcorrelate'
    input_spec = TCorrelateInputSpec
    output_spec = TCorrelateOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.xset,
                                                  suffix=self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]


class BrickStatInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dmaskave',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)

    mask = File(desc='-mask dset = use dset as mask to include/exclude voxels',
        argstr='-mask %s',
        position=2,
        exists=True)

    min = traits.Bool(desc='print the minimum value in dataset',
        argstr='-min',
        position=1)


class BrickStatOutputSpec(TraitedSpec):
    min_val = traits.Float(desc='output')


class BrickStat(AFNICommand):
    """Compute maximum and/or minimum voxel values of an input dataset

    For complete details, see the `3dBrickStat Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dBrickStat.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> brickstat = afni.BrickStat()
    >>> brickstat.inputs.in_file = example_data('functional.nii')
    >>> brickstat.inputs.mask = example_data('skeleton_mask.nii.gz')
    >>> brickstat.inputs.min = True
    >>> res = brickstat.run() # doctest: +SKIP

    """
    _cmd = '3dBrickStat'
    input_spec = BrickStatInputSpec
    output_spec = BrickStatOutputSpec

    def aggregate_outputs(self, runtime=None, needed_outputs=None):

        outputs = self._outputs()

        outfile = os.path.join(os.getcwd(), 'stat_result.json')

        if runtime is None:
            try:
                min_val = load_json(outfile)['stat']
            except IOError:
                return self.run().outputs
        else:
            min_val = []
            for line in runtime.stdout.split('\n'):
                if line:
                    values = line.split()
                    if len(values) > 1:
                        min_val.append([float(val) for val in values])
                    else:
                        min_val.extend([float(val) for val in values])

            if len(min_val) == 1:
                min_val = min_val[0]
            save_json(outfile, dict(stat=min_val))
        outputs.min_val = min_val

        return outputs


class ROIStatsInputSpec(AFNITraitedSpec):
    in_file = File(desc='input file to 3dROIstats',
        argstr='%s',
        position=-1,
        mandatory=True,
        exists=True)

    mask = File(desc='input mask',
        argstr='-mask %s',
        position=3,
        exists=True)

    mask_f2short = traits.Bool(
        desc='Tells the program to convert a float mask ' +
            'to short integers, by simple rounding.',
        argstr='-mask_f2short',
        position=2)

    quiet = traits.Bool(desc='execute quietly',
        argstr='-quiet',
        position=1)


class ROIStatsOutputSpec(TraitedSpec):
    stats = File(desc='output', exists=True)


class ROIStats(AFNICommand):
    """Display statistics over masked regions

    For complete details, see the `3dROIstats Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dROIstats.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> roistats = afni.ROIStats()
    >>> roistats.inputs.in_file = example_data('functional.nii')
    >>> roistats.inputs.mask = example_data('skeleton_mask.nii.gz')
    >>> roistats.inputs.quiet=True
    >>> res = roistats.run() # doctest: +SKIP

    """
    _cmd = '3dROIstats'
    input_spec = ROIStatsInputSpec
    output_spec = ROIStatsOutputSpec

    def aggregate_outputs(self, runtime=None, needed_outputs=None):

        outputs = self._outputs()

        outfile = os.path.join(os.getcwd(), 'stat_result.json')

        if runtime is None:
            try:
                stats = load_json(outfile)['stat']
            except IOError:
                return self.run().outputs
        else:
            stats = []
            for line in runtime.stdout.split('\n'):
                if line:
                    values = line.split()
                    if len(values) > 1:
                        stats.append([float(val) for val in values])
                    else:
                        stats.extend([float(val) for val in values])

            if len(stats) == 1:
                stats = stats[0]
            of = os.path.join(os.getcwd(), 'TS.1D')
            f = open(of, 'w')

            for st in stats:
                f.write(str(st) + '\n')
            f.close()
            save_json(outfile, dict(stat=of))
        outputs.stats = of

        return outputs


"""
3dcalc -a ${rest}.nii.gz[${TRstart}..${TRend}] -expr 'a' -prefix $
{rest}_dr.nii.gz

3dcalc -a ${rest}_mc.nii.gz -b ${rest}_mask.nii.gz -expr 'a*b' -prefix
${rest}_ss.nii.gz
"""


class CalcInputSpec(AFNITraitedSpec):
    in_file_a = File(desc='input file to 3dcalc',
        argstr='-a %s', position=0, mandatory=True, exists=True)
    in_file_b = File(desc='operand file to 3dcalc',
        argstr=' -b %s', position=1, exists=True)
    expr = traits.Str(desc='expr', argstr='-expr %s', position=2,
        mandatory=True)
    out_file = File(desc='output file from 3dFourier', argstr='-prefix %s',
        position=-1, genfile=True)
    start_idx = traits.Int(desc='start index for in_file_a',
        requires=['stop_idx'])
    stop_idx = traits.Int(desc='stop index for in_file_a',
        requires=['start_idx'])
    single_idx = traits.Int(desc='volume index for in_file_a')
    suffix = traits.Str('_calc', desc="out_file suffix", usedefault=True)


class CalcOutputSpec(TraitedSpec):
    out_file = File(desc=' output file', exists=True)


class Calc(AFNICommand):
    """This program does voxel-by-voxel arithmetic on 3D datasets

    For complete details, see the `3dcalc Documentation.
    <http://afni.nimh.nih.gov/pub/dist/doc/program_help/3dcalc.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> from nipype.testing import  example_data
    >>> calc = afni.Calc()
    >>> calc.inputs.in_file_a = example_data('functional.nii')
    >>> calc.inputs.in_file_b = example_data('functional2.nii.gz')
    >>> calc.inputs.expr='a*b'
    >>> calc.inputs.out_file =  'functional_calc.nii.gz'
    >>> res = calc.run() # doctest: +SKIP

    """

    _cmd = '3dcalc'
    input_spec = CalcInputSpec
    output_spec = CalcOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file_a,
                                                  suffix=self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]

    def _format_arg(self, name, trait_spec, value):
        if name == 'in_file_a':
            arg = trait_spec.argstr % value
            if isdefined(self.inputs.start_idx):
                arg += '[%d..%d]' % (self.inputs.start_idx,
                    self.inputs.stop_idx)
            if isdefined(self.inputs.single_idx):
                arg += '[%d]' % (self.inputs.single_idx)
            return arg
        return super(Calc, self)._format_arg(name, trait_spec, value)

    def _parse_inputs(self, skip=None):
        """Skip the arguments without argstr metadata
        """
        return super(Calc, self)._parse_inputs(
            skip=('start_idx', 'stop_idx', 'other'))


class BlurInMaskInputSpec(AFNITraitedSpec):
    in_file = File(
        desc='input file to 3dSkullStrip',
        argstr='-input %s',
        position=1,
        mandatory=True,
        exists=True)
    out_file = File(
        desc='output to the file',
        argstr='-prefix %s',
        position=-1,
        genfile=True)
    mask = File(
        desc='Mask dataset, if desired.  Blurring will occur only within the mask.  Voxels NOT in the mask will be set to zero in the output.',
        argstr='-mask %s')
    multimask = File(
        desc='Multi-mask dataset -- each distinct nonzero value in dataset will be treated as a separate mask for blurring purposes.',
        argstr='-Mmask %s')
    automask = traits.Bool(
        desc='Create an automask from the input dataset.',
        argstr='-automask')
    fwhm = traits.Float(
        desc='fwhm kernel size', 
        argstr='-FWHM %f',
        mandatory=True)
    preserve = traits.Bool(
        desc = 'Normally, voxels not in the mask will be set to zero in the output.  If you want the original values in the dataset to be preserved in the output, use this option.',
        argstr = '-preserve')
    float_out = traits.Bool(
        desc= 'Save dataset as floats, no matter what the input data type is.',
        argstr='-float')
    options = traits.Str(desc='options', argstr='%s', position=2)
    suffix = traits.Str('_blurmask', desc="out_file suffix", usedefault=True)

class BlurInMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True)

class BlurInMask(AFNICommand):
    _cmd = '3dBlurInMask'
    input_spec  = BlurInMaskInputSpec
    output_spec = BlurInMaskOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file,
                suffix = self.inputs.suffix)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
    

class TCorrMapInputSpec(AFNITraitedSpec):
    in_file = File(exists=True, argstr='-input %s', mandatory=True)
    seeds = File(exists=True, argstr='-seed %s',xor=('seeds_width'))
    mask = File(exists=True, argstr='-mask %s')
    automask = traits.Bool(argstr='-automask')
    polort = traits.Int(argstr='-polort %d')
    bandpass = traits.Tuple((traits.Float(), traits.Float()),
                             argstr='-bpass %f %f')
    regress_out_timeseries = traits.File(exists=True, argstr='-ort %s')
    blur_fwhm = traits.Float(argstr='-Gblur %f')
    seeds_width = traits.Float(argstr='-Mseed %f', xor=('seeds'))
    
    #outputs
    mean_file = File(genfile=True,argstr='-Mean %s',suffix='_mean')
    zmean = File(genfile=True,argstr='-Zmean %s',suffix='_zmean')
    qmean = File(genfile=True,argstr='-Qmean %s',suffix='_qmean')
    pmean = File(genfile=True,argstr='-Pmean %s',suffix='_pmean')

    
    _thresh_opts = ('absolute_threshold','var_absolute_threshold','var_absolute_threshold_normalize')
    thresholds = traits.List(traits.Int())
    absolute_threshold = File(
        genfile=True,
        argstr='-Thresh %f %s', suffix='_thresh',
        xor=_thresh_opts)
    var_absolute_threshold = File(
        genfile=True,
        argstr='-VarThresh %f %f %f %s', suffix='_varthresh',
        xor=_thresh_opts)
    var_absolute_threshold_normalize = File(
        genfile=True,
        argstr='-VarThreshN %f %f %f %s', suffix='_varthreshn',
        xor=_thresh_opts)

    correlation_maps = File(
        genfile=True, argstr='-CorrMap %s', suffix='_corrmap')
    correlation_maps_masked = File(
        genfile=True, argstr='-CorrMask %s', suffix='_corrmask')

    _expr_opts = ('average_expr','average_expr_nonzero','sum_expr')
    expr = traits.Str()
    average_expr = File(
        genfile=True,
        argstr='-Aexpr %s %s', suffix='_aexpr',
        xor=_expr_opts)
    average_expr_nonzero  = File(
        genfile=True,
        argstr='-Cexpr %s %s', suffix='_cexpr',
        xor=_expr_opts)
    sum_expr = File(
        genfile=True,
        argstr='-Sexpr %s %s', suffix='_sexpr',
        xor=_expr_opts)
    histogram_bin_numbers = traits.Int()
    histogram = File(
        genfile=True,
        argstr='-Hist %d %s', suffix='_hist')

class TCorrMapOutputSpec(TraitedSpec):

    mean_file = File(genfile=True)
    zmean = File()
    qmean = File()
    pmean = File()
    absolute_threshold = File()
    var_absolute_threshold = File()
    var_absolute_threshold_normalize = File()
    correlation_maps = File()
    correlation_maps_masked = File()
    average_expr = File()
    average_expr_nonzero = File()
    sum_expr = File()
    histogram = File()
    
class TCorrMap(AFNICommand):
    _cmd = '3dTcorrMap'
    input_spec  = TCorrMapInputSpec
    output_spec = TCorrMapOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name in self.inputs._thresh_opts:
            return trait_spec.argstr % self.inputs.thresholds + [value]
        elif name in self.inputs._expr_opts:
            return trait_spec.argstr % (self.inputs.expr, value)
        else:
            return super(TCorrMap, self)._format_arg(name,trait_spec,value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for o in self._outputs().get().keys():
            ov = getattr(self.inputs,o)
            if not isdefined(ov):
                ov = self._gen_fname(
                    o, suffix=self.input_spec.class_traits()[o].suffix)
            outputs[o]=ov
        return outputs

    def _parse_inputs(self,skip=None):
        outs = self._list_outputs()
        #skip under
        if skip==None:
            skip=[]
        skip.extend([k for k in self._outputs().get().keys() if not isdefined(outs[k])])
        return super(TCorrMap, self)._parse_inputs(skip=skip)
        
    def _gen_filename(self, name):
        if name in self._outputs().get().keys():
            return self._list_outputs()[name]


class AutoboxInputSpec(AFNITraitedSpec):
    in_file = File(exists=True, mandatory=True, argstr='-input %s')
    padding = traits.Int(argstr='-npad %d')
    out_file = File(argstr="-prefix %s")
    no_clustering = traits.Bool(argstr='-noclust')

class AutoboxOuputSpec(TraitedSpec):
    x_min = traits.Int()
    x_max = traits.Int()
    y_min = traits.Int()
    y_max = traits.Int()
    z_min = traits.Int()
    z_max = traits.Int()

    out_file = File()

class Autobox(AFNICommand):
    _cmd = '3dAutobox'
    input_spec  = AutoboxInputSpec
    output_spec = AutoboxOuputSpec
    
    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = self._outputs()
        pattern = 'x=(?P<x_min>-?\d+)\.\.(?P<x_max>-?\d+)  y=(?P<y_min>-?\d+)\.\.(?P<y_max>-?\d+)  z=(?P<z_min>-?\d+)\.\.(?P<z_max>-?\d+)'
        for line in runtime.stderr.split('\n'):
            m = re.search(pattern,line)
            if m:
                d=m.groupdict()
                for k in d.keys():
                    d[k]=int(d[k])
                outputs.set(**d)
        if isdefined(self.inputs.out_file):
            outputs.set(out_file = os.path.abspath(self.inputs.out_file))
        return outputs
