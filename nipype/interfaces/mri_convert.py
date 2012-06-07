from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    InputMultiPath, traits, TraitedSpec,
                                    OutputMultiPath, isdefined,
                                    File, Directory)
import os
from copy import deepcopy
from nipype.utils.filemanip import split_filename
import re, glob

class MRIConvertInputSpec(CommandLineInputSpec):
    in_folder = Directory(desc='folder with DICOM images to convert',
        argstr='%s/*.dcm',
        position=-1,
        mandatory=True,
        exists=True)

    output_dir = Directory(exists=True, argstr='-o %s', 
                           genfile=True)

    _file_types = ['fsl','spm','meta','nifti','analyze','bv']
    out_filetype = traits.Enum(_file_types[0], _file_types, argstr='-f %s',
                               usedefault=True, mandatory=True)
    filename_format = traits.Str(
        argstr='-F %s',
        desc="""Use this format  for  name  of  output  file:  
+/-  PatientName, PatientId,  SeriesDate,  SeriesTime,  StudyId,
StudyDescription, SeriesNumber, SequenceName, ProtocolName, 
SeriesDescription""")

    rescale = traits.Bool(
        argstr='-r',
        desc='Apply rescale slope and intercept to data')

    four_d = traits.Bool(True, argstr='-d', usedefault=True,
                         desc='Save output volumes as 4D files')
    gzip_output = traits.Bool(desc='compress output, only for nifti nii files')
    nii = traits.Bool(True, usedefault=True, argstr='-n',
                      desc='save as nii file')
    header_only = traits.Bool(argstr='-a')
    skip_volumes = traits.Int(
        argstr='-s %d',
        desc = 'Skip n first volumes.')
    match_series = traits.Str(
        argstr='-m \'%s\'', 
        desc='Only convert files whose series description include this string')
    id_for_name = traits.Bool(argstr='-u')

    args = traits.Str(argstr='%s', desc='Additional parameters to the command')

class MRIConvertOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(File(exists=True))

class MRIConvert(CommandLine):
    input_spec=MRIConvertInputSpec
    output_spec=MRIConvertOutputSpec

    _cmd = 'mcverter'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outdir=self._gen_filename('output_dir')
        if self.inputs.gzip_output and \
                self.inputs.out_filetype in ['fsl','nifti'] and \
                self.inputs.nii:
            outputs['out_files'] = sorted(glob.glob(os.path.join(outdir,'*.nii.gz')))
        else:
            outputs['out_files'] = sorted(glob.glob(os.path.join(outdir,'*.nii')))
        return outputs

    def _gen_filename(self, name):
        if name == 'output_dir':
            if isdefined(self.inputs.output_dir):
                return self.inputs.output_dir
            else:
                return os.getcwd()
        return None

    @property
    def cmdline(self):
        cmd = super(MRIConvert, self).cmdline
        if self.inputs.gzip_output and \
                self.inputs.out_filetype in ['fsl','nifti'] and \
                self.inputs.nii:
            cmd += ';gzip -f %s;' % os.path.join(self._gen_filename('output_dir'), '*.nii')
        return cmd
