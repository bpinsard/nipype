"""Autogenerated file - DO NOT EDIT         
If you spot a bug, please report it on the mailing list and/or change the generator."""

from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class CastInputSpec(CommandLineInputSpec):
    InputVolume = File(position="0", desc="Input volume, the volume to cast.", exists=True, argstr="--InputVolume %s")
    OutputVolume = traits.Either(traits.Bool, File(), position="1", hash_files=False, desc="Output volume, cast to the new type.", argstr="--OutputVolume %s")
    type = traits.Enum("Char", "UnsignedChar", "Short", "UnsignedShort", "Int", "UnsignedInt", "Float", "Double", desc="Type for the new output volume.", argstr="--type %s")


class CastOutputSpec(TraitedSpec):
    OutputVolume = File(position="1", desc="Output volume, cast to the new type.", exists=True)


class Cast(SlicerCommandLine):
    """title: Cast Image

category: Filtering.Arithmetic

description: 
Cast a volume to a given data type.
Use at your own risk when casting an input volume into a lower precision type!
Allows casting to the same type as the input volume.

version: 0.1.0.$Revision: 2104 $(alpha)

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/Cast

contributor: Nicole Aucoin, BWH (Ron Kikinis, BWH)

acknowledgements: 
This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.


"""

    input_spec = CastInputSpec
    output_spec = CastOutputSpec
    _cmd = " Cast "
    _outputs_filenames = {'OutputVolume':'OutputVolume.nii'}


class AddInputSpec(CommandLineInputSpec):
    inputVolume1 = File(position="0", desc="Input volume 1", exists=True, argstr="--inputVolume1 %s")
    inputVolume2 = File(position="1", desc="Input volume 2", exists=True, argstr="--inputVolume2 %s")
    outputVolume = traits.Either(traits.Bool, File(), position="2", hash_files=False, desc="Volume1 + Volume2", argstr="--outputVolume %s")
    order = traits.Enum("0", "1", "2", "3", desc="Interpolation order if two images are in different coordinate frames or have different sampling.", argstr="--order %s")


class AddOutputSpec(TraitedSpec):
    outputVolume = File(position="2", desc="Volume1 + Volume2", exists=True)


class Add(SlicerCommandLine):
    """title: Add Images

category: Filtering.Arithmetic

description: 
Adds two images. Although all image types are supported on input, only signed types are produced. The two images do not have to have the same dimensions.


version: 0.1.0.$Revision: 18864 $(alpha)

documentation-url: http://slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/Add

contributor: Bill Lorensen

acknowledgements: 
This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.


"""

    input_spec = AddInputSpec
    output_spec = AddOutputSpec
    _cmd = " Add "
    _outputs_filenames = {'outputVolume':'outputVolume.nii'}
