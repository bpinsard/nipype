# -*- coding: utf8 -*- 
"""Autogenerated file - DO NOT EDIT
If you spot a bug, please report it on the mailing list and/or change the generator."""

from nipype.interfaces.base import CommandLine, CommandLineInputSpec, SEMLikeCommandLine, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os


class N4ITKBiasFieldCorrectionInputSpec(CommandLineInputSpec):
    inputimage = File(desc="Input image where you observe signal inhomegeneity", exists=True, argstr="--inputimage %s")
    maskimage = File(desc="Binary mask that defines the structure of your interest. NOTE: This parameter is OPTIONAL. If the mask is not specified, the module will use internally Otsu thresholding to define this mask. Better processing results can often be obtained when a meaningful mask is defined.", exists=True, argstr="--maskimage %s")
    outputimage = traits.Either(traits.Bool, File(), hash_files=False, desc="Result of processing", argstr="--outputimage %s")
    outputbiasfield = traits.Either(traits.Bool, File(), hash_files=False, desc="Recovered bias field (OPTIONAL)", argstr="--outputbiasfield %s")
    iterations = InputMultiPath(traits.Int, desc="Maximum number of iterations at each level of resolution. Larger values will increase execution time, but may lead to better results.", sep=",", argstr="--iterations %s")
    convergencethreshold = traits.Float(desc="Stopping criterion for the iterative bias estimation. Larger values will lead to smaller execution time.", argstr="--convergencethreshold %f")
    meshresolution = InputMultiPath(traits.Float, desc="Resolution of the initial bspline grid defined as a sequence of three numbers. The actual resolution will be defined by adding the bspline order (default is 3) to the resolution in each dimension specified here. For example, 1,1,1 will result in a 4x4x4 grid of control points. This parameter may need to be adjusted based on your input image. In the multi-resolution N4 framework, the resolution of the bspline grid at subsequent iterations will be doubled. The number of resolutions is implicitly defined by Number of iterations parameter (the size of this list is the number of resolutions)", sep=",", argstr="--meshresolution %s")
    splinedistance = traits.Float(desc="An alternative means to define the spline grid, by setting the distance between the control points. This parameter is used only if the grid resolution is not specified.", argstr="--splinedistance %f")
    shrinkfactor = traits.Int(desc="Defines how much the image should be upsampled before estimating the inhomogeneity field. Increase if you want to reduce the execution time. 1 corresponds to the original resolution. Larger values will significantly reduce the computation time.", argstr="--shrinkfactor %d")
    bsplineorder = traits.Int(desc="Order of B-spline used in the approximation. Larger values will lead to longer execution times, may result in overfitting and poor result.", argstr="--bsplineorder %d")
    weightimage = File(desc="Weight Image", exists=True, argstr="--weightimage %s")
    histogramsharpening = InputMultiPath(traits.Float, desc="A vector of up to three values. Non-zero values correspond to Bias Field Full Width at Half Maximum, Wiener filter noise, and Number of histogram bins.", sep=",", argstr="--histogramsharpening %s")


class N4ITKBiasFieldCorrectionOutputSpec(TraitedSpec):
    outputimage = File(desc="Result of processing", exists=True)
    outputbiasfield = File(desc="Recovered bias field (OPTIONAL)", exists=True)


class N4ITKBiasFieldCorrection(SEMLikeCommandLine):
    """title: N4ITK MRI Bias correction

category: Filtering

description: Performs image bias correction using N4 algorithm. This module is based on the ITK filters contributed in the following publication:  Tustison N, Gee J "N4ITK: Nick's N3 ITK Implementation For MRI Bias Field Correction", The Insight Journal 2009 January-June, http://hdl.handle.net/10380/3053

version: 9

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/N4ITKBiasFieldCorrection

contributor: Nick Tustison (UPenn), Andrey Fedorov (SPL, BWH), Ron Kikinis (SPL, BWH)

acknowledgements: The development of this module was partially supported by NIH grants R01 AA016748-01, R01 CA111288 and U01 CA151261 as well as by NA-MIC, NAC, NCIGT and the Slicer community.

"""

    input_spec = N4ITKBiasFieldCorrectionInputSpec
    output_spec = N4ITKBiasFieldCorrectionOutputSpec
    _cmd = "N4ITKBiasFieldCorrection "
    _outputs_filenames = {'outputimage':'outputimage.nii','outputbiasfield':'outputbiasfield.nii'}
