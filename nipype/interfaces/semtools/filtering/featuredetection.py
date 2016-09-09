# -*- coding: utf-8 -*-
# -*- coding: utf8 -*-
"""Autogenerated file - DO NOT EDIT
If you spot a bug, please report it on the mailing list and/or change the generator."""

import os

from ...base import (CommandLine, CommandLineInputSpec, SEMLikeCommandLine,
                     TraitedSpec, File, Directory, traits, isdefined,
                     InputMultiPath, OutputMultiPath)


class GenerateSummedGradientImageInputSpec(CommandLineInputSpec):
    inputVolume1 = File(desc="input volume 1, usally t1 image", exists=True, argstr="--inputVolume1 %s")
    inputVolume2 = File(desc="input volume 2, usally t2 image", exists=True, argstr="--inputVolume2 %s")
    outputFileName = traits.Either(traits.Bool, File(), hash_files=False, desc="(required) output file name", argstr="--outputFileName %s")
    MaximumGradient = traits.Bool(desc="If set this flag, it will compute maximum gradient between two input volumes instead of sum of it.", argstr="--MaximumGradient ")
    numberOfThreads = traits.Int(desc="Explicitly specify the maximum number of threads to use.", argstr="--numberOfThreads %d")


class GenerateSummedGradientImageOutputSpec(TraitedSpec):
    outputFileName = File(desc="(required) output file name", exists=True)


class GenerateSummedGradientImage(SEMLikeCommandLine):

    """title: GenerateSummedGradient

category: Filtering.FeatureDetection

description: Automatic FeatureImages using neural networks

version: 1.0

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: Greg Harris, Eun Young Kim

"""

    input_spec = GenerateSummedGradientImageInputSpec
    output_spec = GenerateSummedGradientImageOutputSpec
    _cmd = " GenerateSummedGradientImage "
    _outputs_filenames = {'outputFileName': 'outputFileName'}
    _redirect_x = False


class CannySegmentationLevelSetImageFilterInputSpec(CommandLineInputSpec):
    inputVolume = File(exists=True, argstr="--inputVolume %s")
    initialModel = File(exists=True, argstr="--initialModel %s")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, argstr="--outputVolume %s")
    outputSpeedVolume = traits.Either(traits.Bool, File(), hash_files=False, argstr="--outputSpeedVolume %s")
    cannyThreshold = traits.Float(desc="Canny Threshold Value", argstr="--cannyThreshold %f")
    cannyVariance = traits.Float(desc="Canny variance", argstr="--cannyVariance %f")
    advectionWeight = traits.Float(desc="Controls the smoothness of the resulting mask, small number are more smooth, large numbers allow more sharp corners.  ", argstr="--advectionWeight %f")
    initialModelIsovalue = traits.Float(desc="The identification of the input model iso-surface.  (for a binary image with 0s and 1s use 0.5) (for a binary image with 0s and 255's use 127.5).", argstr="--initialModelIsovalue %f")
    maxIterations = traits.Int(desc="The", argstr="--maxIterations %d")


class CannySegmentationLevelSetImageFilterOutputSpec(TraitedSpec):
    outputVolume = File(exists=True)
    outputSpeedVolume = File(exists=True)


class CannySegmentationLevelSetImageFilter(SEMLikeCommandLine):

    """title: Canny Level Set Image Filter

category: Filtering.FeatureDetection

description: The CannySegmentationLevelSet is commonly used to refine a manually generated manual mask.

version: 0.3.0

license: CC

contributor: Regina Kim

acknowledgements: This command module was derived from Insight/Examples/Segmentation/CannySegmentationLevelSetImageFilter.cxx (copyright) Insight Software Consortium.  See http://wiki.na-mic.org/Wiki/index.php/Slicer3:Execution_Model_Documentation for more detailed descriptions.

"""

    input_spec = CannySegmentationLevelSetImageFilterInputSpec
    output_spec = CannySegmentationLevelSetImageFilterOutputSpec
    _cmd = " CannySegmentationLevelSetImageFilter "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii', 'outputSpeedVolume': 'outputSpeedVolume.nii'}
    _redirect_x = False


class DilateImageInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Required: input image", exists=True, argstr="--inputVolume %s")
    inputMaskVolume = File(desc="Required: input brain mask image", exists=True, argstr="--inputMaskVolume %s")
    inputRadius = traits.Int(desc="Required: input neighborhood radius", argstr="--inputRadius %d")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class DilateImageOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class DilateImage(SEMLikeCommandLine):

    """title: Dilate Image

category: Filtering.FeatureDetection

description: Uses mathematical morphology to dilate the input images.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was developed by Mark Scully and Jeremy Bockholt.

"""

    input_spec = DilateImageInputSpec
    output_spec = DilateImageOutputSpec
    _cmd = " DilateImage "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False


class TextureFromNoiseImageFilterInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Required: input image", exists=True, argstr="--inputVolume %s")
    inputRadius = traits.Int(desc="Required: input neighborhood radius", argstr="--inputRadius %d")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class TextureFromNoiseImageFilterOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class TextureFromNoiseImageFilter(SEMLikeCommandLine):

    """title: TextureFromNoiseImageFilter

category: Filtering.FeatureDetection

description: Calculate the local noise in an image.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was developed by Eunyoung Regina Kim

"""

    input_spec = TextureFromNoiseImageFilterInputSpec
    output_spec = TextureFromNoiseImageFilterOutputSpec
    _cmd = " TextureFromNoiseImageFilter "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False


class FlippedDifferenceInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Required: input image", exists=True, argstr="--inputVolume %s")
    inputMaskVolume = File(desc="Required: input brain mask image", exists=True, argstr="--inputMaskVolume %s")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class FlippedDifferenceOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class FlippedDifference(SEMLikeCommandLine):

    """title: Flip Image

category: Filtering.FeatureDetection

description: Difference between an image and the axially flipped version of that image.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was developed by Mark Scully and Jeremy Bockholt.

"""

    input_spec = FlippedDifferenceInputSpec
    output_spec = FlippedDifferenceOutputSpec
    _cmd = " FlippedDifference "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False


class ErodeImageInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Required: input image", exists=True, argstr="--inputVolume %s")
    inputMaskVolume = File(desc="Required: input brain mask image", exists=True, argstr="--inputMaskVolume %s")
    inputRadius = traits.Int(desc="Required: input neighborhood radius", argstr="--inputRadius %d")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class ErodeImageOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class ErodeImage(SEMLikeCommandLine):

    """title: Erode Image

category: Filtering.FeatureDetection

description: Uses mathematical morphology to erode the input images.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was developed by Mark Scully and Jeremy Bockholt.

"""

    input_spec = ErodeImageInputSpec
    output_spec = ErodeImageOutputSpec
    _cmd = " ErodeImage "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False


class GenerateBrainClippedImageInputSpec(CommandLineInputSpec):
    inputImg = File(desc="input volume 1, usally t1 image", exists=True, argstr="--inputImg %s")
    inputMsk = File(desc="input volume 2, usally t2 image", exists=True, argstr="--inputMsk %s")
    outputFileName = traits.Either(traits.Bool, File(), hash_files=False, desc="(required) output file name", argstr="--outputFileName %s")
    numberOfThreads = traits.Int(desc="Explicitly specify the maximum number of threads to use.", argstr="--numberOfThreads %d")


class GenerateBrainClippedImageOutputSpec(TraitedSpec):
    outputFileName = File(desc="(required) output file name", exists=True)


class GenerateBrainClippedImage(SEMLikeCommandLine):

    """title: GenerateBrainClippedImage

category: Filtering.FeatureDetection

description: Automatic FeatureImages using neural networks

version: 1.0

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: Eun Young Kim

"""

    input_spec = GenerateBrainClippedImageInputSpec
    output_spec = GenerateBrainClippedImageOutputSpec
    _cmd = " GenerateBrainClippedImage "
    _outputs_filenames = {'outputFileName': 'outputFileName'}
    _redirect_x = False


class NeighborhoodMedianInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Required: input image", exists=True, argstr="--inputVolume %s")
    inputMaskVolume = File(desc="Required: input brain mask image", exists=True, argstr="--inputMaskVolume %s")
    inputRadius = traits.Int(desc="Required: input neighborhood radius", argstr="--inputRadius %d")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class NeighborhoodMedianOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class NeighborhoodMedian(SEMLikeCommandLine):

    """title: Neighborhood Median

category: Filtering.FeatureDetection

description: Calculates the median, for the given neighborhood size, at each voxel of the input image.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was developed by Mark Scully and Jeremy Bockholt.

"""

    input_spec = NeighborhoodMedianInputSpec
    output_spec = NeighborhoodMedianOutputSpec
    _cmd = " NeighborhoodMedian "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False


class GenerateTestImageInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="input volume 1, usally t1 image", exists=True, argstr="--inputVolume %s")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="(required) output file name", argstr="--outputVolume %s")
    lowerBoundOfOutputVolume = traits.Float(argstr="--lowerBoundOfOutputVolume %f")
    upperBoundOfOutputVolume = traits.Float(argstr="--upperBoundOfOutputVolume %f")
    outputVolumeSize = traits.Float(desc="output Volume Size", argstr="--outputVolumeSize %f")


class GenerateTestImageOutputSpec(TraitedSpec):
    outputVolume = File(desc="(required) output file name", exists=True)


class GenerateTestImage(SEMLikeCommandLine):

    """title: DownSampleImage

category: Filtering.FeatureDetection

description: Down sample image for testing

version: 1.0

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: Eun Young Kim

"""

    input_spec = GenerateTestImageInputSpec
    output_spec = GenerateTestImageOutputSpec
    _cmd = " GenerateTestImage "
    _outputs_filenames = {'outputVolume': 'outputVolume'}
    _redirect_x = False


class NeighborhoodMeanInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Required: input image", exists=True, argstr="--inputVolume %s")
    inputMaskVolume = File(desc="Required: input brain mask image", exists=True, argstr="--inputMaskVolume %s")
    inputRadius = traits.Int(desc="Required: input neighborhood radius", argstr="--inputRadius %d")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class NeighborhoodMeanOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class NeighborhoodMean(SEMLikeCommandLine):

    """title: Neighborhood Mean

category: Filtering.FeatureDetection

description: Calculates the mean, for the given neighborhood size, at each voxel of the T1, T2, and FLAIR.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was developed by Mark Scully and Jeremy Bockholt.

"""

    input_spec = NeighborhoodMeanInputSpec
    output_spec = NeighborhoodMeanOutputSpec
    _cmd = " NeighborhoodMean "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False


class HammerAttributeCreatorInputSpec(CommandLineInputSpec):
    Scale = traits.Int(desc="Determine Scale of Ball", argstr="--Scale %d")
    Strength = traits.Float(desc="Determine Strength of Edges", argstr="--Strength %f")
    inputGMVolume = File(desc="Required: input grey matter posterior image", exists=True, argstr="--inputGMVolume %s")
    inputWMVolume = File(desc="Required: input white matter posterior image", exists=True, argstr="--inputWMVolume %s")
    inputCSFVolume = File(desc="Required: input CSF posterior image", exists=True, argstr="--inputCSFVolume %s")
    outputVolumeBase = traits.Str(desc="Required: output image base name to be appended for each feature vector.", argstr="--outputVolumeBase %s")


class HammerAttributeCreatorOutputSpec(TraitedSpec):
    pass


class HammerAttributeCreator(SEMLikeCommandLine):

    """title: HAMMER Feature Vectors

category: Filtering.FeatureDetection

description: Create the feature vectors used by HAMMER.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This was extracted from the Hammer Registration source code, and wrapped up by Hans J. Johnson.

"""

    input_spec = HammerAttributeCreatorInputSpec
    output_spec = HammerAttributeCreatorOutputSpec
    _cmd = " HammerAttributeCreator "
    _outputs_filenames = {}
    _redirect_x = False


class TextureMeasureFilterInputSpec(CommandLineInputSpec):
    inputVolume = File(exists=True, argstr="--inputVolume %s")
    inputMaskVolume = File(exists=True, argstr="--inputMaskVolume %s")
    distance = traits.Int(argstr="--distance %d")
    insideROIValue = traits.Float(argstr="--insideROIValue %f")
    outputFilename = traits.Either(traits.Bool, File(), hash_files=False, argstr="--outputFilename %s")


class TextureMeasureFilterOutputSpec(TraitedSpec):
    outputFilename = File(exists=True)


class TextureMeasureFilter(SEMLikeCommandLine):

    """title: Canny Level Set Image Filter

category: Filtering.FeatureDetection

description: The CannySegmentationLevelSet is commonly used to refine a manually generated manual mask.

version: 0.3.0

license: CC

contributor: Regina Kim

acknowledgements: This command module was derived from Insight/Examples/Segmentation/CannySegmentationLevelSetImageFilter.cxx (copyright) Insight Software Consortium.  See http://wiki.na-mic.org/Wiki/index.php/Slicer3:Execution_Model_Documentation for more detailed descriptions.

"""

    input_spec = TextureMeasureFilterInputSpec
    output_spec = TextureMeasureFilterOutputSpec
    _cmd = " TextureMeasureFilter "
    _outputs_filenames = {'outputFilename': 'outputFilename'}
    _redirect_x = False


class DilateMaskInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Required: input image", exists=True, argstr="--inputVolume %s")
    inputBinaryVolume = File(desc="Required: input brain mask image", exists=True, argstr="--inputBinaryVolume %s")
    sizeStructuralElement = traits.Int(desc="size of structural element. sizeStructuralElement=1 means that 3x3x3 structuring element for 3D", argstr="--sizeStructuralElement %d")
    lowerThreshold = traits.Float(desc="Required: lowerThreshold value", argstr="--lowerThreshold %f")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class DilateMaskOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class DilateMask(SEMLikeCommandLine):

    """title: Dilate Image

category: Filtering.FeatureDetection

description: Uses mathematical morphology to dilate the input images.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was developed by Mark Scully and Jeremy Bockholt.

"""

    input_spec = DilateMaskInputSpec
    output_spec = DilateMaskOutputSpec
    _cmd = " DilateMask "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False


class DumpBinaryTrainingVectorsInputSpec(CommandLineInputSpec):
    inputHeaderFilename = File(desc="Required: input header file name", exists=True, argstr="--inputHeaderFilename %s")
    inputVectorFilename = File(desc="Required: input vector filename", exists=True, argstr="--inputVectorFilename %s")


class DumpBinaryTrainingVectorsOutputSpec(TraitedSpec):
    pass


class DumpBinaryTrainingVectors(SEMLikeCommandLine):

    """title: Erode Image

category: Filtering.FeatureDetection

description: Uses mathematical morphology to erode the input images.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was developed by Mark Scully and Jeremy Bockholt.

"""

    input_spec = DumpBinaryTrainingVectorsInputSpec
    output_spec = DumpBinaryTrainingVectorsOutputSpec
    _cmd = " DumpBinaryTrainingVectors "
    _outputs_filenames = {}
    _redirect_x = False


class DistanceMapsInputSpec(CommandLineInputSpec):
    inputLabelVolume = File(desc="Required: input tissue label image", exists=True, argstr="--inputLabelVolume %s")
    inputMaskVolume = File(desc="Required: input brain mask image", exists=True, argstr="--inputMaskVolume %s")
    inputTissueLabel = traits.Int(desc="Required: input integer value of tissue type used to calculate distance", argstr="--inputTissueLabel %d")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class DistanceMapsOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class DistanceMaps(SEMLikeCommandLine):

    """title: Mauerer Distance

category: Filtering.FeatureDetection

description: Get the distance from a voxel to the nearest voxel of a given tissue type.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was developed by Mark Scully and Jeremy Bockholt.

"""

    input_spec = DistanceMapsInputSpec
    output_spec = DistanceMapsOutputSpec
    _cmd = " DistanceMaps "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False


class STAPLEAnalysisInputSpec(CommandLineInputSpec):
    inputDimension = traits.Int(desc="Required: input image Dimension 2 or 3", argstr="--inputDimension %d")
    inputLabelVolume = InputMultiPath(File(exists=True), desc="Required: input label volume", argstr="--inputLabelVolume %s...")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class STAPLEAnalysisOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class STAPLEAnalysis(SEMLikeCommandLine):

    """title: Dilate Image

category: Filtering.FeatureDetection

description: Uses mathematical morphology to dilate the input images.

version: 0.1.0.$Revision: 1 $(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was developed by Mark Scully and Jeremy Bockholt.

"""

    input_spec = STAPLEAnalysisInputSpec
    output_spec = STAPLEAnalysisOutputSpec
    _cmd = " STAPLEAnalysis "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False


class GradientAnisotropicDiffusionImageFilterInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Required: input image", exists=True, argstr="--inputVolume %s")
    numberOfIterations = traits.Int(desc="Optional value for number of Iterations", argstr="--numberOfIterations %d")
    timeStep = traits.Float(desc="Time step for diffusion process", argstr="--timeStep %f")
    conductance = traits.Float(desc="Conductance for diffusion process", argstr="--conductance %f")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class GradientAnisotropicDiffusionImageFilterOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class GradientAnisotropicDiffusionImageFilter(SEMLikeCommandLine):

    """title: GradientAnisopropicDiffusionFilter

category: Filtering.FeatureDetection

description: Image Smoothing using Gradient Anisotropic Diffuesion Filer

contributor: This tool was developed by Eun Young Kim by modifying ITK Example

"""

    input_spec = GradientAnisotropicDiffusionImageFilterInputSpec
    output_spec = GradientAnisotropicDiffusionImageFilterOutputSpec
    _cmd = " GradientAnisotropicDiffusionImageFilter "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False


class CannyEdgeInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Required: input tissue label image", exists=True, argstr="--inputVolume %s")
    variance = traits.Float(desc="Variance and Maximum error are used in the Gaussian smoothing of the input image.  See  itkDiscreteGaussianImageFilter for information on these parameters.", argstr="--variance %f")
    upperThreshold = traits.Float(
        desc="Threshold is the lowest allowed value in the output image.  Its data type is the same as the data type of the output image. Any values below the Threshold level will be replaced with the OutsideValue parameter value, whose default is zero.  ", argstr="--upperThreshold %f")
    lowerThreshold = traits.Float(
        desc="Threshold is the lowest allowed value in the output image.  Its data type is the same as the data type of the output image. Any values below the Threshold level will be replaced with the OutsideValue parameter value, whose default is zero.  ", argstr="--lowerThreshold %f")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: output image", argstr="--outputVolume %s")


class CannyEdgeOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: output image", exists=True)


class CannyEdge(SEMLikeCommandLine):

    """title: Canny Edge Detection

category: Filtering.FeatureDetection

description: Get the distance from a voxel to the nearest voxel of a given tissue type.

version: 0.1.0.(alpha)

documentation-url: http:://www.na-mic.org/

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

contributor: This tool was written by Hans J. Johnson.

"""

    input_spec = CannyEdgeInputSpec
    output_spec = CannyEdgeOutputSpec
    _cmd = " CannyEdge "
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False
