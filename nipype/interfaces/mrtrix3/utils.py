from nipype.interfaces.base import traits, File, InputMultiPath, isdefined
from .base import MRtrixCommandInputSpec, MRtrixCommandOutputSpec, MRtrixCommand


class LabelConfigInputSpec(MRtrixCommandInputSpec):
    
     labels_in = File(
         argstr='%s',
         position=-3,
         exists = True,
         mandatory = True,
         desc='the input image')
     config = File(
         argstr='%s',
         position=-2,
         exists=True,
         mandatory = True,
         desc="""the MRtrix connectome configuration file specifying
               desired nodes & indices)""")
     labels_out = File(
         argstr='%s',
         position=-1,
         name_source = 'labels_in',
         name_template = '%s_lconf',
         desc='the output labels reconfigured to increment of 1')

     # Options for importing information from parcellation lookup tables

     lut_basic = File(
         argstr='-lut_basic %s',
         exists=True,
         desc = 'get information from a basic lookup table consisting of index / name pairs')

     lut_freesurfer = File(
         argstr = '-lut_freesurfer %s',
         exists=True,
         desc = 'get information from a FreeSurfer lookup table (typically "FreeSurferColorLUT.txt")')

     lut_aal = File(
         argstr = '-lut_aal %s',
         exists=True,
         desc= 'get information from the AAL lookup table (typically "ROI_MNI_V4.txt")')

     lut_itksnap = File(
         argstr = '-lut_itksnap %s',
         exists=True,
         desc = """get information from an ITK-SNAP lookup table (this includes the IIT atlas
     file "LUT_GM.txt") """)

     spine = File(
         argstr = '-spine %s',
         exists=True,
         desc = """
     provide a manually-defined segmentation of the base of the spine where the
     streamlines terminate, so that this can become a node in the connection
     matrix.""")

class LabelConfigOuputSpec(MRtrixCommandOutputSpec):
    labels_out = File(exists=True)

class LabelConfig(MRtrixCommand):
    """
     prepare a parcellated image for connectome construction by modifying the
     image values; typically this involves making the parcellation intensities
     increment from 1 to coincide with rows and columns of a matrix. The
     configuration file passed as the second argument specifies the indices
     that should be assigned to different structures; examples of such
     configuration files are provided in
     src//dwi//tractography//connectomics//example_configs//
     """

    input_spec = LabelConfigInputSpec
    output_spec = LabelConfigOuputSpec

    _cmd = 'labelconfig'


class MRInfoInputSpec(MRtrixCommandInputSpec):
     """
     display header information, or extract specific information from the
     header.

     By default, all information contained in each image header will be printed
     to the console in a reader-friendly format.

     Alternatively, command-line options may be used to extract specific
     details from the header(s); these are printed to the console in a format
     more appropriate for scripting purposes or piping to file. If multiple
     options and/or images are provided, the requested header fields will be
     printed in the order in which they appear in the help page, with all
     requested details from each input image in sequence printed before the
     next image is processed.

     The command can also write the diffusion gradient table from a single
     input image to file; either in the MRtrix or FSL format (bvecs/bvals file
     pair; includes appropriate diffusion gradient vector reorientation)
     """
     in_files = InputMultiPath(
          argstr = '%s',
          desc = 'the input image(s).')
     
     #OPTIONS

     norealign = traits.Bool(
          argstr = '-norealign',
          desc = """
     do not realign transform to near-default RAS coordinate system (the
     default behaviour on image load). This is useful to inspect the transform
     and strides as they are actually stored in the header, rather than as
     MRtrix interprets them.""")

     format = traits.Bool(
          argstr = '-format',
          desc = 'image file format')
     ndim = traits.Bool(
          argstr = '-ndim',
          desc = 'number of image dimensions')
     dimensions = traits.Bool(
          argstr = '-dimensions',
          desc = 'image dimensions along each axis')
     voxel_size =  traits.Bool(
          argstr = '-vox',
          desc = 'voxel size along each image dimension')
     datatype_long = traits.Bool(
          argstr = '-datatype_long',
          desc = 'data type used for image data storage (long description)')
     datatype_short = traits.Bool(
          argstr = '-datatype_short',
          desc = 'data type used for image data storage (short specifier)')
     stride = traits.Bool(
          argstr = '-stride',
          desc = 'data strides i.e. order and direction of axes data layout')
     offset = traits.Bool(
          argstr = '-offset', desc = 'image intensity offset')
     multiplier = traits.Bool(
          argstr = '-multiplier',
          desc = 'image intensity multiplier')
     comments = traits.Bool(
          argstr = '-comments',
          desc = 'any comments embedded in the image header')
     properties = traits.Bool(
          argstr = '-properties',
          desc = 'any text properties embedded in the image header')
     transform = traits.Bool(
          argstr = '-transform',
          desc = 'the image transform')
     dwgrad = traits.Bool(
          argstr = '-dwgrad',
          desc = 'the diffusion-weighting gradient table')

     export_grad_mrtrix = File(
          argstr = '-export_grad_mrtrix %s',
          desc = 'export the diffusion-weighted gradient table to file in MRtrix format')
     export_grad_fsl = traits.Tuple(
          File(),File(),
          argstr = '-export_grad_fsl %s %s',
          desc = 'export the diffusion-weighted gradient table to files in FSL (bvecs / bvals) format')

class MRInfoOutputSpec(MRtrixCommandOutputSpec):
     export_grad_mrtrix = File()
     bvecs = File()
     bvals = File()

class MRInfo(MRtrixCommand):
    input_spec = MRInfoInputSpec
    output_spec = MRInfoOutputSpec
    _cmd = 'mrinfo'
 
