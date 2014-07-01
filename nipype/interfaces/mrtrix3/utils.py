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
