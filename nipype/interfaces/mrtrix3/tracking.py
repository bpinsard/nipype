from nipype.interfaces.base import traits, File, InputMultiPath, isdefined
from .base import MRtrixCommandInputSpec, MRtrixCommandOutputSpec, MRtrixCommand


class TckgenInputSpec(MRtrixCommandInputSpec):
    
    source = File(
        argstr = '%s',
        position=-2,
        mandatory = True,
        exists = True,
        desc="""the image containing the source data. 
                     The type of data depends on the algorithm used:
                     - FACT: the directions file (each triplet of volumes is
                     the X,Y,Z direction of a fibre population).
                     - iFOD1/2 & SD_Stream: the SH image resulting from CSD.
                     - Nulldist & SeedTest: any image (will not be used).
                     - TensorDet / TensorProb: the DWI image.""")
    
    tracks = File(
        argstr = '%s',
        position=-1,
        name_source = 'source',
        name_template = '%s_track.tck',
        keep_extension = True,
        desc='the output file containing the tracks generated.')

    algorithm = traits.Enum(
        'FACT', 'iFOD1', 'iFOD2', 'Nulldist', 'SD_Stream', 'Seedtest', 'Tensor_Det', 'Tensor_Prob',
        argstr = '-algorithm %s',
        desc="""
     specify the tractography algorithm to use. Valid choices are: FACT, iFOD1,
     iFOD2, Nulldist, SD_Stream, Seedtest, Tensor_Det, Tensor_Prob (default:
     iFOD2).""")

    #Region Of Interest processing options

    include = traits.Either(
        File(exists=True),traits.Tuple([traits.Float]*4),
        argstr = '-include %s',
        desc="""
     specify an inclusion region of interest, as either a binary mask image, or
     as a sphere using 4 comma-separared values (x,y,z,radius). Streamlines
     must traverse ALL inclusion regions to be accepted.""")

    exclude = traits.Either(
        File(exists=True),traits.Tuple([traits.Float]*4),
        argstr = '-exclude %s',
        desc="""
     specify an exclusion region of interest, as either a binary mask image, or
     as a sphere using 4 comma-separared values (x,y,z,radius). Streamlines
     that enter ANY exclude region will be discarded.""")

    mask = traits.Either(
        File(exists=True),traits.Tuple([traits.Float]*4),
        argstr = '-mask %s',
        desc="""
     specify a masking region of interest, as either a binary mask image, or as
     a sphere using 4 comma-separared values (x,y,z,radius). If defined,
     streamlines exiting the mask will be truncated.""")

    #Streamlines tractography options

    gradient = File(
        argstr='-grad %s',
        exists = True,
        desc= """
     specify the diffusion encoding scheme (may be required for Tensor_Det and
     Tensor_Prob, ignored otherwise)""")

    step = traits.Float(
        argstr='-step %f',
        desc="""
     set the step size of the algorithm in mm (default is 0.1 x voxelsize; for
     iFOD2: 0.5 x voxelsize).""")

    angle = traits.Float(
        argstr= '-angle %f',
        desc="""
     set the maximum angle between successive steps (default is 90deg x
     stepsize / voxelsize).""")

    number = traits.Int(
        argstr='-number %d',
        desc="""
     set the desired number of tracks. The program will continue to generate
     tracks until this number of tracks have been selected and written to the
     output file.""")

    maxnum = traits.Int(
        argstr = '-maxnum %d',
        desc = """
     set the maximum number of tracks to generate. The program will not
     generate more tracks than this number, even if the desired number of
     tracks hasn't yet been reached (default is 100 x number).""")

    maxlength = traits.Float(
        argstr = '-maxlength %f',
        desc = """
    set the maximum length of any track in mm (default is 100 x voxelsize).""")

    minlength = traits.Float(
        argstr = '-minlength %f',
        desc = """
    set the minimum length of any track in mm (default is 5 x voxelsize).""")

    cutoff = traits.Float(
        argstr = '-cutoff %f',
        desc="""set the FA or FOD amplitude cutoff for terminating tracks
                (default is 0.1).""")

    initcutoff = traits.Float(
        argstr = '-initcutoff %f',
        desc = """
     set the minimum FA or FOD amplitude for initiating tracks (default is
     twice the normal cutoff).""")

    trials = traits.Int(
        argstr = '-trials %d',
        desc= """
     set the maximum number of sampling trials at each point (only used for
     probabilistic tracking).""")

    unidirectional = traits.Bool(
        argstr='-unidirectional',
        desc="""
     track from the seed point in one direction only (default is to track in
     both directions).""")

    initdirection = traits.Tuple(
        (traits.Float(),)*3,
        argstr = '-initdirection %f',
        sep = ',',
        desc = """
     specify an initial direction for the tracking (this should be supplied as
     a vector of 3 comma-separated values.""")

    noprecomputed = traits.Bool(
        argstr='-noprecomputed',
        desc="""
     do NOT pre-compute legendre polynomial values. Warning: this will slow
     down the algorithm by a factor of approximately 4.""")

    power = traits.Float(
        argstr = '-power %f',
        desc = 'raise the FOD to the power specified (default is 1/nsamples).')

    samples = traits.Int(
        argstr = '-samples %d',
        desc="""
     set the number of FOD samples to take per step for the 2nd order (iFOD2)
     method (Default: 4).""")

    rk4 = traits.Bool(
        argstr = '-rk4',
        desc="""
     use 4th-order Runge-Kutta integration (slower, but eliminates curvature
     overshoot in 1st-order deterministic methods)""")

    stop = traits.Bool(
        argstr = '-stop',
        desc = 'stop propagating a streamline once it has traversed all include regions')
    
    downsample = traits.Float(
        argstr = '-downsample %f',
        desc='downsample the generated streamlines to reduce output file size')

    #Anatomically-Constrained Tractography options

    act = File(
        argstr = '-act %s',
        exists = True,
        desc="""
     use the Anatomically-Constrained Tractography framework during tracking;
     provided image must be in the 5TT (five-tissue-type) format""")

    backtrack = traits.Bool(
        argstr='-backtrack',
        requires = ['act'],
        desc="""
     allow tracks to be truncated and re-tracked if a poor structural
     termination is encountered""")

    crop_at_gmwmi = traits.Bool(
        argstr = '-crop_at_gmwmi',
        requires = ['act'],
        desc = 'crop streamline endpoints more precisely as they cross the GM-WM interface')

    #Tractography seeding options

    seeding = traits.Tuple(
        [traits.Float]*4,
        argstr = '-seed_sphere %s',
        sep=',',
        desc='spherical seed as four comma-separated values (XYZ position and radius)')

    seed_image = File(
        argstr='-seed_image %s',
        exists = True,
        desc = """
     seed streamlines entirely at random within a mask image (this is the same
     behaviour as the streamline seeding in MRtrix 0.2)""")

    seed_random_per_voxel = traits.Tuple(
        File(exists=True), traits.Int(),
        argstr = '-seed_random_per_voxel %s %d',
        desc = """
     seed a fixed number of streamlines per voxel in a mask image; random
     placement of seeds in each voxel""")

    seed_grid_per_voxel = traits.Tuple(
        File(exists=True), traits.Int(),
        argstr = '-seed_grid_per_voxel %s %d',
        desc="""
     seed a fixed number of streamlines per voxel in a mask image; place seeds
     on a 3D mesh grid (grid_size argument is per axis; so a grid_size of 3
     results in 27 seeds per voxel)""")

    seed_rejection = File(
        argstr = '-seed_rejection %s',
        exists = True,
        desc = """
     seed from an image using rejection sampling (higher values = more probable
     to seed from)""")

    seed_gmwmi = File(
        argstr = '-seed_gmwmi %s',
        exists = True,
        desc = """
     seed from the grey matter - white matter interface (only valid if using
     ACT framework)""")

    seed_dynamic = File(
        argstr = '-seed_dynamic %s',
        exists = True,
        xor = ['seeding','seed_image','seed_random_per_voxel',
               'seed_grid_per_voxel','seed_rejection','seed_gmwmi'],
        desc = """
     determine seed points dynamically using the SIFT model (must NOT provide
     any other seeding mechanism)""")

    max_seed_attempts = traits.Int(
        argstr = '-max_seed_attempts %d',
        desc = """
     set the maximum number of times that the tracking algorithm should attempt
     to find an appropriate tracking direction from a given seed point""")

    output_seeds = File(
        argstr = '-output_seeds %s',
#        name_source='source',
        desc = 'output the seed location of all successful streamlines to a file')
    

class TckgenOutputSpec(MRtrixCommandOutputSpec):
    tracks=File(exists = True,)
    output_seeds = File()
    

class Tckgen(MRtrixCommand):
    """
    perform streamlines tractography.
    """

    input_spec = TckgenInputSpec
    output_spec = TckgenOutputSpec

    _cmd = 'tckgen'

class Tck2ConnectomeInputSpec(MRtrixCommandInputSpec):
    
     tracks_in = File(
         argstr='%s', position=-3,
         mandatory = True,
         exists = True,
         desc='the input track file')

     nodes_in = File(
         argstr='%s', position=-2,
         mandatory = True,
         exists = True,         
         desc='the input node parcellation image')

     connectome_out = File(
         argstr='%s', position=-1,
         mandatory = True,
         desc = 'the output .csv file containing edge weights')

     #Structural connectome streamline assignment option

     _assign_opts = ['assignment_voxel_lookup',
                     'assignment_radial_search',
                     'assignment_reverse_search',
                     'assignment_forward_search']

     assignment_voxel_lookup = traits.Bool(
         argstr = '-assignment_voxel_lookup',
         xor = _assign_opts,
         desc = 'use a simple voxel lookup value at each streamline endpoint')

     assignment_radial_search = traits.Bool(
         argstr = '-assignment_radial_search radius',
         xor = _assign_opts,
         desc= """
     perform a radial search from each streamline endpoint to locate the
     nearest node.
     Argument is the maximum radius in mm; if no node is found within this
     radius, the streamline endpoint is not assigned to any node.""")
     
     assignment_reverse_search = traits.Bool(
         argstr = '-assignment_reverse_search max_dist',
         xor = _assign_opts,
         desc = """
     traverse from each streamline endpoint inwards along the streamline, in
     search of the last node traversed by the streamline. Argument is the
     maximum traversal length in mm (set to 0 to allow search to continue to
     the streamline midpoint).""")

     assignment_forward_search = traits.Bool(
         argstr = '-assignment_forward_search max_dist',
         xor = _assign_opts,
         desc = """
     project the streamline forwards from the endpoint in search of a
     parcellation node voxel. Argument is the maximum traversal length in mm.
""")
     
     # Structural connectome metric option

     metric = traits.Enum(
         'count', 'meanlength', 'invlength', 'invnodevolume',
         'invlength_invnodevolume', 'mean_scalar',
         argstr = '-metric %s',
         desc = 'specify the edge weight metric.')

     image = File(
         argstr = '-image %s',
         desc = 'provide the associated image for the mean_scalar metric')
     
     tck_weights_in = File(
         argstr = '-tck_weights_in %s',
         desc = 'specify a text scalar file containing the streamline weights')

     keep_unassigned = traits.Bool(
         argstr = '-keep_unassigned',
         desc = """
     By default, the program discards the information regarding those
     streamlines that are not successfully assigned to a node pair. Set this
     option to keep these values (will be the first row/column in the output
     matrix)""")

     keep_unassigned = traits.Bool(
         argstr = '-zero_diagonal',
         desc = """set all diagonal entries in the matrix to zero 
   (these represent streamlines that connect to the same node at both ends)""")

    
class Tck2ConnectomeOutputSpec(MRtrixCommandOutputSpec):
    connectome_out = File(
        desc = 'the output .csv file containing edge weights')

class Tck2Connectome(MRtrixCommand):
    """
    generate a connectome matrix from a streamlines file and a node
    parcellation image
    """

    input_spec = Tck2ConnectomeInputSpec
    output_spec = Tck2ConnectomeOutputSpec
    _cmd = 'tck2connectome'

class TckMapInputSpec(MRtrixCommandInputSpec):

    tracks_in = File(
        argstr='%s', position=-2,
        exists = True,
        mandatory=True,
        desc='the input track file.')

    out_file= File(
        argstr='%s', position=-1,
        mandatory=True,
        desc='the output track-weighted image')



    #Options for the header of the output image
    
    template = File(
        argstr='-template %s',
        desc = """
     an image file to be used as a template for the output (the output image
     will have the same transform and field of view).""")

    voxelsize = traits.Tuple(
        (traits.Float(),)*3,
        argstr = '-vox %f,%f,%f',
        desc = """
     provide either an isotropic voxel size (in mm), or comma-separated list of
     3 voxel dimensions.""")

    datatype = traits.Enum(
        'short','int','float','double', # are these real possible values?
        argstr = '-datatype %s',
        desc = 'specify output image data type.')

    #Options for the dimensionality of the output image

    colour = traits.Bool(
        argstr = '-colour',
        desc = 'perform track mapping in directionally-encoded colour space')

    #Options for the TWI image contrast properties

    contrast = traits.Enum(
        'tdi', 'precise_tdi', 'endpoint', 'length', 'invlength',
        'scalar_map','scalar_map_count', 'fod_amp', 'curvature',
        argstr = '-contrast %s',
        desc = 'define the desired form of contrast for the output image',
        usedefault=True)

    scalar_image = File(
        argstr = '-image %s',
        desc = """
     provide the scalar image map for generating images with 'scalar_map'
     contrast, or the spherical harmonics image for 'fod_amp' contrast""")

    stat_vox = traits.Enum(
        'sum', 'min', 'mean', 'max',
        argstr = '-stat_vox %s',
        desc = """
     define the statistic for choosing the final voxel intensities for a given
     contrast type given the individual values from the tracks passing through
     each voxel.""",
        usedefault=True)

    stat_tck = traits.Enum(
        'sum', 'min', 'mean', 'max', 'median', 'mean_nonzero', 'gaussian',
        'ends_min', 'ends_mean', 'ends_max', 'ends_prod',
        argstr = '-stat_tck %s',
        desc = """
     define the statistic for choosing the contribution to be made by each
     streamline as a function of the samples taken along their lengths
     Only has an effect for 'scalar_map', 'fod_amp' and 'curvature' contrast
     types (default: mean)""")

    fwhm_tck = traits.Float(
        argstr = '-fwhm_tck %f',
        desc = """
     when using gaussian-smoothed per-track statistic, specify the desired
     full-width half-maximum of the Gaussian smoothing kernel (in mm)""")

    map_zero = traits.Bool(
        argstr = '-map_zero',
        desc = """
     if a streamline has zero contribution based on the contrast & statistic,
     typically it is not mapped; use this option to still contribute to the map
     even if this is the case (these non-contributing voxels can then influence
     the mean value in each voxel of the map)""")

class TckMapOutputSpec(MRtrixCommandOutputSpec):
    out_file= File(
        exists = True,
        desc='the output track-weighted image')

class TckMap(MRtrixCommand):
    """
    Use track data as a form of contrast for producing a high-resolution image.
    """

    input_spec = TckMapInputSpec
    output_spec = TckMapOutputSpec
    _cmd = 'tckmap'
    
