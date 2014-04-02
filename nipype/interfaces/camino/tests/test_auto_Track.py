# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.camino.dti import Track

def test_Track_inputs():
    input_map = dict(anisfile=dict(argstr='-anisfile %s',
    ),
    anisthresh=dict(argstr='-anisthresh %f',
    ),
    args=dict(argstr='%s',
    ),
    curvethresh=dict(argstr='-curvethresh %f',
    ),
    data_dims=dict(argstr='-datadims %s',
    units='voxels',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    gzip=dict(argstr='-gzip',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-inputfile %s',
    mandatory=True,
    position=1,
    ),
    inputdatatype=dict(argstr='-inputdatatype %s',
    ),
    inputmodel=dict(argstr='-inputmodel %s',
    usedefault=True,
    ),
    ipthresh=dict(argstr='-ipthresh %f',
    ),
    maxcomponents=dict(argstr='-maxcomponents %d',
    units='NA',
    ),
    numpds=dict(argstr='-numpds %d',
    units='NA',
    ),
    out_file=dict(argstr='-outputfile %s',
    genfile=True,
    position=-1,
    ),
    output_root=dict(argstr='-outputroot %s',
    position=-1,
    ),
    outputtracts=dict(argstr='-outputtracts %s',
    ),
    seed_file=dict(argstr='-seedfile %s',
    position=2,
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    voxel_dims=dict(argstr='-voxeldims %s',
    units='mm',
    ),
    )
    inputs = Track.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_Track_outputs():
    output_map = dict(tracked=dict(),
    )
    outputs = Track.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

