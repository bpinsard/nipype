# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.cmtk.cmtk import CreateMatrix
def test_CreateMatrix_inputs():
    input_map = dict(resolution_network_file=dict(mandatory=True,
    ),
    out_intersection_matrix_mat_file=dict(genfile=True,
    ),
    out_median_fiber_length_matrix_mat_file=dict(genfile=True,
    ),
    tract_file=dict(mandatory=True,
    ),
    out_matrix_file=dict(genfile=True,
    ),
    out_matrix_mat_file=dict(usedefault=True,
    ),
    count_region_intersections=dict(usedefault=True,
    ),
    out_endpoint_array_name=dict(genfile=True,
    ),
    roi_file=dict(mandatory=True,
    ),
    out_fiber_length_std_matrix_mat_file=dict(genfile=True,
    ),
    out_mean_fiber_length_matrix_mat_file=dict(genfile=True,
    ),
    )
    inputs = CreateMatrix.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_CreateMatrix_outputs():
    output_map = dict(matrix_file=dict(),
    median_fiber_length_matrix_mat_file=dict(),
    fiber_labels_noorphans=dict(),
    matlab_matrix_files=dict(),
    filtered_tractographies=dict(),
    filtered_tractography=dict(),
    fiber_length_std_matrix_mat_file=dict(),
    fiber_label_file=dict(),
    endpoint_file_mm=dict(),
    mean_fiber_length_matrix_mat_file=dict(),
    matrix_files=dict(),
    intersection_matrix_mat_file=dict(),
    filtered_tractography_by_intersections=dict(),
    matrix_mat_file=dict(),
    endpoint_file=dict(),
    fiber_length_file=dict(),
    stats_file=dict(),
    intersection_matrix_file=dict(),
    )
    outputs = CreateMatrix.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
