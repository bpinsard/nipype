# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.cmtk.nx import NetworkXMetrics
def test_NetworkXMetrics_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    treat_as_weighted_graph=dict(usedefault=True,
    ),
    out_node_metrics_matlab=dict(genfile=True,
    ),
    out_edge_metrics_matlab=dict(genfile=True,
    ),
    out_k_shell=dict(usedefault=True,
    ),
    compute_clique_related_measures=dict(usedefault=True,
    ),
    out_k_crust=dict(usedefault=True,
    ),
    out_k_core=dict(usedefault=True,
    ),
    in_file=dict(mandatory=True,
    ),
    out_pickled_extra_measures=dict(usedefault=True,
    ),
    out_global_metrics_matlab=dict(genfile=True,
    ),
    )
    inputs = NetworkXMetrics.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_NetworkXMetrics_outputs():
    output_map = dict(k_networks=dict(),
    edge_measures_matlab=dict(),
    node_measure_networks=dict(),
    pickled_extra_measures=dict(),
    matlab_matrix_files=dict(),
    k_crust=dict(),
    node_measures_matlab=dict(),
    gpickled_network_files=dict(),
    k_shell=dict(),
    edge_measure_networks=dict(),
    global_measures_matlab=dict(),
    k_core=dict(),
    matlab_dict_measures=dict(),
    )
    outputs = NetworkXMetrics.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
