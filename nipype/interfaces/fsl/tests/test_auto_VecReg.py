# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.fsl.dti import VecReg
def test_VecReg_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    warp_field=dict(argstr='-w %s',
    ),
    affine_mat=dict(argstr='-t %s',
    ),
    out_file=dict(argstr='-o %s',
    hash_files=False,
    genfile=True,
    ),
    rotation_warp=dict(argstr='--rotwarp=%s',
    ),
    args=dict(argstr='%s',
    ),
    mask=dict(argstr='-m %s',
    ),
    rotation_mat=dict(argstr='--rotmat=%s',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(mandatory=True,
    argstr='-i %s',
    ),
    ref_vol=dict(mandatory=True,
    argstr='-r %s',
    ),
    output_type=dict(),
    ref_mask=dict(argstr='--refmask=%s',
    ),
    interpolation=dict(argstr='--interp=%s',
    ),
    )
    inputs = VecReg.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_VecReg_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = VecReg.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
