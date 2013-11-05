# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.fsl.preprocess import ApplyWarp
def test_ApplyWarp_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    supersample=dict(argstr='--super',
    ),
    superlevel=dict(argstr='--superlevel=%s',
    ),
    out_file=dict(argstr='--out=%s',
    hash_files=False,
    genfile=True,
    ),
    datatype=dict(argstr='--datatype=%s',
    ),
    args=dict(argstr='%s',
    ),
    interp=dict(argstr='--interp=%s',
    ),
    field_file=dict(argstr='--warp=%s',
    ),
    ref_file=dict(mandatory=True,
    argstr='--ref=%s',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    premat=dict(argstr='--premat=%s',
    ),
    mask_file=dict(argstr='--mask=%s',
    ),
    output_type=dict(),
    postmat=dict(argstr='--postmat=%s',
    ),
    relwarp=dict(xor=['abswarp'],
    argstr='--rel',
    ),
    abswarp=dict(xor=['relwarp'],
    argstr='--abs',
    ),
    in_file=dict(mandatory=True,
    argstr='--in=%s',
    ),
    )
    inputs = ApplyWarp.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_ApplyWarp_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = ApplyWarp.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
