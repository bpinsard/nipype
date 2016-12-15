# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..diffusion import dtiestim


def test_dtiestim_inputs():
    input_map = dict(B0=dict(argstr='--B0 %s',
    hash_files=False,
    ),
    B0_mask_output=dict(argstr='--B0_mask_output %s',
    hash_files=False,
    ),
    DTI_double=dict(argstr='--DTI_double ',
    ),
    args=dict(argstr='%s',
    ),
    bad_region_mask=dict(argstr='--bad_region_mask %s',
    ),
    brain_mask=dict(argstr='--brain_mask %s',
    ),
    correction=dict(argstr='--correction %s',
    ),
    defaultTensor=dict(argstr='--defaultTensor %s',
    sep=',',
    ),
    dwi_image=dict(argstr='--dwi_image %s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    idwi=dict(argstr='--idwi %s',
    hash_files=False,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    method=dict(argstr='--method %s',
    ),
    shiftNeg=dict(argstr='--shiftNeg ',
    ),
    shiftNegCoeff=dict(argstr='--shiftNegCoeff %f',
    ),
    sigma=dict(argstr='--sigma %f',
    ),
    step=dict(argstr='--step %f',
    ),
    tensor_output=dict(argstr='--tensor_output %s',
    hash_files=False,
    ),
    terminal_output=dict(nohash=True,
    ),
    threshold=dict(argstr='--threshold %d',
    ),
    verbose=dict(argstr='--verbose ',
    ),
    weight_iterations=dict(argstr='--weight_iterations %d',
    ),
    )
    inputs = dtiestim.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_dtiestim_outputs():
    output_map = dict(B0=dict(),
    B0_mask_output=dict(),
    idwi=dict(),
    tensor_output=dict(),
    )
    outputs = dtiestim.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
