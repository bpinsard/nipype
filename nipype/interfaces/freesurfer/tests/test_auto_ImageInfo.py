# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.freesurfer.utils import ImageInfo
def test_ImageInfo_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    args=dict(argstr='%s',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(position=1,
    argstr='%s',
    ),
    subjects_dir=dict(),
    )
    inputs = ImageInfo.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_ImageInfo_outputs():
    output_map = dict(info=dict(),
    orientation=dict(),
    data_type=dict(),
    file_format=dict(),
    ph_enc_dir=dict(),
    TR=dict(),
    dimensions=dict(),
    TI=dict(),
    TE=dict(),
    vox_sizes=dict(),
    out_file=dict(),
    )
    outputs = ImageInfo.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
