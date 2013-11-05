# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.freesurfer.model import SegStats
def test_SegStats_inputs():
    input_map = dict(exclude_ctx_gm_wm=dict(argstr='--excl-ctxgmwm',
    ),
    calc_snr=dict(argstr='--snr',
    ),
    frame=dict(argstr='--frame %d',
    ),
    cortex_vol_from_surf=dict(argstr='--surf-ctx-vol',
    ),
    sf_avg_file=dict(argstr='--sfavg %s',
    ),
    etiv=dict(argstr='--etiv',
    ),
    exclude_id=dict(argstr='--excludeid %d',
    ),
    etiv_only=dict(),
    avgwf_txt_file=dict(argstr='--avgwf %s',
    ),
    default_color_table=dict(xor=('color_table_file', 'default_color_table', 'gca_color_table'),
    argstr='--ctab-default',
    ),
    mask_erode=dict(argstr='--maskerode %d',
    ),
    brain_vol=dict(),
    in_file=dict(argstr='--i %s',
    ),
    gca_color_table=dict(xor=('color_table_file', 'default_color_table', 'gca_color_table'),
    argstr='--ctab-gca %s',
    ),
    partial_volume_file=dict(argstr='--pv %f',
    ),
    color_table_file=dict(xor=('color_table_file', 'default_color_table', 'gca_color_table'),
    argstr='--ctab %s',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    surf_label=dict(mandatory=True,
    xor=('segmentation_file', 'annot', 'surf_label'),
    argstr='--slabel %s %s %s',
    ),
    segmentation_file=dict(xor=('segmentation_file', 'annot', 'surf_label'),
    mandatory=True,
    argstr='--seg %s',
    ),
    args=dict(argstr='%s',
    ),
    avgwf_file=dict(argstr='--avgwfvol %s',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    subjects_dir=dict(),
    multiply=dict(argstr='--mul %f',
    ),
    mask_invert=dict(argstr='--maskinvert',
    ),
    mask_sign=dict(),
    non_empty_only=dict(argstr='--nonempty',
    ),
    calc_power=dict(argstr='--%s',
    ),
    mask_frame=dict(requires=['mask_file'],
    ),
    segment_id=dict(argstr='--id %s...',
    ),
    annot=dict(mandatory=True,
    xor=('segmentation_file', 'annot', 'surf_label'),
    argstr='--annot %s %s %s',
    ),
    wm_vol_from_surf=dict(argstr='--surf-wm-vol',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    mask_thresh=dict(argstr='--maskthresh %f',
    ),
    mask_file=dict(argstr='--mask %s',
    ),
    vox=dict(argstr='--vox %s',
    ),
    summary_file=dict(argstr='--sum %s',
    genfile=True,
    ),
    )
    inputs = SegStats.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_SegStats_outputs():
    output_map = dict(sf_avg_file=dict(),
    avgwf_txt_file=dict(),
    summary_file=dict(),
    avgwf_file=dict(),
    )
    outputs = SegStats.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
