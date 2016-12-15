# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..model import SegStats


def test_SegStats_inputs():
    input_map = dict(annot=dict(argstr='--annot %s %s %s',
    mandatory=True,
    xor=(u'segmentation_file', u'annot', u'surf_label'),
    ),
    args=dict(argstr='%s',
    ),
    avgwf_file=dict(argstr='--avgwfvol %s',
    ),
    avgwf_txt_file=dict(argstr='--avgwf %s',
    ),
    brain_vol=dict(argstr='--%s',
    ),
    brainmask_file=dict(argstr='--brainmask %s',
    ),
    calc_power=dict(argstr='--%s',
    ),
    calc_snr=dict(argstr='--snr',
    ),
    color_table_file=dict(argstr='--ctab %s',
    xor=(u'color_table_file', u'default_color_table', u'gca_color_table'),
    ),
    cortex_vol_from_surf=dict(argstr='--surf-ctx-vol',
    ),
    default_color_table=dict(argstr='--ctab-default',
    xor=(u'color_table_file', u'default_color_table', u'gca_color_table'),
    ),
    empty=dict(argstr='--empty',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    etiv=dict(argstr='--etiv',
    ),
    etiv_only=dict(),
    euler=dict(argstr='--euler',
    ),
    exclude_ctx_gm_wm=dict(argstr='--excl-ctxgmwm',
    ),
    exclude_id=dict(argstr='--excludeid %d',
    ),
    frame=dict(argstr='--frame %d',
    ),
    gca_color_table=dict(argstr='--ctab-gca %s',
    xor=(u'color_table_file', u'default_color_table', u'gca_color_table'),
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='--i %s',
    ),
    in_intensity=dict(argstr='--in %s --in-intensity-name %s',
    ),
    intensity_units=dict(argstr='--in-intensity-units %s',
    requires=[u'in_intensity'],
    ),
    mask_erode=dict(argstr='--maskerode %d',
    ),
    mask_file=dict(argstr='--mask %s',
    ),
    mask_frame=dict(requires=[u'mask_file'],
    ),
    mask_invert=dict(argstr='--maskinvert',
    ),
    mask_sign=dict(),
    mask_thresh=dict(argstr='--maskthresh %f',
    ),
    multiply=dict(argstr='--mul %f',
    ),
    non_empty_only=dict(argstr='--nonempty',
    ),
    partial_volume_file=dict(argstr='--pv %s',
    ),
    segment_id=dict(argstr='--id %s...',
    ),
    segmentation_file=dict(argstr='--seg %s',
    mandatory=True,
    xor=(u'segmentation_file', u'annot', u'surf_label'),
    ),
    sf_avg_file=dict(argstr='--sfavg %s',
    ),
    subcort_gm=dict(argstr='--subcortgray',
    ),
    subjects_dir=dict(),
    summary_file=dict(argstr='--sum %s',
    genfile=True,
    position=-1,
    ),
    supratent=dict(argstr='--supratent',
    ),
    surf_label=dict(argstr='--slabel %s %s %s',
    mandatory=True,
    xor=(u'segmentation_file', u'annot', u'surf_label'),
    ),
    terminal_output=dict(nohash=True,
    ),
    total_gray=dict(argstr='--totalgray',
    ),
    vox=dict(argstr='--vox %s',
    ),
    wm_vol_from_surf=dict(argstr='--surf-wm-vol',
    ),
    )
    inputs = SegStats.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_SegStats_outputs():
    output_map = dict(avgwf_file=dict(),
    avgwf_txt_file=dict(),
    sf_avg_file=dict(),
    summary_file=dict(),
    )
    outputs = SegStats.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
