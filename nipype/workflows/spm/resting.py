# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os

import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.fsl as fsl          # fsl
from nipype.algorithms.misc import TSNR
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.lif as lif

def extract_noise_ts(in_file, noise_mask_files):
    """Extract mean timeseries from ROI files
    """
    import os
    from nibabel import load
    import numpy as np
    import scipy as sp
    from scipy.signal import detrend
    imgseries = load(in_file)
    data=imgseries.get_data()
    for idx,mask_file in enumerate(noise_mask_files):
        noise_mask = load(mask_file)
        voxel_timecourses = data[np.nonzero(noise_mask.get_data())]
        for timecourse in voxel_timecourses:
            timecourse[:] = detrend(timecourse, type='constant')
        tc[:,idx]=np.mean(voxel_timecourse)
    components_file = os.path.join(os.getcwd(), 'noise_components.txt')
    np.savetxt(components_file, tc)
    return components_file


def create_resting_reg_preproc(name='restpreproc'):
    workflow = pe.Workflow(name=name)


    inputnode = pe.Node(util.IdentityInterface(fields=['in_files',
                                                       'noise_rois',
                                                       'slice_order',
                                                       'ref_slice',
                                                       'num_slices',
                                                       'tr',
                                                       'smooth_fwhm']),
                        name='inputspec')
    n_slicetiming = pe.Node(interface=spm.SliceTiming(), name='slicetiming')

    n_realign = pe.Node(interface=spm.Realign(), name='realign')
    
    n_smooth = pe.Node(interface=spm.Smooth(), name='smooth')

    n_tsnr = pe.Node(TSNR(regress_poly=2), name='tsnr')

    n_regoutnoise = pe.Node(util.Function(input_names=['in_file',
                                                       'noise_mask_files'],
                                          output_names=['noise_components'],
                                          function=extract_noise_ts),
                            name='regoutnoise')
    
    n_noisecorr = pe.Node(fsl.FilterRegressor(filter_all=True),
                          name='noisecorr')

    n_bandpass_filter = pe.Node(fsl.TemporalFilter(),
                                name='bandpass_filter')
    
    outputnode = pe.Node(interface=util.IdentityInterface(fields=[
        'preprocessed_file',
        ]),
                         name='outputspec')

    def acqtime(tr,nslices):
        return tr-tr/nslices

    workflow.connect([
        (inputnode, n_slicetiming, [('in_files','in_files'),
                                    ('tr','time_repetition'),
                                    ('ref_slice','ref_slice'),
                                    ('slice_order','slice_order'),
                                    (('tr',acqtime,'num_slices'),'time_acquisition'),
                                    ('num_slices','num_slices')]),
        (n_slicetiming, n_realign, [('timecorrected_files','in_files')]),
        (n_realign, n_tsnr, [('realigned_files','in_file')]),
        (inputnode, n_regoutnoise, [('noise_rois','noise_mask_files')]),
        (n_tsnr, n_regoutnoise, [('detrended_file','in_file')]),
        (n_regoutnoise, n_noisecorr, [('noise_components','design_file')]),
        (n_tsnr, n_noisecorr, [('detrended_file','in_file')]),
        (n_noisecorr, n_bandpass_filter, [('out_file','in_file')]),
        (n_bandpass_filter, n_smooth, [('out_file','in_files')]),
        (inputnode, n_smooth, [('smooth_fwhm','fwhm')]),
        (n_smooth, outputnode, [('smoothed_files','preprocessed_file')]),
        ])

    return workflow


def create_compcorr(name='compcorr'):
    from nipype.workflows.fsl.resting import extract_noise_components
    from nipype.algorithms.misc import TSNR

    wkfl = pe.Workflow(name=name)
    inputnode = pe.Node(util.IdentityInterface(fields=['in_file',
                                                       'num_components']),
                           name='inputspec')
    outputnode = pe.Node(util.IdentityInterface(fields=['corrected_file']),
                         name='outputspec')
                            
    tsnr = pe.MapNode(TSNR(regress_poly=2), name='tsnr', iterfield=['in_file'])
    getthresh = pe.MapNode(interface=fsl.ImageStats(op_string='-p 98'),
                        name='getthreshold', iterfield=['in_file'])
    threshold_stddev = pe.MapNode(fsl.Threshold(), name='threshold',
                               iterfield=['in_file','thresh'])
    compcor = pe.MapNode(util.Function(input_names=['realigned_file',
                                                 'noise_mask_file',
                                                 'num_components'],
                                     output_names=['noise_components'],
                                     function=extract_noise_components),
                         name='compcorr',
                         iterfield=['realigned_file','noise_mask_file'])
    remove_noise = pe.MapNode(fsl.FilterRegressor(filter_all=True),
                              name='remove_noise',
                              iterfield=['in_file','design_file'])


    wkfl.connect([
        (inputnode,tsnr,[('in_file','in_file')]),
        (inputnode, compcor, [('in_file','realigned_file'),
                              ('num_components','num_components')]),
        (tsnr, threshold_stddev,[('stddev_file', 'in_file')]),
        (tsnr, getthresh, [('stddev_file', 'in_file')]),
        (tsnr, remove_noise, [('detrended_file','in_file')]),
        (getthresh, threshold_stddev,[('out_stat', 'thresh')]),
        (threshold_stddev, compcor, [('out_file',  'noise_mask_file')]),
        (compcor,remove_noise, [('noise_components', 'design_file')]),
        (remove_noise, outputnode, [('out_file','corrected_file')]),
        ])

    return wkfl
    

def create_corsica_noise_corr(name='corsica_noisecorr'):
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(util.IdentityInterface(fields=['in_files',
                                                       'mask',
                                                       'noise_rois']),
                        name='inputspec')
    ica=pe.MapNode(interface=lif.SICA(),name='ica',iterfield=['in_file','mask'])
    corsica=pe.MapNode(interface=lif.CORSICA(), name='corsica',
                       iterfield=['in_file','noise_rois','sica_file'])


    outputnode = pe.Node(util.IdentityInterface(fields=['corrected_file',
                                                        'sica_file',
                                                        'components']),
                         name='outputspec')

    
    workflow.connect([
        (inputnode, ica, [('in_files', 'in_file'),
                          ('mask', 'mask')]),
        (inputnode, corsica, [('in_files', 'in_file'),
                              ('noise_rois', 'noise_rois')]),
        (ica, corsica, [('sica_file', 'sica_file')]),
        (corsica, outputnode, [('corrected_file', 'corrected_file')]),
        (ica, outputnode, [('sica_file','sica_file')]),
        ])
    return workflow

def create_corsica_preproc(name='corsica'):
    """create a ICA/CorsICA preprocessing pipeline
    The noise removal is based on CORSICA Perlbarg et al. (2007)
    """
    
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(util.IdentityInterface(fields=['in_files',
                                                      'noise_rois',
                                                       'smooth_fwhm']),
                        name='inputspec')
    n_slicetiming = pe.Node(
        interface=spm.SliceTiming(),
        name='slicetiming')

    n_realign = pe.Node(interface=spm.Realign(), name='realign')
    
    n_smooth = pe.Node(
        interface=spm.Smooth(),
        name='smooth')

    corsica_noise_corr = create_corsica_noise_corr()

    outputnode = pe.Node(util.IdentityInterface(fields=['preprocessed_file',
                                                       'sica_file',
                                                       'components']),
                         name='outputspec')
    workflow.connect([
        (inputnode, n_slicetiming, [('in_files','in_files')]),
        (n_slicetiming, n_realign, [('timecorrected_files','in_files')]),
        (n_realign, n_smooth, [('realigned_files','in_files')]),
        (inputnode, n_smooth, [('smooth_fwhm','fwhm')]),
        (n_smooth, corsica_noise_corr, [('smoothed_files',
                                         'inputspec.in_files')]),
        (corsica_noise_corr, outputnode, [('outputspec.corrected_file',
                                           'preprocessed_file')]),
        (corsica_noise_corr, outputnode, [('outputspec.sica_file',
                                           'sica_file')]),
        ])

    return workflow
