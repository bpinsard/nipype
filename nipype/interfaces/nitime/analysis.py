
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

Interfaces to functionality from nitime for time-series analysis of fmri data

- nitime.analysis.CoherenceAnalyzer: Coherence/y
- nitime.fmri.io:
- nitime.viz.drawmatrix_channels

"""

import warnings
import numpy as np
import tempfile
from nipype.utils.misc import package_check
package_check('matplotlib')

from nipype.interfaces.base import (TraitedSpec, File, Undefined, traits,
                                    BaseInterface, isdefined,
                                    BaseInterfaceInputSpec,
                                    InputMultiPath, OutputMultiPath)

from nipype.utils.filemanip import fname_presuffix, savepkl, loadpkl
import nibabel as nb
import os


have_nitime = True
try:
    package_check('nitime')
except Exception, e:
    have_nitime = False
    warnings.warn('nitime not installed')
else:
    import nitime.analysis as nta
    from nitime.algorithms.correlation import seed_corrcoef
    from nitime.timeseries import TimeSeries
    import nitime.viz as viz


have_nipy = True
try:
    package_check('nipy')
except Exception, e:
    have_nipy = False
    warnings.warn('nipy not installed')
else:
    from nipy.algorithms.statistics import bootstrap


def pca(data):
#    from nipy.algorithms.utils import pca
#    return np.squeeze(pca.pca(data,axis=0,ncomp=1)['basis_projections'])
    evl,evc=np.linalg.eig(data.dot(data.T))
    return evc.T.dot(data)[0]

def mean(data):
    return np.mean(data,0)


class CoherenceAnalyzerInputSpec(BaseInterfaceInputSpec):

    #Input either csv file, or time-series object and use _xor_inputs to
    #discriminate
    _xor_inputs=('in_file','in_TS')
    in_file = File(desc=('csv file with ROIs on the columns and ',
                   'time-points on the rows. ROI names at the top row'),
                   exists=True,
                   requires=('TR',))

    #If you gave just a file name, you need to specify the sampling_rate:
    TR = traits.Float(desc=('The TR used to collect the data',
                            'in your csv file <in_file>'))

    in_TS = traits.Any(desc='a nitime TimeSeries object')

    NFFT = traits.Range(low=32,value=64, usedefault=True,
                        desc=('This is the size of the window used for ',
                        'the spectral estimation. Use values between ',
                        '32 and the number of samples in your time-series.',
                        '(Defaults to 64.)'))
    n_overlap = traits.Range(low=0,value=0,usedefault=True,
                             desc=('The number of samples which overlap',
                             'between subsequent windows.(Defaults to 0)'))

    frequency_range = traits.List(value=[0.02, 0.15],usedefault=True,
                                  minlen=2,
                                  maxlen=2,
                                  desc=('The range of frequencies over',
                                        'which the analysis will average.',
                                        '[low,high] (Default [0.02,0.15]'))

    output_csv_file = File(desc='File to write outputs (coherence,time-delay) with file-names: file_name_ {coherence,timedelay}')

    output_figure_file = File(desc='File to write output figures (coherence,time-delay) with file-names: file_name_{coherence,timedelay}. Possible formats: .png,.svg,.pdf,.jpg,...')

    figure_type = traits.Enum('matrix','network',usedefault=True,
                              desc=("The type of plot to generate, where ",
                                    "'matrix' denotes a matrix image and",
                                    "'network' denotes a graph representation.",
                                    " Default: 'matrix'"))

class CoherenceAnalyzerOutputSpec(TraitedSpec):
    coherence_array = traits.Array(desc=('The pairwise coherence values',
                                         'between the ROIs'))

    timedelay_array = traits.Array(desc=('The pairwise time delays between the,'
                                         'ROIs (in seconds)'))

    coherence_csv = File(desc = ('A csv file containing the pairwise ',
                                        'coherence values'))

    timedelay_csv = File(desc = ('A csv file containing the pairwise ',
                                        'time delay values'))

    coherence_fig = File(desc = ('Figure representing coherence values'))
    timedelay_fig = File(desc = ('Figure representing coherence values'))


class CoherenceAnalyzer(BaseInterface):

    input_spec = CoherenceAnalyzerInputSpec
    output_spec = CoherenceAnalyzerOutputSpec

    def _read_csv(self):
        """
        Read from csv in_file and return an array and ROI names

        The input file should have a first row containing the names of the
        ROIs (strings)

        the rest of the data will be read in and transposed so that the rows
        (TRs) will becomes the second (and last) dimension of the array

        """
        #Check that input conforms to expectations:
        first_row = open(self.inputs.in_file).readline()
        if not first_row[1].isalpha():
            raise ValueError("First row of in_file should contain ROI names as strings of characters")

        roi_names = open(self.inputs.in_file).readline().replace('\"','').strip('\n').split(',')
        #Transpose, so that the time is the last dimension:
        data = np.loadtxt(self.inputs.in_file,skiprows=1,delimiter=',').T

        return data,roi_names

    def _csv2ts(self):
        """ Read data from the in_file and generate a nitime TimeSeries object"""
        data,roi_names = self._read_csv()

        TS = TimeSeries(data=data,
                        sampling_interval=self.inputs.TR,
                        time_unit='s')

        TS.metadata = dict(ROIs=roi_names)

        return TS


    #Rewrite _run_interface, but not run
    def _run_interface(self, runtime):
        lb, ub = self.inputs.frequency_range

        if self.inputs.in_TS is Undefined:
            # get TS form csv and inputs.TR
            TS = self._csv2ts()

        else:
            # get TS from inputs.in_TS
            TS = self.inputs.in_TS

        # deal with creating or storing ROI names:
        if not TS.metadata.has_key('ROIs'):
            self.ROIs=['roi_%d' % x for x,_ in enumerate(TS.data)]
        else:
            self.ROIs=TS.metadata['ROIs']

        A = nta.CoherenceAnalyzer(TS,
                                  method=dict(this_method='welch',
                                              NFFT=self.inputs.NFFT,
                                              n_overlap=self.inputs.n_overlap))

        freq_idx = np.where((A.frequencies>self.inputs.frequency_range[0]) *
                            (A.frequencies<self.inputs.frequency_range[1]))[0]

        #Get the coherence matrix from the analyzer, averaging on the last
        #(frequency) dimension: (roi X roi array)
        self.coherence = np.mean(A.coherence[:,:,freq_idx],-1)
        # Get the time delay from analyzer, (roi X roi array)
        self.delay = np.mean(A.delay[:,:,freq_idx],-1)
        return runtime

    #Rewrite _list_outputs (look at BET)
    def _list_outputs(self):
        outputs = self.output_spec().get()

        #if isdefined(self.inputs.output_csv_file):

            #write to a csv file and assign a value to self.coherence_file (a
            #file name + path)

        #Always defined (the arrays):
        outputs['coherence_array']=self.coherence
        outputs['timedelay_array']=self.delay

        #Conditional
        if isdefined(self.inputs.output_csv_file) and hasattr(self,'coherence'):
            # we need to make a function that we call here that writes the
            # coherence values to this file "coherence_csv" and makes the
            # time_delay csv file??
            self._make_output_files()
            outputs['coherence_csv']=fname_presuffix(self.inputs.output_csv_file,suffix='_coherence')

            outputs['timedelay_csv']=fname_presuffix(self.inputs.output_csv_file,suffix='_delay')

        if isdefined(self.inputs.output_figure_file) and hasattr(self,
                                                                 'coherence'):
            self._make_output_figures()
            outputs['coherence_fig'] = fname_presuffix(self.inputs.output_figure_file,suffix='_coherence')
            outputs['timedelay_fig'] = fname_presuffix(self.inputs.output_figure_file,suffix='_delay')

        return outputs
    def _make_output_files(self):
        """
        Generate the output csv files.
        """
        for this in zip([self.coherence,self.delay],['coherence','delay']):
            tmp_f = tempfile.mkstemp()[1]
            np.savetxt(tmp_f,this[0],delimiter=',')

            fid = open(fname_presuffix(self.inputs.output_csv_file,
                                       suffix='_%s'%this[1]),'w+')
            # this writes ROIs as header line
            fid.write(','+','.join(self.ROIs)+'\n')
            # this writes ROI and data to a line
            for r, line in zip(self.ROIs, open(tmp_f)):
                fid.write('%s,%s'%(r,line))
            fid.close()


    def _make_output_figures(self):
        """
        Generate the desired figure and save the files according to
        self.inputs.output_figure_file

        """

        if self.inputs.figure_type == 'matrix':
            fig_coh = viz.drawmatrix_channels(self.coherence,
                                channel_names=self.ROIs,
                                color_anchor=0)

            fig_coh.savefig(fname_presuffix(self.inputs.output_figure_file,
                                    suffix='_coherence'))

            fig_dt = viz.drawmatrix_channels(self.delay,
                                channel_names=self.ROIs,
                                color_anchor=0)

            fig_dt.savefig(fname_presuffix(self.inputs.output_figure_file,
                                    suffix='_delay'))
        else:
            fig_coh = viz.drawgraph_channels(self.coherence,
                                channel_names=self.ROIs)

            fig_coh.savefig(fname_presuffix(self.inputs.output_figure_file,
                                    suffix='_coherence'))

            fig_dt = viz.drawgraph_channels(self.delay,
                                channel_names=self.ROIs)

            fig_dt.savefig(fname_presuffix(self.inputs.output_figure_file,
                                    suffix='_delay'))

class GetTimeSeriesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   desc="Run file from which to extract timeseries")
    rois_files = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = "ROIs integer files describing regions of interest from which to extract timeseries.")
    mask = File(desc = "A mask file to restrain the ROIs.")
    mask_threshold = traits.Float(
        0, usedefault=True,
        desc = "The default threshold to binarize mask image to create mask. Can be useful to create mask from probability image.")
    rois_labels_files = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = "text files with ROIs labels")
    
    aggregating_function = traits.Function(
        desc = """Function to apply to voxel timeseries to get ROIs timeseries.
(default: np.mean) 
ex: np.median, a custom pca component selection function.""")
    
    bootstrap_estimation = traits.Bool(
        False, usedefault = True,
        desc = "Perform a bootstrap estimate of the timeseries with the aggregating function")
    bootstrap_nsamples = traits.Int(
        100, usedefault = True,
        desc = """Number of samples to perform bootstrap estimates.
If greater than the number of possible combinations, the exact resampling will be performed.""")
    
    sampling_interval = traits.Float(2,desc="TR of the data in sec")
    
    
class GetTimeSeriesOutputSpec(TraitedSpec):
    timeseries = File(desc = """The timeseries file as a compressed umpy (.npz) file with fields : TODO""")
    
    """
    timeseries_file = File(desc = "The timeseries file as a compressed numpy (.npz) file with fields : TODO")
    
    timeseries = InputMultiPath(traits.Either(traits.Any(),
    traits.List(traits.Any())),
    desc="Timeseries object or list of timeseries")
"""
    
class GetTimeSeries(BaseInterface):
    # getting time series data from nifti files and ROIs
    input_spec = GetTimeSeriesInputSpec
    output_spec = GetTimeSeriesOutputSpec
    
    def _run_interface(self, runtime):
        #load data and rois files
        run_nii = nb.load(self.inputs.in_file)
        run_data = run_nii.get_data()
        rois_nii = [nb.load(rois_file) for rois_file in self.inputs.rois_files]
        if self.inputs.mask:
            mask_nii = nb.load(self.inputs.mask)
            mask_data = mask_nii.get_data() > self.inputs.mask_threshold
            rois_data = [nii.get_data()*mask_data for nii in rois_nii]
            del mask_nii, mask_data
        else:
            rois_data = [nii.get_data() for nii in rois_nii]
        
        if self.inputs.rois_labels_files:
            rois_labels = [[l.strip('\n') for l in open(labels_file).readlines()] for labels_file in self.inputs.rois_labels_files]
        else:
            rois_labels = [[]]*len(rois_data)


        timeseries = dict()
        pvalues = dict()
        labels_list = []

        #extract timeseries
        for rois,labels in zip(rois_data,rois_labels):
            if labels == []:
                labels = range(1,rois.max()+1)
            for roi,label in enumerate(labels):
                if timeseries.has_key(label):
                    raise ValueError("Duplicate ROIs label "+label)
                ts = np.array(run_data[ rois == roi+1 ])
                pval = 1
                if ts.ndim < 2:
                    ts = np.array(ts[np.newaxis,:])
                if self.inputs.aggregating_function:
                    if ts.shape[0] > 1:
                        if self.inputs.bootstrap_estimation:
                            std,lb,ub,samples = bootstrap.generic_bootstrap(
                                ts,self.inputs.aggregating_function,
                                self.inputs.bootstrap_nsamples)
                            ts = samples.mean(axis=0)
                            del samples
                            pval = std
                        else:
                            ts = self.inputs.aggregating_function(ts)
                    elif ts.shape[0] > 0:
                        ts = np.squeeze(ts)
                    else:
                        ts = np.zeros(run_data.shape[-1])

                timeseries[label] = ts
                pvalues[label] = pval
                labels_list.append(label)
        out_data = dict(timeseries = timeseries,
                        pvalues = pvalues,
                        labels = labels_list)
        fname = self._list_outputs()['timeseries']
        savepkl(fname,out_data)
        
        del run_nii, run_data, rois_nii, rois_data
        return runtime

        
    def _list_outputs(self):
        outputs = self.output_spec().get()

        outputs['timeseries'] = fname_presuffix(self.inputs.in_file,
                                                suffix='_ts',
                                                newpath = os.getcwd(),
                                                use_ext=False) + '.pklz'
        return outputs


class CorrelationAnalysisInputSpec(BaseInterfaceInputSpec):
    timeseries = File(
        exists = True,
        desc = 'Timeseries file produced by GetTimeSeries interface.')
    
    bootstrap_estimation = traits.Bool(
        False, usedefault = True,
        desc = "Perform a bootstrap estimate of the correlation")
    bootstrap_nsamples = traits.Int(
        100, usedefault = True,
        desc = """Number of samples to perform bootstrap estimates.
If greater than the number of possible combinations, the exact resampling will be performed.""")


class CorrelationAnalysisOutputSpec(TraitedSpec):
    correlations = File(desc = 'File containing the correlation values extracted from the timeseries and theirs confidences measures.')

def corr_to_partialcorr(corr):
    u = np.linalg.inv(corr)
    d = np.diag(u)
    if np.any(d<0):
        raise ValueError('Ill conditionned matrix (negative inverse diagonal)')
    d = np.diag(1/np.sqrt(d))
    return 2*np.eye(corr.shape[0]) - d.dot(u).dot(d)


class CorrelationAnalysis(BaseInterface):
    input_spec = CorrelationAnalysisInputSpec
    output_spec = CorrelationAnalysisOutputSpec
    
    def _run_interface(self,runtime):
        tsfile = loadpkl(self.inputs.timeseries)
        ts = np.array([tsfile['timeseries'][roi] for roi in tsfile['labels']])
        if self.inputs.bootstrap_estimation:
            std,lb,ub,samples = bootstrap.generic_bootstrap(
                ts, np.corrcoef,
                self.inputs.bootstrap_nsamples)
            pval = std
            corr = samples.mean(0)
            print np.count_nonzero(np.isnan(samples)) ,'nan in correlations'
            try:
                samples_part = np.array([corr_to_partialcorr(s) for s in samples])
                partialcorr = samples_part.mean(0)
                partialpval = samples_part.std(0)
            except ValueError:                
                partialcorr = None
                partialpval = None
        else:
            corr = np.corrcoef(ts)
            try:
                partialcorr = corr_to_partialcorr(corr)
            except ValueError:
                partialcorr = None
            pval = []
            partialpval = []
        
        fname = self._list_outputs()['correlations']
        out_data = dict(labels = tsfile['labels'],
                        corr = corr,
                        partialcorr = partialcorr,
                        pval = pval,
                        partialpval = partialpval)
        savepkl(fname, out_data)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()

        outputs['correlations'] = fname_presuffix(self.inputs.timeseries,
                                                      suffix='_corr',
                                                      newpath = os.getcwd())
        return outputs

class HomogeneityAnalysisInputSpec(BaseInterfaceInputSpec):
    timeseries = File(
        exists = True,
        desc = 'Timeseries file produced by GetTimeSeries interface.')
    


class HomogeneityAnalysisOutputSpec(TraitedSpec):
    kendall_W = File(
        exists = True,
        desc = 'Homogeneity measures in each region of interest')
    corr = File(
        exists = True,
        desc = 'Correlation stats in each region of interest')


class HomogeneityAnalysis(BaseInterface):
    input_spec = HomogeneityAnalysisInputSpec
    output_spec = HomogeneityAnalysisOutputSpec
    
    def _run_interface(self,runtime):
        tsfile = loadpkl(self.inputs.timeseries)
        kcc = dict()
        mean_corr = dict()
        min_corr = dict()
        for roi in tsfile['labels']:
            ts = tsfile['timeseries'][roi]
            if ts.shape[0] > 1:
                #normalize
                ts -= ts.mean(1)[:,np.newaxis]
                ts /= ts.std(1)[:,np.newaxis]

                #compute rank data
                from bottleneck import rankdata
                y=rankdata(ts,axis=1)
                
                y = (y-y.mean(0)[np.newaxis,:])**2
                n,t = ts.shape
                k = (t*(t**2-1)*n**2)/(12*(n-1))
                kcc[roi]=1-y.sum()/k
                
                corr=np.corrcoef(ts)[np.tri(ts.shape[0],k=-1,dtype=bool)]
                mean_corr[roi]=corr.mean()
                min_corr[roi]=corr.min()
            else:
                kcc[roi]=1
                mean_corr[roi]=1

        savepkl(self._list_outputs()['kendall_W'], dict(kcc=kcc))
        savepkl(self._list_outputs()['corr'], dict(mean_corr = mean_corr,
                                                   min_corr = min_corr))
        return runtime

    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['kendall_W'] = fname_presuffix(self.inputs.timeseries,
                                                suffix='_kendall',
                                                newpath = os.getcwd())
        outputs['corr'] = fname_presuffix(self.inputs.timeseries,
                                          suffix='_corr',
                                          newpath = os.getcwd())

        return outputs

class IntegrationAnalysisInputSpec(BaseInterfaceInputSpec):
    _xor_inputs = ['correlations','correlation_file']
    correlations = traits.Any(desc='correlation matrix')
    correlations_file = File(
        exists=True,
        desc='Output file of correlation analysis interface.')

    networks = traits.Dict(desc='Network dictionnary containing region id.')

class IntegrationAnalysisOutputSpec(TraitedSpec):
    integration = traits.Float(desc='total integration measure')
    inter_network_integration = traits.Dict(
        desc = 'integration measure inter networks')
    intra_network_integration = traits.Dict(
        desc = 'integration measure intra networks')

class IntegrationAnalysis(BaseInterface):
    input_spec  = IntegrationAnalysisInputSpec
    output_spec = IntegrationAnalysisOutputSpec

    def _run_interface(self,runtime):
        if self.inputs.correlations:
            correlations = self.inputs.correlations
        else:
            correlations = loadpkl(self.inputs.correlations_file)
        
        nets = dict()
        if correlations.has_key('nets'):
            nets = correlations['keys']
        if isinstance(self.inputs.networks,dict):
            nets = self.inputs.networks

        integ_anlzr = nta.IntegrationAnalyzer(correlations['corr'],
                                              networks=nets)
        self.total, self.inter, self.intra = integ_anlzr.integration
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['integration'] = self.total
        outputs['inter_network_integration'] = self.inter
        outputs['intra_network_integration'] = self.intra
        return outputs

class GraphAnalysis(BaseInterface):
    #TODO
    pass
    

class SeedCorrelationAnalysisInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = "Input files from which to extract seed correlation maps.")
    mask = File(desc = 'Mask to select voxels to compute correlation maps. It not specified, a implicit masking will be performed with NaN or zeros timeseries.')
    seeds_coordinates = traits.List(
        traits.Tuple(traits.Int,traits.Int,traits.Int),
        desc = 'Voxel coordinates of the seeds.')
    seeds_radius = traits.Float(
        1,usedefault=True,
        desc='radius used to compute voxels average for seeds coordinates')
    seeds_maps = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = "Seeds to compute correlation maps.")
    seeds_maps_type = traits.Enum(
        'regions','voxels',
        desc = 'Either to extract seed signal from regions with different values in the maps, or to extract seed signal from individual nonzero voxels in maps.')
    # TODO: add time axis bootstrap estimation of correlation maps
    
class SeedCorrelationAnalysisOutputSpec(TraitedSpec):
    seed_correlation_maps = OutputMultiPath(
        traits.List(File(exists=True)),
        desc='Seed correlation maps.')

class SeedCorrelationAnalysis(BaseInterface):
    input_spec = SeedCorrelationAnalysisInputSpec
    output_spec = SeedCorrelationAnalysisOutputSpec
    
    
    def _run_interface(self,runtime):
        run_niis = [nb.load(f) for f in self.inputs.in_files]
        if np.diff([nii.get_shape() for nii in run_niis],axis=0).sum()>0:
            raise ValueError('Input files must have same shape')
        data = np.array([nii.get_data() for nii in run_niis])
        # load mask or create implicit mask
        if self.inputs.mask:
            mask_nii = nb.load(self.inputs.mask)
            if mask_nii.get_shape() != run_niis[0].get_shape()[:-1]:
                raise ValueError('Mask file must have same shape as input files')
            mask = mask_nii.get_data() > 0
        else:
            mask = np.ones(run_niis[0].shape,dtype=bool)
        #implicit masking : remove  nans and zeros time series
        mask *= np.prod([(np.isnan(d).sum(axis=-1)==0)* \
                             (d.var(-1)>0) for d in data],
                        axis=0)
        mask = mask > 0

        seeds_ts = [[]]*len(run_niis)
        coords = list(self.inputs.seeds_coordinates)
        radius = self.inputs.seeds_radius
        tile = (np.mgrid[-radius:radius+1,
                          -radius:radius+1,
                          -radius:radius+1]**2).sum(0) <= radius**2
        for seed in self.inputs.seeds_coordinates:
            print seed,radius,data.shape,tile.shape
            for di,d in enumerate(data):
                ts = d[seed[0]-radius:seed[0]+radius+1,
                       seed[1]-radius:seed[1]+radius+1,
                       seed[2]-radius:seed[2]+radius+1][tile].mean(0)
                seeds_ts[di].append(ts)

        #load seeds maps
        if self.inputs.seeds_maps:
            seed_niis = [nb.load(f) for f in self.inputs.seeds_maps]
            
            for smapidx,seed_nii in enumerate(seed_niis):
                smap = seed_nii.get_data()*mask
                seeds_ts.append([])
                if self.inputs.seeds_maps_type == 'voxels':
                    for di,d in enumerate(data):
                        for ts in d[smap>0]:
                            seeds_ts[di].append(ts)
                        coords.append([tuple(c) for c in  np.array(np.where(smap)).T])
                else:
                    rois = np.unique(smap)[1:]
                    for r in rois:
                        m = smap==r
                        coords.append(tuple([int(c.mean()) for c in np.where(m)]))
                        for di,d in enumerate(data):
                            seeds_ts[di].append(d[m].mean(0))
        
        #apply mask to data
        data = data[:,mask,:]
        self.map_fnames = []        
        seed_map=np.empty(mask.shape+(len(coords),), np.float32)+np.nan
        for di,d in enumerate(data):
            for tsi,ts in enumerate(seeds_ts[di]):
                seed_map[mask,tsi] = seed_corrcoef(ts,d)
            fname = fname_presuffix(
                self.inputs.in_files[di],
                suffix = 'seed_maps.nii',
                use_ext = False, newpath = os.getcwd())
            nii = nb.Nifti1Image(seed_map,mask_nii.get_affine())
            nb.save(nii,fname)
            self.map_fnames.append(fname)
        return runtime

    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['seed_correlation_maps'] = self.map_fnames
        return outputs
