import os
import warnings

import nibabel as nb
import numpy as np
import nipy.labs.correlation as corr
import scipy.stats as sstats

from nipype.interfaces.base import (TraitedSpec, BaseInterface, traits,
                                    BaseInterfaceInputSpec, isdefined, File)

class CorrelationDistributionInputSpec(BaseInterfaceInputSpec):

    in_file = File(
        exists = True,
        desc = 'The session file on which to compute correlations')

    masks = traits.List(
        File(exists=True),
        desc = 'Mask files used to sample intra-mask and inter-mask correlations and compute respective distributions.')

    frequency_bands = traits.List([],
        traits.Tuple(traits.Float(),traits.Float()),
        desc='The list of frequency bands to filter data before computing the correlation. All results will be outputed as b0 for the whole frequency band (no filtering) and b[1...] for the frequency bands provided here.')
    min_distance = traits.Float(
        0, usedefault = True,
        desc = """The min distance threshold to compute correlation distribution""")

    nbins = traits.Int(
        1000, usedefault = True,
        desc = 'The number of bins of the distribution')

    nsamples = traits.Int(1000, usedefault = True,
                          desc = """The number of voxel to sample per mask.
                          If mask nonzeros voxels are less than that this,
                          there will be less samples""")
    connexity = traits.Int(1, usedefault = True,
                           desc = 'The connexity to use to average signal to simulate measures of correlation using ROIs.')

    mask_threshold = traits.Float(0, usedefault = True,
                                  desc = 'Threshold for the masks')

    tr = traits.Float(desc = 'Time of repetition of the data, used to filter the data')

class CorrelationDistributionOutputSpec(TraitedSpec):

    intra_correlation_samples = File(exists=True,
        desc = "Correlation samples files, contains also distance between sampled voxels")
    intra_correlation_distributions = traits.List(
        File(exists=True),
        desc = "Correlation distributions outputs")

    inter_correlation_samples = File(exists=True,
        desc = "Correlation samples files, contains also distance between sampled voxels")
    inter_correlation_distributions = traits.List(
        File(exists=True),
        desc = "Correlation distributions outputs")


    mean = traits.Dict(
        traits.String(),traits.List(traits.List(traits.Float())),
        desc = 'mean of the correlations')
    variance = traits.Dict(
        traits.String(),traits.List(traits.List(traits.Float())),
        desc = 'variance of the correlations')
    skewness = traits.Dict(
        traits.String(),traits.List(traits.List(traits.Float())),
        desc = 'skewness of the correlations')
    kurtosis = traits.Dict(
        traits.String(),traits.List(traits.List(traits.Float())),
        desc = 'kurtosis of the correlations')
    

class CorrelationDistribution(BaseInterface):
    input_spec = CorrelationDistributionInputSpec
    output_spec = CorrelationDistributionOutputSpec

    def _average_ngbr(data,conxt):
        # average the data over a 6,18 or 26 neighborhood
        from scipy.signal import convolve
        import numpy as np
        w = 1./conxt
        if conxt == 6:
            kern = np.zeros((3,3,3))
            kern[1,1],kern[1,:,1],kern[:,1,1] = w,w,w
        elif conxt == 18:
            kern = np.zeros((3,3,3))
            kern[1], kern[:,1], kern[:,:,1] = w,w,w
        elif conxt == 26:
            kern = np.ones((3,3,3))*w
        else:
            raise ValueError('connexity must be 6,18 or 26')
        for vol in range(data.shape[-1]):
            data[...,vol] = convolve(data[...,vol],kern,'same')
        return data
        

    def _run_interface(self,runtime):

        nii = nb.load(self.inputs.in_file)
        data = nii.get_data()
        maskniis = [nb.load(m) for m in self.inputs.masks]
        masks = np.array([m.get_data() for m in maskniis])
        masks = masks.transpose((1,2,3,0)) > self.inputs.mask_threshold
        
        # average data over a neighborhood to simulate the use of ROIs
        if self.inputs.connexity > 1:
            data = self._average_ngbr(data,self.inputs.connexity)

        freqbands = self.inputs.frequency_bands
        voxel_size = np.sqrt(np.square(nii.get_affine()).sum(0))[:3]
        
        [intra_corr, inter_corr, intra_dists, inter_dists] = corr.sample_correlation(
            data,masks,voxel_size=voxel_size,
            nsamples=self.inputs.nsamples,
            frequency_bands=freqbands,
            tr = self.inputs.tr)
        
        del data,masks
        bfname = os.path.splitext(os.path.basename(self.inputs.in_file))[0]
        self._intra_prob_names, self._inter_prob_names=[],[]
        #save distance/correlation data
        self._intra_corr_name=os.path.join(os.getcwd(),
                                           bfname + ('_corr_intra.npz'))
        np.savez_compressed(self._intra_corr_name,np.concatenate((
                    intra_dists.reshape((1,)+intra_dists.shape),
                    intra_corr)).astype(np.float16))

        self._inter_corr_name=os.path.join(os.getcwd(),
                                           bfname + ('_corr_inter.npz'))
        np.savez_compressed(self._inter_corr_name,np.concatenate((
            inter_dists.reshape((1,)+inter_dists.shape),
            inter_corr)).astype(np.float16))

        nbins = self.inputs.nbins
        min_dist = self.inputs.min_distance
        self._mean = dict(intra=[],inter=[])
        self._variance = dict(intra=[],inter=[])
        self._skewness = dict(intra=[],inter=[])
        self._kurtosis = dict(intra=[],inter=[])
        
        #compute distributions
        intra_prob = np.empty((len(freqbands)+1,len(maskniis), nbins),np.float)
        for bi,band in enumerate(intra_corr):
            self._mean['intra'].append([])
            self._variance['intra'].append([])
            self._skewness['intra'].append([])
            self._kurtosis['intra'].append([])
            for ti,dists in enumerate(intra_dists):
                sample = band[ti,dists>min_dist]
                h,e=np.histogram(sample, nbins, [-1,1], density=True)
                intra_prob[bi,ti] = h
                self._mean['intra'][-1].append(np.mean(sample))
                self._variance['intra'][-1].append(np.var(sample))
                self._skewness['intra'][-1].append(sstats.skew(sample))
                self._kurtosis['intra'][-1].append(sstats.kurtosis(sample))
        for bi,band in enumerate(intra_prob):
            fname = os.path.join(os.getcwd(), bfname + ('_intra_b%d.npy'% bi))
            np.save(fname, intra_prob[bi])
            self._intra_prob_names.append(fname)

        inter_prob = np.empty((len(freqbands)+1,inter_corr.shape[1],nbins),
                              np.float)
        for bi,band in enumerate(inter_corr):
            self._mean['inter'].append([])
            self._variance['inter'].append([])
            self._skewness['inter'].append([])
            self._kurtosis['inter'].append([])
            for ti,dists in enumerate(inter_dists):
                sample = band[ti,dists>min_dist]
                h,e=np.histogram(sample, nbins, [-1,1], density=True)
                inter_prob[bi,ti] = h
                self._mean['inter'][-1].append(np.mean(sample))
                self._variance['inter'][-1].append(np.var(sample))
                self._skewness['inter'][-1].append(sstats.skew(sample))
                self._kurtosis['inter'][-1].append(sstats.kurtosis(sample))
        for bi,band in enumerate(inter_prob):
            fname = os.path.join(os.getcwd(), bfname + ('_inter_b%d.npy'% bi))
            np.save(fname, inter_prob[bi])
            self._inter_prob_names.append(fname)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["intra_correlation_samples"] = self._intra_corr_name
        outputs["inter_correlation_samples"] = self._inter_corr_name
        outputs["intra_correlation_distributions"]= self._intra_prob_names
        outputs["inter_correlation_distributions"]= self._inter_prob_names
        
        outputs['mean'] = self._mean
        outputs['variance'] = self._variance
        outputs['skewness'] = self._skewness
        outputs['kurtosis'] = self._kurtosis
        return outputs


class MultiCorrelationDistributionInputSpec(BaseInterfaceInputSpec):

    in_files = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = 'The session files on which to compute correlations')

    masks = traits.List(
        File(exists=True),
        desc = 'Mask files used to sample intra-mask and inter-mask correlations and compute respective distributions.')

    nbins = traits.Int(
        1000, usedefault = True,
        desc = 'The number of bins of the distribution')

    nsamples = traits.Int(
        1000, usedefault = True,
        desc = """The number of voxel to sample per mask.
                  If mask nonzeros voxels are less than that this,
                  there will be less samples""")
    connexity = traits.Int(
        1, usedefault = True,
        desc = 'The connexity to use to average signal to simulate measures of correlation using ROIs.')

    mask_threshold = traits.Float(
        0, usedefault = True,
        desc = 'Threshold for the masks')


class MultiCorrelationDistributionOutputSpec(TraitedSpec):

    intra_correlation_samples = File(
        exists=True,
        desc = "Correlation samples files, contains also distance between sampled voxels")
    intra_correlation_distributions = traits.List(
        File(exists=True),
        desc = "Correlation distributions outputs")


class MultiCorrelationDistribution(BaseInterface):
    input_spec = MultiCorrelationDistributionInputSpec
    output_spec = MultiCorrelationDistributionOutputSpec


    def _run_interface(self,runtime):

        niis = [nb.load(f) for f in self.inputs.in_files]
        datas = [nii.get_data() for nii in niis]
        maskniis = [nb.load(m) for m in self.inputs.masks]
        masks = np.array([m.get_data() for m in maskniis])
        masks = masks.transpose((1,2,3,0)) > self.inputs.mask_threshold
        
        # average data over a neighborhood to simulate the use of ROIs
        if self.inputs.connexity > 1:
            cnxt = self.inputs.connexity
            datas = [self._average_ngbr(data,cnxt) for data in datas]

        voxel_size = np.sqrt(np.square(niis[0].get_affine()).sum(0))[:3]
        
        [intra_corr, intra_dists] = corr.multi_sample_correlation(
            datas,masks,voxel_size=voxel_size,
            nsamples=self.inputs.nsamples)
        
        del datas,masks
        bfname = os.path.splitext(os.path.basename(self.inputs.in_file))[0]
        #save distance/correlation data
        self._intra_corr_name=os.path.join(os.getcwd(),
                                           bfname + ('_corr_intra.npz'))
        np.savez_compressed(self._intra_corr_name,np.concatenate((
                    intra_dists.reshape((1,)+intra_dists.shape),
                    intra_corr)).astype(np.float16))

        nbins = self.inputs.nbins
        min_dist = self.inputs.min_distance
        self._mean = []
        self._variance = []
        self._skewness = []
        self._kurtosis = []
        
        #compute distributions
        intra_prob = np.empty((len(datas),len(maskniis), nbins),np.float)
        for di,dcorr in enumerate(intra_corr):
            self._mean['intra'].append([])
            self._variance['intra'].append([])
            self._skewness['intra'].append([])
            self._kurtosis['intra'].append([])
            for ti,dists in enumerate(intra_dists):
                sample = dcorr[ti,dists>min_dist]
                h,e=np.histogram(sample, nbins, [-1,1], density=True)
                intra_prob[di,ti] = h
                self._mean['intra'][-1].append(np.mean(sample))
                self._variance['intra'][-1].append(np.var(sample))
                self._skewness['intra'][-1].append(sstats.skew(sample))
                self._kurtosis['intra'][-1].append(sstats.kurtosis(sample))
        self._intra_prob_name = os.path.join(os.getcwd(), 
                                             bfname + '_intra_dist.npy')
        np.save(self._intra_prob_name, intra_prob)

        return runtime

    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["intra_correlation_samples"] = self._intra_corr_name
        outputs["intra_correlation_distributions"]= self._intra_prob_name
        
        outputs['mean'] = self._mean
        outputs['variance'] = self._variance
        outputs['skewness'] = self._skewness
        outputs['kurtosis'] = self._kurtosis
        return outputs
