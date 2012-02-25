import os
import warnings

import nibabel as nb
import numpy as np
import nipy.labs.correlation as corr
import nipy.algorithms.utils.preprocess as preproc
import scipy.stats as sstats

from nipype.interfaces.base import (TraitedSpec, BaseInterface, traits,
                                    BaseInterfaceInputSpec, isdefined, File,
                                    InputMultiPath, OutputMultiPath)
from nipype.utils.filemanip import (fname_presuffix, filename_to_list,
                                    list_to_filename, split_filename,
                                    savepkl, loadpkl)


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

    sampling_masks = traits.List(
        File(exists=True),
        desc = 'Mask files used to sample intra-mask and inter-mask correlations and compute respective distributions.')

    mask = File(desc='optional mask to restrict the sampling masks')

    nbins = traits.Int(
        1000, usedefault = True,
        desc = 'The number of bins of the distribution')

    min_distance = traits.Float(
        0, usedefault = True,
        desc = """The min distance threshold to compute correlation distribution""")

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
    intra_correlation_distributions = File(
        exists=True,
        desc = "Correlation distributions outputs")

    mean = traits.List(
        traits.List(traits.Float()),
        desc = '')

    variance = traits.List(
        traits.List(traits.Float()),
        desc = '')
    skewness = traits.List(
        traits.List(traits.Float()),
        desc = '')
    kurtosis = traits.List(
        traits.List(traits.Float()),
        desc = '')


class MultiCorrelationDistribution(BaseInterface):
    input_spec = MultiCorrelationDistributionInputSpec
    output_spec = MultiCorrelationDistributionOutputSpec


    def _run_interface(self,runtime):

        niis = [nb.load(f) for f in self.inputs.in_files]
        datas = [nii.get_data() for nii in niis]
        maskniis = [nb.load(m) for m in self.inputs.sampling_masks]
        if self.inputs.mask:
            mask = nb.load(self.inputs.mask).get_data()>0
            masks = np.array([m.get_data()*mask for m in maskniis])
        else:
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
        bfname = 'multi'
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
        intra_prob = np.empty((len(intra_corr),len(maskniis), nbins),np.float)
        for di,dcorr in enumerate(intra_corr):
            self._mean.append([])
            self._variance.append([])
            self._skewness.append([])
            self._kurtosis.append([])
            for ti,dists in enumerate(intra_dists):
                sample = dcorr[ti,dists>min_dist]
                h,e=np.histogram(sample, nbins, [-1,1], density=True)
                intra_prob[di,ti] = h
                self._mean[di].append(np.mean(sample))
                self._variance[di].append(np.var(sample))
                self._skewness[di].append(sstats.skew(sample))
                self._kurtosis[di].append(sstats.kurtosis(sample))
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


class CorrelationDistributionMapsInput(BaseInterfaceInputSpec):

    in_files = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = 'The session files on which to compute correlations')
    
    mask = File(exists = True,
                desc = 'The brain mask used to compute correlation to seed voxels.')

    seed_masks = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = 'Mask files used to sample seeds to which whole-brain correlation is computed.')

    nsamples = traits.Int(
        -1, usedefault = True,
        desc = """The number of voxel to sample per seed_mask.
                  If -1 all voxels in seed masks will be used""")

    mask_threshold = traits.Float(
        0, usedefault = True,
        desc = 'Threshold for the seed masks')

    nbins = traits.Int(
        1000, usedefault = True,
        desc = 'The number of bins to compute the global distribution.')
    
    min_distance = traits.Float(
        0, usedefault = True,
        desc = """The min distance threshold to compute correlation distribution""")

    sampling_method = traits.Enum(
      'mask', 'rois', usedefault=True,
      desc = """Either:
mask : sampling nsamples from a mask, 
rois : sampling a ratio (rois_sampling_ratio) of the voxels of regions of interest""")
    rois_sampling_ratio = traits.Float(
      0.1, usedefault=True, 
      desc = 'which ratio of the rois voxels to samples')

class CorrelationDistributionMapsOutput(TraitedSpec):
    correlation_mean_maps = OutputMultiPath(
        traits.List(File(exists=True)),
        desc = 'correlation mean maps from run files provided')
#    correlation_median_maps = OutputMultiPath(
#        traits.List(File(exists=True)),
#        desc = 'correlation median maps from run files provided')
    correlation_variance_maps = OutputMultiPath(
        traits.List(File(exists=True)),
        desc = 'correlation variance maps from run files provided')
    correlation_std_maps = OutputMultiPath(
        traits.List(File(exists=True)),
        desc = 'correlation standard deviation maps from run files provided')
    correlation_skew_maps = OutputMultiPath(
        traits.List(File(exists=True)),
        desc = 'correlation skewness maps from run files provided')
    correlation_kurtosis_maps = OutputMultiPath(
        traits.List(File(exists=True)),
        desc = 'correlation kurtosis maps from run files provided')

#    correlation_kl_maps = OutputMultiPath(
#        traits.List(File(exists=True)),
#        desc = 'correlation kullback leibler divergence from mean-var normal fistribution maps')
    
    seed_maps = File(
        exists=True,
        desc = 'seed maps used to sample datas')

    all_cmaps = OutputMultiPath(
        traits.List(File(exists=True)),
        desc = 'all correlation sample in mask per file per seed map')

    global_correlation_distribution = OutputMultiPath(
        traits.List(File(exists=True)),
        desc = 'global correlation density function for all voxels indexed per seed mask ')

import gc 
class CorrelationDistributionMaps(BaseInterface):

    input_spec = CorrelationDistributionMapsInput
    output_spec = CorrelationDistributionMapsOutput

    def _run_interface(self,runtime):
        niis = [nb.load(f) for f in self.inputs.in_files]
        seed_niis = [nb.load(f) for f in self.inputs.seed_masks]
        maskthr = self.inputs.mask_threshold

        masknii = nb.load(self.inputs.mask)
        mask = masknii.get_data()>0

        if self.inputs.sampling_method == 'rois':
          seed_masks = [m.get_data()[mask] for m in seed_niis]
        else:
          seed_masks = [m.get_data()[mask]>maskthr for m in seed_niis]
        nseeds = self.inputs.nsamples

        nseed_masks = len(seed_masks)
        statsshape = mask.shape+(nseed_masks,)
        means = np.zeros(statsshape, np.float32)
#        medians = np.zeros(statsshape, np.float32)
        variances = np.zeros(statsshape, np.float32)
        stds = np.zeros(statsshape, np.float32)
        skews = np.zeros(statsshape, np.float32)
        kurtosiss = np.zeros(statsshape, np.float32)
#        klmaps = np.zeros(statsshape, np.float32)


        # sample the same seeds for all data
        seed_indices = []
        seed_maps = np.zeros(statsshape, np.int16) #16bits is min for most ni software
        for si,seed_mask in enumerate(seed_masks):
            seed_indices.append(corr.sample_voxels(
                    seed_mask, nseeds,
                    sampling_method = self.inputs.sampling_method,
                    rois_sampling_ratio = self.inputs.rois_sampling_ratio))
            seed_maps[...,si][[c[seed_indices[-1]] for c in np.where(mask)]] = True
        #save seed maps
        nb.save(nb.Nifti1Image(seed_maps,masknii.get_affine()),
                os.path.join(os.getcwd(),'seed_maps.nii'))

        # compute coordinates for distance
        voxel_size = np.sqrt(np.square(masknii.get_affine()[:3,:3]).sum(0))
        coords = np.array(np.where(mask)+(np.ones(np.count_nonzero(mask)),))
        coords = masknii.get_affine().dot(coords)[:3].T
        
        # compute all distances from seeds to all voxels
        all_dists = []
        for smi, seeds in enumerate(seed_indices):
            dists = np.empty((coords.shape[0],seeds.shape[0]),np.float16)
            for si, seed in enumerate(seeds):
                dists[:,si] = np.sqrt(((coords-coords[seed])**2).sum(1))
            all_dists.append(dists)

            dists_f = fname_presuffix(
                self.inputs.seed_masks[smi], suffix='_dists.npz',
                newpath=os.getcwd(),use_ext=False)
            np.savez_compressed(dists_f, dists=dists)

        del coords

        
        gstats =[[]]*len(seed_indices)
        # compute seed-based correlation maps
        for nii in niis:
            data = nii.get_data()[mask].astype(np.float32)
            fname = nii.get_filename()
            for si, seeds in enumerate(seed_indices):
                cmaps = corr.seeds_correlation(seeds,data)
                if self.inputs.min_distance > 0:
                    cmaps = np.ma.array(
                        cmaps, mask = all_dists[si] < self.inputs.min_distance)
                mean = np.array(cmaps.mean(1))
                var = np.array(cmaps.var(1))
                std = var**0.5
                #create maps
                means[mask,si] = mean
                variances[mask,si] = var
                stds[mask,si] = std
#                medians[mask,si] = np.array(np.ma.median(cmaps,1))
                ms = cmaps-mean[...,np.newaxis]
                skews[mask,si] = np.array((ms**3).mean(-1))/std**3
                kurtosiss[mask,si]=np.array((ms**4).mean(-1))/std**4
#                from nipy.algorithms.statistics.kl import norm_kl_divergence
#                klmaps[mask,si] = norm_kl_divergence(cmaps, mn=mean, vr=var)

                #global stats 
                gstats[si] = dict(
                    mean = float(mean.mean()),
                    var = float(cmaps.var()))
                gstats[si]['std'] = gstats[si]['var']**0.5
                ms = cmaps-gstats[si]['mean']
                gstats[si]['skew'] = float((ms**3).mean()/gstats[si]['std']**3)
                gstats[si]['kurtosis']=float((ms**4).mean()/gstats[si]['std']**4)
                #histogram
                gstats[si]['histogram']= np.histogram(cmaps.flatten(),
                                                      self.inputs.nbins,
                                                      [-1,1], density=True)[0]
                
                all_f = fname_presuffix(
                    fname, suffix='_seeds%d_all_cmaps.npz'%si,
                    newpath=os.getcwd(),use_ext=False)
                np.savez_compressed(all_f, cmaps=cmaps.__array__())
                del cmaps, ms, mean, var, std
                gc.collect()
            #writing maps
            nb.save(nb.Nifti1Image(means,masknii.get_affine()),
                    fname_presuffix(fname, suffix = '_meancorr.nii',
                                    newpath=os.getcwd(),use_ext=False))
#            nb.save(nb.Nifti1Image(medians,masknii.get_affine()),
#                    fname_presuffix(fname, suffix = '_mediancorr.nii',
#                                    newpath=os.getcwd(),use_ext=False))
            nb.save(nb.Nifti1Image(variances,masknii.get_affine()),
                    fname_presuffix(fname, suffix = '_varcorr.nii',
                                    newpath=os.getcwd(),use_ext=False))
            nb.save(nb.Nifti1Image(stds,masknii.get_affine()),
                    fname_presuffix(fname, suffix = '_stdcorr.nii',
                                    newpath=os.getcwd(),use_ext=False))
            nb.save(nb.Nifti1Image(skews,masknii.get_affine()),
                    fname_presuffix(fname, suffix = '_skewcorr.nii',
                                    newpath=os.getcwd(),use_ext=False))
            nb.save(nb.Nifti1Image(kurtosiss,masknii.get_affine()),
                    fname_presuffix(fname, suffix = '_kurtosiscorr.nii',
                                    newpath=os.getcwd(),use_ext=False))
#            nb.save(nb.Nifti1Image(klmaps,masknii.get_affine()),
#                    fname_presuffix(fname, suffix = '_klcorr.nii',
#                                    newpath=os.getcwd(),use_ext=False))
            
            #writing global stats
            stats_f = fname_presuffix(fname, suffix='_global_corrdist.pklz',
                                    newpath=os.getcwd(),use_ext=False)
            savepkl(stats_f,gstats)
        del dists
        gc.collect()
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['seed_maps'] = os.path.join(os.getcwd(),'seed_maps.nii')
        
        outputs['all_cmaps']=[
                fname_presuffix(fname, suffix='_seeds%d_all_cmaps.npz'%si,
                                newpath=os.getcwd(),use_ext=False) \
                    for fname in self.inputs.in_files \
                    for si in range(len(self.inputs.seed_masks)) ]
                                
        outputs['correlation_mean_maps'] = [fname_presuffix(
                f, use_ext=False, newpath=os.getcwd(),
                suffix = '_meancorr.nii') for f in self.inputs.in_files]
 #       outputs['correlation_median_maps'] = [fname_presuffix(
 #               f, use_ext=False, newpath=os.getcwd(),
 #               suffix = '_mediancorr.nii') for f in self.inputs.in_files]
        outputs['correlation_variance_maps'] = [fname_presuffix(
                f, use_ext=False, newpath=os.getcwd(),
                suffix = '_varcorr.nii') for f in self.inputs.in_files]
        outputs['correlation_std_maps'] = [fname_presuffix(
                f, use_ext=False, newpath=os.getcwd(),
                suffix = '_stdcorr.nii') for f in self.inputs.in_files]
        outputs['correlation_skew_maps'] = [fname_presuffix(
                f, use_ext=False, newpath=os.getcwd(),
                suffix = '_skewcorr.nii') for f in self.inputs.in_files]
        outputs['correlation_kurtosis_maps'] = [fname_presuffix(
                f, use_ext=False, newpath=os.getcwd(),
                suffix = '_kurtosiscorr.nii') for f in self.inputs.in_files]
#        outputs['correlation_kl_maps'] = [fname_presuffix(
#                f, use_ext=False, newpath=os.getcwd(),
#                suffix = '_klcorr.nii') for f in self.inputs.in_files]

        outputs['global_correlation_distribution'] = [fname_presuffix(
                f, use_ext=False, newpath=os.getcwd(),
                suffix ='_global_corrdist.pklz') for f in self.inputs.in_files]
        return outputs



class MotionMapsInputSpec(BaseInterfaceInputSpec):

    mask = File(
        exists=True,
        desc='Mask file to select voxel to regress out.')
    motion = File(
        exists=True,
        desc='Motion parameters files')
    motion_source = traits.Enum(
        'spm','fsl','afni',
        desc = 'software used to estimate motion',
        usedefault = True)

    motion_transform = traits.Enum(
        'raw', 'bw_derivatives', 'fw_derivatives',
        desc = 'Which transform to apply to motion')

    slicing_axis = traits.Int(
        2, usedefault = True, desc = 'Axis for outplane motion measure')


class MotionMapsOutputSpec(TraitedSpec):
    
    drms_mean_map = File(
        exists = True,
        desc='mean delta-root-mean-square of the translation in each voxel')

    drms_max_map = File(
        exists = True,
        desc='max delta-root-mean-square of the translation in each voxel')

    outplane_mean_map = File(
        exists = True,
        desc='mean outplane translation in each voxel')

    outplane_max_map = File(
        exists = True,
        desc='max outplane translation in each voxel')



class MotionMaps(BaseInterface):
    input_spec = MotionMapsInputSpec
    output_spec = MotionMapsOutputSpec
    

    def _run_interface(self,runtime):
        motion = np.loadtxt(self.inputs.motion)
        motion = preproc.motion_parameter_standardize(motion,self.inputs.motion_source)
        masknii = nb.load(self.inputs.mask)
        mask = masknii.get_data()>0
        affine = masknii.get_affine()
        voxels_motion = preproc.compute_voxels_motion(motion,mask,affine)

        outputs = self._list_outputs()
        out = np.zeros(mask.shape, np.float) + np.nan
 
        regsh = voxels_motion.shape
        if self.inputs.motion_transform == 'bw_derivatives':
            voxels_motion = np.concatenate(
                (np.zeros((regsh[0],1,regsh[2])),
                 np.diff(voxels_motion, axis=1)), axis=1)
        elif self.inputs.motion_transform == 'fw_derivatives':
            voxels_motion = np.concatenate(
                (np.diff(voxels_motion, axis=1),
                 np.zeros((regsh[0],1,regsh[2]))), axis=1)

        drms = np.sqrt(np.square(voxels_motion[...,:3]).sum(axis=2))
        out[mask] = drms.mean(1)
        nb.save(nb.Nifti1Image(out,affine), outputs['drms_mean_map'])
        out[mask] = drms.max(1)
        nb.save(nb.Nifti1Image(out,affine), outputs['drms_max_map'])
        
        outplane = np.dot(
            np.linalg.inv(affine),
            voxels_motion.transpose((0,2,1)))[self.inputs.slicing_axis]
        out[mask] = outplane.mean(1)
        nb.save(nb.Nifti1Image(out,affine), outputs['outplane_mean_map'])
        out[mask] = outplane.max(1)
        nb.save(nb.Nifti1Image(out,affine), outputs['outplane_max_map'])
        
        return runtime
            

    def _list_outputs(self):
        outputs = self._outputs().get()
        ol = [('drms_mean_map', '_drms_mean'),
              ('drms_max_map', '_drms_max'),
              ('outplane_mean_map', '_outplane_mean'),
              ('outplane_max_map', '_outplane_max'),]
        for k,p in ol:
            outputs[k] = fname_presuffix(
                self.inputs.motion, use_ext=False,
                newpath=os.getcwd(),suffix = p+'.nii')

        return outputs


class MotionEstimatesInputSpec(BaseInterfaceInputSpec):
    pass
class MotionEstimatesOutputSpec(TraitedSpec):
    pass
class MotionEstimate(BaseInterface):
    
    def _run_interface(self,runtime):
        motion = np.loadtxt(self.inputs.motion)
        motion = preproc.motion_parameter_standardize(motion,self.inputs.motion_source)
        masknii = nb.load(self.inputs.mask)
        mask = masknii.get_data()>0
        affine = masknii.get_affine()
        voxels_motion = preproc.compute_voxels_motion(motion,mask,affine)
        
        return runtime
