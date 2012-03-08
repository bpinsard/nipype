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

class SampleSeedsMapInputSpec(BaseInterfaceInputSpec):
    mask = File(exists = True,
                desc = 'Global mask to restrict seeds location.')
    seed_masks = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = 'Mask files used to sample seeds to which whole-brain correlation is computed.')
    sampling_ratio = traits.Float(
      0.1, usedefault=True, 
      desc = 'which ratio of the mask voxels to samples. 1 is all')
    nsamples = traits.Int(
        -1, usedefault = True,
        desc = """The number of voxel to sample per seed_mask.
                  If -1 the sampling ratio will be used""")
    mask_threshold = traits.Float(
        0, usedefault = True,
        desc = 'Threshold for the seed masks')


class SampleSeedsMapOutputSpec(TraitedSpec):
    seed_maps = OutputMultiPath(
        File(exists=True),
        desc = 'seed maps used to sample datas')
    distances = OutputMultiPath(
        traits.List(File(exists=True)),
        desc = 'distance from all seeds to every voxel in the mask')
    
class SampleSeedsMap(BaseInterface):
    input_spec  = SampleSeedsMapInputSpec
    output_spec = SampleSeedsMapOutputSpec
    
    def _run_interface(self,runtime):
        seed_niis = [nb.load(f) for f in self.inputs.seed_masks]
        maskthr = self.inputs.mask_threshold
        masknii = nb.load(self.inputs.mask)
        mask = masknii.get_data()>maskthr

        seed_masks = [(m.get_data()>maskthr)*mask for m in seed_niis]

        # compute coordinates for distance
        voxel_size = np.sqrt(np.square(masknii.get_affine()[:3,:3]).sum(0))
        coords = np.array(np.where(mask)+(np.ones(np.count_nonzero(mask)),))
        coords = masknii.get_affine().dot(coords)[:3].T

        for si,seed_mask in enumerate(seed_masks):
            nvox = np.count_nonzero(seed_mask)
            nsamp = self.inputs.nsamples
            if nsamp == -1:
                nsamp = nvox*self.inputs.sampling_ratio
            
            randind = np.random.permutation(nvox)[:nsamp]
            seeds = [c[randind] for c in np.nonzero(seed_mask)]
            #save seed maps
            seed_map = np.zeros(seed_mask.shape, np.int16) #16bits min for most ni soft
            seed_map[seeds] = 1
            nb.save(nb.Nifti1Image(seed_map,masknii.get_affine()),
                    fname_presuffix(self.inputs.seed_masks[si],
                                    suffix='_seeds',newpath=os.getcwd()))

            dists = np.sqrt(((coords[:,np.newaxis,:].repeat(nsamp,1)-
                              coords[seed_map[mask]>0][np.newaxis])**2).sum(2))
            dists_f = fname_presuffix(
                self.inputs.seed_masks[si], suffix='_distances.npz',
                newpath=os.getcwd(),use_ext=False)
            np.savez_compressed(dists_f, dists=dists)
            
            del dists, seed_map
        del coords, seed_masks, seed_niis, mask, masknii

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['seed_maps'] = [fname_presuffix(f,suffix='_seeds',newpath=os.getcwd()) for f in self.inputs.seed_masks]
        outputs['distances'] = [fname_presuffix(f,suffix='_distances.npz',use_ext=False,newpath=os.getcwd()) for f in self.inputs.seed_masks]
        return outputs

class CorrelationDistributionMapsInput(BaseInterfaceInputSpec):

    in_files = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = 'The session files on which to compute correlations')
    
    mask = File(
        exists = True,
        desc = 'The brain mask used to compute correlation to seed voxels.')

    seed_masks = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc='mask files with seeds to compute correlation with brain voxels')

    distances = InputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc = 'distances files from seed sampling interface')

    nbins = traits.Int(
        1000, usedefault = True,
        desc = 'The number of bins to compute the global distribution.')
    
    min_distance = traits.Float(
        0, usedefault = True,
        desc = """The min distance threshold to compute correlation distribution""")

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
    
    distances = OutputMultiPath(
        traits.List(File(exists=True)),
        desc = 'distance from all seeds to every voxel in the mask')

    correlations = OutputMultiPath(
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

        masknii = nb.load(self.inputs.mask)
        mask = masknii.get_data()>0

        seed_masks = [m.get_data()[mask]>0 for m in seed_niis]
        nseed_masks = len(seed_masks)
        statsshape = mask.shape+(nseed_masks,)
        means = np.zeros(statsshape, np.float32)
#        medians = np.zeros(statsshape, np.float32)
        variances = np.zeros(statsshape, np.float32)
        stds = np.zeros(statsshape, np.float32)
        skews = np.zeros(statsshape, np.float32)
        kurtosiss = np.zeros(statsshape, np.float32)
#        klmaps = np.zeros(statsshape, np.float32)

        # compute all distances from seeds to all voxels
        all_dists = [np.load(f) for f in self.inputs.distances]
        
        gstats =[[]]*len(seed_masks)
        # compute seed-based correlation maps
        for nii in niis:
            data = nii.get_data()[mask].astype(np.float32)
            fname = nii.get_filename()
            for si, seed_mask in enumerate(seed_masks):
                seeds = np.where(seed_mask>0)[0]
                cmaps = corr.seeds_correlation(seeds, data)
                if self.inputs.min_distance > 0:
                    cmaps = np.ma.array(
                        cmaps, 
                        mask = all_dists[si]['dists']<self.inputs.min_distance)
                mean = np.array(cmaps.mean(1))
                var = np.array(cmaps.var(1))
                std = var**0.5
                #create maps
                means[mask,si] = mean
                variances[mask,si] = var
                stds[mask,si] = std
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
                    fname, suffix='_seeds%d_correlations.npz'%si,
                    newpath=os.getcwd(),use_ext=False)
                np.savez_compressed(all_f, corrs=cmaps.__array__())
                del cmaps, ms, mean, var, std, seeds
                gc.collect()
            #writing maps
            nb.save(nb.Nifti1Image(means,masknii.get_affine()),
                    fname_presuffix(fname, suffix = '_meancorr.nii',
                                    newpath=os.getcwd(),use_ext=False))
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
        gc.collect()
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        
        outputs['correlations']=[
                fname_presuffix(fname, suffix='_seeds%d_correlations.npz'%si,
                                newpath=os.getcwd(),use_ext=False) \
                    for fname in self.inputs.in_files \
                    for si in range(len(self.inputs.seed_masks)) ]
                                
        outputs['correlation_mean_maps'] = [fname_presuffix(
                f, use_ext=False, newpath=os.getcwd(),
                suffix = '_meancorr.nii') for f in self.inputs.in_files]
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
    mask = File(exists=True,
                desc = 'brain mask file')
    motion = File(exists=True,
                  desc='motion parameters file')
    motion_source = traits.Enum(
        'spm','fsl','afni',
        desc = 'software used to estimate motion',
        usedefault = True)

    global_head_radius = traits.Float(
        85.0, usedefault = True,
        desc='head radius for global parameters computation.')
    
    slicing_axis = traits.Int(
        2, usedefault = True, desc = 'Axis for outplane motion measure')

class MotionEstimatesOutputSpec(TraitedSpec):
    global_motion_max = traits.Float(
        desc='drms of global motion max over run')
    global_motion_mean = traits.Float(
        desc='drms of global motion mean over run')
    voxel_motion_max = traits.Float(
        desc='drms of voxel motion max over run')
    voxel_motion_mean = traits.Float(
        desc='drms of voxel motion mean over run')
    voxel_motion_maxmean = traits.Float(
        desc='max drms of voxel motions mean over run')
    voxel_motion_meanmax = traits.Float(
        desc='mean drms of voxel motions max over run')

    outplane_motion_max = traits.Float(
        desc='voxel outplane motion max over run')
    outplane_motion_mean = traits.Float(
        desc='voxel outplane motion mean over run')
    outplane_motion_maxmean = traits.Float(
        desc='max voxel outplane motion mean over run')
    outplane_motion_meanmax = traits.Float(
        desc='mean voxel outplane motion max over run')


class MotionEstimate(BaseInterface):
    input_spec = MotionEstimatesInputSpec
    output_spec = MotionEstimatesOutputSpec
    
    def _run_interface(self,runtime):
        motion = np.loadtxt(self.inputs.motion)
        masknii = nb.load(self.inputs.mask)
        mask = masknii.get_data()>0
        affine = masknii.get_affine()
        motion = preproc.motion_parameter_standardize(
            motion,self.inputs.motion_source)

        voxels_motion = preproc.compute_voxels_motion(motion,mask,affine)
        motion=np.diff(motion,axis=0)
        # global motion parameter (Van Dijk like)
        rot_mm = self.inputs.global_head_radius * motion[:,3:]
        global_motion_drms = (rot_mm**2 + motion[:,:3]**2).sum(0)
        self._global_motion_max = global_motion_drms.max()
        self._global_motion_mean = global_motion_drms.mean()
        
        #voxel motion parameters
        voxels_drms = np.sqrt((np.diff(voxels_motion,axis=1)**2).sum(-1))
        self._voxels_motion_max = voxels_drms.max()
        self._voxels_motion_mean = voxels_drms.mean()
        self._voxels_motion_maxmean = voxels_drms.max(0).mean()
        self._voxels_motion_meanmax = voxels_drms.mean(0).max()

        
        outplane = np.abs(np.diff(np.dot(
            np.linalg.inv(affine),
            voxels_motion.transpose((0,2,1)))[self.inputs.slicing_axis],1,1))
        self._outplane_motion_max = outplane.max()
        self._outplane_motion_mean = outplane.mean()
        self._outplane_motion_meanmax = outplane.max(0).mean()
        self._outplane_motion_maxmean = outplane.mean(0).max()


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['global_motion_mean'] = self._global_motion_mean
        outputs['global_motion_max'] = self._global_motion_max

        outputs['voxel_motion_mean'] = self._voxels_motion_mean
        outputs['voxel_motion_max'] = self._voxels_motion_max
        outputs['voxel_motion_meanmax'] = self._voxels_motion_meanmax
        outputs['voxel_motion_maxmean'] = self._voxels_motion_maxmean

        outputs['outplane_motion_mean'] = self._outplane_motion_mean
        outputs['outplane_motion_max'] = self._outplane_motion_max
        outputs['outplane_motion_meanmax'] = self._outplane_motion_meanmax
        outputs['outplane_motion_maxmean'] = self._outplane_motion_maxmean


        return outputs
