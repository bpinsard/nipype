from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
import os,sys
from string import Template


from nipype.utils.filemanip import fname_presuffix, filename_to_list

code_path, filename = os.path.split(os.path.abspath(__file__))
nbw_path='/coconut/applis/src/nbw/st_pub/'
MatlabCommand.set_default_paths([
    nbw_path+'data_analysis/',
    nbw_path+'gui/',
    nbw_path+'sica/',
    nbw_path+'visu/',
    code_path+'/overwrite/',
    code_path])

class SICABase(BaseInterface):
    def _parse_inputs(self, skip=(), only=()):
        spm_dict = ""
        metadata=dict(field=lambda t : t is not None)
        for name, spec in self.inputs.traits(**metadata).items():
            if skip and name in skip:
                continue
            if only and name not in only:
                continue
            value = getattr(self.inputs, name)
            if not isdefined(value):
                continue
            field = spec.field
            if isinstance(value,str):
                spm_dict += "'{0}','{1}',".format(field,value)
            elif isinstance(value,bool):
                spm_dict += "'{0}',{1},".format(field,int(value))
            else:
                spm_dict += "'{0}',{1},".format(field,value)
        return spm_dict[:-1]


class SICAInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    mask = File(exists=True, field='mask')
    sica_file = File('sica.mat', usedefault=True)
    sica_comp_filename=File('sica_comp.nii',desc='ICA components image filename', usedefault=True)
    TR = traits.Float(3.,desc='the repetition time (TR) of the acquisition, in seconds',field='TR')
    filter_high = traits.Float(0,desc='(optional, default 0) cut-off frequency of a high-pass fIltering. A 0 Value Will Result In No Filtering.',field='high', usedefault=True)
    filter_low = traits.Float(0,desc='(optional, default 0) cut-off frequency of a low-pass filtering. A 0 value will result in no filtering.',field='low', usedefault=True)

    slice_correction = traits.Bool(False,desc='a correction for slice mean intensity. Use this correction if some mean slice intensities are not stable across time ("spiked" artefacts involving 1 slice)',field='slice_correction', usedefault=True)
    suppress_volumes=traits.Int(0,desc='number of volumes to suppress at the begining of a run.',field='suppress_vol', usedefault=True)
    detrend = traits.Int(2,desc='(optional, default 2) order of the polynomial for polynomial drifts correction',field='detrend', usedefault=True);
    norm = traits.Int(desc='2: corrects for differences in mean between runs, 0-1: corrects for differences in variances',field='norm', usedefault=True);
    algo = traits.Enum('Infomax','Fastica-Def','Fastica-Sym', desc='(optional, default \'Infomax\') the type of algorithm to be used for the sica decomposition: \'Infomax\', \'Fastica-Def\' or \'Fastica-Sym\'.',field='algo', usedefault=True)
    type_nb_comp = traits.Bool(False,desc='False, to choose directly the number of component to compute. True, to choose the ratio of the variance to keep',field='type_nb_comp', usedefault=True)
    param_nb_comp = traits.Int(40,desc='if type_nb_comp = 0, number of components to compute if type_nb_comp = 1, ratio of the variance to keep (default, 90 %)',field='param_nb_comp', usedefault=True)
    

class SICAOutputSpec(TraitedSpec):
    sica_file = File(exists=True)
    components = OutputMultiPath(traits.Either(traits.List(File(exists=True)),File(exists=True)), desc='component files')
    

class SICA(SICABase):
    input_spec=SICAInputSpec
    output_spec=SICAOutputSpec

    def _run_interface(self, runtime):
        opts=self._parse_inputs(skip=['in_file','sica_file','sica_comp_files_fmt','filter_low','filter_high'])
        filters=self._parse_inputs(only=['filter_high','filter_low'])
        comp_filename=os.path.join(os.getcwd(),self.inputs.sica_comp_filename)
        d=dict(in_file=self.inputs.in_file,
               sica_file=self.inputs.sica_file,
               opts=opts,
               filters=filters,
               comp_filename=comp_filename)
        script = Template("""
        opts=struct($opts);
        opts.filter=struct($filters);
        in_file{1} = '$in_file';
        sica = st_script_sica(in_file, opts);
        save('$sica_file','sica');
        write_sica_comps(sica,'$comp_filename');
        exit;
        """).substitute(d)

        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['sica_file'] = os.path.abspath(self.inputs.sica_file)
        outputs['components'] = os.path.join(os.getcwd(),'sica_comp.nii')
        return outputs

class CORSICAInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    sica_file = File(exists=True)
    noise_rois = InputMultiPath(File(exists=True))
    n_cluster = traits.Int(-1,desc='default floor(nbvox/10), where nbvox is the number of voxels in the region)', field='nb_clust')
    n_kmeans = traits.Int(3,desc='(default 3) the number of repetition for kmeans clustering.', field='nb_kmeans')
    score_type = traits.Enum('freq','inertia', desc='(default \'freq\') type of computed score. \'freq\' for the frequency of selection of the regressor and \'inertia\' for the relative part of inertia explained by the clusters "selecting" the regressor', field='type_score')
    score_thresh = traits.Float(0.25,desc='(default -1) the threshold of the scores to select the components. value between 0 and 1. =-1 for automatic threshold by Otsu algorithm.', field='scoreThres')

    noise_components_mat=File('_noise_components.mat', usedefault=True)

    baseline = traits.Int(0,desc='value of artificial baseline (default 1000)',field='baseline', usedefault=True)
    add_residuals  = traits.Bool(True, desc='whether to add or not the residuals to the recontructed data', field='addres', usedefault = True)


class CORSICAOutputSpec(TraitedSpec):
    corrected_file = File(exists=True, desc='corrected file')
    noise_components_mat = File(exists=True)
    
class CORSICA(SICABase):
    input_spec=CORSICAInputSpec
    output_spec=CORSICAOutputSpec

    def _run_interface(self, runtime):
        opts=self._parse_inputs(skip=['in_file','sica_file','mask_file'])
        outputs = self._list_outputs()
        noise_rois=filename_to_list(self.inputs.noise_rois)
        noise_rois=('\',\'').join(noise_rois)
        d=dict(sica_file=self.inputs.sica_file,
               mask_file=noise_rois,
               corrected_file=outputs['corrected_file'],
               noise_components_mat=outputs['noise_components_mat'],
               opts=opts)
        script = Template("""
        opts=struct($opts);
        load('$sica_file');
        mask = {'$mask_file'};
        compSel_corsica = st_script_spatial_sel_comp(sica, mask,opts);
        ncomps=compSel_corsica.numcomp;
        disp(ncomps);
        save('$noise_components_mat', 'ncomps');
        data=st_suppress_comp(sica,compSel_corsica);
        
        ni = nifti;
        ni.dat = file_array('$corrected_file', ...
          [sica.header.dim size(sica.A,1)], ...
          [16 0], ...
          0,1,0);
        ni.mat = sica.header.mat;
        ni.mat0= sica.header.mat;
        ni.descrip = 'Corsica corrected data';
        create(ni);
        for t=1:size(data,4); ni.dat(:,:,:,t) = data(:,:,:,t); end
        exit;
        """).substitute(d)

        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime


    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['noise_components_mat'] = fname_presuffix(self.inputs.in_file, suffix=self.inputs.noise_components_mat, newpath=os.getcwd(), use_ext=False)
        outputs['corrected_file'] = fname_presuffix(self.inputs.in_file, prefix='c', newpath=os.getcwd())
        return outputs
    
    
    
