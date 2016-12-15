
# classes per constructor that take a folder as source and encapsulate other interfaces to convert multiple sequences and output a list of these grouped by type.
from ..base import TraitedSpec, InputMultiPath, File, \
    Directory, traits, BaseInterface, isdefined, Undefined, \
    BaseInterfaceInputSpec, DynamicTraitedSpec, OutputMultiPath
import dcmstack, dicom
from nipype.interfaces.io import IOBase, add_traits
import glob, os, inspect

default_group_keys = dcmstack.dcmstack.default_group_keys+(
    'SeriesDescription','SequenceName',)

class DefaultRules(object):
    def localizer(self,f): 
        return 'loca' in f.get('SeriesDescription','').lower()
    def t1(self,f):
        return '3dt1' in f.get('SeriesDescription','').lower()
    def t1_nd(self,f):
        return '3dt1' in f.get('SeriesDescription','').lower() and \
                 'ND' in f.get('ImageType') and \
                 'NORM' not in f.get('ImageType')
    def t1_nd_norm(self,f):
        return '3dt1' in f.get('SeriesDescription','').lower() and \
            'ND' in f.get('ImageType') and \
            'NORM' in f.get('ImageType')
    def t1_dis2d_norm(self,f):
        return '3dt1' in f.get('SeriesDescription','').lower() and \
            'DIS2D' in f.get('ImageType') and \
            'NORM' in f.get('ImageType')
    def t2_tse(self,f):
        return 't2tse' in f.get('SeriesDescription','').lower()
    def t2_gre(self,f):
        return 't2gre' in f.get('SeriesDescription','').lower()
    def t2_flair(self,f):
        return 't2flair' in f.get('SeriesDescription','').lower()

    def dti(self,f):
        return 'dti' in f.get('SeriesDescription','').lower() or \
               'diff' in f.get('SeriesDescription','').lower()
    def fieldmap(self,f):
        return 'b0map' in f.get('SeriesDescription','').lower()
    def fmri(self,f): 
        return 'fmri' in f.get('SeriesDescription','').lower()
    def moco(self,f): 
        return 'moco' in f.get('SeriesDescription','').lower()

class DCMStackStudyInputSpec(BaseInterfaceInputSpec):
    
    dicom_folder = Directory(
        exists=True, 
        mandatory=True,
        desc='dicom folder in which to search dicom files')
    extension = traits.Str('*',usedefault=True)
    
    group_keys = traits.Tuple(
        default_group_keys,
        usedefault=True,
        desc='list of dicom fields for grouping and selecting sequences')
    
    no_sampling = traits.Bool(
        False, usedefault=True,
        desc='do not sample files, group all, will be slower but can be useful if not sorted in subfolders')
    
    list_files = traits.Bool(
        False, usedefault=True,
        desc='list files instead of directories, in case files are not sorted in subfolders.')

    subfolders_path = traits.Str(
        os.path.join('*',''),usedefault=True,
        desc='the wilcard to find subfolders to be change in case of deeper nester folders')

    sort_by = traits.Str('StudyTime',usedefault=True,
                         desc='dicom key to use to sort sequences matching')

class DCMStackStudy(IOBase):
    
    input_spec = DCMStackStudyInputSpec
    output_spec = DynamicTraitedSpec

    def __init__(self, sequences=None, **inputs):
        super(DCMStackStudy, self).__init__(**inputs)
        self._sequences = sequences
        if self._sequences is None or not self._sequences:
            self._sequences = DefaultRules()
        self._output_names=[m[0] for m in inspect.getmembers(
                self._sequences,inspect.ismethod)]

    def _add_output_traits(self, base):
        undefined_traits = {}
        for key in self._output_names:
            base.add_trait(key, traits.Any)
            undefined_traits[key] = Undefined
        base.trait_set(trait_change_notify=False, **undefined_traits)
        return base

    def _run_interface(self,runtime):
        subfolders = glob.glob(os.path.join(self.inputs.dicom_folder,
                                            self.inputs.subfolders_path))
        self._sample_files = []
        for d in subfolders:
            ffs = sorted([f for f in glob.glob(os.path.join(d,self.inputs.extension)) if os.path.isfile(f)])
            if len(ffs)>0:
                for f in ffs:
                    try:
                        dcm = dicom.read_file(f,stop_before_pixels=True,defer_size=1024)
                        if dcm.get('ImageType') != None:
                            self._sample_files.append((d,f))
                            break
                        del dcm
                    except (IOError,dicom.filereader.InvalidDicomError):
                        pass
        
        self._groups = dcmstack.parse_and_group(
            [f[1] for f in self._sample_files],
            group_by = self.inputs.group_keys,)

        self._out = {}
        for key,rule in inspect.getmembers(self._sequences,inspect.ismethod):
            matching_groups = self._lookup_sequence(rule)
            if len(matching_groups) > 0:
                folders = [os.path.dirname(g[0][2]) \
                               for g in matching_groups.values()]
                stimes = [g[0][0].get('SeriesTime') \
                              for g in matching_groups.values()]
                idxs = sorted(range(len(stimes)),key=stimes.__getitem__)
                self._out[key] = [folders[i] for i in idxs]
            else:
                self._out[key] = Undefined
        del self._groups
        return runtime

    def _lookup_sequence(self,rule):
        key2idx = dict((k,i) for i,k in enumerate(self.inputs.group_keys))
        matching_groups = dict()
        for g,g2 in self._groups.items():
            dcm = g2[0][0]
            if dcm.get('ImageType') != None : # not an image
                mtch = rule(dcm)
                if mtch:
                    matching_groups[g]=g2
        return matching_groups

    def _list_outputs(self):
        outputs = self._outputs().get()
        for key in self._output_names:
            outputs[key] = self._out[key]
        return outputs
