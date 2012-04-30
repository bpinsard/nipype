import numpy as np
from ..interfaces.base import BaseInterfaceInputSpec, TraitedSpec, \
    BaseInterface, traits, File
from scipy.stats import rankdata


class KendallInputSpec(BaseInterfaceInputSpec):
    in_files = traits.List(traits.List(File(exists = True)),
        desc = 'List of subject\'s list of session data files to be analyzed.Can be npy, npz or pklz files for now.')
    
    _xor_inputs = ('data_key','data_transform')

    data_key = traits.String(
        desc = 'Data key or filename to select in loaded dict/data/file.')

    data_transform = traits.Function(
        desc = 'Function to apply to the loaded object before ICC analysis.')

class KendallOutputSpec(TraitedSpec):
    kcc_file = File(
        exists = True,
        desc = 'File containing ICC values corresponding to values of input files.')

class Kendall(BaseInterface):
    """ Interface to compute Kendall'W coefficient of concordance (KCC).
    Can be used for test-retest studies to assess agreement among differente measures. Comparing different KCC values requires the same number of repeated measures.
    """
    input_spec  = KendallInputSpec
    output_spec = KendallOutputSpec
    
    def _run_interface(self, runtime):
        files = [[self._loadfile(f) for f in sessions] for sessions in self.inputs.in_files]
        if self.inputs.data_key is not Undefined:
            data = [[s[self.inputs.data_key] for s in sessions] for sessions in files]
        elif self.inputs.data_transform is not Undefined:
            data = [[self.inputs.data_transform(s) for s in sessions] for sessions in files]
        else:
            data = files

        k = kcc(data)

        outs = self._list_outputs()
        np.save(outs['kcc'],kcc)
        return runtime

    def _loadfile(self,f):
        p,b,e = split_filename(f)
        if e == '.csv':
            ff = np.loadtxt(f)
        elif e == '.npy' or e == '.npz':
            ff = np.load(f)
        elif e == '.pkl' or e == '.pklz':
            ff = loadpkl(f)
        return ff

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['kcc_file'] = 'kcc.npy'
        return outputs


def kcc(data, axis=-1, repeat_axis=0):
    """Compute Kendall coefficient of concordance."""
    #move values axis to last
    if axis != -1:
        data = np.rollaxis(data,axis,data.ndim)
    #move repeated measures axis to first
    if repeat_axis != 0:
        data = np.rollaxis(data,repeat_axis, 1)
    y = np.empty(data.shape)
    for i in np.ndindex(y.shape[:-1]):
        y[i] = rankdata(data[i])

    n,m = data.shape[0],data.shape[-1]
    y = (y-y.mean(0)[np.newaxis])**2
    k = (m*(m**2-1)*n**2)/(12*(n-1))
    return 1-y.sum(-1).sum(0)/k
