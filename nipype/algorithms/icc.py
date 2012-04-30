from numpy import ones, kron, mean, eye, hstack, dot, tile
from scipy.linalg import pinv
from ..interfaces.base import BaseInterfaceInputSpec, TraitedSpec, \
    BaseInterface, traits, File, Undefined, isdefined
import nibabel as nb
import numpy as np
import os
from nipype.utils.filemanip import fname_presuffix, split_filename, loadpkl


class VolumeICCInputSpec(BaseInterfaceInputSpec):
    subjects_sessions = traits.List(traits.List(File(exists=True)),
                           desc="n subjects m sessions 3D stat files",
                           mandatory=True)
    mask = File(exists=True, mandatory=True)


class VolumeICCOutputSpec(TraitedSpec):
    icc_map = File(exists=True)
    sessions_F_map = File(exists=True, desc="F statistics for the effect of session")
    sessions_df_1 = traits.Int()
    sessions_df_2 = traits.Int()


class VolumeICC(BaseInterface):
    '''
    Calculates Interclass Correlation Coefficient (3,1) as defined in
    P. E. Shrout & Joseph L. Fleiss (1979). "Intraclass Correlations: Uses in
    Assessing Rater Reliability". Psychological Bulletin 86 (2): 420-428. This
    particular implementation is aimed at relaibility (test-retest) studies.
    '''
    input_spec = VolumeICCInputSpec
    output_spec = VolumeICCOutputSpec

    def _run_interface(self, runtime):
        maskdata = nb.load(self.inputs.mask).get_data()
        maskdata = np.logical_not(np.logical_or(maskdata == 0, np.isnan(maskdata)))

        session_datas = [[nb.load(fname).get_data()[maskdata].reshape(-1, 1) for fname in sessions] for sessions in self.inputs.subjects_sessions]
        list_of_sessions = [np.hstack(session_data) for session_data in session_datas]
        all_data = np.dstack(list_of_sessions)
        icc = np.zeros(session_datas[0][0].shape)
        session_F = np.zeros(session_datas[0][0].shape)

        for x in range(icc.shape[0]):
            Y = all_data[x, :, :]
            icc[x], session_F[x], self._df1, self._df2 = ICC_rep_anova(Y)

        nim = nb.load(self.inputs.subjects_sessions[0][0])
        new_data = np.zeros(nim.get_shape())
        new_data[maskdata] = icc.reshape(-1,)
        new_img = nb.Nifti1Image(new_data, nim.get_affine(), nim.get_header())
        nb.save(new_img, 'icc_map.nii')

        new_data = np.zeros(nim.get_shape())
        new_data[maskdata] = session_F.reshape(-1,)
        new_img = nb.Nifti1Image(new_data, nim.get_affine(), nim.get_header())
        nb.save(new_img, 'sessions_F_map.nii')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['icc_map'] = os.path.abspath('icc_map.nii')
        outputs['sessions_F_map'] = os.path.abspath('sessions_F_map.nii')
        outputs['sessions_df_1'] = self._df1
        outputs['sessions_df_2'] = self._df2
        return outputs


def ICC_rep_anova(Y):
    '''
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns

    --------------------------------------------------------------------------
                       One Sample Repeated measure ANOVA
                       Y = XB + E with X = [FaTor / Subjects]
    --------------------------------------------------------------------------
    '''

    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = kron(eye(nb_conditions), ones((nb_subjects, 1)))  # sessions
    x0 = tile(eye(nb_subjects), (nb_conditions, 1))  # subjects
    X = hstack([x, x0])

    # Sum Square Error
    predicted_Y = dot(dot(dot(X, pinv(dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between colums/sessions
    SSC = ((mean(Y, 0) - mean_Y) ** 2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) / (mean square subjeT + (k-1)*-mean square error)
    ICC = (MSR - MSE) / (MSR + dfc * MSE)

    return ICC, session_effect_F, dfc, dfe


class ICCInputSpec(BaseInterfaceInputSpec):
    in_files = traits.List(traits.List(File(exists = True)),
        desc = 'List of subject\'s list of session data files to be analyzed.Can be npy, npz or pklz files for now.')
    
    _xor_inputs = ('data_key','data_transform')

    data_key = traits.String(
        desc = 'Data key or filename to select in loaded dict/data/file.')

    data_transform = traits.Function(
        desc = 'Function to apply to the loaded object before ICC analysis.')

class ICCOutputSpec(TraitedSpec):
    icc_file = File(
        exists = True,
        desc = 'File containing ICC values corresponding to values of input files.')
    fstat_file = File(
        exists = True,
        desc = 'f statistics')
    sessions_df_1 = traits.Int()
    sessions_df_2 = traits.Int()

class ICC(BaseInterface):
    '''
    Calculates Interclass Correlation Coefficient (3,1) as defined in
    P. E. Shrout & Joseph L. Fleiss (1979). "Intraclass Correlations: Uses in
    Assessing Rater Reliability". Psychological Bulletin 86 (2): 420-428. This
    particular implementation is aimed at relaibility (test-retest) studies.
    This is a tentative generic ICC interfaces to compute ICC over other values than statistic map.
    '''    
    input_spec  = ICCInputSpec
    output_spec = ICCOutputSpec
    
    def _run_interface(self, runtime):
        files = [[self._loadfile(f) for f in sessions] for sessions in self.inputs.in_files]
        if self.inputs.data_key is not Undefined:
            data = [[s[self.inputs.data_key] for s in sessions] for sessions in files]
        elif self.inputs.data_transform is not Undefined:
            data = [[self.inputs.data_transform(s) for s in sessions] for sessions in files]
        else:
            data = files
        data = np.concatenate([np.concatenate([s[...,np.newaxis,np.newaxis] for s in sessions],-1) for sessions in data],-2)
        icc = np.zeros(data.shape[:-2])
        fstat = np.zeros(data.shape[:-2])
        for ind in np.ndindex(data.shape[:-2]):
            icc[ind], fstat[ind], self._df1, self._df2 = ICC_rep_anova(data[ind])
        outs = self._list_outputs()
        np.save(outs['icc_file'],icc)
        np.save(outs['fstat_file'],fstat)
        print self._df1, self._df2
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
        outputs['icc_file'] = 'icc.npy'
        outputs['fstat_file'] = 'fstat.npy'
        outputs['sessions_df_1'] = self._df1
        outputs['sessions_df_2'] = self._df2
        return outputs
