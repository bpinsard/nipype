import warnings, re, dicom
from copy import deepcopy
import nibabel as nb

from dcmstack import DicomStack
from dcmstack.dcmmeta import NiftiWrapper
import numpy as np
import struct, itertools

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from nibabel.nicom.dicomwrappers import (wrapper_from_data, 
                                             wrapper_from_file)

import collections

def filenames_to_dicoms(fnames):
    for f in fnames:
        yield dicom.read_file(f)

class DicomStackOnline(DicomStack):

    def _init_dataset(self):
        if hasattr(self,'_is_init') and self._is_init:
            return
        # try to find information about dataset in a single dicom
        self._shape = None        
        self._slice_order = None
        self.frame_idx, self.slice_idx = 0, 0
        
        self._dicom_queue = collections.deque()
        
        dw = wrapper_from_data(self._iter_dicoms(queue_dicoms=True).next())
        self.dw = dw
        nw = NiftiWrapper.from_dicom_wrapper(dw)
        self._affine = nw.nii_img.get_affine()
        self._voxel_size = np.sqrt((self._affine[:3,:3]**2).sum(0))
        self._shape = dw.image_shape
        self._nframes_per_dicom = 1
        self.nframes, self.nslices = 0, 0
        if len(self._shape) < 3:
            self._nslices_per_dicom = 1
            self._nframes_per_dicom = 0
            if not dw.get((0x2001, 0x1018)) is None:
                self.nslices = dw.get((0x2001,0x1018)).value
            elif not dw.get((0x0021,0x104f)) is None:
                self.nslices = dw.get((0x0021,0x104f)).value
            if isinstance(self.nslices, str):
                self.nslices = struct.unpack('i', self.nslices)[0]
                self.nslices = int(self.nslices)
            self._shape += (self.nslices,)
        else:
            self._nslices_per_dicom = self._shape[2]
            self.nslices = self._shape[2]
        if len(self._shape) < 4:
            if not dw.get('NumberOfTemporalPositions') is None:
                self.nframes = int(dw.get('NumberOfTemporalPositions'))
            self._shape += (self.nframes,)
        else:
            self._nframes_per_dicom = self._shape[3]
            self.nframes = self._shape[3]
            
        if dw.is_mosaic:
            self._slice_trigger_times = dw.csa_header['tags'].get(
                'MosaicRefAcqTimes')['items']
            self._slice_order = np.argsort(
                np.array(zip(self._slice_trigger_times,
                             np.arange(self.nslices)),
                         dtype=[('trigger_times','f'),('slice','i')]),
                order=['trigger_times','slice'], axis=0)
        elif dw.is_multiframe:
            self._slice_order = np.arange(self.nslices)
            tr=dw.shared.MRTimingAndRelatedParametersSequence[0].RepetitionTime
            self._slice_trigger_times = np.linspace(
                0, tr*1e-3, self.nslices+1)[:-1]
        else:
            self._slice_locations = []
            self._slice_trigger_times = []
            while not dw.slice_indicator in self._slice_locations:
                self._slice_locations.append(dw.slice_indicator)
                tt = dw.get((0x0018, 0x1060)) #TriggerTime
                if tt is None:
                    tt = dw.get((0x0021, 0x105e)) #RTIA Timer
                if not tt  is None:
                    self._slice_trigger_times.append(float(tt.value))
                df = dicom_source.next()
                dw = wrapper_from_data(df)
            if self.nslices == 0:
                self.nslices = len(self._slice_locations)
            if not len(self._slice_trigger_times):
                self._slice_trigger_times = np.linspace(
                    0, dw.get('RepetitionTime')*1e-3, self.nslices+1)[:-1]
            else:
                self._slice_trigger_times = [self._slice_trigger_times[i] \
                    for i in np.argsort(self._slice_locations)]
            self._slice_order = np.argsort(self._slice_trigger_times)
            self._slice_locations = sorted(self._slice_locations)
        self._is_init = True

        uniq_tt = np.unique(self._slice_trigger_times)
        self._slabs = [(tt,np.where(self._slice_trigger_times==tt)[0].tolist())\
                           for tt in uniq_tt]


    def set_source(self, dicom_source):
        self._dicom_source = iter(dicom_source)
        
    def _iter_dicoms(self, queue_dicoms=False):
        while True:
            if queue_dicoms:
                if not self._dicom_queue:
                    self._dicom_queue.append(self._dicom_source.next())
                while self._dicom_queue:
                    for df in self._dicom_queue:
                        yield df
            else:
                while self._dicom_queue:
                    yield self._dicom_queue.popleft()
                yield self._dicom_source.next()

    def iter_frame(self, data=True, queue_dicoms=False):
        # iterate on each acquired volume
        self._init_dataset()
        frame_data = None
#        for df in dicom_source:

        frame_idx = self.frame_idx
        slice_idx = self.slice_idx

        for df in self._iter_dicoms(queue_dicoms=queue_dicoms):
            dw = wrapper_from_data(df)
            nw = NiftiWrapper.from_dicom_wrapper(dw)
            if self._nframes_per_dicom is 1:
                if data:
                    frame_data = nw.nii_img.get_data()
                frame_idx += 1
            elif self._nframes_per_dicom > 1:
                if data:
                    frames_data = nw.nii_img.get_data()
                for t in xrange(self._shape[-1]):
                    if data:
                        frame_data = frames_data[...,t]
                    frame_idx += 1
                    if not queue_dicoms:
                        self.frame_idx = frame_idx
                        self.slice_idx = slice_idx
                    yield frame_idx-1, nw.nii_img.get_affine(), frame_data
                continue
            else:
                if data:
                    pos = self._slice_locations.index(dw.slice_indicator)
                    if frame_data is None:
                        frame_data = np.empty(self._shape[:3])
                    frame_data[...,pos] = np.squeeze(nw.nii_img.get_data())
                slice_idx += 1
                if self.slice_idx == self.nslices:
                    frame_idx += 1
                    slice_idx = 0
                else:
                    continue
            if not queue_dicoms:
                self.frame_idx = frame_idx
                self.slice_idx = slice_idx
            yield frame_idx-1, nw.nii_img.get_affine(), frame_data
            del dw, nw
    
    def iter_slices(self, data=True, slice_order='acq_time', queue_dicoms=False):
        # iterate on each slice in the temporal order they are acquired
        self._init_dataset()
        slices_buffer = [None]*self.nslices

        slice_seq = self._slice_order
        if slice_order is 'ascending':
            slice_seq = np.arange(0,self.nslices)
        elif slice_order is 'descending':
            slice_seq = np.arange(self.nslices,0,-1)-1

        frame_idx = self.frame_idx
        slice_idx = self.slice_idx

        for df in self._iter_dicoms(queue_dicoms=queue_dicoms):
            dw = wrapper_from_data(df)
            nw = NiftiWrapper.from_dicom_wrapper(dw)
            slice_data = None
            if self._nframes_per_dicom is 1:
                if data:
                    frame_data = nw.nii_img.get_data()
                for sl in slice_seq:
                    if data:
                        slice_data = frame_data[...,sl]
                    yield frame_idx, sl, nw.nii_img.get_affine(), \
                        self._slice_trigger_times[sl], slice_data
                frame_idx += 1
            elif self._nframes_per_dicom > 1:
                if data:
                    frames_data = nw.nii_img.get_data()
                for t in xrange(self._nframes_per_dicom):
                    for sl in slice_seq:
                        if data:
                            slice_data = frames_data[...,sl,t]
                        yield frame_idx, sl, nw.nii_img.get_affine(),\
                            self._slice_trigger_times[sl], slice_data
                    frame_idx += 1
            else:
                # buffer incoming slices to
                pos = self._slice_locations.index(dw.slice_indicator)
                slices_buffer[pos] = dw,nw
                sl = slice_seq[slice_idx]
                while slices_buffer[sl] is not None:
                    dw,nw = slices_buffer[sl]
                    slices_buffer[sl] = None
                    if data:
                        slice_data = nw.nii_img.get_data()[...,0]
                    yield frame_idx, sl, nw.nii_img.get_affine(), \
                        self._slice_trigger_times[sl], slice_data
                    self.slice_idx += 1
                    if self.slice_idx == self.nslices:
                        frame_idx += 1
                        slice_idx = 0
                    sl = slice_seq[self.slice_idx]
            del dw,nw
            if not queue_dicoms:
                self.frame_idx = frame_idx
                self.slice_idx = slice_idx

    def iter_slabs(self, data=True, queue_dicoms=False):
        self._init_dataset()

        if self._slabs is None:
            for fr, sl, aff, tt, data in self.iter_slices(data=data, queue_dicoms=queue_dicoms):
                if not data is None:
                    yield fr, [sl], aff, tt, data[...,np.newaxis]
                else:
                    yield fr, [sl], aff, tt, data
            return

        frame_idx = self.frame_idx
        slice_idx = self.slice_idx

        for df in self._iter_dicoms(queue_dicoms=queue_dicoms):
            dw = wrapper_from_data(df)
            nw = NiftiWrapper.from_dicom_wrapper(dw)
            slice_data = None
            if self._nframes_per_dicom is 1:
                if data:
                    frame_data = nw.nii_img.get_data()
                for sl in self._slabs:
                    if data:
                        slice_data = frame_data[...,sl[1]]
                    yield frame_idx, sl[1], nw.nii_img.get_affine(), \
                        sl[0], slice_data
                frame_idx += 1
            elif self._nframes_per_dicom > 1:
                if data:
                    frames_data = nw.nii_img.get_data()
                for t in xrange(self._nframes_per_dicom):
                    for sl in self._slabs:
                        if data:
                            slice_data = frame_data[...,sl[1],t]
                        yield frame_idx, sl[1], nw.nii_img.get_affine(),\
                            sl[0], slice_data
                    frame_idx += 1
            else:
                raise NotImplementedError(
                    'does not handle slabs stored in separate dicoms')
            del dw,nw
            if not queue_dicoms:
                self.frame_idx = frame_idx
                self.slice_idx = slice_idx
