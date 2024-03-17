import datetime
import logging
import os
import pickle
import random

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
#import utils.train_utils_expressive
#import utils.data_utils_expressive
import utils.train_utils_BEAT
#from model.vocab import Vocab
from model.vocab import Vocab
#from data_loader.data_preprocessor_expressive import DataPreprocessor
import pyarrow
import copy
import pickle5


def word_seq_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    word_seq, text_padded, audio, spectrogram, poses_seq, eid_label, aux_info = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    text_padded = default_collate(text_padded)
    poses_seq = default_collate(poses_seq)
    #vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    eid_label = default_collate(eid_label)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return word_seq, words_lengths, text_padded, audio, spectrogram, poses_seq, eid_label, aux_info


def default_collate_fn(data):
    word_seq, text_padded, audio, spectrogram, poses_seq, eid_label, aux_info = zip(*data)

    text_padded = default_collate(text_padded)
    poses_seq = default_collate(poses_seq)
    #vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    eid_label = default_collate(eid_label)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return torch.tensor([0]), torch.tensor([0]), text_padded, audio, spectrogram, poses_seq, eid_label, aux_info


def audio_classifier_collate_fn(data):
    audio, spectrogram, poses_seq, eid_label, aux_info = zip(*data)

    # text_padded = default_collate(text_padded)
    poses_seq = default_collate(poses_seq)
    #vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    eid_label = default_collate(eid_label)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return audio, spectrogram, poses_seq, eid_label, aux_info


def one_hot_eid(eid):
    index = int(eid.split('_', 4)[-1])
    eid_label = np.zeros(8, dtype=float)
    if index <= 64:
        eid_label[0] = 1
        #x = torch.arange(1)
        #eid_label = F.one_hot(x, num_classes =8)
    elif index > 64 and index <= 72:
        eid_label[1] = 1
        #x = torch.arange(2)
        #eid_label = F.one_hot(x, num_classes =8)
    elif index > 72 and index <= 80:
        eid_label[2] = 1
        #x = torch.arange(3)
        #eid_label = F.one_hot(x, num_classes =8)
    elif index > 80 and index <= 86:
        eid_label[3] = 1
        #x = torch.arange(4)
        #eid_label = F.one_hot(x, num_classes =8)
    elif index > 86 and index <= 94:
        eid_label[4] = 1
        #x = torch.arange(5)
        #eid_label = F.one_hot(x, num_classes =8)
    elif index > 94 and index <= 102:
        eid_label[5] = 1
        #x = torch.arange(6)
        #eid_label = F.one_hot(x, num_classes =8)
    elif index > 102 and index <= 110:
        eid_label[6] = 1
        #x = torch.arange(6)
        #eid_label = F.one_hot(x, num_classes =8)
    elif index > 110 and index <= 118:
        eid_label[7] = 1
        #x = torch.arange(6)
        #eid_label = F.one_hot(x, num_classes =8)
    else:
        assert 'label one_hot error!'
    
    #print(eid_label)
    
    
    return eid_label

class SpeechMotionDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, #mean_pose, mean_dir_vec,
                 speaker_model=None, remove_word_timing=False):


        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses #60
        self.subdivision_stride = subdivision_stride # train 15
        self.skeleton_resampling_fps = pose_resampling_fps # train 15 
        #self.mean_dir_vec = mean_dir_vec # shape: [1,126]
        self.remove_word_timing = remove_word_timing #input_context: both
 
        self.expected_audio_length = int(round(n_poses / pose_resampling_fps * 16000))
        self.expected_spectrogram_length = utils.train_utils_BEAT.calc_spectrogram_length_from_motion_length(
            n_poses, pose_resampling_fps)

        self.lang_model = None
        self.n_samples = 0
        print("Reading data from '{}'...".format(lmdb_dir))
        """
        if len(lmdb_dir) > 1:
            self.train_lmdb_env = lmdb.open(lmdb_dir[0], readonly=True, lock=False)
            with self.train_lmdb_env.begin() as train_txn:
                self.n_samples += train_txn.stat()['entries']
            
            self.val_lmdb_env = lmdb.open(lmdb_dir[1], readonly=True, lock=False)
            with self.val_lmdb_env.begin() as val_txn:
                self.n_samples += val_txn.stat()['entries']
            
            self.test_lmdb_env = lmdb.open(lmdb_dir[2], readonly=True, lock=False)
            with self.test_lmdb_env.begin() as test_txn:
                self.n_samples += test_txn.stat()['entries']    
        """
        preloaded_dir = lmdb_dir + '_cache'
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']      

        if speaker_model is None or speaker_model == 0:
            precomputed_model = lmdb_dir + '_speaker_model.pkl'
            if not os.path.exists(precomputed_model):
                self._make_speaker_model(lmdb_dir, precomputed_model)
            else:
                with open(precomputed_model, 'rb') as f:
                    self.speaker_model = pickle.load(f)
        else:
            self.speaker_model = speaker_model 

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)#.to_buffer()
            # sample = pickle5.loads(sample)
            #word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
            # print('sample is: ', sample)
            word_seq, pose_seq, audio, spectrogram, aux_info = sample

        def extend_word_seq(lang, words, end_time=None):
            n_frames = self.n_poses
            if end_time is None:
                end_time = aux_info['end_time']
            frame_duration = (end_time - aux_info['start_time']) / n_frames

            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            if self.remove_word_timing:
                n_words = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        n_words += 1
                space = int(n_frames / (n_words + 1))
                for i in range(n_words):
                    idx = (i+1) * space
                    extended_word_indices[idx] = lang.get_word_index(words[i][0])
            else:
                prev_idx = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        extended_word_indices[idx] = lang.get_word_index(word[0])
                        # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
                        prev_idx = idx
            return torch.Tensor(extended_word_indices).long()

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        duration = aux_info['end_time'] - aux_info['start_time']
        start_no_frame = aux_info['start_frame_no']
        end_no_frame = aux_info['end_frame_no']
        eid = aux_info['eid']
        #print('start_no_frame is: ', start_no_frame)
        #print('end_no_frame is: ', end_no_frame)
        do_clipping = True

        if do_clipping:
            sample_end_time = aux_info['start_time'] + duration * self.n_poses / pose_seq.shape[0] #vec_seq.shape[0]
            audio = utils.train_utils_BEAT.make_audio_fixed_length(audio, self.expected_audio_length)
            spectrogram = spectrogram[:, 0:self.expected_spectrogram_length]
            #vec_seq = vec_seq[0:self.n_poses]
            #pose_seq = pose_seq[0:self.n_poses]
            sample_end_time = aux_info['start_time'] + duration * self.n_poses / pose_seq.shape[0] #vec_seq.shape[0]
        else:
            sample_end_time = None

        # to tensors
        #word_seq_tensor = words_to_tensor(self.lang_model, word_seq, sample_end_time)
        # extended_word_seq = extend_word_seq(self.lang_model, word_seq, sample_end_time)
        #vec_seq = torch.from_numpy(copy.copy(vec_seq)).reshape((vec_seq.shape[0], -1)).float()
        pose_seq = torch.from_numpy(copy.copy(pose_seq)).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(copy.copy(audio)).float()
        spectrogram = torch.from_numpy(copy.copy(spectrogram)).float()
        
        eid_label = one_hot_eid(eid)
        eid_label = torch.from_numpy(copy.copy(eid_label)).float()
        #eid_label = 
        #print(eid_label)

        # spectrogram = (spectrogram - self.spec_mean.unsqueeze(1)) / self.spec_std.unsqueeze(1)

        #return word_seq_tensor, extended_word_seq, audio, spectrogram, pose_seq, eid_label, aux_info
    
        return audio, spectrogram, pose_seq, eid_label, aux_info

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model
    
    def _make_speaker_model(self, lmdb_dir, cache_path):
        print('  building a speaker model...')
        speaker_model = Vocab('eid', insert_default_tokens=False)

        lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        txn = lmdb_env.begin(write=False)
        cursor = txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            eid = video['eid']
            speaker_model.index_word(eid)

        lmdb_env.close()
        print('    indexed %d videos' % speaker_model.n_words)
        self.speaker_model = speaker_model

        # cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.speaker_model, f)




if __name__ == '__main__':

    train_data_path = "/data/BEAT_Dataset_Full/beat_trainable_dataset/train/"
    n_poses = 60 
    subdivision_stride = 15
    motion_resampling_framerate = 15
    input_context = 'both'
    train_dataset = SpeechMotionDataset(train_data_path,
                                        n_poses=n_poses, #34
                                        subdivision_stride=subdivision_stride, # 10
                                        pose_resampling_fps=motion_resampling_framerate, #15
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=mean_pose,
                                        remove_word_timing=(input_context == 'text') #input_context: both
                                        )