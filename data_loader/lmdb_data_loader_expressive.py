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

#import utils.train_utils_expressive
import utils.data_utils_expressive
from model.vocab import Vocab
from data_loader.data_preprocessor_expressive import DataPreprocessor
import pyarrow
import copy


def word_seq_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    word_seq, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    text_padded = default_collate(text_padded)
    poses_seq = default_collate(poses_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info


def default_collate_fn(data):
    _, text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    text_padded = default_collate(text_padded)
    pose_seq = default_collate(pose_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return torch.tensor([0]), torch.tensor([0]), text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info


class SpeechMotionDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, #mean_pose, mean_dir_vec,
                 speaker_model=None, remove_word_timing=False):

        # self.spec_mean = torch.load('/mnt/lustressd/liuxian.vendor/HA2G/spec_mean.pth')
        # self.spec_std = torch.load('/mnt/lustressd/liuxian.vendor/HA2G/spec_std.pth')
        #mean_dir_vec: [-0.0737964, -0.9968923, -0.1082858,  0.9111595,  0.2399522, -0.102547 , -0.8936886,  0.3131501, -0.1039348,  0.2093927, 0.958293 , 
        # 0.0824881, -0.1689021, -0.0353824, -0.7588258, -0.2794763, -0.2495191, -0.614666 , -0.3877234,  0.005006 , -0.5301695, -0.5098616,  0.2257808,  
        # 0.0053111, -0.2393621, -0.1022204, -0.6583039, -0.4992898,  0.1228059, -0.3292085, -0.4753748,  0.2132857,  0.1742853, -0.2062069,  0.2305175, 
        # -0.5897119, -0.5452555,  0.1303197, -0.2181693, -0.5221036, 0.1211322,  0.1337591, -0.2164441,  0.0743345, -0.6464546, -0.5284583,  0.0457585, 
        # -0.319634 , -0.5074904,  0.1537192, 0.1365934, -0.4354402, -0.3836682, -0.3850554, -0.4927187, -0.2417618, -0.3054556, -0.3556116, -0.281753 , 
        # -0.5164358, -0.3064435,  0.9284261, -0.067134 ,  0.2764367,  0.006997 , -0.7365526,  0.2421269, -0.225798 , -0.6387642,  0.3788997, 0.0283412, 
        # -0.5451686,  0.5753376,  0.1935219,  0.0632555, 0.2122412, -0.0624179, -0.6755542,  0.5212831,  0.1043523, -0.345288 ,  0.5443628,  0.128029 , 
        # 0.2073687,  0.2197118, 0.2821399, -0.580695 ,  0.573988 ,  0.0786667, -0.2133071, 0.5532452, -0.0006157,  0.1598754,  0.2093099,  0.124119, 
        # -0.6504359,  0.5465003,  0.0114155, -0.3203954,  0.5512083, 0.0489287,  0.1676814,  0.4190787, -0.4018607, -0.3912126, 0.4841548, -0.2668508, 
        # -0.3557675,  0.3416916, -0.2419564, -0.5509825,  0.0485515, -0.6343101, -0.6817347, -0.4705639, -0.6380668,  0.4641643,  0.4540192, -0.6486361, 
        # 0.4604001, -0.3256226,  0.1883097,  0.8057457,  0.3257385,  0.1292366, 0.815372] # [1,126]
        
        #mean_pose: [-0.0046788, -0.5397806,  0.007695 , -0.0171913, -0.7060388,-0.0107034,  0.1550734, -0.6823077, -0.0303645, -0.1514748, -0.6819547, 
        # -0.0268262,  0.2094328, -0.469447 , -0.0096073,   -0.2318253, -0.4680838, -0.0444074,  0.1667382, -0.4643363,   -0.1895118, -0.1648597, 
        # -0.4552845, -0.2159728,  0.1387546,   -0.4859474, -0.2506667,  0.1263615, -0.4856088, -0.2675801,   0.1149031, -0.4804542, -0.267329 , 
        # 0.1414847, -0.4727709,   -0.2583424,  0.1262482, -0.4686185, -0.2682536,  0.1150217,   -0.4633611, -0.2640182,  0.1475897, -0.4415648, 
        # -0.2438853,   0.1367996, -0.4383164, -0.248248 ,  0.1267222, -0.435534 ,   -0.2455436,  0.1455485, -0.4557491, -0.2521977,  0.1305471,  
        # -0.4535603, -0.2611591,  0.1184687, -0.4495366, -0.257798 ,   0.1451682, -0.4802511, -0.2081622,  0.1301337, -0.4865308,   -0.2175783, 
        # 0.1208341, -0.4932623, -0.2311025, -0.1409241,-0.4742868, -0.2795303, -0.1287992, -0.4724431, -0.2963172,-0.1159225, -0.4676439, -0.2948754, 
        # -0.1427748, -0.4589126,-0.2861245, -0.126862 , -0.4547355, -0.2962466, -0.1140265,-0.451308 , -0.2913815, -0.1447202, -0.4260471, -0.2697673,
        # -0.1333492, -0.4239912, -0.2738043, -0.1226859, -0.4238346,-0.2706725, -0.1446909, -0.440342 , -0.2789209, -0.1291436,-0.4391063, -0.2876539, 
        # -0.1160435, -0.4376317, -0.2836147,-0.1441438, -0.4729031, -0.2355619, -0.1293268, -0.4793807,-0.2468831, -0.1204146, -0.4847246, -0.2613876, 
        # -0.0056085,-0.9224338, -0.1677302, -0.0352157, -0.963936 , -0.1388849,0.0236298, -0.9650772, -0.1385154, -0.0697098, -0.9514691,-0.055632 , 
        # 0.0568838, -0.9565502, -0.0567985] #[1, 124]
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses #60
        self.subdivision_stride = subdivision_stride # train 15
        self.skeleton_resampling_fps = pose_resampling_fps # train 15 
        #self.mean_dir_vec = mean_dir_vec # shape: [1,126]
        self.remove_word_timing = remove_word_timing #input_context: both
 
        self.expected_audio_length = int(round(n_poses / pose_resampling_fps * 16000))
        self.expected_spectrogram_length = utils.data_utils_expressive.calc_spectrogram_length_from_motion_length(
            n_poses, pose_resampling_fps)

        self.lang_model = None

        print("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            print('Creating the dataset cache...')
            #assert mean_dir_vec is not None
            #if mean_dir_vec.shape[-1] != 3:
                #mean_dir_vec = mean_dir_vec.reshape(mean_dir_vec.shape[:-1] + (-1, 3)) # [42,3]
            #n_poses_extended = int(round(n_poses * 1.25))  # some margin 34 * 1.25 = 42.5->43
            
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps)
            data_sampler.run()
        else:
            print('Found the cache {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

        # make a speaker model
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

            sample = pyarrow.deserialize(sample)
            #word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
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
        #print('start_no_frame is: ', start_no_frame)
        #print('end_no_frame is: ', end_no_frame)
        do_clipping = True

        if do_clipping:
            #sample_end_time = aux_info['start_time'] + duration * self.n_poses / pose_seq.shape[0] #vec_seq.shape[0]
            audio = utils.data_utils_expressive.make_audio_fixed_length(audio, self.expected_audio_length)
            spectrogram = spectrogram[:, 0:self.expected_spectrogram_length]
            #vec_seq = vec_seq[0:self.n_poses]
            pose_seq = pose_seq[0:self.n_poses]
            sample_end_time = aux_info['start_time'] + duration * self.n_poses / pose_seq.shape[0] #vec_seq.shape[0]
        else:
            sample_end_time = None

        # to tensors
        word_seq_tensor = words_to_tensor(self.lang_model, word_seq, sample_end_time)
        extended_word_seq = extend_word_seq(self.lang_model, word_seq, sample_end_time)
        #vec_seq = torch.from_numpy(copy.copy(vec_seq)).reshape((vec_seq.shape[0], -1)).float()
        pose_seq = torch.from_numpy(copy.copy(pose_seq)).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(copy.copy(audio)).float()
        spectrogram = torch.from_numpy(copy.copy(spectrogram)).float()

        # spectrogram = (spectrogram - self.spec_mean.unsqueeze(1)) / self.spec_std.unsqueeze(1)

        return word_seq_tensor, extended_word_seq, pose_seq, audio, spectrogram, aux_info

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
    mean_dir_vec = [-0.0737964, -0.9968923, -0.1082858,  0.9111595,  0.2399522, -0.102547 , -0.8936886,  0.3131501, -0.1039348,  0.2093927, 0.958293 ,  
                    0.0824881, -0.1689021, -0.0353824, -0.7588258, -0.2794763, -0.2495191, -0.614666 , -0.3877234,  0.005006 , -0.5301695, -0.5098616, 
                    0.2257808,  0.0053111, -0.2393621, -0.1022204, -0.6583039, -0.4992898,  0.1228059, -0.3292085, -0.4753748,  0.2132857,  0.1742853, 
                    -0.2062069,  0.2305175, -0.5897119, -0.5452555,  0.1303197, -0.2181693, -0.5221036, 0.1211322,  0.1337591, -0.2164441,  0.0743345, 
                    -0.6464546, -0.5284583,  0.0457585, -0.319634 , -0.5074904,  0.1537192, 0.1365934, -0.4354402, -0.3836682, -0.3850554, -0.4927187, 
                    -0.2417618, -0.3054556, -0.3556116, -0.281753 , -0.5164358, -0.3064435,  0.9284261, -0.067134 ,  0.2764367,  0.006997 , -0.7365526, 
                    0.2421269, -0.225798 , -0.6387642,  0.3788997, 0.0283412, -0.5451686,  0.5753376,  0.1935219,  0.0632555, 0.2122412, -0.0624179, -0.6755542,  
                    0.5212831,  0.1043523, -0.345288 ,  0.5443628,  0.128029 ,  0.2073687,  0.2197118, 0.2821399, -0.580695 ,  0.573988 ,  0.0786667, -0.2133071, 
                    0.5532452, -0.0006157,  0.1598754,  0.2093099,  0.124119, -0.6504359,  0.5465003,  0.0114155, -0.3203954,  0.5512083, 0.0489287,  0.1676814,  
                    0.4190787, -0.4018607, -0.3912126, 0.4841548, -0.2668508, -0.3557675,  0.3416916, -0.2419564, -0.5509825,  0.0485515, -0.6343101, -0.6817347, 
                    -0.4705639, -0.6380668,  0.4641643,  0.4540192, -0.6486361,  0.4604001, -0.3256226,  0.1883097,  0.8057457,  0.3257385,  0.1292366, 0.815372]
    mean_pose = [-0.0046788, -0.5397806,  0.007695 , -0.0171913, -0.7060388,-0.0107034,  0.1550734, -0.6823077, -0.0303645, -0.1514748,   -0.6819547, 
                -0.0268262,  0.2094328, -0.469447 , -0.0096073,   -0.2318253, -0.4680838, -0.0444074,  0.1667382, -0.4643363,   -0.1895118, -0.1648597, 
                -0.4552845, -0.2159728,  0.1387546,   -0.4859474, -0.2506667,  0.1263615, -0.4856088, -0.2675801,   0.1149031, -0.4804542, -0.267329 , 
                0.1414847, -0.4727709,   -0.2583424,  0.1262482, -0.4686185, -0.2682536,  0.1150217,   -0.4633611, -0.2640182,  0.1475897, -0.4415648, 
                -0.2438853,   0.1367996, -0.4383164, -0.248248 ,  0.1267222, -0.435534 ,   -0.2455436,  0.1455485, -0.4557491, -0.2521977,  0.1305471,  
                -0.4535603, -0.2611591,  0.1184687, -0.4495366, -0.257798 ,   0.1451682, -0.4802511, -0.2081622,  0.1301337, -0.4865308,   -0.2175783,  
                0.1208341, -0.4932623, -0.2311025, -0.1409241,-0.4742868, -0.2795303, -0.1287992, -0.4724431, -0.2963172,-0.1159225, -0.4676439, -0.2948754, 
                -0.1427748, -0.4589126,-0.2861245, -0.126862 , -0.4547355, -0.2962466, -0.1140265,-0.451308 , -0.2913815, -0.1447202, -0.4260471, -0.2697673,
                -0.1333492, -0.4239912, -0.2738043, -0.1226859, -0.4238346,-0.2706725, -0.1446909, -0.440342 , -0.2789209, -0.1291436,-0.4391063, -0.2876539, 
                -0.1160435, -0.4376317, -0.2836147,-0.1441438, -0.4729031, -0.2355619, -0.1293268, -0.4793807,-0.2468831, -0.1204146, -0.4847246, -0.2613876, 
                -0.0056085,-0.9224338, -0.1677302, -0.0352157, -0.963936 , -0.1388849,0.0236298, -0.9650772, -0.1385154, -0.0697098, -0.9514691,-0.055632 , 0.0568838, -0.9565502, -0.0567985]
    input_context = 'both'
    train_dataset = SpeechMotionDataset(train_data_path,
                                        n_poses=n_poses, #34
                                        subdivision_stride=subdivision_stride, # 10
                                        pose_resampling_fps=motion_resampling_framerate, #15
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=mean_pose,
                                        remove_word_timing=(input_context == 'text') #input_context: both
                                        )