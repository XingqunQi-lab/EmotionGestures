import logging
import os
import pickle
import random
import subprocess
from collections import defaultdict, namedtuple
from logging.handlers import RotatingFileHandler
from textwrap import wrap

import numpy as np
import re
import time
import math
import soundfile as sf
import librosa.display

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

#import utils.data_utils
from scipy.interpolate import interp1d
from torch.nn.functional import normalize
#import train
#import data_loader.lmdb_data_loader
from model import vocab

def set_logger(log_path=None, log_filename='log'):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        handlers.append(
            RotatingFileHandler(os.path.join(log_path, log_filename), maxBytes=10 * 1024 * 1024, backupCount=5))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s', handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_speaker_model(net):
    try:
        if hasattr(net, 'module'):
            speaker_model = net.module.z_obj
        else:
            speaker_model = net.z_obj
    except AttributeError:
        speaker_model = None

    if not isinstance(speaker_model, vocab.Vocab):
        speaker_model = None

    return speaker_model




def time_since(since):
    now = time.time()
    s = now - since
    return '%s' % as_minutes(s)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


"""
def create_video_and_save(save_path, epoch, prefix, iter_idx, target, output, mean_data, title,
                          audio=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True):
    print('rendering a video...')
    start = time.time()

    fig = plt.figure(figsize=(8, 4))
    axes = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]
    axes[0].view_init(elev=20, azim=-60)
    axes[1].view_init(elev=20, azim=-60)
    fig_title = title

    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')

    # un-normalization and convert to poses
    mean_data = mean_data.flatten()
    output = output + mean_data
    #output_poses = utils.data_utils.convert_dir_vec_to_pose(output)
    target_poses = None
    if target is not None:
        target = target + mean_data
        #target_poses = utils.data_utils.convert_dir_vec_to_pose(target)

    def animate(i):
        for k, name in enumerate(['human', 'generated']):
            if name == 'human' and target is not None and i < len(target):
                pose = target_poses[i]
            elif name == 'generated' and i < len(output):
                pose = output_poses[i]
            else:
                pose = None

            if pose is not None:
                axes[k].clear()
                for j, pair in enumerate(utils.data_utils.dir_vec_pairs):
                    axes[k].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                 [pose[pair[0], 2], pose[pair[1], 2]],
                                 [pose[pair[0], 1], pose[pair[1], 1]],
                                 zdir='z', linewidth=1.5)
                axes[k].set_xlim3d(-0.5, 0.5)
                axes[k].set_ylim3d(0.5, -0.5)
                axes[k].set_zlim3d(0.5, -0.5)
                axes[k].set_xlabel('x')
                axes[k].set_ylabel('z')
                axes[k].set_zlabel('y')
                axes[k].set_title('{} ({}/{})'.format(name, i + 1, len(output)))

    if target is not None:
        num_frames = max(len(target), len(output))
    else:
        num_frames = len(output)
    ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)

    # show audio
    audio_path = None
    if audio is not None:
        assert len(audio.shape) == 1  # 1-channel, raw signal
        audio = audio.astype(np.float32)
        sr = 16000
        audio_path = '{}/{}_{:03d}_{}.wav'.format(save_path, prefix, epoch, iter_idx)
        sf.write(audio_path, audio, sr)

    # save video
    try:
        video_path = '{}/temp_{}_{:03d}_{}.mp4'.format(save_path, prefix, epoch, iter_idx)
        ani.save(video_path, fps=15, dpi=80)  # dpi 150 for a higher resolution
        del ani
        plt.close(fig)
    except RuntimeError:
        assert False, 'RuntimeError'

    # merge audio and video
    if audio is not None:
        merged_video_path = '{}/{}_{:03d}_{}.mp4'.format(save_path, prefix, epoch, iter_idx)
        cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', video_path, '-i', audio_path, '-strict', '-2',
               merged_video_path]
        if clipping_to_shortest_stream:
            cmd.insert(len(cmd) - 1, '-shortest')
        subprocess.call(cmd)
        if delete_audio_file:
            os.remove(audio_path)
        os.remove(video_path)

    print('done, took {:.1f} seconds'.format(time.time() - start))
    return output_poses, target_poses
"""

def normalize_string(s):
    """ lowercase, trim, and remove non-letter characters """
    s = s.lower().strip()
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def remove_tags_marks(text):
    reg_expr = re.compile('<.*?>|[.,:;!?]+')
    clean_text = re.sub(reg_expr, '', text)
    return clean_text


def extract_melspectrogram(y, sr=16000):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, power=2)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
    log_melspec = log_melspec.astype('float16')
    return log_melspec


def calc_spectrogram_length_from_motion_length(n_frames, fps):
    ret = (n_frames / fps * 16000 - 1024) / 512 + 1
    return int(round(ret))


def resample_pose_seq(poses, duration_in_sec, fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
    expected_n = duration_in_sec * fps
    x_new = np.arange(0, n, n / expected_n)
    interpolated_y = f(x_new)
    if hasattr(poses, 'dtype'):
        interpolated_y = interpolated_y.astype(poses.dtype)
    return interpolated_y


def time_stretch_for_words(words, start_time, speech_speed_rate):
    for i in range(len(words)):
        if words[i][1] > start_time:
            words[i][1] = start_time + (words[i][1] - start_time) / speech_speed_rate
        words[i][2] = start_time + (words[i][2] - start_time) / speech_speed_rate

    return words


def make_audio_fixed_length(audio, expected_audio_length):
    n_padding = expected_audio_length - len(audio)
    if n_padding > 0:
        audio = np.pad(audio, (0, n_padding), mode='symmetric')
    else:
        audio = audio[0:expected_audio_length]
    return audio