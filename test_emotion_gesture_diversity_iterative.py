from data_loader.lmdb_loader_BEAT_full import *
from utils.train_utils_BEAT import set_logger
import torch
import argparse
import librosa
import numpy as np
import os
import pprint
import pyarrow
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import time
import torch.nn as nn
from torch.nn import init
#from torch.cuda.amp import autocast
from apex import amp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.vocab_utils import build_vocab
from model.vocab import Vocab
from model.vocab import *
from model import vocab
from utils.train_utils_BEAT import get_speaker_model

from Full_model.Models_memory import Transformer, Motion_Discriminator#, Pose_Discriminator ## memory
from CAVE.BEAT_CVAE import MLP_Reconstruct_v3 as VAE
from model.FGD import MLP_Reconstruct
from model.FHD_score import calculate_frechet_distance, diversity_score
from model.Beat_score_v2 import alignment
from skeleton_classifer.Models import Transformer as skeleton_header
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_acc(input_label, out):
    score, pred = out.topk(1, 1)
    pred0 = pred.squeeze().data
    acc = 100 * torch.true_divide(torch.sum(pred0 == input_label.data), input_label.size(0))
    return acc

def calc_motion(motion):
    motion_offset = motion[:,1:60,:]-motion[:,:59,:]
    
    return motion_offset

def l2_distance_pose(fake, gt):
    overall_l2 = np.mean(
                 np.linalg.norm(gt[:,:,:] - fake[:,:,:], axis=-1))
    return overall_l2

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
              nets (network list)   -- a list of networks
              requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=4):
    #decay = decay_rate ** (epoch // decay_epoch)
    if epoch <= 15: # 4
       base_lr = init_lr
    elif epoch >= 16 and epoch <= 50: # 4; 10
       base_lr = init_lr * 0.2 
    elif epoch >= 51 and epoch <= 80: # 10; 50
       base_lr = init_lr * 0.01
    elif epoch >= 81 and epoch <= 100: # 10; 50
       base_lr = init_lr * 0.005
    elif epoch >= 101 and epoch <= 150: # 10; 50
       base_lr = init_lr * 0.001
    lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class SoftmaxContrastiveLoss(nn.Module):
    def __init__(self):
        super(SoftmaxContrastiveLoss, self).__init__()
        self.cross_ent = nn.CrossEntropyLoss()

    def l2_norm(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        return x_norm

    def l2_sim(self, feature1, feature2):
        Feature = feature1.expand(feature1.size(0), feature1.size(0), feature1.size(1)).transpose(0, 1)
        return torch.norm(Feature - feature2, p=2, dim=2)

    @torch.no_grad()
    def evaluate(self, face_feat, audio_feat, mode='max'):
        assert mode in 'max' or 'confusion', '{} must be in max or confusion'.format(mode)
        face_feat = self.l2_norm(face_feat)
        audio_feat = self.l2_norm(audio_feat)
        cross_dist = 1.0 / (self.l2_sim(face_feat, audio_feat) + 1e-8)

        # print(cross_dist)
        if mode == 'max':
            label = torch.arange(face_feat.size(0)).to(cross_dist.device)
            max_idx = torch.argmax(cross_dist, dim=1)
            # print(max_idx, label)
            acc = torch.sum(label == max_idx).float() / label.size(0)
        else:
            raise ValueError
        # print(acc)
        return acc, cross_dist

    def forward(self, face_feat, audio_feat, contrastive_device, mode='max'):
        assert mode in 'max' or 'confusion', '{} must be in max or confusion'.format(mode)

        face_feat = face_feat.to(contrastive_device)
        audio_feat = audio_feat.to(contrastive_device)
        face_feat = self.l2_norm(face_feat)
        audio_feat = self.l2_norm(audio_feat)
        # print(self.l2_sim(face_feat, audio_feat))
        cross_dist = 1.0 / (self.l2_sim(face_feat, audio_feat) + 1e-8)
        cross_dist = torch.clamp(cross_dist, min=1e-8)

        if mode == 'max':
            label = torch.arange(face_feat.size(0)).to(cross_dist.device)
            loss = F.cross_entropy(cross_dist, label)
        else:
            raise ValueError
        return loss
    


def test_model(args, test_loader, lang_model, device):
    
    
    
    generator = Transformer(args, lang_model, frames = args.n_frames, pose_dim = args.pose_dim, prior_frames =args.n_pre_poses, d_word_vec=512, d_model=512, d_inner=args.latent_dim, n_layers=3, n_head=8, d_k=64, d_v=64)
    #logging.info("Model based on {} have {:.4f}Mb paramerters in total".format('Generator', sum(x.numel()/1e6 for x in generator.parameters())))
    if torch.cuda.device_count() > 1:
       generator = torch.nn.DataParallel(generator)
    generator.to(device)   
    pretrain_model = args.checkpoints_folder_path + args.Cueernt_checkpoint
    loaded_state = torch.load(pretrain_model)
    generator.load_state_dict(loaded_state)
    generator = generator.eval()
    for gm in generator.parameters():
        gm.requires_grad = False
    
    
    FGD = MLP_Reconstruct()
    FGD.load_state_dict({k.replace('module.',''):v for k,v in torch.load("/root/BEAT_Emotion/checkpoints/FGD_v3/checkpoint_iteration8100.pth").items()})
    if torch.cuda.device_count() > 1:
        FGD = torch.nn.DataParallel(FGD)
    FGD.to(device)
    FGD.eval()
    for fm in FGD.parameters():
        fm.requires_grad = False


    skeleton_classifer = skeleton_header(class_dim = args.class_dim, pose_dim = args.pose_dim, d_word_vec=512, d_model=512, d_inner=args.latent_dim, n_layers=3, n_head=8, d_k=64, d_v=64, n_position=60)
    skeleton_classifer.load_state_dict({k.replace('module.',''):v for k,v in torch.load("/root/BEAT_Emotion/checkpoints_v2/skeleton_emotion_classifer_train_test_v2/checkpoint_iteration7500.pth").items()})
    if torch.cuda.device_count() > 1:
        skeleton_classifer = torch.nn.DataParallel(skeleton_classifer)
    skeleton_classifer.to(device)
    skeleton_classifer.eval()
    for sm in skeleton_classifer.parameters():
        sm.requires_grad = False

    Emotion_VAE = VAE()
    Emotion_VAE.load_state_dict({k.replace('module.',''):v for k,v in torch.load("/root/BEAT_Emotion/checkpoints_v2/emotion_CVAE3_v3/checkpoint_iteration35200.pth").items()})
    if torch.cuda.device_count() > 1:
        Emotion_VAE = torch.nn.DataParallel(Emotion_VAE)
    Emotion_VAE.to(device)
    Emotion_VAE.eval()
    for em in Emotion_VAE.parameters():
        em.requires_grad = False


        
    total_error_pose = 0.
    totalSteps = 0
    total_rotation_error = []
    #print('test_loader length is: ', len(test_loader))
    length = len(test_loader) * args.test_batch_size * args.n_frames
    pred_arr = np.empty((length, 512))
    target_arr = np.empty((length, 512))
    alignmenter = alignment(0.3, 2)
    #t_start = int(args.n_pre_poses/ args.motion_resampling_framerate)
    t_start = 0
    t_end = int(args.n_poses/ args.motion_resampling_framerate)
    BL_score = 0.
    all_acc = 0.0
    with torch.no_grad():
        for iter_idx, data in enumerate(test_loader, 0):
            totalSteps += 1
            in_text, text_lengths, in_text_padded, in_audio, in_spec, pose_seq, eid_label, aux_info = data
            pre_pose = pose_seq[:,:args.n_pre_poses, : ].to(device)
            #target_pose = pose_seq[:,args.n_pre_poses:, : ].to(device)
            target_pose = pose_seq.to(device)
            in_spec = in_spec.to(device)
            sample_eid = eid_label.to(device)
            
            eid_label = eid_label.to(device, dtype=torch.long)
            
            sampled_emotion_feature = Emotion_VAE.sample(sample_eid)
            # sampled_emotion_feature = None
            pred_pose, _, _, emotion_prediction, _ = generator(in_spec, in_text_padded, pre_pose, sampled_emotion_feature)#.detach()
            
            


            emotion_prediction = emotion_prediction.detach()
            pred_pose = pred_pose.detach()
            pred_pose_np = pred_pose.data.cpu().numpy().astype(np.float32)
            target_pose_np = target_pose.data.cpu().numpy().astype(np.float32)

            
            ### Emotion_ACC
            fake_emotion_label, fake_emotion_features = skeleton_classifer(pred_pose)        
            emotion_prediction = fake_emotion_label#.detach()                
            eid_label = torch.max(eid_label, 1)[1]
            classifer_acc = compute_acc(eid_label, emotion_prediction)   
            all_acc += classifer_acc.item()       
            ### MPJRE
            rotation_error  = torch.mean(torch.absolute(target_pose.reshape(args.test_batch_size, -1, 6)-pred_pose.reshape(args.test_batch_size, -1, 6)))
            total_rotation_error.append(rotation_error)
            ### FGD
            _, pred_feature = FGD(pred_pose)
            pred_feature = pred_feature.reshape(-1, 512).cpu().detach().numpy()
            _, target_feature = FGD(target_pose)
            target_feature = target_feature.reshape(-1, 512).cpu().detach().numpy()
            idxStart = iter_idx * args.test_batch_size * args.n_frames
            pred_arr[idxStart:(idxStart + args.test_batch_size * args.n_frames)] = pred_feature
            target_arr[idxStart:(idxStart + args.test_batch_size * args.n_frames)] = target_feature
            ### L2 distance
            pred_pose_np = pred_pose.data.cpu().numpy().astype(np.float32)
            target_pose_np = target_pose.data.cpu().numpy().astype(np.float32)
            l2_error_pose = l2_distance_pose(target_pose_np, pred_pose_np)
            total_error_pose = total_error_pose + l2_error_pose
            
            ### Beat alignment
            #in_audio = in_audio.reshape(1, -1)
            #print('in_audio shape is: ', in_audio.shape)
            
            for batch_idx in range(args.test_batch_size):
                audio = in_audio[batch_idx, :]
                #print('audio shape is: ', audio.shape)
                onset_raw, onset_bt, onset_bt_rms = alignmenter.load_audio(audio.cpu().numpy().reshape(-1), t_start, True)
                beat_right_arm, beat_right_shoulder, beat_right_fore_arm, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_fore_arm, beat_left_wrist = alignmenter.load_pose(pred_pose_np[batch_idx, :, :], t_start, t_end, args.motion_resampling_framerate, True)
                BL_score += alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_fore_arm, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_fore_arm, beat_left_wrist, args.motion_resampling_framerate)
            
    avf_BL_score = BL_score / (len(test_loader) * args.test_batch_size)                    
    pred_m = np.mean(pred_arr, axis=0)
    pred_s = np.cov(pred_arr, rowvar=False)
    target_m = np.mean(target_arr, axis=0)
    target_s = np.cov(target_arr, rowvar=False)
    fid_value = calculate_frechet_distance(pred_m, pred_s, target_m, target_s)
    Div_score, Div_interval = diversity_score(pred_arr, device)
    print("Div_score: {:.5f}, Div_interval: ({:.5f}, {:.5f})".format(float(Div_score), float(Div_interval[0]), float(Div_interval[1])))
    total_error_pose = total_error_pose / totalSteps       
    total_rotation_error = sum(total_rotation_error)/len(total_rotation_error)
    avg_acc = float(all_acc)/totalSteps   
    logging.info(">>> Total_pose score: {:.5f},  Total_rotation score: {:.5f}, FGD_score: {:.5f}, Beat score: {:.5f}, Emotion_acc: {:.5f}, Div_score: {:.5f}, Div_interval: ({:.5f}, {:.5f})".format(total_error_pose, total_rotation_error*57.2958, fid_value, avf_BL_score, avg_acc, float(Div_score), float(Div_interval[0]), float(Div_interval[1])))

    
            
            
    
    
    













def main(config):
    args = config

    # random seed
    #if args.random_seed >= 0:
        #utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    set_logger(args.log_save_path, os.path.basename(__file__).replace('.py', '.log'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))
    
    
    # dataset
    collate_fn = default_collate_fn
    dataset_list = [args.train_data_path, args.val_data_path, args.test_data_path]
    train_dataset = SpeechMotionDataset(args.train_data_path,
                                        n_poses=args.n_poses, #60
                                        subdivision_stride=args.subdivision_stride, # 30
                                        pose_resampling_fps=args.motion_resampling_framerate, #15
                                        remove_word_timing=(args.input_context == 'text') #input_context: both
                                        )

    val_dataset = SpeechMotionDataset(args.val_data_path,#args.val_data_path[0]
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      remove_word_timing=(args.input_context == 'text')
                                      )

    test_dataset = SpeechMotionDataset(args.test_data_path,
                                    n_poses=args.n_poses,
                                    subdivision_stride=args.subdivision_stride,
                                    pose_resampling_fps=args.motion_resampling_framerate, speaker_model=train_dataset.speaker_model,
                                    remove_word_timing=(args.input_context == 'text'))
    
    vocab_cache_path = os.path.join('/data/BEAT_Dataset_Full/', 'vocab_all.pkl')

    lang_model = build_vocab('words', [test_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    test_dataset.set_lang_model(lang_model)     

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size,
                              shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )
    
    # train
    # pose_dim = 282  # 47 x 6
    pose_dim = args.pose_dim
    args.checkpoints_folder_path = "/root/BEAT_Emotion/checkpoints_v2/fullmodel_resnet_disentangle_emotion_infonce_TM3_v1/"
    args.Cueernt_checkpoint = ''

    test_model(args, test_loader, device)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ##dataset
    parser.add_argument("--motion_resampling_framerate", type=int, default=15)
    parser.add_argument("--n_poses", type=int, default=60)
    parser.add_argument("--n_frames", type=int, default=60)
    parser.add_argument("--n_pre_poses", type=int, default=10)
    parser.add_argument("--class_dim", type=int, default = 8)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--subdivision_stride", type=int, default=30)
    parser.add_argument("--loader_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default = 128)
    parser.add_argument("--test_batch_size", type=int, default = 1024)
    parser.add_argument("--lr", type=float, default = 0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--pose_dis_warm_epoch", type=int, default=0)
    parser.add_argument("--loss_regression_weight", type=int, default=100)
    parser.add_argument("--dropout_prob", type=float, default=0.1) 
    parser.add_argument("--freeze_wordembed", type=bool, default=False)
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument("--pose_dim", type=int, default = 282)
    parser.add_argument("--log_save_path", type=str, default = '/root/BEAT_Emotion/log_checkpoints_v2/Beat_Score_supplemental/fullmodel_resnet_disentangle_emotion_infonce_SM_TM_v3/')
    #parser.add_argument("--model_save_path", type=str, default = '/root/BEAT_Emotion/checkpoints_v2/fullmodel_resnet_disentangle_wemotion_v5/')
    parser.add_argument("--latent_dim", type=int, default=2048) 
    parser.add_argument("--wordembed_path", type=str, default='/root/HA2G/data/fasttext/crawl-300d-2M-subword.bin') 
    parser.add_argument("--wordembed_dim", type=int, default=300)
    parser.add_argument("--input_context", type=str, default= 'text')
    parser.add_argument("--train_data_path", type=str, default = '/data/BEAT_Dataset_Full/beat_trainable_dataset/train/')
    parser.add_argument("--val_data_path", type=str, default = '/data/BEAT_Dataset_Full/beat_trainable_dataset/val/')
    parser.add_argument("--test_data_path", type=str, default = '/data/BEAT_Dataset_Full/beat_trainable_dataset/test/')
    
    args = parser.parse_args()
    
    
    
       
    main(args)