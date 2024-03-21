#from data_loader.lmdb_data_loader_expressive import *
from data_loader.lmdb_loader_BEAT_full import *
from utils.vocab_utils import build_vocab
from utils.train_utils_BEAT import set_logger
import torch
import argparse
import numpy as np
import os
import pprint
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import time
from model.audio_emotion_classifer import EmotionNet
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
#import torchmetrics
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_weights( net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

def compute_acc(input_label, out):
    score, pred = out.topk(1, 1)
    pred0 = pred.squeeze().data
    acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
    # acc = 100 * torch.true_divide(torch.sum(pred0 == input_label.data), input_label.size(0))
    return acc

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def vis_confusion_matrix(conf_matrix, Emotion_kinds):
    # 绘制混淆矩阵
    Emotion=8#这个数值是具体的分类数
    labels = ['neutral', 'happiness', 'anger', 'sadness', 'contempt', 'surprise', 'fear', 'disgust']#每种类别的标签

    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2	#数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(Emotion_kinds):
        for y in range(Emotion_kinds):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                    verticalalignment='center',
                    horizontalalignment='center',
                    color="white" if info > thresh else "black")
                 
    
    plt.yticks(range(Emotion_kinds), labels)
    plt.xticks(range(Emotion_kinds), labels,rotation=30)#X轴字体倾斜45°
    plt.tight_layout()#保证图不重叠
    plt.savefig('/home/xingqunqi/Emotion_Gesture/test_confusion_matrix.jpg',dpi=512)
    plt.show()
    plt.close()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def train_K_fold(args, train_dataset, test_dataset, kf, collate_fn, device):
    start = time.time()
    
  
    # model = EmotionNet() 
    # pretrain_model = "/root/BEAT_Emotion/checkpoints/audio_emotion_classifer_train_test/checkpoint_iteration10400.pth"
    # model.load_state_dict({k.replace('module.',''):v for k,v in torch.load("/home/xingqunqi/Emotion_Gesture/BEAT_Emotion/checkpoints_v2/audio_classifer/checkpoint_iteration13700.pth").items()})




    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                              shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )

    for i, (train_index, val_index) in enumerate(kf.split(train_dataset)):

        model = EmotionNet()    
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-5)
        global_iter = 0
        criterion = nn.CrossEntropyLoss()        
        
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_index)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_index)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn, sampler=train_sampler
                              )
        val_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn, sampler=val_sampler
                              )
        for epoch in range(args.total_epoch):
        
            
            for iter_idx, data in enumerate(train_loader, 0):
                model.train()
                model_optimizer.zero_grad()
                global_iter += 1
                in_audio, in_spec, pose_seq, eid_label, aux_info = data 
                #print(len(aux_info['eid']))
                in_spec = in_spec.to(device)
                eid_label = eid_label.to(device, dtype=torch.long)
                #label = torch.randn(args.batch_size, 8).to(device, dtype=torch.long)
                # print('in_spec shape is: ', in_spec.shape)
                output = model(in_spec)
                # print('output shape is: ', output)
                #print(eid_label.shape)
                eid_label = torch.max(eid_label, 1)[1]    
                # print(eid_label)
                loss = criterion(output,eid_label) * 100

                
                
                loss.backward()
                model_optimizer.step()     
                
                if global_iter % 100 == 0 and epoch >= 0 :#and args.epoch % 2 == 0:
                    total = 0
                    correct = 0
                    all_val_acc = 0.0
                    with torch.no_grad():
                        for iter_idx, data in enumerate(val_loader, 0):
                            total += 1
                            in_audio, in_spec, pose_seq, eid_label, aux_info = data 
                            #print(len(aux_info['eid']))
                            in_spec = in_spec.to(device)
                            eid_label = eid_label.to(device, dtype=torch.long)
                            #label = torch.randn(args.batch_size, 8).to(device, dtype=torch.long)
                            output = model(in_spec)
                            eid_label = torch.max(eid_label, 1)[1]    
                            val_acc = compute_acc(eid_label, output)
                            all_val_acc += val_acc.item()
                            
                    accuracy = float(all_val_acc)/total
                    #print(avg_acc)
                    logging.info('Fold {}, Epoch {}, Val Accuracy: {:.2f}%'.format(i+1, epoch, accuracy))
                    args.GfileName = args.model_save_path + '/checkpoint_fold{}_epoch{}_iteraction{}.pth'.format(i+1, epoch, global_iter) 
                
                    torch.save(model.state_dict(), args.GfileName)
                    test_model(args, model, test_loader, epoch, global_iter, i+1)


                  


def test_model(args, model, test_loader, epoch, global_iter, fold):
# def test_model(args, test_loader, device, test_num):
    #all_val_acc = 0.0
    #model.eval()
    val_iter = 0
    Emotion_kinds = 8
    conf_matrix = torch.zeros(Emotion_kinds, Emotion_kinds)


    all_val_acc = 0.0
    model.eval()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_loader, 0):
            val_iter += 1
            in_audio, in_spec, pose_seq, eid_label, aux_info = data 
            #print(len(aux_info['eid']))
            in_spec = in_spec.to(device)
            eid_label = eid_label.to(device, dtype=torch.long)

            output = model(in_spec)

            eid_label = torch.max(eid_label, 1)[1]    
            val_acc = compute_acc(eid_label, output)
            all_val_acc += val_acc.item()
  
            
            
            conf_matrix = confusion_matrix(output, eid_label, conf_matrix)
            conf_matrix = conf_matrix.cpu()
    conf_matrix = np.array(conf_matrix.cpu())
    corrects = conf_matrix.diagonal(offset=0)
    per_kinds = conf_matrix.sum(axis=1)
    # print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)),test_num))
    # # print(conf_matrix)

    # # 获取每种Emotion的识别准确率
    # logging.info("每种情感总个数：",per_kinds)
    # logging.info("每种情感预测正确的个数：",corrects)
    # logging.info("每种情感的识别准确率为：{0}".format([rate*100 for rate in corrects/per_kinds]))
    # vis_confusion_matrix(conf_matrix, Emotion_kinds)

            
            
            
            
    # epoch = 1
    # global_iter = 13700
    avg_acc = float(all_val_acc)/val_iter   
    logging.info('Fold {}, Epoch {}/{}, Iteraction {}, Test Accuracy: {:.2f}%'.format(fold, epoch, args.total_epoch, global_iter, avg_acc))

def main(config):
    args = config

    # random seed
    #if args.random_seed >= 0:
        #utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))
    
    
    # dataset
    collate_fn = audio_classifier_collate_fn
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
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      remove_word_timing=(args.input_context == 'text')
                                       )
    
    Full_dataset = ConcatDataset([train_dataset, val_dataset])

    
    kf = KFold(n_splits=10, shuffle=True)
    
   
    train_K_fold(args, Full_dataset, test_dataset, kf, collate_fn, device)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ##dataset
    parser.add_argument("--motion_resampling_framerate", type=int, default=15)
    parser.add_argument("--n_poses", type=int, default=60)
    parser.add_argument("--n_pre_poses", type=int, default=15)
    parser.add_argument("--subdivision_stride", type=int, default=30)
    parser.add_argument("--loader_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default = 128)
    parser.add_argument("--lr", type=float, default = 0.0003)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument('--total_epoch', type=int, default=60)
    parser.add_argument("--pose_dim", type=int, default = 4)
    parser.add_argument("--model_save_path", type=str, default = '/data/xingqunqi/BEAT_Emotion/checkpoints/audio_emotion_classifer_10_fold_v1')
    parser.add_argument("--latent_dim", type=int, default=128) 
    parser.add_argument("--wordembed_path", type=str, default='/mnt/data/xingqun.qi/BEAT_dataset/crawl-300d-2M-subword.bin') 
    parser.add_argument("--wordembed_dim", type=int, default=300)
    parser.add_argument("--input_context", type=str, default= 'text')
    parser.add_argument("--train_data_path", type=str, default = '/data/xingqunqi/BEAT_Emotion/train/')
    parser.add_argument("--val_data_path", type=str, default = '/data/xingqunqi/BEAT_Emotion/val/')
    parser.add_argument("--test_data_path", type=str, default = '/data/xingqunqi/BEAT_Emotion/test/')
    
    args = parser.parse_args()
    
    
    
       
    main(args)
