''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from .Layers import EncoderLayer, DecoderLayer
#from .Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F
import torch_dct as dct #https://github.com/zh217/torch-dct
from .tcn import TemporalConvNet
from .ResNetBlocks import SEBasicBlock, SEBottleneck
from .ResNetSE34V2 import ResNetSE
#from torch.cuda.amp import autocast
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, *_ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.register_buffer('pos_table2', self._get_sinusoid_encoding_table(n_position, d_hid))
        # self.register_buffer('pos_table3', self._get_sinusoid_encoding_table(n_position, d_hid))
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self,x):
        p=self.pos_table[:,:x.size(1)].clone().detach()
        return x + p

    def forward2(self, x, n_person):
        # if x.shape[1]==135:
        #     p=self.pos_table3[:, :int(x.shape[1]/n_person)].clone().detach()
        #     p=p.repeat(1,n_person,1)
        # else:
        p=self.pos_table2[:, :int(x.shape[1]/n_person)].clone().detach()
        p=p.repeat(1,n_person,1)
        return x + p

class Audio_ConvEncoder(nn.Module):
    ''' A conv encoder model to handle the audio specturm. '''
    def __init__(self,frames,d_model):
        super(Audio_ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, frames , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(frames)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(frames, frames , kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(frames)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 31, d_model) # 32 * 18 -> dim of MFCC after conv
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(d_model,d_model)   
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        B, n_frames, H, W = x.shape
        #print(x.shape)
        x = x.reshape(B, n_frames,-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class Audio_ResNetEncoder(nn.Module):
    ''' A ResNet encoder model to handle the audio specturm. '''

    def __init__(self, frames, d_model):
        super(Audio_ResNetEncoder, self).__init__()
        #num_filters = [32, 64, 128, 256]
        num_filters = [32, 64, 128]
        #self.feat_extractor = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters)
        self.feat_extractor = ResNetSE(SEBasicBlock, [3, 4, 6], num_filters)
        
        self.final_conv1 = nn.Conv2d(num_filters[2], frames , kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(frames)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 31, d_model) # 32 * 18 -> dim of MFCC after conv
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(d_model,d_model)
        #self.fc = nn.Sequential(
            #nn.Linear(32 * 31, 512),
            #nn.Dropout(0.2),
            #nn.Linear(512, 256),
            #nn.Dropout(0.2),
            #nn.Linear(256, 128),
            #nn.Dropout(0.2),
            #nn.Linear(pose_dim, pose_dim)
            #)
        
    def forward(self, audio_spectrum):
        #audio_spectrum = audio_spectrum.unsqueeze(1)
        x = self.feat_extractor(audio_spectrum)
        x = self.final_conv1(x)
        x = self.bn1(x)
        B, n_frames, H, W = x.shape
        x = x.reshape(B, n_frames, -1)
        #x = self.fc1(x)
        #x = self.dropout(x)
        #out = self.fc(x)
        x = self.fc1(x)
        x = self.dropout(x)
        out = self.fc2(x)
        
        #print(out.shape)
        return out  # to (batch x seq x dim)
    
    







class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layers
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], 512)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()
        self.fc1 = nn.Sequential(
            nn.Linear(60, 60)
        )
    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        #print('hahah')
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2))#.transpose(1, 2)
        #print('y shape is: ', y.shape)
        y = self.fc1(y).transpose(1, 2)
        y = self.decoder(y).contiguous()
        #print('text embedding shape is: ', y.shape)
        return y




class Prior_ConvEncoder(nn.Module):
    ''' A conv encoder model to handle the prior pose sequence. '''
    def __init__(self,prior_frames, frames, pose_dim, d_model):
        super(Prior_ConvEncoder, self).__init__()
        self.conv1 = nn.Conv1d(prior_frames, frames , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(frames)
        #self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(frames, frames , kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(frames)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    
        self.fc1 = nn.Linear(pose_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        #x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        #x = self.maxpool2(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SP_Memory_Net_v1(nn.Module):
    ''' A Memory encoder model to handle the prior pose sequence. '''
    def __init__(self, args, prior_frames, pred_frames, pose_dim, d_model):
        super(SP_Memory_Net_v1, self).__init__()
        self.prior_frames = prior_frames
        self.pred_frames = pred_frames
        self.pose_dim = pose_dim
        self.d_model = d_model
        self.chunk_length = args.chunk
        #self.spatial_chunk_encoder = nn.Sequential(
            #nn.Linear(self.chunk_length * d_model, d_model),
            #nn.Dropout(0.2),    
        #)
        self.spatial_chunk_encoder = nn.Sequential(
            nn.Linear(self.chunk_length * pose_dim, pose_dim),
            nn.Dropout(0.2),    
            nn.Linear(pose_dim, pose_dim),
        )
    def forward(self, initial_feature, pred_feature):
        B,_,_ = initial_feature.shape
        pred_feature_clone = pred_feature.clone()
        #print('initial_feature[:, self.prior_frames-self.chunk_length:, :] shape is: ', initial_feature[:, self.prior_frames-self.chunk_length:, :].shape) # B,self.chunk_length,282
        memory_encoding = self.spatial_chunk_encoder(initial_feature[:, self.prior_frames-self.chunk_length:, :].reshape(B,-1)) # B, 282 * 10
        memory_encoding = memory_encoding.unsqueeze(1)
        for b_index in range(B):
            for c_index in range(self.chunk_length):
                #print('memory_encoding[b_index, :, :] shape is ', memory_encoding[b_index, :, :].shape)
                #print('pred_feature[b_index, c_index, :] shape is ', pred_feature[b_index, c_index, :].shape)
                score = torch.mm(memory_encoding[b_index, :, :], pred_feature_clone[b_index, c_index, :].unsqueeze(1))
                soft_score = torch.sigmoid(score.squeeze(1))
                #print('soft_score is: ', soft_score)
                #print('pred_feature[b_index, c_index, :] shape is: ', pred_feature[b_index, c_index, :].shape)
                #print('memory_encoding[b_index, :, :] shape is: ', memory_encoding[b_index, :, :].squeeze(0).shape)
                spatial_feature = soft_score * pred_feature_clone[b_index, c_index, :] + (1-soft_score) * memory_encoding[b_index, :, :].squeeze(0)
                pred_feature[b_index, c_index, :] = spatial_feature
        
        return pred_feature
                








                
            
class TM_Memory_Net(nn.Module):
    ''' A Memory encoder model to handle the prior pose sequence. '''
    def __init__(self, args, prior_frames, pred_frames, pose_dim, d_model):
        super(TM_Memory_Net, self).__init__()
        self.prior_frames = prior_frames
        self.pred_frames = pred_frames
        self.pose_dim = pose_dim
        self.d_model = d_model
        self.chunk_length = args.chunk
        self.temporal_chunk_encoder = nn.Sequential(
            nn.Linear(self.chunk_length * self.pose_dim, self.pose_dim),
            nn.Dropout(0.2),    
            nn.Linear(self.pose_dim, self.pose_dim),
        )
        self.temporal_memory_encoder = nn.Sequential(
            nn.Linear(self.chunk_length * self.pose_dim, self.chunk_length),
            nn.Dropout(0.2),    
            nn.Linear(self.chunk_length, self.chunk_length),
        )
    def forward(self, initial_feature, pred_feature):
        B,_,_ = initial_feature.shape
        pred_feature_clone = pred_feature.clone()
        memory_encoding = self.temporal_chunk_encoder(initial_feature[:, self.prior_frames-self.chunk_length:, :].reshape(B,-1)) # B, 512
        pred_encoding = self.temporal_memory_encoder(pred_feature_clone[:, :self.chunk_length, :].reshape(B,-1)) # B, self.chunk_length
        #memory_encoding = memory_encoding.unsqueeze(1)
        score = torch.mm(memory_encoding.t(), pred_encoding)
        score = torch.mm(memory_encoding, score)
        soft_score = F.softmax(score, dim=1)
        temporal_feature = pred_feature_clone[:, :self.chunk_length, :] + torch.mul(pred_feature_clone[:, :self.chunk_length, :], soft_score.unsqueeze(2))
        pred_feature[:, :self.chunk_length, :] = temporal_feature
        return pred_feature           
        
    
        
    

class Prior_MemoryEncoder(nn.Module):
    ''' A conv encoder model to handle the prior pose sequence. '''
    def __init__(self, args, prior_frames, frames, pose_dim, d_model):
        super(Prior_MemoryEncoder, self).__init__()
        #self.conv1 = nn.Conv1d(prior_frames, frames , kernel_size=3, stride=1, padding=1)
        #self.relu = nn.ReLU(inplace=True)
        #self.bn1 = nn.BatchNorm1d(frames)
        #self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv2 = nn.Conv1d(frames, frames , kernel_size=3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm1d(frames)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    
        #self.fc1 = nn.Linear(pose_dim, d_model)
        #self.fc2 = nn.Linear(d_model, d_model)
        #self.dropout = nn.Dropout(0.2)
        
        #self.prior_header = nn.Sequential(
            #nn.Linear(pose_dim, d_model),
           #nn.Dropout(0.2),         
        #)
        self.post_header = nn.Sequential(
            nn.Linear(pose_dim, d_model),
            nn.Dropout(0.2),  
            nn.Linear(d_model, d_model),       
        )
        self.pred_length = frames - prior_frames
        self.pred_conv = nn.Sequential(
            nn.Conv1d(prior_frames, self.pred_length , kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),   
            nn.BatchNorm1d(self.pred_length),     
            nn.Conv1d(self.pred_length, self.pred_length , kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),   
            nn.BatchNorm1d(self.pred_length), 
        )
        self.spatial_memory = SP_Memory_Net_v1(args, prior_frames, self.pred_length, pose_dim, d_model)

        self.temporal_memory = TM_Memory_Net(args, prior_frames, self.pred_length, pose_dim, d_model)

    def forward(self, x):
        
        #initial_feature = self.prior_header(x)
        initial_feature = x
        pred_feature = self.pred_conv(x)
        pred_feature = self.spatial_memory(initial_feature, pred_feature) # spatial
        pred_feature = self.temporal_memory(initial_feature, pred_feature)# temporal
        out = torch.cat((initial_feature, pred_feature), 1)
        out = self.post_header(out)
        return out


        

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, use_wscale = True):

        super().__init__()
        self.position_embeddings = nn.Embedding(n_position, d_model)
        #self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        #self.motion_IN = AdaIN(l_dim, 64, use_wscale)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, src_seq, src_mask, return_attns=False, global_feature=False):
        
        enc_slf_attn_list = []
        # -- Forward
        #src_seq = self.layer_norm(src_seq)
        #if global_feature:
            #enc_output = self.dropout(self.position_enc.forward2(src_seq,n_person))
            #enc_output = self.dropout(src_seq)
        #else:
        enc_output = self.dropout(self.position_enc(src_seq))
        #enc_output = self.layer_norm(enc_output)
        #enc_output=self.dropout(src_seq+position_embeddings)
        #enc_output = self.dropout(self.layer_norm(enc_output))
        #i = 0
        for enc_layer in self.layer_stack:
            #enc_output = self.motion_IN(enc_output, motion_code)
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            

        if return_attns:
            return enc_output, enc_slf_attn_list
        
        return enc_output,

class Decoder(nn.Module):

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()

        #self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        #self.device=device

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = (trg_seq)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, dec_enc_attn_list

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, args, lang_model, frames = 60, pose_dim = 282, prior_frames = 10, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, dropout=0.2, n_position=60):

        super().__init__()
        
        #self.device=device
        
        self.d_model=d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.audio_encoder = Audio_ResNetEncoder(frames, self.d_model)
        self.text_encoder = TextEncoderTCN(args, lang_model.n_words, args.wordembed_dim, pre_trained_embedding=lang_model.word_embedding_weights,
                                           dropout=args.dropout_prob)
        self.emotion_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(0.2),
            nn.Linear(self.d_model, self.d_model),
            #nn.Dropout(0.2),
            #nn.Linear(d_model, pose_dim),
            #nn.Dropout(0.2),
            #nn.Linear(pose_dim, pose_dim)
            )
        self.emotion_classifer_header = nn.Sequential(
            #nn.Linear(256, d_model),
            #nn.ReLU(True),
            nn.Linear(frames*self.d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64,8)
            #nn.Linear(pose_dim, pose_dim)
            )
        self.semantic_proj = nn.Sequential(
            nn.Linear( self.d_model, self.d_model),
            nn.Dropout(0.2),
            nn.Linear(self.d_model, self.d_model),
            #nn.Dropout(0.2),
            #nn.Linear(d_model, pose_dim),
            #nn.Dropout(0.2),
            #nn.Linear(pose_dim, pose_dim)
            )
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_model),
            #nn.Dropout(0.2),
            #nn.Linear(d_model, pose_dim),
            #nn.Dropout(0.2),
            #nn.Linear(pose_dim, pose_dim)
            )
        self.prior_seq_encoder = Prior_MemoryEncoder(args, prior_frames, frames, pose_dim, self.d_model)
        self.post_projector = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*4),
            nn.Dropout(0.2),
            nn.Linear(self.d_model*4, self.d_model),
            nn.Dropout(0.2),
            nn.Linear(self.d_model, pose_dim),
            nn.Dropout(0.2),
            nn.Linear(pose_dim, pose_dim)
            )
        #self.proj_inverse=nn.Linear(d_model,126)
        #self.l1=nn.Linear(d_model, d_model*4)
        #self.l2=nn.Linear(d_model*4, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)



        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
    #@autocast() 
    #def forward(self, input_spectrum, text, prior_seq):
    def forward(self, input_spectrum, text, prior_seq, sampled_emotion_feature = None):
        

        src_mask = None
        
        
        text_embedding = self.text_encoder(text)

        
        input_spectrum = input_spectrum.unsqueeze(1)

        
        spectrum_feature = self.audio_encoder(input_spectrum)

        prior_seq = self.prior_seq_encoder(prior_seq)

        
        emotion_feature = self.emotion_proj(spectrum_feature)
        semantic_festure = self.semantic_proj(spectrum_feature)
        B, N, D = emotion_feature.shape
        
        emotion_prediction = self.emotion_classifer_header(emotion_feature.reshape(B, -1))
        
        
        # ## sample
        # fusion_feature = sampled_emotion_feature + semantic_festure
        
        ### without sample
        #fusion_feature = emotion_feature + semantic_festure
        
        if sampled_emotion_feature is not None:
            fusion_feature = sampled_emotion_feature + semantic_festure
            #print("hhaha")
        else:
            fusion_feature = emotion_feature + semantic_festure
        
        fusion_feature = self.fusion_proj(fusion_feature)
        
        enc_output, *_=self.encoder(fusion_feature, src_mask)
        dec_output, *_=self.decoder(prior_seq, None, enc_output, None)


        dec_output=self.post_projector(dec_output)
              
        return dec_output, emotion_feature, semantic_festure, emotion_prediction, text_embedding



class Motion_Discriminator(nn.Module):
    def __init__(
            self, frames = 59, pose_dim = 282, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=2, n_head=8, d_k=64, d_v=64, dropout=0.2, n_position=59):

        super().__init__()
        #self.device=device     
        self.d_model=d_model
        self.encoder = Encoder(
                n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=src_pad_idx, dropout=dropout)
            
        self.fc1 = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(inplace=True)
            #nn.Linear(64, 1)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(frames * 64, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            #nn.ReLU(inplace=True),
            #nn.Linear(64, 1)
            )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)         
    #@autocast() 
    def forward(self, x):
        B, T, D = x.shape
        x, *_ = self.encoder(x,src_mask=None)
        x = self.fc1(x)
        #print(x.shape)
        x = x.reshape(B, -1)
        
        
        x = self.fc2(x)
        #output = torch.sigmoid(x)
        return x

        




if __name__ == '__main__':
    def calc_motion(tensor):
        res = tensor[:,:1,:] - tensor[:,:-1,:]
        return res
    use = None
    device='cuda'
    input_spectrum = torch.randn((64,1, 128,70)).to(device)
    prior_seq = torch.randn((64,4, 126)).to(device)
    model = Transformer(frames = 34, d_word_vec=256, d_model=256, d_inner=1024, n_layers=3, n_head=8, d_k=64, d_v=64).to(device)
    discriminator = Trans_Discriminator(d_word_vec=126, d_model=126, d_inner=512, n_layers=3, n_head=8, d_k=32, d_v=32).to(device)
    gen_seq = model.forward(input_spectrum,prior_seq)
    print('gen_seq shape is: ', gen_seq.shape)
    fake_motion = calc_motion(gen_seq)
    print('fake_motion shape is: ', fake_motion.shape)
    fake_score = discriminator(fake_motion)
    print('fake_score is: ', fake_score.shape)
    
    