from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .module_tome_patch import apply_patch as tome_patch
from .module_tome_utils import parse_r
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL
import numpy as np
import copy
allgather = AllGather.apply
allgather2 = AllGather2.apply

logger = logging.getLogger(__name__)

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class VTRModel(nn.Module):
    def __init__(self, config):
        super(VTRModel, self).__init__()
        
        self.config = config
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        self.lora_dim = config.lora_dim
        logger.info("v_LoRA: {} dim".format(self.lora_dim))
        
        assert backbone in _PT_NAME
        model_path = os.path.join(config.pretrained_path, _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.merge_layer = [int(_l) for _l in config.merge_layer.split('-')]
        self.merge_frame_num = [int(_l) for _l in config.merge_frame_num.split('-')]
        frame_num_list=[]
        frame_num = config.max_frames
        for _l in range(len(self.merge_layer)):
            frame_num_list.append(frame_num)
            frame_num = frame_num // self.merge_frame_num[_l]
        logger.info('Position_embedding: {}'.format(frame_num_list))
        
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, self.lora_dim, 
                        self.merge_layer, config.frame_pos, frame_num_list)
            
        self.loss_fct = CrossEn(config)

        self.clip.load_state_dict(state_dict, strict=False)

        self.tome_r = config.tome_r
        self.tome_tracesource = config.tome_tracesource
        self.tome_propattn = config.tome_propattn
        logger.info("tome: {} r | {} tracesource | {} propattn".format(self.tome_r, self.tome_tracesource, self.tome_propattn))
        
        logger.info("merge_layer: {}".format(config.merge_layer))
        logger.info("merge_frame_num: {}".format(config.merge_frame_num))
        logger.info("merge_token_proportion: {}".format(config.merge_token_proportion))
        logger.info("frame_pos: {}".format(config.frame_pos))

        self.merge_token_proportion = [int(_l) / 100 for _l in config.merge_token_proportion.split('-')]
        self.frame_pos = config.frame_pos
        
        self.merge_layer = [int(_l) for _l in config.merge_layer.split('-')]
        self.merge_frame_num = [int(_l) for _l in config.merge_frame_num.split('-')]
        self.TVPt_Video_Positional_embedding = []
        if config.base_encoder == "ViT-B/32":
            patch_num = 50
        else:
            patch_num = 197
        cls_num = 1
        frame_num = config.max_frames
        self.patch_list = [patch_num]
        self.frame_list = [frame_num]
        for _l in range(12):
            if _l not in self.merge_layer:
                if _l < self.merge_layer[0]:
                    patch_num = patch_num - self.tome_r
                    self.patch_list.append(patch_num)
                    self.frame_list.append(frame_num)
                else:
                    patch_num = patch_num - int(patch_num * self.merge_token_proportion[1])
                    self.patch_list.append(patch_num)
                    self.frame_list.append(frame_num)
            else:
                M_frame_num = self.merge_frame_num.pop(0)
                M_token_num = int(patch_num * M_frame_num * self.merge_token_proportion[0])

                assert frame_num % M_frame_num == 0
                patch_num = patch_num * M_frame_num - M_token_num
                cls_num = cls_num * M_frame_num
                frame_num = frame_num // M_frame_num
                self.patch_list.append(patch_num)
                self.frame_list.append(frame_num)

                patch_num = patch_num - int(patch_num * self.merge_token_proportion[1])
                self.patch_list.append(patch_num)
                self.frame_list.append(frame_num)
        
        self.merge_layer = [int(_l) for _l in config.merge_layer.split('-')]
        self.merge_frame_num = [int(_l) for _l in config.merge_frame_num.split('-')]
            
        tome_patch(self.clip, trace_source=self.tome_tracesource, prop_attn=self.tome_propattn)
        
    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        cls = self.get_text_feat(text_ids, text_mask)
        video_feat = self.get_video_feat(video, video_mask)
        
        cls = allgather(cls, self.config)
        video_feat = allgather(video_feat, self.config)
        torch.distributed.barrier()
        
        logit_scale = self.clip.logit_scale.exp()
        loss = 0.
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True)
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])

        loss_t2v = self.loss_fct(t2v_logits * logit_scale)
        loss_v2t = self.loss_fct(t2v_logits.T * logit_scale)
        loss = (loss_t2v + loss_v2t) / 2
        
        return loss

    def stage1_eval(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        cls = self.get_text_feat(text_ids, text_mask)
        video = self.get_video_feat(video, video_mask)

        return cls, video

    def stage2_eval(self, cls, text_mask, video_feat, video_mask):
        logit_scale = self.clip.logit_scale.exp()
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True) 
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True) 

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])
        
        return t2v_logits * logit_scale

    def get_text_feat(self, text_ids, orig_mask):
        b = text_ids.size(0)
        x = self.clip.token_embedding(text_ids) 
        max_t_len = x.size(1)
        pos_emd = self.clip.positional_embedding[:max_t_len, :]
        x = x + pos_emd

        mask = orig_mask
        text_length = max_t_len
        attn_mask = self.clip.build_attention_mask(text_length).repeat(x.size(0), 1, 1).to(mask.device)
        inf = torch.zeros((text_length, text_length)).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        attn_mask = torch.where(mask>0, attn_mask, inf)
    
        x = self.clip.transformer(x, attn_mask)

        hidden = self.clip.ln_final(x) @ self.clip.text_projection
        cls = hidden[torch.arange(hidden.shape[0]), text_ids.argmax(dim=-1)]

        cls = cls.float()
        cls = cls.view(b, -1, cls.size(-1)).squeeze(1)
        return cls

    def get_video_feat(self, video, video_mask):
        self.clip._tome_info["size"] = None
        self.clip._tome_info["source"] = None
        self.clip._tome_info["cls_num"] = 1
        self.clip._tome_info["frame_num"] = self.frame_list[0]
        self.clip._tome_info["token_num"] = self.patch_list[0]

        self.merge_frame_num = [int(_l) for _l in self.config.merge_frame_num.split('-')]
        
        b, n_f = video_mask.size()
        org_n_f = n_f
        x = video
            
        x = self.clip.visual.conv1(x)  

        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)  
        x = torch.cat(
            [self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  
        
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        
        _, token_len, d_v = x.size()

        pos_count = 0
        for res_i, res_block in enumerate(self.clip.visual.transformer.resblocks):
            if res_i not in self.merge_layer:
                if res_i < self.merge_layer[0]:
                    x = res_block(x, M_frame_num=1, M_token_num=[self.tome_r])
                else:
                    M_token_num = int(self.clip._tome_info["token_num"] * self.merge_token_proportion[1])
                    M_token_num = min((self.clip._tome_info["token_num"] - self.clip._tome_info["cls_num"]) // 2, M_token_num)
                    x = res_block(x, M_frame_num=1, M_token_num=[M_token_num])
            else:
                M_frame_num = self.merge_frame_num.pop(0)
                M_token_num_0 = int(self.clip._tome_info["token_num"] * M_frame_num * self.merge_token_proportion[0])
                M_token_num_0 = min((self.clip._tome_info["token_num"] - self.clip._tome_info["cls_num"]) * M_frame_num // 2, M_token_num_0)
                M_token_num_1 = int((self.clip._tome_info["token_num"] * M_frame_num - M_token_num_0) * self.merge_token_proportion[1])
                M_token_num_1 = min( ( (self.clip._tome_info["token_num"] - self.clip._tome_info["cls_num"]) * M_frame_num - M_token_num_0) // 2, M_token_num_1)
                    
                x = res_block(x, M_frame_num=M_frame_num, M_token_num=[M_token_num_0, M_token_num_1], frame_pos=self.frame_pos)
        
        n_f = self.clip._tome_info["frame_num"]
        token_len = self.clip._tome_info["token_num"]
        cls_num = self.clip._tome_info["cls_num"]
        x = x.view(b, n_f, token_len, d_v)[:,:,:cls_num,:].reshape(b,org_n_f,d_v)
        hidden = self.clip.visual.ln_post(x) @ self.clip.visual.proj
        video_feat = hidden.float()
        
        video_feat = video_feat.contiguous()
        
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        video_feat = self.get_video_avg_feat(video_feat, video_mask)
        
        return video_feat

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
