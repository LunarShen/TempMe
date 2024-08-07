from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

import torch
import torch.nn.functional as F
from tvr.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tvr.dataloaders.data_dataloaders import DATALOADER_DICT
from tvr.dataloaders.dataloader_msrvtt_retrieval import MSRVTTDataset
from tvr.models.modeling import VTRModel, AllGather
from tvr.models.optimization_adamw import AdamW, get_cosine_schedule_with_warmup
from tvr.utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from tvr.utils.comm import is_main_process, synchronize
from tvr.utils.logger import setup_logger
from tvr.utils.metric_logger import MetricLogger

from scipy.special import softmax

allgather = AllGather.apply

global logger

def get_args(description='Temporal Token Merging for Efficient Text-Video Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos', help='video path')
    parser.add_argument('--pretrained_path', type=str, default="your_path", help='pretrained model path')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--clip_lr', type=float, default=6e-4, help='learning rate')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')

    parser.add_argument("--device", default='cuda', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    parser.add_argument("--local-rank", default=0, type=int, help="distribted training")
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument('--n_display', type=int, default=50, help='Information display frequence')
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str, help="Choose a CLIP version")

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    
    parser.add_argument('--lora_dim', type=int, default=8)

    parser.add_argument('--tome_r', type=int, default=2)
    parser.add_argument('--tome_tracesource', type=bool, default=False)
    parser.add_argument('--tome_propattn', type=bool, default=True)

    ### 12--9-->6--10-->3--11-->1
    parser.add_argument('--merge_layer', type=str, default='8-9-10') # start from 0
    parser.add_argument('--merge_frame_num', type=str, default='2-2-3')

    ### R_c = 100% - 30% = 70%; R_I = 100% - 10% = 90%
    parser.add_argument('--merge_token_proportion', type=str, default='30-10')
    parser.add_argument('--frame_pos', type=int, default=1)
    
    args = parser.parse_args()

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.cuda.is_available():
        torch.distributed.barrier()
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def build_model(args):
    model = VTRModel(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)
    return model


def build_dataloader(args):
    ## ####################################
    # dataloader loading
    ## ####################################
    tokenizer = ClipTokenizer()
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if isinstance(test_length, int):
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    elif len(test_length) == 2:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %dv %dt", test_length[0], test_length[1])
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d %d", len(test_dataloader[0]), len(test_dataloader[1]))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %dv %dt", val_length[0], val_length[1])

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", len(train_dataloader) * args.epochs)
    else:
        train_dataloader, train_sampler = None, None

    return test_dataloader, val_dataloader, train_dataloader, train_sampler


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    if hasattr(model, 'module'):
        model = model.module
    clip_lr = args.clip_lr  # 1e-7
    weight_decay = args.weight_decay  # 0.2
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())

    for name, param in param_optimizer:
        if "TVPt" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    
    optimizer_parameters_prompt = []
    enabled_prompt = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled_prompt.append(name)
            optimizer_parameters_prompt.append(param)
    logger.info(f"Tuned Parameters: {sorted(enabled_prompt)}")

    optimizer_grouped_params = [
        {'params': optimizer_parameters_prompt, 'lr': args.clip_lr}
    ]

    optimizer = AdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    num_warmup_steps = int(warmup_proportion * num_train_optimization_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(args.output_dir, "{}.pth".format(type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def prompt_save_model(epoch, args, model, type_name=""):
    assert "Not Implement" == 0
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(args.output_dir, "{}.pth".format(type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                scheduler, global_step, max_steps, val_dataloader):
    global logger
    global best_score
    global best_score_list
    global meters
    global sim_matrix_num
    global sim_name_list

    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0

    end = time.time()
    logit_scale = 0
    for step, batch in enumerate(train_dataloader, start=1):
        global_step += 1
        data_time = time.time() - end

        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text_ids, text_mask, video, video_mask, inds, idx = batch
        loss = model(text_ids, text_mask, video, video_mask, idx, global_step)

        optimizer.zero_grad()
        
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        with torch.autograd.detect_anomaly():
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule

        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        reduced_l = reduce_loss(loss, args)
        meters.update(time=batch_time, data=data_time, loss=float(reduced_l))

        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (global_step % log_step == 0 or global_step == 1) and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}/{max_epoch}",
                        "iteration: {iteration}/{max_iteration}",
                        "{meters}",
                        "lr: {lr}",
                        "logit_scale: {logit_scale:.2f}"
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    max_epoch=args.epochs,
                    iteration=global_step,
                    max_iteration=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(scheduler.get_last_lr())))]),
                    logit_scale=logit_scale,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if (global_step % (log_step * 3) == 0)  or global_step == 1:
            max_R1 = eval_epoch(args, model, val_dataloader, args.device)
            if args.local_rank == 0:
                for list_idx in range(sim_matrix_num):
                    if best_score_list[list_idx] < max_R1[list_idx]:
                        best_score_list[list_idx] = max_R1[list_idx]
                    logger.info("The R1 is: {:.4f}\t| {:.4f}\tin {}".format(max_R1[list_idx], best_score_list[list_idx],sim_name_list[list_idx]))

                if best_score < max(max_R1):
                    best_score = max(max_R1)
                    output_model_file = save_model(epoch, args, model, type_name="best")
                logger.info("The best R1 is: {:.4f} at all".format(best_score))

            synchronize()
            model.train()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def eval_epoch(args, model, test_dataloader, device):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
    # ----------------------------
    # 1. cache the features
    # ----------------------------
    batch_cls, batch_mask_t = [], []
    batch_video_feat, batch_mask_v = [], []
    batch_ids = []

    with torch.no_grad():
        tic = time.time()

        sim_matrix = []

        logger.info('[start] extract')
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text_ids, text_mask, video, video_mask, inds, _ = batch
            cls, video_feat = model.stage1_eval(text_ids, text_mask, video, video_mask)
            batch_cls.append(cls)
            batch_mask_t.append(text_mask)
            batch_video_feat.append(video_feat)
            batch_mask_v.append(video_mask)
            batch_ids.append(inds)

        torch.distributed.barrier()
        
        batch_ids = allgather(torch.cat(batch_ids, dim=0), args).squeeze()
        
        batch_cls = allgather(torch.cat(batch_cls, dim=0), args)
        batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
        batch_video_feat = allgather(torch.cat(batch_video_feat, dim=0), args)
        batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
        
        batch_cls[batch_ids] = batch_cls.clone()
        batch_mask_t[batch_ids] = batch_mask_t.clone()
        batch_video_feat[batch_ids] = batch_video_feat.clone()
        batch_mask_v[batch_ids] = batch_mask_v.clone()
        
        batch_cls = batch_cls[:batch_ids.max() + 1, ...]
        batch_mask_t = batch_mask_t[:batch_ids.max() + 1, ...]
        batch_video_feat = batch_video_feat[:batch_ids.max() + 1, ...]
        batch_mask_v = batch_mask_v[:batch_ids.max() + 1, ...]
        logger.info('[finish] extract')
        
        logger.info('[start] calculate the similarity')
        with torch.no_grad():
            mini_batch = args.batch_size_val
            sim_matrix = []
            
            batch_cls_split = torch.split(batch_cls, mini_batch)
            batch_mask_t_split = torch.split(batch_mask_t, mini_batch)
            batch_video_feat_split = torch.split(batch_video_feat, mini_batch)
            batch_mask_v_split = torch.split(batch_mask_v, mini_batch)
            
            for cls, text_mask in tqdm(zip(batch_cls_split, batch_mask_t_split)):
                each_row = []
                for video_feat, video_mask in zip(batch_video_feat_split, batch_mask_v_split):
                    logits = model.stage2_eval(cls, text_mask, video_feat, video_mask)
                    logits = logits.cpu().detach().numpy()
                    each_row.append(logits)
                each_row = np.concatenate(tuple(each_row), axis=-1)
                sim_matrix.append(each_row)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        logger.info('[finish] calculate the similarity')
        
        
    logger.info('[start] compute_metrics')
    logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1])) 
    global sim_name_list
    
    max_R1=[]
    list_idx = 0
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    logger.info("Eval {} ...".format(sim_name_list[list_idx]))
    logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['R50'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['R50'], vt_metrics['MR'], vt_metrics['MeanR']))
    max_R1.append(tv_metrics['R1'])

    return max_R1

def main():
    global logger
    global best_score
    global best_score_list
    global meters
    global sim_matrix_num
    global sim_name_list

    sim_name_list = ['base'] 
    sim_matrix_num = len(sim_name_list)

    meters = MetricLogger(delimiter="  ")
    args = get_args()
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)

    args = set_seed_logger(args)

    model = build_model(args)

    test_dataloader, val_dataloader, train_dataloader, train_sampler = build_dataloader(args)
    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        _max_steps = len(train_dataloader) * args.epochs
        optimizer, scheduler, model = prep_optimizer(args, model, _max_steps, args.local_rank)

        best_score = 0.00001
        best_score_list = [0.00001 for _ in range(sim_matrix_num)]
        best_output_model_file = "None"
        global_step = 0
        for epoch in range(args.epochs):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            synchronize()
            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader,
                                               args.device, args.world_size, optimizer,
                                               scheduler, global_step, max_steps, val_dataloader)
            torch.cuda.empty_cache()

            max_R1 = eval_epoch(args, model, val_dataloader, args.device)
            torch.cuda.empty_cache()
            synchronize()

            if args.local_rank == 0:
                for list_idx in range(sim_matrix_num):
                    if best_score_list[list_idx] < max_R1[list_idx]:
                        best_score_list[list_idx] = max_R1[list_idx]
                    logger.info("The R1 is: {:.4f}\t| {:.4f}\tin {}".format(max_R1[list_idx], best_score_list[list_idx],sim_name_list[list_idx]))

                if best_score < max(max_R1):
                    best_score = max(max_R1)
                    output_model_file = save_model(epoch, args, model, type_name="best")
                logger.info("The best R1 is: {:.4f} at all".format(best_score))

            synchronize()

        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')

        if args.local_rank == 0:
            with open("{}_{}.txt".format(args.output_dir, best_score),'w') as f:
                f.write(' ')

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, args.device)


if __name__ == "__main__":
    main()
