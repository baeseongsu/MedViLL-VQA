"""BERT for report generation finetuning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
from multiprocessing.sharedctypes import Value
import os

now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")
print("START", now)
import glob
import logging
import json
import argparse
from tqdm import tqdm, trange
from pathlib import Path
import copy
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import wandb

# custom pkgs
import data_loader
from loader_utils import batch_list_to_batch_tensors
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.model import MedViLLForVQA
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear


# import sys
# import math
# from transformers import BertConfig, AutoConfig
# from pytorch_pretrained_bert.model import BertForPreTrainingLossMask
# from data_parallel import DataParallelImbalance
# import utils
# import pickle
# from collections import defaultdict


logger = logging.getLogger(__name__)


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split(".")[-1]) for fn in fn_model_list]) & set([int(Path(fn).stem.split(".")[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rank():
    import torch.distributed as dist

    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def main(args):

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.global_rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '3412'
    print("global_rank: {}, local rank: {}".format(args.global_rank, args.local_rank))

    # Load pre-trained model (from origianl github)
    if args.model_recover_path != None:
        args.config_path = os.path.join(os.path.dirname(args.model_recover_path), "config.json")
    else:
        raise ValueError()
        # args.model_recover_path = "/home/edlab/jhmoon/mimic_mv_real/mimic-cxr/pre-train/_2/base_PAR_36,128/pytorch_model.bin"
        # args.config_path = os.path.join(os.path.dirname(args.model_recover_path), "config.json")

    # set experiment name
    if args.model_recover_path != None:
        if args.exp_name == "":
            args.exp_name = args.model_recover_path.split("/")[-2]
    else:
        raise ValueError()

    # set output directory
    args.output_dir = os.path.join(args.output_dir, f"""{args.model_recover_path.split("/")[-3]}_{args.exp_name}""")
    print("args.output_dir", args.output_dir)

    # set max sequence length
    args.max_seq_length = args.max_len_b + args.len_vis_input + 3  # +3 for 2x[SEP] and [CLS]

    # define file path
    if args.vqa_dataset == "vqa-rad":
        args.src_file = "/home/data_storage/mimic-cxr/dataset/data_RAD"
        args.img_path = "/home/data_storage/mimic-cxr/dataset/vqa_image/vqa_512_3ch"
        # args.train_dataset = "/home/data_storage/mimic-cxr/dataset/data_RAD/trainet.json"

    elif args.vqa_dataset == "slake":
        args.src_file = None
        args.img_path = None

    elif args.vqa_dataset == "vqa-mimic":
        # TODO:
        semantic_type = args.vqa_mimic
        assert semantic_type in ["all", "verify", "choose", "query"]
        if args.vqa_mimic == "all":
            args.src_file = "/home/data_storage/EHR_VQG/SAMPLED_QA/20220717_all/csv"
        elif args.vqa_mimic in ["verify", "choose"]:
            args.src_file = f"/home/data_storage/EHR_VQG/SAMPLED_QA/20220628_{semantic_type}/csv/"
        elif args.vqa_mimic == "query":
            args.src_file = f"/home/data_storage/EHR_VQG/SAMPLED_QA/20220627_{semantic_type}/full/csv/"
        else:
            raise ValueError()

        args.img_path = "/home/data_storage/mimic-cxr-jpg/2.0.0/resized_512"
    else:
        raise ValueError()

    print(" # PID :", os.getpid())
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, "opt.json"), "w"), sort_keys=True, indent=2)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, args.log_file), filemode="w", format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    args.device = torch.device("cuda", args.local_rank)

    torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.global_rank)

    logger.info("device: {} distributed training: {}, 16-bits training: {}".format(device, bool(args.local_rank != -1), args.fp16))
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)
    torch.backends.cudnn.benchmark

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # fix random seed
    set_seed(seed=args.seed)

    if args.wandb and is_main_process():
        # wandb.init(config=args, project="report_gen", entity="mimic-cxr", name=args.exp_name, reinit=True)
        wandb.init(
            config=args,
            entity=args.wandb_entity_name,
            project=args.wandb_project_name,
            name=args.exp_name,
            reinit=True,
        )

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    # build preprocessing pipeline
    from data_loader import PipelineForVQARAD, PipelineForVQAMIMIC

    preproc_pipeline = PipelineForVQAMIMIC(
        args=args,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
        len_vis_input=args.len_vis_input,
    )

    # get train dataset
    from data_loader import VQARADDataset, VQAMIMICDataset

    if not args.vqa_eval:
        train_dataset = VQAMIMICDataset(
            args=args,
            split="train",
            file_src=args.src_file,  # "/home/data_storage/mimic-cxr/dataset/data_RAD"
            img_root=args.img_path,
            batch_size=args.train_batch_size,
            tokenizer=tokenizer,
            preproc_pipeline=preproc_pipeline,
        )

        # get train sampler
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

        # get train dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            sampler=train_sampler,
            collate_fn=batch_list_to_batch_tensors,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        t_total = int(len(train_dataloader) * args.num_train_epochs * 1.0 / args.gradient_accumulation_steps)

    # get eval dataset
    eval_dataset = VQAMIMICDataset(
        args=args,
        split="test",
        file_src=args.src_file,  # "/home/data_storage/mimic-cxr/dataset/data_RAD"
        img_root=args.img_path,
        batch_size=args.train_batch_size,
        tokenizer=tokenizer,
        preproc_pipeline=preproc_pipeline,
    )

    # define dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=batch_list_to_batch_tensors,
        drop_last=True,
    )

    # prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    if args.vqa_dataset == "vqa-rad":
        cls_num_labels = 458
    elif args.vqa_dataset == "vqa-mimic":
        cls_num_labels = 106  # TODO: vqa-rad (458), vqa-mimic (??)
    type_vocab_size = 2
    relax_projection = 4 if args.relax_projection else 0
    # task_idx_proj = 3 if args.tasks == "report_generation" else 0
    task_idx_proj = 0

    # BERT model will be loaded! from scratch
    if args.model_recover_path is None:
        raise ValueError()
    else:
        # print("Task :", args.tasks, args.s2s_prob)
        model_recover = None
        for model_recover_path in glob.glob(args.model_recover_path.strip()):
            print("Recover path: ", model_recover_path)
            logger.info("***** Recover model: %s *****", args.model_recover_path)
            model_recover = torch.load(model_recover_path)

            for key in list(model_recover.keys()):
                model_recover[key.replace("enc.", "").replace("mlm.", "cls.")] = model_recover.pop(key)

        model = MedViLLForVQA.from_pretrained(
            args.bert_model,
            state_dict=model_recover,
            args=args,
            num_labels=cls_num_labels,  # cls_num_labels
            type_vocab_size=type_vocab_size,
            relax_projection=relax_projection,
            config_path=args.config_path,
            task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings,
            label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding,
            # cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank),
            drop_prob=args.drop_prob,
            # len_vis_input=args.len_vis_input,
            # tasks=args.tasks,
        )

        model.load_state_dict(model_recover, strict=False)
        print("The pretrained model loaded and fine-tuning.")
        del model_recover
        torch.cuda.empty_cache()

    model.to(device)
    if args.wandb and is_main_process():
        wandb.watch(model)

    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if not args.vqa_eval:
        # get optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, schedule=args.sche_mode, t_total=t_total)
        if recover_step:
            logger.info("***** Recover optimizer: %d *****", recover_step)
            optim_recover = torch.load(os.path.join(args.output_dir, "optim.{0}.bin".format(recover_step)))
            if hasattr(optim_recover, "state_dict"):
                optim_recover = optim_recover.state_dict()
            optimizer.load_state_dict(optim_recover)
            if args.loss_scale == 0:
                logger.info("***** Recover optimizer: dynamic_loss_scale *****")
                optimizer.dynamic_loss_scale = True

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.vqa_eval:
        # total_acc, closed_acc, open_acc = vqa_eval(args, device, logger, bi_uni_pipeline, tokenizer, model, results_dict)
        eval_vqa_acc, eval_closed_acc, eval_open_acc = evaluate_vqa_model(args=args, model=model, eval_dataloader=eval_dataloader)
        eval_metrics = {
            "eval/acc (total)": eval_vqa_acc,
            "eval/acc (closed)": eval_closed_acc,
            "eval/acc (open)": eval_open_acc,
        }
        print(eval_metrics)
    else:
        if args.do_train:

            logger.info("***** Running training *****")
            model.train()
            print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

            global_step = 0

            if recover_step:
                start_epoch = recover_step + 1
                print("Recoverd epoch", start_epoch)
            else:
                start_epoch = 1

            # training epoch loop
            for i_epoch in trange(start_epoch, args.num_train_epochs + 1, desc="Epoch"):
                if args.local_rank != -1:
                    train_sampler.set_epoch(i_epoch - 1)
                iter_bar = tqdm(train_dataloader, desc="Iter (loss=X.XXX)")

                # init metrics
                train_loss = []
                total_vqa_score, total_closed_score, total_open_score = [], [], []

                for step, batch in enumerate(iter_bar):

                    # prepare inputs
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, attention_mask, img, vis_pe, ans_labels, ans_type, _organ = batch

                    # half precision
                    if args.fp16:
                        img = img.half()
                        vis_pe = vis_pe.half()

                    # run model
                    loss_tuple = model(
                        img,
                        vis_pe,
                        input_ids,
                        segment_ids,
                        attention_mask,
                        ans_labels,
                        ans_type=ans_type,
                    )

                    # _masked_lm_loss, vqa_loss, vqa_acc, closed_acc, open_acc = loss_tuple
                    vqa_loss, vqa_score, vqa_logits, closed_score, open_score = loss_tuple

                    # loss
                    loss = vqa_loss.mean()
                    train_loss.append(loss.item())
                    iter_bar.set_description("Iter (loss=%5.3f)" % (loss.item()))
                    if args.wandb and is_main_process():
                        wandb.log({"train/loss": loss, "global_step": global_step})

                    # accuracy (step)
                    total_vqa_score += vqa_score
                    total_closed_score += closed_score
                    total_open_score += open_score

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                        if args.fp16:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if global_step % 500 == 0:
                        # evaulation (stepwise)
                        eval_vqa_acc, eval_closed_acc, eval_open_acc = evaluate_vqa_model(args=args, model=model, eval_dataloader=eval_dataloader)
                        eval_metrics = {
                            "eval/acc (total)": eval_vqa_acc,
                            "eval/acc (closed)": eval_closed_acc,
                            "eval/acc (open)": eval_open_acc,
                        }
                        print(eval_metrics)

                        if args.wandb and is_main_process():
                            wandb.log(eval_metrics, step=global_step)

                # accuracy (epoch)
                train_epoch_vqa_acc = sum(total_vqa_score) / len(total_vqa_score)
                if len(total_closed_score) > 0:
                    train_epoch_closed_acc = sum(total_closed_score) / len(total_closed_score)
                else:
                    train_epoch_closed_acc = 0
                if len(total_open_score) > 0:
                    train_epoch_open_acc = sum(total_open_score) / len(total_open_score)
                else:
                    train_epoch_open_acc = 0
                train_epoch_metrics = {
                    "train/epoch acc (total)": train_epoch_vqa_acc,
                    "train/epoch acc (closed)": train_epoch_closed_acc,
                    "train/epoch acc (open)": train_epoch_open_acc,
                }
                # print(train_epoch_metrics)
                if args.wandb and is_main_process():
                    wandb.log(train_epoch_metrics, step=global_step)
                    wandb.log({"learning rate": lr_this_step}, step=global_step)

                logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
                model_to_save = model.module if hasattr(model, "module") else model  # Only save the model it-self
                output_config_file = os.path.join(args.output_dir, "config.json")

                with open(output_config_file, "w") as f:
                    f.write(model_to_save.config.to_json_string())

                output_model_file = os.path.join(args.output_dir, "model.{0}.bin".format(i_epoch))
                # output_optim_file = os.path.join(args.output_dir, "optim.{0}.bin".format(i_epoch))
                if args.global_rank in (-1, 0):  # save model if the first device or no dist
                    torch.save(copy.deepcopy(model_to_save).cpu().state_dict(), output_model_file)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                if args.world_size > 1:
                    torch.distributed.barrier()

                # evaulation (epoch)
                eval_vqa_acc, eval_closed_acc, eval_open_acc = evaluate_vqa_model(args=args, model=model, eval_dataloader=eval_dataloader)
                eval_metrics = {
                    "eval/acc (total)": eval_vqa_acc,
                    "eval/acc (closed)": eval_closed_acc,
                    "eval/acc (open)": eval_open_acc,
                }
                print(eval_metrics)

                if args.wandb and is_main_process():
                    wandb.log(eval_metrics, step=global_step)


def evaluate_vqa_model(args, model, eval_dataloader):
    """
    Modified Version (2022.07.10, Seongsu Bae)
    """

    logger.info("***** Running Evaulation *****")
    model.eval()
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

    # init metrics
    test_loss = []
    total_vqa_score, total_closed_score, total_open_score = [], [], []
    total_vqa_logits, total_vqa_labels = [], []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_dataloader)):

            # prepare inputs
            batch = [t.to(args.device) for t in batch]
            input_ids, segment_ids, attention_mask, img, vis_pe, ans_labels, ans_type, _organ = batch

            # half precision
            if args.fp16:
                img = img.half()
                vis_pe = vis_pe.half()

            # run model
            loss_tuple = model(
                img,
                vis_pe,
                input_ids,
                segment_ids,
                attention_mask,
                ans_labels,
                ans_type=ans_type,
            )
            vqa_loss, vqa_score, vqa_logits, closed_score, open_score = loss_tuple

            loss = vqa_loss.mean()
            test_loss.append(loss.item())

            total_vqa_score += vqa_score
            total_closed_score += closed_score
            total_open_score += open_score

            total_vqa_logits += vqa_logits.cpu()
            total_vqa_labels += ans_labels.cpu()

    # assert len(total_vqa_score) == len(total_closed_score) + len(total_open_score)
    total_vqa_labels = torch.stack(total_vqa_labels, axis=0)
    total_vqa_labels = total_vqa_labels.numpy()

    total_vqa_logits = torch.stack(total_vqa_logits, axis=0)
    total_vqa_logits = total_vqa_logits.numpy()

    from sklearn.metrics import classification_report

    reports = classification_report(total_vqa_labels, total_vqa_logits >= 0.5)
    print(reports)

    # save report
    save_file_path = os.path.join(args.output_dir, "report.bin")
    torch.save(reports, save_file_path)
    # with open(save_file, "wb") as f:
    #     pickle.dump(reports, f, protocol=pickle.HIGHEST_PROTOCOL)
    # print(classification_report(y_true, y_pred, target_names=label_names))

    eval_vqa_acc = sum(total_vqa_score) / len(total_vqa_score)
    if len(total_closed_score) != 0:
        eval_closed_acc = sum(total_closed_score) / len(total_closed_score)
    else:
        eval_closed_acc = 0
    if len(total_open_score) != 0:
        eval_open_acc = sum(total_open_score) / len(total_open_score)
    else:
        eval_open_acc = 0

    # empty cache
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    # # save the evaulation result
    # if args.model_recover_path is None:
    #     save_file = f"vqa_tr_random{args.random_bootstrap_testnum}_finetune_only_50ep.pickle"
    # elif args.model_recover_path.split("/")[-1].split(".")[0] == "pytorch_model":
    #     save_file = f"final_vqa_tr_ft_random{args.random_bootstrap_testnum}_{args.model_recover_path.split('/')[-3]}_{args.model_recover_path.split('/')[-2]}_{args.vqa_rad}_100ep.pickle"
    # else:
    #     save_file = f"vqa_random{args.random_bootstrap_testnum}_{args.model_recover_path.split('/')[-2]}_{args.model_recover_path.split('.')[-2]}ep.pickle"
    # with open(save_file, "wb") as f:
    #     pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return eval_vqa_acc, eval_closed_acc, eval_open_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")

    # model
    parser.add_argument("--model_recover_path", default=None, type=str, help="The file of fine-tuned pretraining model. ex)'./pretrained_model/pytorch_model.bin'")
    parser.add_argument("--len_vis_input", type=int, default=256, help="The length of visual token input")
    parser.add_argument("--img_encoding", type=str, default="fully_use_cnn", choices=["random_sample", "fully_use_cnn"])
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--img_hidden_sz", type=int, default=2048, help="Whether to use amp for fp16")
    parser.add_argument("--img_postion", default=True, help="It will produce img_position.")
    parser.add_argument("--drop_prob", default=0.1, type=float)

    # truncate_config for input
    parser.add_argument("--max_len_b", type=int, default=253, help="Truncate_config: maximum length of segment B. (Language)")
    parser.add_argument("--trunc_seg", type=str, default="b", help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument("--always_truncate_tail", action="store_true", help="Truncate_config: Whether we should always truncate tail.")

    # dataset
    parser.add_argument("--vqa_dataset", default="vqa-rad", type=str, choices=["vqa-rad", "vqa-mimic"])
    # dataset (vqa-mimic)
    parser.add_argument("--vqa_mimic", default="all", type=str, choices=["all", "verify", "choose", "query"])
    # dataset (vqa-rad)
    parser.add_argument("--vqa_rad", default="all", type=str, choices=["all", "chest", "head", "abd"])

    # training
    parser.add_argument("--do_train", action="store_true", default=True, help="Whether to run training. This should ALWAYS be set to True.")
    parser.add_argument("--num_train_epochs", default=50, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)

    parser.add_argument("--sche_mode", type=str, default="warmup_linear", help="warmup_linear | warmup_constant | warmup_cosine")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")  # 3e-5
    parser.add_argument("--label_smoothing", default=0, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="The weight decay rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--fp16", action="store_true", default=False, help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--fp32_embedding", action="store_true", default=False, help="Whether to use 32-bit float precision instead of 32-bit for embeddings")

    # NOTE: NOT USED ANYMORE
    # parser.add_argument("--tasks", default="vqa", help="report_generation | vqa")
    # parser.add_argument("--from_scratch", action="store_true", default=False, help="Initialize parameters with random values (i.e., training from scratch).")
    # parser.add_argument("--mask_prob", default=0.0, type=float, help="Number of prediction is sometimes less than max_pred when sequence is short.")
    # parser.add_argument("--mlm_task", type=str, default=True, help="The model will train only mlm task!! | True | False")
    # parser.add_argument("--generation_dataset", default="mimic-cxr", type=str, help=["mimic-cxr, openi"])
    # parser.add_argument("--data_set", default="train", type=str, help="train | valid")
    # parser.add_argument("--finetune_decay", action="store_true", help="Weight decay to the original weights.")
    # parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

    # parser.add_argument("--bar", default=False, type=str, help="True or False")
    # parser.add_argument("--random_bootstrap_testnum", type=int, default=30, help="Random bootstrap num of Iteration")

    parser.add_argument("--vqa_eval", action="store_true", help="vqa_eval | True | False")

    # parser.add_argument("--max_pred", type=int, default=3, help="Max tokens of prediction.")
    # parser.add_argument(
    #     "--s2s_prob", default=0, type=float, help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!"
    # )
    # parser.add_argument("--bi_prob", default=1, type=float, help="Percentage of examples that are bidirectional LM.")
    # parser.add_argument("--new_segment_ids", default=False, action="store_true", help="Use new segment ids for bi-uni-directional LM.")

    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n" "0 (default value): dynamic loss scaling.\n" "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument("--amp", action="store_true", default=False, help="Whether to use amp for fp16")

    # logging / directory
    parser.add_argument("--wandb", action="store_true", default=False, help="Whether to use wandb logging")
    parser.add_argument("--wandb_entity_name", type=str, default="ehr-vqg")
    parser.add_argument("--wandb_project_name", type=str, default="phase1-qa-dataset-debugging-seongsu")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--image_root", type=str, default="../../data/mimic/re_512_3ch/Train")
    parser.add_argument("--output_dir", default="/home/edlab/ssbae/medvill/vqa_finetune", type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_file", default="training.log", type=str, help="The output directory where the log will be written.")

    parser.add_argument("--num_workers", default=8, type=int, help="Number of workers for the data loader.")
    parser.add_argument("--max_position_embeddings", type=int, default=None, help="max position embeddings")

    # parser.add_argument("--split", type=str, nargs="+", default=["train", "valid"])

    # parser.add_argument('--dist_url', default='file://[PT_OUTPUT_DIR]/nonexistent_file', type = str, help = 'url used to set up distributed training')
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    # parser.add_argument("--use_num_imgs", default=-1, type=int)
    # parser.add_argument("--max_drop_worst_ratio", default=0, type=float)
    # parser.add_argument("--drop_after", default=6, type=int)
    parser.add_argument("--relax_projection", action="store_true", help="Use different projection layers for tasks.")

    args = parser.parse_args()

    main(args)
