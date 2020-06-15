# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import os
import time
from pprint import pprint

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from dcase_util.data import ProbabilityEncoder
from data_utils.Desed import DESED
from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from TestModel import _load_model
from evaluation_measures import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics
from models.models import PS
from models.RNN import MS
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms


def adjust_learning_rate(ms_optimizer, ps_optimizer, rampup_value, rampdown_value=1):
    """ adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        rampdown_value: float, the float between 1 and 0 that should decrease linearly
    Returns:

    """
    # We commented parts on betas and weight decay to match 2nd system of last year from Orange
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    # beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    # beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    # weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value
    # decay = 0.8 ** (c_epoch // 10)
    # lr = cfg.max_learning_rate * decay

    for param_group in ms_optimizer.param_groups:
        param_group['lr'] = param_group['lr'] = lr
    for param_group in ps_optimizer.param_groups:
        param_group['lr'] = param_group['lr'] = lr

def train(train_loader, model, encoder, ms_optimizer, ps_optimizer, c_epoch, ema_model=None, mask_weak=None, mask_strong=None, adjust_lr=False):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        ema_model: torch.Module, student model, should return a weak and strong prediction
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        adjust_lr: bool, Whether or not to adjust the learning rate during training (params in config)
    """
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(class_criterion, consistency_criterion)

    meters = AverageMeterSet()
    log.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    for i, ((batch_input, ema_batch_input), target) in enumerate(train_loader):
        global_step = c_epoch * len(train_loader) + i
        rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup*len(train_loader))

        if adjust_lr:
            adjust_learning_rate(ms_optimizer, ps_optimizer, rampup_value)
        meters.update('lr', ms_optimizer.param_groups[0]['lr'])
        batch_input, ema_batch_input, target = to_cuda_if_available(batch_input, ema_batch_input, target)
        # Outputs
        enc_strong, enc_weak, feature = encoder(batch_input)
        noisy_feature = feature + to_cuda_if_available(torch.randn(feature.size()) / 6)
        
        feature = feature.detach()
        noisy_feature = noisy_feature.detach()
        
        strong_pred_ema, weak_pred_ema = ema_model(noisy_feature)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()
        strong_pred, weak_pred = model(feature)

        loss = None
        # Weak BCE Loss
        target_weak = target.max(-2)[0]  # Take the max in the time axis
        if mask_weak is not None:
            weak_class_loss = class_criterion(weak_pred[mask_weak], target_weak[mask_weak])
            ema_class_loss = class_criterion(weak_pred_ema[mask_weak], target_weak[mask_weak])
            loss = weak_class_loss

            if i == 0:
                log.debug(f"target: {target.mean(-2)} \n Target_weak: {target_weak} \n "
                          f"Target weak mask: {target_weak[mask_weak]} \n "
                          f"Target strong mask: {target[mask_strong].sum(-2)}\n"
                          f"weak loss: {weak_class_loss} \t rampup_value: {rampup_value}"
                          f"tensor mean: {batch_input.mean()}")
            meters.update('weak_class_loss', weak_class_loss.item())
            meters.update('Weak EMA loss', ema_class_loss.item())

        # Strong BCE loss
        if mask_strong is not None:
            strong_class_loss = class_criterion(strong_pred[mask_strong], target[mask_strong])
            meters.update('Strong loss', strong_class_loss.item())

            strong_ema_class_loss = class_criterion(strong_pred_ema[mask_strong], target[mask_strong])
            meters.update('Strong EMA loss', strong_ema_class_loss.item())

            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if ema_model is not None:
            consistency_cost = cfg.max_consistency_cost * rampup_value
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about strong predictions (all data)
            consistency_loss_strong = consistency_cost * consistency_criterion(strong_pred, strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about weak predictions (all data)
            consistency_loss_weak = consistency_cost *  consistency_criterion(weak_pred, weak_pred_ema)
            meters.update('Consistency weak', consistency_loss_weak.item())
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        ms_optimizer.zero_grad()
        ps_optimizer.zero_grad()
        loss.backward()
        ms_optimizer.step()
        ps_optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start
    log.info(f"Epoch: {c_epoch}\t Time {epoch_time:.2f}\t {meters}")
    return loss

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(1 - alpha, params.data)

def generate_label(pred):
    p = []
    for b in range(pred.size(0)):
        pred_bin = ProbabilityEncoder().binarization(pred[b].detach().cpu().numpy(),
                                                     binarization_type="global_threshold",
                                                     threshold=0.5)
        p.append(pred_bin)

    p = torch.FloatTensor(p)
    p = to_cuda_if_available(p)
    return p

def get_dfs(desed_dataset, nb_files=None, separated_sources=False):
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    audio_weak_ss = None
    audio_unlabel_ss = None
    audio_validation_ss = None
    audio_synthetic_ss = None
    if separated_sources:
        audio_weak_ss = cfg.weak_ss
        audio_unlabel_ss = cfg.unlabel_ss
        audio_validation_ss = cfg.validation_ss
        audio_synthetic_ss = cfg.synthetic_ss

    weak_df = desed_dataset.initialize_and_get_df(cfg.weak, audio_dir_ss=audio_weak_ss, nb_files=nb_files)
    unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel, audio_dir_ss=audio_unlabel_ss, nb_files=nb_files)
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic, audio_dir_ss=audio_synthetic_ss,
                                                       nb_files=nb_files, download=False)
    log.debug(f"synthetic: {synthetic_df.head()}")
    validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir,
                                                        audio_dir_ss=audio_validation_ss, nb_files=nb_files)
    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=1.0, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)
    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    log.debug(valid_synth_df.event_label.value_counts())

    data_dfs = {"weak": weak_df,
                "unlabel": unlabel_df,
                "synthetic": synthetic_df,
                "train_synthetic": train_synth_df,
                "valid_synthetic": valid_synth_df,
                "validation": validation_df,
                }

    return data_dfs


if __name__ == '__main__':
    torch.manual_seed(2020)
    np.random.seed(2020)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Baseline 2020")
    logger.info(f"Starting time: {datetime.datetime.now()}")
    parser = argparse.ArgumentParser(description="")

    f_args = parser.parse_args()
    pprint(vars(f_args))

    add_dir_model_name = "gl-mt-ema"

    store_dir = os.path.join("stored_data", add_dir_model_name)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(saved_pred_dir, exist_ok=True)

    n_channel = 1
    add_axis_conv = 0

    PS_kwargs = {"n_in_channel": n_channel, "nclass": len(cfg.classes), "attention": True,
                 "activation": "Relu",
                 "conv_dropout": 0.0,
                 "kernel_size": [5, 5, 3], "padding": [2, 2, 1], "stride": 3 * [1],
                 "nb_filters": [160, 160, 160],
                 "pooling": [[1, 8], [1, 4], [1, 4]]}

    pooling_time_ratio = 1  # 2 * 2

    out_nb_frames_1s = cfg.sample_rate / cfg.hop_size / pooling_time_ratio
    median_window = max(int(cfg.median_window_s * out_nb_frames_1s), 1)
    logger.debug(f"median_window: {median_window}")
    # ##############
    # DATA
    # ##############
    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                    compute_log=False)
    dfs = get_dfs(dataset)

    # Meta path for psds
    durations_validation = get_durations_df(cfg.validation, cfg.audio_validation_dir)
    durations_synth = get_durations_df(cfg.synthetic)
    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio)
    encod_func = many_hot_encoder.encode_strong_df

    # Normalisation per audio or on the full dataset
    if cfg.scaler_type == "dataset":
        transforms = get_transforms(cfg.max_frames, add_axis=add_axis_conv)
        weak_data = DataLoadDf(dfs["weak"], encod_func, transforms)
        unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms)
        train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms)
        scaler_args = []
        scaler_synth = Scaler()
        scaler = Scaler()
        # # Only on real data since that's our final goal and test data are real
        scaler.calculate_scaler(ConcatDataset([weak_data, unlabel_data]))
        scaler_synth.calculate_scaler(ConcatDataset([train_synth_data]))
        logger.debug(f"scaler mean: {scaler.mean_}")
    else:
        scaler_args = ["global", "min-max"]
        scaler = ScalerPerAudio(*scaler_args)

    transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv,
                                noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    transforms_synth = get_transforms(cfg.max_frames, scaler_synth, add_axis_conv,
                                noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)

    weak_data = DataLoadDf(dfs["weak"], encod_func, transforms, in_memory=cfg.in_memory)
    unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms, in_memory=cfg.in_memory_unlab)
    train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms_synth, in_memory=cfg.in_memory)
    valid_synth_data = DataLoadDf(dfs["validation"], encod_func, transforms_valid,
                                  return_indexes=True, in_memory=cfg.in_memory)
    logger.debug(f"len synth: {len(train_synth_data)}, len_unlab: {len(unlabel_data)}, len weak: {len(weak_data)}")

    list_dataset = [weak_data, train_synth_data, unlabel_data]
    batch_sizes = [cfg.batch_size//4, cfg.batch_size//4, cfg.batch_size//2]
    strong_mask = slice(cfg.batch_size//4, cfg.batch_size//2)
    weak_mask = slice(batch_sizes[0]+batch_sizes[1])  # Assume weak data is always the first one

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset, batch_sizes=batch_sizes)
    training_loader = DataLoader(concat_dataset, batch_sampler=sampler)
    valid_synth_loader = DataLoader(valid_synth_data, batch_size=cfg.batch_size)

    # ##############
    # Model
    # ##############
    
    PS_model = PS(**PS_kwargs)
    state = torch.load('stored_data/pretrained_model/feature_extractor.pt')
    PS_model.load_state_dict(state["PS"]["state_dict"])
    logger.info("Model loaded at epoch: {}".format(state["epoch"]))

    MS_model = MS()
    PS_total_params = sum(p.numel() for p in PS_model.parameters() if p.requires_grad)
    MS_total_params = sum(p.numel() for p in MS_model.parameters() if p.requires_grad)
    logger.info(PS_model)
    logger.info("number of parameters in the PS model: {}".format(PS_total_params))
    logger.info(MS_model)
    logger.info("number of parameters in the MS model: {}".format(MS_total_params))
    
    MS_model.apply(weights_init)

    MS_ema = MS()
    MS_ema.apply(weights_init)
    for param in MS_ema.parameters():
        param.detach_()

    optim_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
    ms_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, MS_model.parameters()), **optim_kwargs)
    ps_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, PS_model.parameters()), **optim_kwargs)
    bce_loss = nn.BCELoss()

    
    state = {
        'MS': {"name": MS_model.__class__.__name__,
                  'args': '',
                  "kwargs": '',
                  'state_dict': MS_model.state_dict()},
        'PS': {"name": PS_model.__class__.__name__,
                  'args': '',
                  "kwargs": PS_kwargs,
                  'state_dict': PS_model.state_dict()},
        'model_ema': {"name": MS_ema.__class__.__name__,
                      'args': '',
                      "kwargs": '',
                      'state_dict': MS_ema.state_dict()},
        'ms_optimizer': {"name": ms_optim.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': ms_optim.state_dict()},
        'ps_optimizer': {"name": ps_optim.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': ps_optim.state_dict()},
        "pooling_time_ratio": pooling_time_ratio,
        "scaler": {
            "type": type(scaler).__name__,
            "args": scaler_args,
            "state_dict": scaler.state_dict()},
        "many_hot_encoder": many_hot_encoder.state_dict(),
        "median_window": median_window,
        "desed": dataset.state_dict()
    }
    

    save_best_cb = SaveBest("sup")
    if cfg.early_stopping is not None:
        early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup", init_patience=cfg.es_init_wait)

    # ##############
    # Train
    # ##############
    results = pd.DataFrame(columns=["loss", "valid_synth_f1", "weak_metric", "global_valid"])
    for epoch in range(cfg.n_epoch):
        MS_model.train()
        MS_ema.train()
        PS_model.train()
        MS_model, MS_ema, PS_model = to_cuda_if_available(MS_model, MS_ema, PS_model)

        loss_value = train(training_loader, MS_model, PS_model, ms_optim, ps_optim, epoch,
                           ema_model=MS_ema, mask_weak=weak_mask, mask_strong=strong_mask, adjust_lr=cfg.adjust_lr)

        # Validation
        ema = MS_ema.eval()
        MS_m = MS_model.eval()
        PS_m = PS_model.eval()
        logger.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(PS_m, ema, MS_m, valid_synth_loader, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      median_window=median_window, save_predictions=None)
        # Validation with synthetic data (dropping feature_filename for psds)
        valid_synth = dfs["validation"].drop("feature_filename", axis=1)
        valid_synth_f1, psds_m_f1 = compute_metrics(predictions, valid_synth, durations_validation)

        # Update state
        state['MS']['state_dict'] = MS_model.state_dict()
        state['PS']['state_dict'] = PS_model.state_dict()
        state['model_ema']['state_dict'] = MS_ema.state_dict()
        state['ms_optimizer']['state_dict'] = ms_optim.state_dict()
        state['ps_optimizer']['state_dict'] = ps_optim.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_synth_f1
        state['valid_f1_psds'] = psds_m_f1

        # Callbacks
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "ms_epoch_" + str(epoch))
            torch.save(state, model_fname)

        if cfg.save_best:
            if save_best_cb.apply(valid_synth_f1):
                model_fname = os.path.join(saved_model_dir, "ms_best")
                torch.save(state, model_fname)
            results.loc[epoch, "global_valid"] = valid_synth_f1
        results.loc[epoch, "loss"] = loss_value.item()
        results.loc[epoch, "valid_synth_f1"] = valid_synth_f1

        if cfg.early_stopping:
            if early_stopping_call.apply(valid_synth_f1):
                logger.warn("EARLY STOPPING")
                break

