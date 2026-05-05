import re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import mne 
from typing import List, Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix, accuracy_score
import warnings
from sklearn.model_selection import train_test_split
import os
import csv
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import iirnotch, butter, filtfilt
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import dataHandler
import postProcessing
np.set_printoptions(threshold=np.inf)

# ── CHANGE 1: Dataset paths ────────────────────────────────────────────────
# Update these to match where your files are on your Mac
FTR_DATASET_PATH   = '/Users/mikayla/Downloads/archive/seizure_256Hz_dataset'
SEIZURE_EVENTS_CSV = '/Users/mikayla/Downloads/archive/seizure_events.csv'
# ──────────────────────────────────────────────────────────────────────────


def compute_fold_stats(loader, batch_size=32, device="cpu"):
    """
    Computes per-channel mean & std over ALL windows in train_dataset
    in a streaming (batch-by-batch) way.
    Returns (mu, sigma) each torch.Tensor of shape (1, C, 1).
    """
    sum_c   = None
    sumsq_c = None
    total   = 0
    
    for Xb, _ in loader:
        B, C, L = Xb.shape
        xb = Xb.reshape(B, C, L).to(torch.float64)
        s  = xb.sum(dim=(0,2))
        ss = (xb * xb).sum(dim=(0,2))
        if sum_c is None:
            sum_c, sumsq_c = s, ss
        else:
            sum_c   += s
            sumsq_c += ss
        total += B * L

    mu_c  = sum_c   / total
    var_c = sumsq_c / total - mu_c**2
    std_c = torch.sqrt(torch.clamp(var_c, min=1e-8))
    mu  = mu_c.view(1, C, 1).float()
    std = std_c.view(1, C, 1).float()
    return mu, std


class EpiDeNet(nn.Module):
    def __init__(self, eeg_channels=23, eeg_out_channels=16, num_classes=2, p_dropout=0.5):
        super(EpiDeNet, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 4, (1,4), (1,1), padding='same'),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,8), stride=(1,8)),

            nn.Conv2d(4, 16, (1,16), (1,1), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),

            nn.Conv2d(16,16,(1,8),(1,1), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),

            nn.Conv2d(16,16,(16,1),(1,1), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,1), stride=(4,1)),

            nn.Conv2d(16, 16, (8,1),(1,1),padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(p_dropout)
        self.fcn = nn.Linear(16, num_classes)

    def forward(self, x1):
        out1 = self.net1(x1)
        out = self.fcn(out1)
        return out


class SSWCELoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        self.ce    = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)[:, 1]
        y_true = targets.float()
        TP = torch.sum(probs * y_true)
        FN = torch.sum((1 - probs) * y_true)
        TN = torch.sum((1 - probs) * (1 - y_true))
        FP = torch.sum(probs * (1 - y_true))
        sensitivity  = TP / (TP + FN + self.eps)
        specificity  = TN / (TN + FP + self.eps)
        loss = ce_loss \
             + self.alpha * (1.0 - specificity) \
             + self.beta  * (1.0 - sensitivity)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction="mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor):
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        mod = (1 - p_t) ** self.gamma
        alpha_factor = torch.ones_like(targets, dtype=torch.float).to(logits.device)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_factor * mod * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedFocalLoss(FocalLoss):
    def __init__(self, gamma, alpha, class_weights, reduction="mean"):
        super().__init__(gamma, alpha, reduction)
        self.class_weights = class_weights
    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weights.to(logits.device),
            reduction="none"
        )
        p_t = torch.exp(-ce)
        mod = (1 - p_t)**self.gamma
        alpha_factor = torch.where(targets==1, self.alpha, 1-self.alpha)
        loss = alpha_factor * mod * ce
        return loss.mean() if self.reduction=="mean" else loss.sum()


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        if x.dim() == 4:
            b, c, h, w = x.shape
            y = x.mean(dim=(2,3))
        else:
            b, c = x.shape
            y = x
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1) if x.dim()==4 else x * y


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksizes=(3,8,16)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, (1, k), padding='same')
            for k in ksizes
        ])
        self.bn   = nn.BatchNorm2d(out_ch * len(ksizes))
        self.relu = nn.ReLU()

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        y = torch.cat(outs, dim=1)
        y = self.bn(y)
        return self.relu(y)


class ResidualBlockSE(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, (1, kernel_size), padding='same')
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, (1, kernel_size), padding='same')
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SqueezeExcite(channels)

    def forward(self, x):
        identity = x
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.se(y)
        return self.relu(y + identity)


class SeizurePredictionNet(nn.Module):
    def __init__(self, eeg_channels=23, base_filters=16, num_classes=2, p_dropout=0.3):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, base_filters, (1, 5), padding=(0,2)),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
        )
        self.ms_block = MultiScaleTemporalBlock(base_filters, base_filters, ksizes=(3,8,16))
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(base_filters*3, base_filters, (eeg_channels,1)),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlockSE(base_filters, kernel_size=8),
            ResidualBlockSE(base_filters, kernel_size=8),
        )
        self.pool    = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p_dropout)
        self.fc      = nn.Linear(base_filters, num_classes)

    def forward(self, x):
        y = self.input_conv(x)
        y = self.ms_block(y)
        y = self.spatial_conv(y)
        y = self.res_blocks(y)
        y = self.pool(y).flatten(1)
        y = self.dropout(y)
        return self.fc(y)


name_of_run = 'All_Channels_Win5_Val10'


def run(subject='01', run_idx=0):

    # ── CHANGE 2: Load data from .ftr files + seizure_events.csv ──────────
    seizure_dict = dataHandler.parse_seizure_events_csv(SEIZURE_EVENTS_CSV)
    records      = dataHandler.build_file_records_from_ftr(
                       subject_id=f'chb{subject}',
                       seizure_dict=seizure_dict,
                       ftr_path=FTR_DATASET_PATH
                   )
    base_path = FTR_DATASET_PATH   # used for data extraction below
    # ──────────────────────────────────────────────────────────────────────

    preictal_duration_min          = 30
    interictal_preictal_buffer_min = 10
    postictal_buffer_min           = 10

    interictal = dataHandler.extract_interictal_data(
        records,
        preictal_duration_sec=preictal_duration_min * 60,
        interictal_buffer_sec=interictal_preictal_buffer_min * 60,
        postictal_duration_min=postictal_buffer_min
    )
    preictal = dataHandler.extract_preictal_segments_distance(
        records,
        preictal_duration_min=preictal_duration_min,
        postictal_duration_min=10,
        min_length_min=10
    )

     # ── DEBUG: show what data was found ──────────────────────────────────
    print(f"Interictal segments: { {k: len(v) for k,v in interictal.items()} }")
    print(f"Preictal segments: {len(preictal)}")
    # ─────────────────────────────────────────────────────────────────────

    abs_times_per_file = dataHandler.helper_absolute_time(records)
    file_abs_starts    = {fn: info["abs_start"] for fn, info in abs_times_per_file.items()}
    tn_chunks          = dataHandler.combine_interictal_preictal(interictal, preictal)

    # ── CHANGE 3: get_subject_channels now reads .ftr column headers ──────
    always_present_channels_for_subject = dataHandler.get_subject_channels(
        records, base_path
    )
    # ──────────────────────────────────────────────────────────────────────

    interictal_data_list, preictal_data_list = dataHandler.process_tn_chunks(
        tn_chunks, base_path, always_present_channels_for_subject
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id  = f"{subject}_run{run_idx}_{int(time.time())}"
    log_dir  = f"runs/{name_of_run}/{run_id}"
    ckpt_dir = f"checkpoints/{run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # hyperparams
    n_chunks      = len(interictal_data_list)
    window_len    = 10
    sampling_rate = 0.25
    val_frac      = 0.1
    batch_size    = 32
    epochs        = 100
    patience      = 30
    lr            = 1e-4
    weight_decay  = 1e-4
    dropout       = 0.5
    k             = 8
    n             = 10
    R             = 30
    SPH           = 5
    SOP           = 30
    threshold_inference_probability = 0.5
    alpha         = 0.5
    beta          = 1.0

    hparams = {
        'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay,
        'dropout': dropout, 'epochs': epochs, 'patience': patience,
        'k': k, 'n': n, 'R': R, 'SPH': SPH, 'SOP': SOP,
    }
    writer.add_hparams(hparams, {})

    all_fold_metrics      = []
    all_fold_metrics_post = []
    test_results          = []
    test_results_post     = []
    training_history      = []

    total_sens = 0
    total_fa   = 0
    total_ih   = 0.0
    lead_times = []
    total_alarms    = []
    y_pred_forecast = []
    run_summaries   = []

    for fold in range(n_chunks):

        test_pair   = (interictal_data_list[fold], preictal_data_list[fold])
        test_loader = dataHandler.make_test_loader(test_pair, window_len, batch_size)

        train_pairs = [
            (interictal_data_list[i], preictal_data_list[i])
            for i in range(n_chunks) if i != fold
        ]
        train_loader, val_loader = dataHandler.make_train_val_loaders_random_val(
            train_pairs, window_len, val_frac=val_frac, batch_size=batch_size
        )

        mu, std = compute_fold_stats(train_loader, batch_size=batch_size)
        mu, std = mu.to(device), std.to(device)

        N_CHANNELS = len(always_present_channels_for_subject)
        model = EpiDeNet(eeg_channels=N_CHANNELS, p_dropout=dropout).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = SSWCELoss(alpha, beta)

        global_step  = 0
        best_val_loss = float('inf')
        best_f1       = float('-inf')
        best_val_acc  = 0.0
        wait          = 0

        for epoch in range(epochs):
            print(f"[Fold {fold:2d}] Epoch {epoch+1}/{epochs}")

            model.train()
            train_losses = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                Xb = (Xb - mu) / std
                Xb = Xb.unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                writer.add_scalar(f"train/fold{fold}/batch_loss", loss.item(), global_step)
                global_step += 1

            avg_train_loss = np.mean(train_losses)
            writer.add_scalar(f"train/fold{fold}/epoch_loss", avg_train_loss, epoch)

            model.eval()
            val_losses = []
            y_true_val, y_score_val, y_pred_val = [], [], []
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device), yv.to(device)
                    Xv = (Xv - mu) / std
                    Xv = Xv.unsqueeze(1)
                    val_losses.append(criterion(model(Xv), yv).item())
                    logits = model(Xv)
                    probs  = torch.softmax(logits, dim=1)[:,1]
                    preds  = (probs >= threshold_inference_probability).long()
                    y_true_val .append(yv.cpu().numpy())
                    y_score_val.append(probs.cpu().numpy())
                    y_pred_val .append(preds.cpu().numpy())

            y_true_val  = np.concatenate(y_true_val)
            y_score_val = np.concatenate(y_score_val)
            y_pred_val  = np.concatenate(y_pred_val)

            avg_val_loss = np.mean(val_losses)
            val_auc      = roc_auc_score(y_true_val, y_score_val)
            val_acc      = accuracy_score(y_true_val, y_pred_val)
            val_sens     = recall_score(y_true_val, y_pred_val, pos_label=1)
            val_spec     = recall_score(y_true_val, y_pred_val, pos_label=0)
            val_f1       = f1_score(y_true_val, y_pred_val, pos_label=1)
            val_fpr      = 1 - val_spec

            writer.add_scalar(f"val/epoch_loss/fold{fold}", avg_val_loss, epoch)
            writer.add_scalar(f"val/auc/fold{fold}",        val_auc,      epoch)
            writer.add_scalar(f"val/accuracy/fold{fold}",   val_acc,      epoch)
            writer.add_scalar(f"val/sensitivity/fold{fold}", val_sens,    epoch)
            writer.add_scalar(f"val/fpr/fold{fold}",         val_fpr,     epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                wait = 0
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"best_fold{fold}.pt"))
            else:
                wait += 1
                if wait >= patience:
                    print(f"[Fold {fold:2d}] Early stopping at epoch {epoch}")
                    break

        ckpt_path = os.path.join(ckpt_dir, f"best_fold{fold}.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        y_true_all    = []
        y_score_all   = []
        y_pred_all    = []
        y_pred_kn_all = []
        test_losses   = []

        with torch.no_grad():
            for Xt, yt in test_loader:
                Xt, yt = Xt.to(device), yt.to(device)
                Xt = (Xt - mu) / std
                Xt = Xt.unsqueeze(1)
                logits = model(Xt)
                probs  = torch.softmax(logits, dim=1)[:,1]
                preds  = (probs >= threshold_inference_probability).long()
                loss   = criterion(logits, yt)
                test_losses.append(loss.item())
                y_true_all   .append(yt.cpu().numpy())
                y_score_all  .append(probs.cpu().numpy())
                y_pred_all   .append(preds.cpu().numpy())
                y_pred_kn_all.append(preds.cpu().numpy())

        y_true    = np.concatenate(y_true_all)
        y_score   = np.concatenate(y_score_all)
        y_pred    = np.concatenate(y_pred_all)
        y_pred_kn = np.concatenate(y_pred_kn_all)

        y_pred_kn   = postProcessing.k_of_n_filter(y_pred_kn, k=k, n=n)
        y_pred_post = postProcessing.apply_refractory(window_len=window_len, alarm_seq=y_pred_kn, R=R)
        y_pred_riskLevels = postProcessing.risk_levels(y_pred, n=n)

        auc  = roc_auc_score(y_true, y_score)
        sens = recall_score(y_true, y_pred, pos_label=1)
        spec = recall_score(y_true, y_pred, pos_label=0)
        f1   = f1_score(y_true, y_pred, pos_label=1)
        fpr  = 1 - spec
        acc  = accuracy_score(y_true, y_pred)

        sens_post = recall_score(y_true, y_pred_post, pos_label=1)
        spec_post = recall_score(y_true, y_pred_post, pos_label=0)
        fpr_post  = 1 - spec_post
        acc_post  = accuracy_score(y_true, y_pred_post)

        all_fold_metrics.append((auc, sens, spec, fpr, acc, f1))
        all_fold_metrics_post.append((auc, sens_post, spec_post, fpr_post, acc_post))

        sens, fp, interictal_seconds, lead_time, alarms, boundry_window_idx, first_preictal_window_idx = \
            postProcessing.evaluate_seizure_fold_SPH_SOP_complex(
                y_pred_post,
                tn_chunks[fold][0],
                tn_chunks[fold][1],
                window_len=window_len,
                SPH=SPH*60,
                SOP=SOP*60,
                file_abs_starts=file_abs_starts
            )

        total_alarms.append(alarms)
        total_sens += sens
        total_fa   += fp
        total_ih   += interictal_seconds
        if not np.isnan(lead_time):
            lead_times.append(lead_time)

        avg_test_loss = np.mean(test_losses)
        alarm_idx = [e["window-index"] for e in alarms]
        times = dataHandler.chunk_absolute_times(records, tn_chunks[fold])

        y_pred_forecast.append({
            "fold":                    fold,
            "first_preictal_window_idx": first_preictal_window_idx,
            "boundary_window_idx":     boundry_window_idx,
            "alarm_window_idx":        alarm_idx,
            "window_size":             window_len,
            "window_predictions":      y_pred,
            "risk":                    y_pred_riskLevels,
            "start_end_times_of_files": times,
            "interictal_chunks":       tn_chunks[fold][0],
            "preictal_chunks":         tn_chunks[fold][1],
            "file_abs_starts":         file_abs_starts
        })

    writer.close()

    all_fold_metrics = np.array(all_fold_metrics)
    mean_auc, mean_sens, mean_spec, mean_fpr, mean_acc, mean_f1 = all_fold_metrics.mean(axis=0)
    std_auc,  std_sens,  std_spec,  std_fpr,  std_acc,  std_f1  = all_fold_metrics.std(axis=0)

    all_fold_metrics_post = np.array(all_fold_metrics_post)
    mean_auc_post, mean_sens_post, mean_spec_post, mean_fpr_post, mean_acc_post = all_fold_metrics_post.mean(axis=0)
    std_auc_post,  std_sens_post,  std_spec_post,  std_fpr_post,  std_acc_post  = all_fold_metrics_post.std(axis=0)

    overall_sens = total_sens
    overall_fpr  = total_fa / total_ih if total_ih > 0 else float('nan')
    mean_lead    = float(np.mean(lead_times)) if lead_times else float('nan')
    std_lead     = float(np.std(lead_times))  if lead_times else float('nan')

    run_summaries.append({
        "run":       run_idx,
        "mean_auc":  mean_auc,  "std_auc":  std_auc,
        "mean_sens": mean_sens, "std_sens": std_sens,
        "mean_spec": mean_spec, "std_spec": std_spec,
        "mean_fpr":  mean_fpr,  "std_fpr":  std_fpr,
        "mean_acc":  mean_acc,  "std_acc":  std_acc,
        "mean_f1":   mean_f1,   "std_f1":   std_f1,
        "overall_sens_seizures": (overall_sens / n_chunks),
        "total_fa":    total_fa,
        "total_ih":    total_ih / 3600,
        "overall_fpr": overall_fpr * 3600,
        "mean_lead":   mean_lead / 60,
        "std_lead":    std_lead  / 60,
    })

    df_training_history   = pd.DataFrame(training_history)
    df_test_results       = pd.DataFrame(test_results)
    df_test_results_post  = pd.DataFrame(test_results_post)
    df_risk_levels        = pd.DataFrame(y_pred_forecast)
    df_test_results       = pd.concat([df_test_results, df_test_results_post], axis=0)
    df_training_history.to_csv(f"training_history_{subject}.csv", index=False)
    df_test_results.to_csv(f"test_results_{subject}.csv", index=False)

    out_dir = Path(name_of_run)
    if out_dir.exists() and out_dir.is_file():
        out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name_of_run}_risk_levels_Subject_{subject}_run{run_idx}.csv"
    df_risk_levels.to_csv(out_path, index=False)
    print("Saved metrics to training_history.csv")

    summary_txt = []
    summary_txt.append(f"Subject: {subject}, Window Size: {window_len}s, Dist Interictal/Preictal: {interictal_preictal_buffer_min}min, Postictal: {postictal_buffer_min}, Epochs: {epochs}, Batch: {batch_size}, k: {k}, n: {n}, R: {R}min, SPH: {SPH}min, SOP: {SOP}min")
    summary_txt.append("=== Summary across folds ===")
    summary_txt.append(f"AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    summary_txt.append(f"Sensitivity = {mean_sens:.3f} ± {std_sens:.3f}")
    summary_txt.append(f"Specificity = {mean_spec:.3f} ± {std_spec:.3f}")
    summary_txt.append(f"False-alarm rate = {mean_fpr:.3f} ± {std_fpr:.3f}")
    summary_txt.append(f"Acc = {mean_acc:.3f} ± {std_acc:.3f}")
    summary_txt.append("")
    summary_txt.append("=== Summary across folds postprocessing ===")
    summary_txt.append(f"AUC = {mean_auc_post:.3f} ± {std_auc_post:.3f}")
    summary_txt.append(f"Sensitivity = {mean_sens_post:.3f} ± {std_sens_post:.3f}")
    summary_txt.append(f"Specificity = {mean_spec_post:.3f} ± {std_spec_post:.3f}")
    summary_txt.append(f"False-alarm rate = {mean_fpr_post:.3f} ± {std_fpr_post:.3f}")
    summary_txt.append(f"Acc = {mean_acc_post:.3f} ± {std_acc_post:.3f}")
    summary_txt.append("")
    summary_txt.append(f"Across {n_chunks} folds:")
    summary_txt.append(f"  Seizures predicted: {overall_sens}/{n_chunks}")
    summary_txt.append(f"  Total false alarms: {total_fa}")
    summary_txt.append(f"  Total interictal hours: {total_ih/3600:.2f}h")
    summary_txt.append(f"  Overall FPR: {overall_fpr*3600:.2f} per hour")
    summary_txt.append(f"  Mean lead time: {mean_lead/60:.2f} ± {std_lead/60:.2f} minutes")
    summary_txt.append("=== Per File summary ===")
    for fold in range(n_chunks):
        summary_txt.append(f"===Used Files in Fold {fold}:====")
        summary_txt.append(f"==Interictal files: ")
        for c in tn_chunks[fold][0]:
            summary_txt.append(f"File: {c['file']}: start: {c['start']} end: {c['end']}")
        summary_txt.append(f"==Preictal files: ")
        if len(tn_chunks[fold][1]) == 2:
            summary_txt.append(f"File: {tn_chunks[fold][1][0]['file']}: start: {tn_chunks[fold][1][0]['start']} end: {tn_chunks[fold][1][0]['end']}")
            summary_txt.append(f"File: {tn_chunks[fold][1][1]['file']}: start: {tn_chunks[fold][1][1]['start']} end: {tn_chunks[fold][1][1]['end']}")
        else:
            summary_txt.append(f"File: {tn_chunks[fold][1]['file']}: start: {tn_chunks[fold][1]['start']} end: {tn_chunks[fold][1]['end']}")
        summary_txt.append(f"==Total Alarms in This fold: {len(total_alarms[fold])}")
        for e in total_alarms[fold]:
            summary_txt.append(f"Alarm in {e['segment']:10s} {e['file']:15s} at local {e['local_time']:.1f}s (abs {e['abs_time']:.1f})")

    path = "final_metrics.txt"
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode) as fout:
        if mode == "a":
            fout.write("\n")
        fout.write("\n".join(summary_txt))
        fout.write("\n")

    print("Wrote summary metrics to final_metrics.txt")
    return run_summaries


def run_multiple_subjects():
    subjects = ['01','02','03','04','05','06','07','08','09','10','11',
                '13','14','15','16','17','18','19','20','21','22','23']
    runs_per_subject  = 4
    all_subject_results = []

    for subj in subjects:
        print(f"\n=== Running Subject {subj} ===")
        df_runs = run_per_subj(subj, runs_per_subject)
        avg = df_runs.mean(numeric_only=True)
        std = df_runs.std(numeric_only=True)
        result = {
            "subject":   subj,
            "runs":      runs_per_subject,
            "mean_auc":  avg["mean_auc"],  "std_auc":  std["mean_auc"],
            "mean_sens": avg["mean_sens"], "std_sens": std["mean_sens"],
            "mean_spec": avg["mean_spec"], "std_spec": std["mean_spec"],
            "mean_fpr":  avg["mean_fpr"],  "std_fpr":  std["mean_fpr"],
            "mean_acc":  avg["mean_acc"],  "std_acc":  std["mean_acc"],
            "mean_f1":   avg["mean_f1"],   "std_f1":   std["mean_f1"],
            "overall_sens_seizures": avg["overall_sens_seizures"],
            "total_fa":    avg["total_fa"],
            "total_ih":    avg["total_ih"],
            "overall_fpr": avg["overall_fpr"],
            "mean_lead":   avg["mean_lead"],
            "std_lead":    std["mean_lead"]
        }
        all_subject_results.append(result)

    df_summary = pd.DataFrame(all_subject_results)
    df_summary.to_csv(f"per_subject_summary_{name_of_run}.csv", index=False)


def run_per_subj(subj, runs_per_subject):
    run_summaries = []
    for run_idx in range(runs_per_subject):
        summary = run(subj, run_idx)
        run_summaries.extend(summary)
    print(run_summaries)
    return pd.DataFrame(run_summaries)


def load_for_inference(checkpoint_path: str,
                       n_channels: int = 8,
                       dropout: float = 0.0,
                       device: str = 'cpu') -> EpiDeNet:
    """
    Load a trained EpiDeNet checkpoint for real-time ADS1299 inference.
    dropout=0.0 at inference time (deterministic output).
    """
    m = EpiDeNet(eeg_channels=n_channels, p_dropout=dropout).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    m.load_state_dict(state)
    m.eval()
    return m


if __name__ == "__main__":
    # To run all subjects:
    # run_multiple_subjects()

    # To run a single subject first (recommended for testing):
    run('01', 0)