import re
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import csv
from scipy.stats import spearmanr, gmean
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path

def extract_params(
        string,
        pattern=r'lr=(?P<learning_rate>[\d\.e-]+)_wd=(?P<weight_decay>[\d\.e-]+)_bs=(?P<num_gpus>\d+)-(?P<local_batch_size>\d+)'):
    match = re.match(pattern, string)
    if match:
        hyper_params = match.groupdict()
        
        param_key = ""
        for k, v in hyper_params.items():
            if k not in ["num_gpus", "local_batch_size"]:
                param_key += f'{k}={v}-'

        num_gpus = int(hyper_params['num_gpus'])
        local_bs = int(hyper_params['local_batch_size'])
        param_key += f'bs={num_gpus*local_bs}'

        return param_key
    else:
        return None

######## draw scatters #############
def draw(preds:np.ndarray, gts: np.ndarray, save_path: str, **kwargs):
    fig_title = ''
    for k, v in kwargs.items():
        fig_title = fig_title + f'_{str(k)}={str(v)}'

    preds = preds.tolist()
    gts = gts.tolist()

    min_val = min(preds + gts)
    max_val = max(preds + gts)
    plt.figure(figsize=(5, 5))
    plt.scatter(preds, gts, s=10)
    plt.plot(np.linspace(min_val, max_val, 100), np.linspace(min_val, max_val, 100), linestyle='--', color='red')
    plt.ylabel('real')
    plt.xlabel('prediction')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.title(fig_title)
    plt.savefig(f'{save_path}/res.png') # TODO: NOTE TO Change
    plt.clf()
    plt.close()

def main():
    all_tasks = [
        "castelli_eform_small",
        "mp_gvrh_small",
        "mp_gvrh_large",
        "mp_dielectric_small",
        "mp_dielectric_large",
        "mp_phonons_small",
        "mp_phonons_large",
    ]

    root_path = 'record_extrapolation_v2/mex-0.75_cos-mlp_nce-gradclip=10'
    # root_path = 'record_shared_etp_v3/mex-100'

    csvfile = open(os.path.join(root_path, 'result.csv'), 'w', newline='') # TODO: NOTE TO Change
    writer = csv.writer(csvfile)
    writer.writerow(['tasks'] + all_tasks)

    ROOT = Path(root_path)
    res_files = ROOT.rglob("ocp_predictions.npz") # TODO: NOTE TO Change

    params_task_metrics = {} # e.g. {params: {'task_name': {'mae': [], 'spearman': [], }}}
    for f in tqdm(res_files):
        # ========== compute matrics and draw ==============
        res = np.load(f)
        preds = np.array(res['y']).reshape(-1)
        gts = np.array(res['y_gts']).reshape(-1)

        maes_all = np.abs(preds - gts)
        mae = np.mean(maes_all)
        spear, _ = spearmanr(preds, gts)
        # erro geometric mean
        maes_all_wo_zero = np.where(maes_all == 0, 1e-10, maes_all)
        egm = gmean(maes_all_wo_zero)

        draw(preds, gts, f.parent, mae=round(mae, 4), gm=round(egm, 4))
        # ==================== end ========================

        relative_path_parts = f.relative_to(root_path).parts
        
        # extract hyper-paraemters str and task name
        hyper_params_str = None
        task_name = None
        for part in relative_path_parts:
            tmp = extract_params(
                part,
                pattern=r'(?P<model_name>[\w-]+)_lr=(?P<learning_rate>[\d\.e-]+)_wd=(?P<weight_decay>[\d\.e-]+)_bs=(?P<num_gpus>\d+)-(?P<local_batch_size>\d+)'
            )
            if tmp is not None:
                hyper_params_str = tmp
            
            if part in all_tasks:
                task_name = part
        if hyper_params_str is None or task_name is None:
            print(f'bad file: {f}', flush=True)
            continue
        
        if hyper_params_str not in params_task_metrics:
            params_task_metrics[hyper_params_str] = defaultdict(lambda: defaultdict(list))
        existing_task_dict = params_task_metrics[hyper_params_str][task_name] # {} or {'mae': [], ...}
        existing_task_dict['maes'].append(mae)
        existing_task_dict['spearmans'].append(spear)
        existing_task_dict['gms'].append(egm)
        params_task_metrics[hyper_params_str][task_name] = existing_task_dict
    
    for params, task_res in params_task_metrics.items():
        writer.writerow([])
        writer.writerow([params])

        maes, spearmans, gms = ['mae'], ['spearman'], ['gm']
        for task_name in all_tasks:
            if len(task_res[task_name]['maes']) == 0:
                maes.append('')
                spearmans.append('')
                gms.append('')
            else:
                maes.append(f"{np.round(np.mean(task_res[task_name]['maes']), 4)}({np.round(np.std(task_res[task_name]['maes']), 4)})")
                spearmans.append(f"{np.round(np.mean(task_res[task_name]['spearmans']), 4)}({np.round(np.std(task_res[task_name]['spearmans']), 4)})")
                gms.append(f"{np.round(np.mean(task_res[task_name]['gms']), 4)}({np.round(np.std(task_res[task_name]['gms']), 4)})")
        
        writer.writerow(maes)
        writer.writerow(spearmans)
        writer.writerow(gms)

if __name__ == "__main__":
    # predict_res()
    # move_res('record/tmp', 'record_extrapolation/eqv2-scratch_global-conr=0.5')
    main()