"""
update_notebooks.py — Add auto-save and reload cells to all training notebooks.
Run from: ~/nilm_project/
Usage: python3 update_notebooks.py
"""
import json
import os
from pathlib import Path

notebooks = [
    'notebooks/colab/01_train_cnn.ipynb',
    'notebooks/colab/02_train_gru.ipynb',
    'notebooks/colab/03_train_bigru.ipynb',
    'notebooks/colab/04_train_lstm.ipynb',
    'notebooks/colab/05_train_bilstm.ipynb',
    'notebooks/colab/06_train_cnn_lstm.ipynb',
    'notebooks/colab/07_train_nilmformer.ipynb',
    'notebooks/colab/08_train_biwave.ipynb',
]

# ============================================================
# AUTO-SAVE code to inject inside training loop
# Inserted after: best_state = {k: v.cpu().clone() ...}
# ============================================================
AUTOSAVE_CODE = (
    "\n"
    "        # Auto-save checkpoint every MR improvement\n"
    "        if is_best:\n"
    "            torch.save({\n"
    "                'model_name': MODEL_NAME,\n"
    "                'model_state_dict': best_state,\n"
    "                'best_epoch': best_epoch,\n"
    "                'best_val_mr': best_mr,\n"
    "                'n_params': n_params,\n"
    "                'norm_stats': {\n"
    "                    'agg_mean': norm_stats.agg_mean,\n"
    "                    'agg_std': norm_stats.agg_std,\n"
    "                    'appliance_max': norm_stats.appliance_max,\n"
    "                },\n"
    "            }, f'experiments/checkpoints/{MODEL_NAME}_best.pth')\n"
    "\n"
    "        # Auto-save history every 5 epochs\n"
    "        if epoch % 5 == 0 or is_best:\n"
    "            import json as _json\n"
    "            _hist = {**history,\n"
    "                'best_epoch': best_epoch,\n"
    "                'best_val_mr': best_mr,\n"
    "                'model': MODEL_NAME}\n"
    "            with open(f'experiments/results/{MODEL_NAME}_history.json','w') as _f:\n"
    "                _json.dump(_hist, _f)\n"
)

TARGET_LINE = "        best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}"

# ============================================================
# RELOAD cell source lines
# ============================================================
RELOAD_SOURCE = [
    "# Reload from Drive if session restarted\n",
    "import json, os, torch\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.dates as mdates\n",
    "import matplotlib.patches as mpatches\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from config import INPUT_CHANNELS, WINDOW_SIZE, N_APPLIANCES, APPLIANCE_NAMES, APPLIANCES\n",
    "from metrics import MetricsTracker\n",
    "from train import validate_one_epoch, get_model_registry\n",
    "from dataset import load_clean_df, split_train_val, build_dataloaders\n",
    "\n",
    "COLORS = {'kettle':'#D94040','fridge':'#2E9E5A',\n",
    "          'washing_machine':'#E8922A','dishwasher':'#7B4FBF','microwave':'#CC3399'}\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "plt.rcParams['axes.grid'] = True\n",
    "plt.rcParams['grid.alpha'] = 0.3\n",
    "\n",
    "# 1. Reload history\n",
    "hist_path = f'experiments/results/{MODEL_NAME}_history.json'\n",
    "assert os.path.exists(hist_path), f'No history at {hist_path} - run training first'\n",
    "with open(hist_path) as f:\n",
    "    history = json.load(f)\n",
    "best_epoch = history['best_epoch']\n",
    "best_mr = history['best_val_mr']\n",
    "total_time = history.get('training_time_seconds', 0)\n",
    "print(f'History: {len(history[\"epoch\"])} epochs | Best MR: {best_mr:.4f} @ ep{best_epoch}')\n",
    "\n",
    "# 2. Reload model\n",
    "ckpt_path = f'experiments/checkpoints/{MODEL_NAME}_best.pth'\n",
    "assert os.path.exists(ckpt_path), f'No checkpoint at {ckpt_path} - run training first'\n",
    "registry = get_model_registry()\n",
    "model = registry[MODEL_NAME]().to(DEVICE)\n",
    "ckpt = torch.load(ckpt_path, map_location=DEVICE)\n",
    "model.load_state_dict(ckpt['model_state_dict'])\n",
    "model.eval()\n",
    "n_params = sum(p.numel() for p in model.parameters())\n",
    "print(f'Model: {MODEL_NAME} | {n_params:,} params | epoch {ckpt[\"best_epoch\"]}')\n",
    "\n",
    "# 3. Reload data\n",
    "from preprocessing import load_ukdale_house, preprocess_house\n",
    "from dataset import save_clean_df\n",
    "cached = load_clean_df('UK-DALE', 1)\n",
    "if cached is not None:\n",
    "    clean_df = cached\n",
    "else:\n",
    "    raw_df = load_ukdale_house(house=1)\n",
    "    clean_df = preprocess_house(raw_df)\n",
    "    save_clean_df(clean_df, 'UK-DALE', 1)\n",
    "train_df, val_df = split_train_val(clean_df, val_fraction=0.15)\n",
    "_, val_loader, norm_stats = build_dataloaders(\n",
    "    train_df, val_df, batch_size=256, train_stride=120,\n",
    "    val_stride=480, num_workers=2, add_temporal_features=True)\n",
    "print(f'Data: {len(val_df):,} val rows')\n",
    "\n",
    "# 4. Collect predictions\n",
    "app_max = {a: float(APPLIANCES[a]['max_power']) for a in APPLIANCE_NAMES}\n",
    "tracker = MetricsTracker(APPLIANCE_NAMES, app_max)\n",
    "_, final_metrics = validate_one_epoch(model, val_loader, DEVICE, tracker)\n",
    "all_pp, all_tp, all_ps, all_ts = [], [], [], []\n",
    "with torch.no_grad():\n",
    "    for x, yp, ys in val_loader:\n",
    "        pp, ps, pg = model(x.to(DEVICE))\n",
    "        all_pp.append(pp.cpu().numpy())\n",
    "        all_tp.append(yp.numpy())\n",
    "        all_ps.append((torch.sigmoid(ps)>=0.5).float().cpu().numpy())\n",
    "        all_ts.append(ys.numpy())\n",
    "pred_power = np.concatenate(all_pp)\n",
    "true_power = np.concatenate(all_tp)\n",
    "pred_state = np.concatenate(all_ps)\n",
    "true_state = np.concatenate(all_ts)\n",
    "print(f'Predictions: {pred_power.shape[0]:,} samples')\n",
    "print('\\nFinal metrics:')\n",
    "tracker.print_table(final_metrics)\n",
]

RELOAD_MARKDOWN = [
    "## Reload from Drive\n",
    "**Run this cell if the session restarted** — restores all variables without retraining."
]


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }


def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }


def update_notebook(nb_path):
    print(f"\nProcessing: {nb_path}")

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False

    # ---- 1. Add auto-save inside training loop ----
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if TARGET_LINE in src and 'Auto-save checkpoint' not in src:
            new_src = src.replace(TARGET_LINE, TARGET_LINE + AUTOSAVE_CODE)
            cell['source'] = [new_src]
            print("  + Auto-save injected into training loop")
            changed = True

    # ---- 2. Add reload cell before Training Curves section ----
    already_has_reload = any(
        'Reload from Drive' in ''.join(c.get('source', []))
        for c in nb['cells']
    )

    if not already_has_reload:
        insert_idx = None
        for i, cell in enumerate(nb['cells']):
            src = ''.join(cell.get('source', []))
            if 'Training Curves' in src and cell['cell_type'] == 'markdown':
                insert_idx = i
                break

        if insert_idx is not None:
            nb['cells'].insert(insert_idx, make_code_cell(RELOAD_SOURCE))
            nb['cells'].insert(insert_idx, make_markdown_cell(RELOAD_MARKDOWN))
            print("  + Reload cell added before Training Curves section")
            changed = True
        else:
            print("  ! Could not find Training Curves section")

    if changed:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  Saved: {nb_path}")
    else:
        print("  Already up to date")


if __name__ == '__main__':
    for nb_path in notebooks:
        if Path(nb_path).exists():
            update_notebook(nb_path)
        else:
            print(f"NOT FOUND: {nb_path}")

    print("\nDone! All notebooks updated.")
    print("Commit with: git add . && git commit -m 'fix(notebooks): auto-save + reload cells'")
