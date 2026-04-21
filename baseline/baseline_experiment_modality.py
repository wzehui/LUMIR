import os
import yaml
import numpy as np
import pandas as pd
import datetime
from backbone.data.session_dataset import SessionDataset
from backbone.eval.evaluation import Evaluation, EvaluationReport,metrics
from backbone.grurec.grurec_with_embeddings import GRURecWithEmbeddings
from backbone.transformer.sasrec.sasrec_with_embeddings import SASRecWithEmbeddings
from backbone.transformer.bert.bert_with_embeddings import BERTWithEmbeddings

import hashlib
from pathlib import Path
import json

from representation.utils_embedding import modality2embedding

### MODIFICATION 1 START
# Prediction cache
import pickle
### MODIFICATION 1 END

# === Config Load ===
# CONFIG_PATH = "../configs/representation_config.yaml"
CONFIG_PATH = "../configs/representation_config_ins.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Settings
INCLUDE = {
    # "LLM2GRU4Rec",
    # "LLM2SASRec",
    "LLM2BERT4Rec",
}

# DATASET_FILENAME = "../yelp/dataset/dataset.pickle"
DATASET_FILENAME = "../instagram/dataset/dataset.pickle"
EXPERIMENTS_FOLDER = "../results"

CORES = 20
EARLY_STOPPING_PATIENCE = 5
IS_VERBOSE = True
FILTER_PROMPT_ITEMS = True
MAX_SESSION_LENGTH_FOR_DECAY_PRECOMPUTATION = 500
PRED_BATCH_SIZE = 1000
PRED_SEEN = False
TRAIN_VAL_FRACTION = 0.1
TOP_Ks = [10, 20]

# Embedding paths
EMBEDDING_PATHS = {
    # "concate": "embedding_meta-review-photo_768_2048.csv.gz"
    # "concate": "embedding_meta-review-photo_768_2048_ins.csv.gz"
    # "weighted_concate": "embedding_weighted_concat_meta-review-photo_raw_4096_2048.csv.gz"
    # "weighted_concate": "embedding_reweighted_concat_meta-review-photo_fixedpca_4096_2048.csv.gz"
    # "flava": "embedding_flava_text3_meta-review-photo_raw_4096_2048.csv.gz"
    # "CoMM": "embedding_comm_2048.csv.gz"
    # "lumir": "lumir_epoch15.csv.gz"
    "lumir": "lumir_epoch15_ins.csv.gz"
    # "CoMM": "lumir_epoch15_comm.csv.gz"
}

# Main dataset (no embeddings)
dataset = SessionDataset.from_pickle(DATASET_FILENAME)

# Model mapping
MODEL_CLASS_MAPPING = {
    "LLM2GRU4Rec": GRURecWithEmbeddings,
    "LLM2SASRec": SASRecWithEmbeddings,
    "LLM2BERT4Rec": BERTWithEmbeddings,
}

# Best configs
BEST_CONFIGS = {
    "LLM2GRU4Rec": {
        # "concate": [
        #     {
        #         "N": 25,
        #         "activation": "gelu",
        #         "emb_dim": 352,
        #         "hidden_dim": 672,
        #         "fit_batch_size": 254,
        #         "optimizer_kwargs": {
        #             "lr": 0.0018776141247698085,
        #             "weight_decay": 0.0010704879723381286,
        #         },
        #     }  # Best value: 0.02584421918934782
        # ],
        # "weighted_concate": [
        #     {
        #         "N": 25,
        #         "activation": "relu",
        #         "emb_dim": 672,
        #         "hidden_dim": 736,
        #         "fit_batch_size": 288,
        #         "optimizer_kwargs": {
        #             "lr": 0.004216067715029253,
        #             "weight_decay": 0.0005075974813608443,
        #         },
        #     }  # Best value: 0.030531633773127412
        # ],
        # "CoMM": [
        #     {
        #         "N": 25,
        #         "activation": "gelu",
        #         "emb_dim": 736,
        #         "hidden_dim": 1056,
        #         "fit_batch_size": 416,
        #         "optimizer_kwargs": {
        #             "lr": 0.002012321304211598,
        #             "weight_decay": 0.0009585471904629478,
        #         },
        #     }  # Best value: 0.033232428331381625
        # ],
        "flava": [
            {
                "N": 25,
                "activation": "relu",
                "emb_dim": 512,
                "hidden_dim": 544,
                "fit_batch_size": 800,
                "optimizer_kwargs": {
                    "lr": 0.0032484889425194226,
                    "weight_decay": 0.0006680525478770676,
                },
            }  # Best value: 0.029993488432342664
        ],
        # "lumir": [
        #     {
        #         "N": 25,
        #         "activation": "gelu",
        #         "emb_dim": 608,
        #         "hidden_dim": 672,
        #         "fit_batch_size": 576,
        #         "optimizer_kwargs": {
        #             "lr": 0.001901535356471273,
        #             "weight_decay": 0.0006423085168201819,
        #         },
        #     }  # Best value: 0.03346078243663244
        # ],
        # "CoMM": [
        #     {
        #         "N": 25,
        #         "activation": "gelu",
        #         "emb_dim": 544,
        #         "hidden_dim": 576,
        #         "fit_batch_size": 512,
        #         "optimizer_kwargs": {
        #             "lr": 0.0013113694857123934,
        #             "weight_decay": 0.0005078748115340037,
        #         },
        #     }  # Best value: 0.03162061579175318
        # ]
    },
    "LLM2SASRec": {
        # "concate": [
        #     {
        #         "N": 25,
        #         "L": 6,
        #         "activation": "relu",
        #         "drop_rate": 0.31876246983684786,
        #         "emb_dim": 624,           # head_dim(104) × h(6)
        #         "fit_batch_size": 96,
        #         "h": 6,
        #         "optimizer_kwargs": {
        #             "lr": 8.764810229080126e-05,
        #             "weight_decay": 0.0006056760976805056,
        #         },
        #         "clipnorm": 2,
        #         "transformer_layer_kwargs": {"layout": "NFDR"},
        #     }  # Best value: 0.030032421432499522
        # ],
        # "weighted_concate": [
        #     {
        #         "N": 25,
        #         "L": 5,
        #         "activation": "gelu",
        #         "drop_rate": 0.4915596398777161,
        #         "emb_dim": 672,  # head_dim(112) × h(6)
        #         "fit_batch_size": 80,
        #         "h": 6,
        #         "optimizer_kwargs": {
        #             "lr": 4.905836261208772e-05,
        #             "weight_decay": 0.00012877577753329307,
        #         },
        #         "clipnorm": 9,
        #         "transformer_layer_kwargs": {"layout": "NFDR"},
        #     }  # Best value: 0.03461170362565737
        # ],
        # "CoMM": [
        #     {
        #         "N": 25,
        #         "L": 12,
        #         "activation": "gelu",
        #         "drop_rate": 0.15452445610784762,
        #         "emb_dim": 384,  # head_dim(64) × h(6)
        #         "fit_batch_size": 112,
        #         "h": 6,
        #         "optimizer_kwargs": {
        #             "lr": 8.089033927314092e-05,
        #             "weight_decay": 6.345451851412472e-05,
        #         },
        #         "clipnorm": 6,
        #         "transformer_layer_kwargs": {"layout": "NFDR"},
        #     }  # Best value: 0.031898776558618454
        # ],
        # "flava": [
        #     {
        #         "N": 25,
        #         "L": 4,
        #         "activation": "gelu",
        #         "drop_rate": 0.2547876138720607,
        #         "emb_dim": 480,  # head_dim(80) × h(6)
        #         "fit_batch_size": 128,
        #         "h": 6,
        #         "optimizer_kwargs": {
        #             "lr": 8.673286576632917e-05,
        #             "weight_decay": 7.322879032448214e-05,
        #         },
        #         "clipnorm": 7,
        #         "transformer_layer_kwargs": {"layout": "NFDR"},
        #     }  # Best value: 0.030312650238027348
        # ],
        # "lumir": [
        #     {
        #         "N": 25,
        #         "L": 8,
        #         "activation": "gelu",
        #         "drop_rate": 0.24426388100717972,
        #         "emb_dim": 320,           # head_dim(64) × h(5)
        #         "fit_batch_size": 112,
        #         "h": 5,
        #         "optimizer_kwargs": {
        #             "lr": 9.671751761230975e-05,
        #             "weight_decay": 2.182276288752246e-06,
        #         },
        #         "clipnorm": 9,
        #         "transformer_layer_kwargs": {"layout": "NFDR"},
        #     }  # Best value: 0.03890402234854255
        # ],
        # "lumir": [
        #     {
        #         "N": 25,
        #         "L": 11,
        #         "activation": "gelu",
        #         "drop_rate": 0.21668718912750046,
        #         "emb_dim": 960,           # head_dim(80) × h(12)
        #         "fit_batch_size": 112,
        #         "h": 12,
        #         "optimizer_kwargs": {
        #             "lr": 3.044132054647906e-05,
        #             "weight_decay": 7.55837049270248e-06,
        #         },
        #         "clipnorm": 7,
        #         "transformer_layer_kwargs": {"layout": "NFDR"},
        #     }  # Best value: 0.037867509
        # ],
        # "CoMM": [
        #     {
        #         "N": 25,
        #         "L": 5,
        #         "activation": "gelu",
        #         "drop_rate": 0.39362680935565525,
        #         "emb_dim": 640,  # head_dim(80) × h(8)
        #         "fit_batch_size": 80,
        #         "h": 8,
        #         "optimizer_kwargs": {
        #             "lr": 5.816023046432624e-05,
        #             "weight_decay": 0.00012171043280644375,
        #         },
        #         "clipnorm": 2,
        #         "transformer_layer_kwargs": {"layout": "NFDR"},
        #     }  # Best value: 0.03782929861994359
        # ],
    },
    "LLM2BERT4Rec": {
        # "concate": [
        #     {
        #         "N": 25,
        #         "L": 6,
        #         "activation": "gelu",
        #         "drop_rate": 0.25321037303056404,
        #         "emb_dim": 256,  # head_dim × h = 64 × 4
        #         "fit_batch_size": 128,
        #         "h": 4,
        #         "mask_prob": 0.29227250088950113,
        #         "optimizer_kwargs": {
        #             "lr": 8.224192411036129e-06,
        #             "weight_decay": 1.0329738631009728e-05,
        #         },
        #         "clipnorm": 48,
        #         "transformer_layer_kwargs": {"layout": "FDRN"},
        #     }  # Best value: 0.03606599307124843
        # ],
        # "concate": [
        #     {
        #         "N": 25,
        #         "L": 7,
        #         "activation": "relu",
        #         "drop_rate": 0.3772751101857168,
        #
        #         # head_dim × h = 80 × 7
        #         "emb_dim": 560,
        #
        #         "fit_batch_size": 80,
        #         "h": 7,
        #         "mask_prob": 0.28483987053875504,
        #
        #         "optimizer_kwargs": {
        #             "lr": 7.87174110262265e-06,
        #             "weight_decay": 3.904567767693391e-05,
        #         },
        #
        #         "clipnorm": 10,
        #         "transformer_layer_kwargs": {"layout": "FDRN"},
        #     }
        # ],  # ins
        # "weighted_concate": [
        #     {
        #         "N": 25,
        #         "L": 8,
        #         "activation": "relu",
        #         "drop_rate": 0.4606051090981257,
        #
        #         # head_dim × h = 96 × 5
        #         "emb_dim": 480,
        #
        #         "fit_batch_size": 64,
        #         "h": 5,
        #         "mask_prob": 0.25673988848637647,
        #
        #         "optimizer_kwargs": {
        #             "lr": 6.139423598363368e-06,
        #             "weight_decay": 4.402572999664845e-05,
        #         },
        #
        #         "clipnorm": 31,
        #         "transformer_layer_kwargs": {"layout": "FDRN"},
        #     }   # Best value: 0.037729657606505644
        # ],
        # "weighted_concate": [
        #     {
        #         "N": 25,
        #         "L": 7,
        #         "activation": "gelu",
        #         "drop_rate": 0.36184498074973886,
        #
        #         # head_dim × h = 48 × 6
        #         "emb_dim": 288,
        #
        #         "fit_batch_size": 96,
        #         "h": 6,
        #         "mask_prob": 0.2570918209455599,
        #
        #         "optimizer_kwargs": {
        #             "lr": 6.842363989421189e-06,
        #             "weight_decay": 4.2861406302643726e-05,
        #         },
        #
        #         "clipnorm": 19,
        #         "transformer_layer_kwargs": {"layout": "FDRN"},
        #     }  # Best value: 0.038714962307848926
        # ],
        # "flava": [
        #     {
        #         "N": 25,
        #         "L": 5,
        #         "activation": "relu",
        #         "drop_rate": 0.41698096911528393,
        #
        #         # head_dim × h = 48 × 5
        #         "emb_dim": 240,
        #
        #         "fit_batch_size": 128,
        #         "h": 5,
        #         "mask_prob": 0.25694502772328776,
        #
        #         "optimizer_kwargs": {
        #             "lr": 8.368077183882665e-06,
        #             "weight_decay": 6.629376125475801e-06,
        #         },
        #
        #         "clipnorm": 35,
        #         "transformer_layer_kwargs": {"layout": "FDRN"},
        #     }   # Best value: 0.03546848045820686
        # ],
        # "CoMM": [
        #     {
        #         "N": 25,
        #         "L": 8,
        #         "activation": "gelu",
        #         "drop_rate": 0.2631659950842709,
        #
        #         # head_dim × h = 80 × 5
        #         "emb_dim": 400,
        #
        #         "fit_batch_size": 80,
        #         "h": 5,
        #         "mask_prob": 0.22261714958293688,
        #
        #         "optimizer_kwargs": {
        #             "lr": 6.707737168560691e-06,
        #             "weight_decay": 2.006982325704669e-05,
        #         },
        #
        #         "clipnorm": 12,
        #         "transformer_layer_kwargs": {"layout": "FDRN"},
        #     }   # Best value: 0.037001627881705204
        # ],
        # "lumir": [
        #     {
        #         "N": 25,
        #         "L": 6,
        #         "activation": "relu",
        #         "drop_rate": 0.39018300621622615,
        #         "emb_dim": 576,  # head_dim(72) × h(8)
        #         "fit_batch_size": 80,
        #         "h": 8,
        #         "mask_prob": 0.29006447501450106,
        #         "optimizer_kwargs": {
        #             "lr": 5.009822428306333e-06,
        #             "weight_decay": 4.3005416458197435e-05,
        #         },
        #         "clipnorm": 44,
        #         "transformer_layer_kwargs": {"layout": "FDRN"},
        #     }  # Optuna best value: 0.050056043569725685
        # ],
        "lumir": [
            {
                "N": 30,
                "L": 5,
                "activation": "gelu",
                "drop_rate": 0.4463625724615874,
                "emb_dim": 104 * 8,  # head_dim(104) × h(8) = 832
                "fit_batch_size": 128,
                "h": 8,
                "mask_prob": 0.1839294921147962,
                "optimizer_kwargs": {
                    "lr": 9.643913797107524e-06,
                    "weight_decay": 0.00038878839042258504,
                },
                "clipnorm": 48,
                "transformer_layer_kwargs": {"layout": "FDRN"},
            }  # ins best value: 0.060774939325710375
        ],
        # "lumir": [
        #     {
        #         "N": 25,
        #         "L": 6,
        #         "activation": "gelu",
        #         "drop_rate": 0.3846038046182684,
        #
        #         # emb_dim = head_dim × h = 112 × 5 = 560
        #         "emb_dim": 560,
        #
        #         "fit_batch_size": 112,
        #         "h": 5,
        #         "mask_prob": 0.1503818747870349,
        #
        #         "optimizer_kwargs": {
        #             "lr": 8.435078077725219e-06,
        #             "weight_decay": 2.375954822158069e-05,
        #         },
        #
        #         "clipnorm": 36,
        #         "transformer_layer_kwargs": {"layout": "FDRN"},
        #     }
        # ]
        # 'CoMM':[
        #     {
        #         "N": 25,
        #         "L": 6,
        #         "activation": "gelu",
        #         "drop_rate": 0.4766172426386313,
        #         "emb_dim": 288,  # head_dim(72) × h(4)
        #         "fit_batch_size": 80,
        #         "h": 4,
        #         "mask_prob": 0.20373935182758954,
        #         "optimizer_kwargs": {
        #             "lr": 6.872822359854046e-06,
        #             "weight_decay": 7.767947526449707e-05,
        #         },
        #         "clipnorm": 16,
        #         "transformer_layer_kwargs": {"layout": "FDRN"},
        #     }   # Best value: 0.04566670424589091
        # ]
    },
}

# Core function to train & predict
def _stable_config_id(param_set: dict) -> str:
    """Generate a stable short hash ID from a config dict for tracking runs."""
    filtered = {k: v for k, v in param_set.items() if k not in ["cores", "is_verbose"]}
    payload = json.dumps(filtered, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]

def _ensure_dir(path: str):
    """Create directory if it does not exist."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

### MODIFICATION 2 START
# Cache settings
PRED_CACHE_DIR = os.path.join(EXPERIMENTS_FOLDER, "pred_cache")
os.makedirs(PRED_CACHE_DIR, exist_ok=True)

USE_PRED_CACHE = False
SAVE_PRED_CACHE = True

# Optional safety switch
# If True, and cache missing, will train to produce predictions.
# If False, and cache missing, will raise error so you never retrain by accident.
TRAIN_IF_NO_CACHE = True


def _pred_cache_path(model_name: str, config_id: str, trial_id: int, max_topk: int) -> str:
    safe_model = model_name.replace("/", "_")
    return os.path.join(
        PRED_CACHE_DIR,
        f"{safe_model}__{config_id}__trial{trial_id}__topk{max_topk}.pkl"
    )


def save_predictions(path: str, predictions) -> None:
    with open(path, "wb") as f:
        pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_predictions(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
### MODIFICATION 2 END

def train_and_predict_n(model_class, model_config, dataset, n_trials=1, result_path="results.csv"):
    """
    Train and evaluate model(s) defined in `model_config` on the given dataset.
    Saves both aggregated metrics and per-sample metrics (with sample IDs) to CSV.

    Args:
        model_class: model constructor (e.g., BERTWithEmbeddings)
        model_config: list of config dicts (hyperparameter sets)
        dataset: dataset object (SessionDataset)
        n_trials: number of random trials (for stability, usually 1)
        result_path: path to the main aggregated results CSV
    """

    all_results_list = []
    all_trials_records = []

    # Directory where we will save per-sample metrics (e.g., per user/session)
    per_sample_dir = result_path.replace(".csv", "_per_sample")
    _ensure_dir(per_sample_dir)

    for param_set in model_config:
        model_for_info = model_class(**param_set)
        model_name = model_for_info.name()
        config_id = _stable_config_id(param_set)

        trial_results = []
        best_ndcg_20 = -np.inf
        best_trial_result = {}

        for trial_num in range(n_trials):

            # === FIXED RANDOM SEED PER TRIAL ===
            seed = config["seed"] + trial_num
            np.random.seed(seed)
            import random
            random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass

            ### MODIFICATION 3 START
            # Replace always-train with cache-first logic
            max_topk = max(TOP_Ks)
            pred_path = _pred_cache_path(model_name, config_id, trial_num + 1, max_topk)

            if USE_PRED_CACHE and os.path.exists(pred_path):
                model_predictions = load_predictions(pred_path)
                print(f"✅ Loaded cached predictions: {pred_path}")
            else:
                if not TRAIN_IF_NO_CACHE:
                    raise RuntimeError(f"Missing cached predictions and TRAIN_IF_NO_CACHE=False: {pred_path}")

                model = model_class(**param_set)
                model.train(dataset.get_train_data())
                model_predictions = model.predict(dataset.get_test_prompts(), top_k=max_topk)

                if SAVE_PRED_CACHE:
                    save_predictions(pred_path, model_predictions)
                    print(f"✅ Saved predictions: {pred_path}")
            ### MODIFICATION 3 END

            dependencies = {
                metrics.MetricDependency.NUM_ITEMS: dataset.get_unique_item_count(),
                metrics.MetricDependency.ITEM_COUNT: dataset.get_item_counts(),
                metrics.MetricDependency.SAMPLE_COUNT: dataset.get_sample_counts(),
            }

            model_report = None
            per_sample_frames = []  # Store per-sample metrics across all top_k values

            for top_k in TOP_Ks:
                # Evaluate model with per-sample metrics enabled
                report: EvaluationReport = Evaluation.eval(
                    predictions=model_predictions,
                    ground_truths=dataset.get_test_ground_truths(),
                    model_name=model_name,
                    top_k=top_k,
                    metrics_per_sample=True,  # critical: instance-level metrics
                    dependencies=dependencies,
                    cores=1,
                )

                # Aggregate (global) results for summary
                if model_report is None:
                    model_report = report
                else:
                    model_report.results.update(report.results)

                # Try to use report.to_per_sample_df() if available
                try:
                    per_df = report.to_per_sample_df()
                except Exception:
                    # Fallback if EvaluationReport lacks that method
                    if getattr(report, "sample_ids", None) is None:
                        raise RuntimeError(
                            "EvaluationReport missing `sample_ids`. "
                            "Add it as described in the modified Evaluation class."
                        )
                    per_df = pd.DataFrame({"sample_id": report.sample_ids})
                    for mname, arr in report.results_per_sample.items():
                        per_df[mname] = np.asarray(arr)
                    per_df["__model__"] = report.model_name
                    per_df["__topk__"] = report.top_k

                # Add metadata for later pairing
                per_df["__trial__"] = trial_num + 1
                per_df["__embedding_key__"] = param_set.get("embedding_type", "N/A")
                per_df["__config_id__"] = config_id
                per_df["__timestamp__"] = datetime.datetime.now().isoformat(timespec="seconds")

                per_sample_frames.append(per_df)

            # === Save per-sample metrics directly into ../results/ ===
            if per_sample_frames:
                per_trial_df = pd.concat(per_sample_frames, ignore_index=True)

                # Keep only meaningful columns
                metric_cols = [
                    c for c in per_trial_df.columns
                    if c.startswith(("NDCG@", "HitRate@", "MRR@"))
                ]
                cols = ["sample_id"] + metric_cols + [
                    "__topk__", "__model__", "__trial__", "__config_id__",
                    "__timestamp__"
                ]
                per_trial_df = per_trial_df[
                    [c for c in cols if c in per_trial_df.columns]]

                # Example filename: ../results/per_sample_LLM2BERT4Rec_trial1.csv
                per_sample_path = os.path.join(
                    os.path.dirname(result_path),
                    f"per_sample_{model_name}_trial{trial_num + 1}.csv"
                )

                # Ensure ../results/ exists; no subfolders created
                os.makedirs(os.path.dirname(per_sample_path), exist_ok=True)

                per_trial_df.to_csv(per_sample_path, index=False)
                print(f"✅ Saved per-sample results to: {per_sample_path}")

            # Store global aggregated results
            trial_results.append(model_report.results)
            trial_record = {
                "trial_id": trial_num + 1,
                "Model Name": model_name,
                "config_id": config_id,
                "embedding_type": param_set.get("embedding_type", "N/A"),
            }
            trial_record.update(model_report.results)
            all_trials_records.append(trial_record)

            # Track best result (by NDCG@20)
            if model_report.results.get('NDCG@20', -np.inf) > best_ndcg_20:
                best_ndcg_20 = model_report.results['NDCG@20']
                best_trial_result = model_report.results.copy()

        # === Summarize mean/std across trials ===
        metrics_summary = {}
        for metric in trial_results[0]:
            metric_values = [trial[metric] for trial in trial_results]
            metrics_summary[f'{metric}_mean'] = np.mean(metric_values)
            metrics_summary[f'{metric}_std'] = np.std(metric_values)

        metrics_summary['Best_NDCG@20'] = best_ndcg_20
        param_set_filtered = {k: v for k, v in param_set.items() if k not in ['cores', 'is_verbose']}

        result_entry = {
            "Model Name": model_name,
            "embedding_type": param_set.get("embedding_type", "N/A"),
            "config_id": config_id,
            **param_set_filtered,
            **{f"Best_{k}": best_trial_result[k] for k in sorted(best_trial_result)},
            **{k: metrics_summary[k] for k in sorted(metrics_summary)}
        }
        all_results_list.append(result_entry)

    # === Save trial-level summary (same as before) ===
    trial_result_path = result_path.replace(".csv", "_trials.csv")
    pd.DataFrame(all_trials_records).to_csv(trial_result_path, index=False)

    results_df = pd.DataFrame(all_results_list)
    metric_order = [
        col for metric in [
            'NDCG@10', 'HitRate@10', 'MRR@10', 'Catalog coverage@10',
            'GiniIndex@10',
            'Serendipity@10', 'Novelty@10',
            'NDCG@20', 'HitRate@20', 'MRR@20', 'Catalog coverage@20',
            'GiniIndex@20',
            'Serendipity@20', 'Novelty@20'
        ]
        for col in [f'Best_{metric}', f'{metric}_mean', f'{metric}_std'] if
        col in results_df.columns
    ]
    columns_order = ['Model Name', 'embedding_type', 'config_id'] + [
        col for col in results_df.columns
        if col not in ['Model Name', 'embedding_type', 'config_id', 'cores', 'is_verbose'] + metric_order
    ] + metric_order

    results_df = results_df[columns_order].drop(columns=['cores', 'is_verbose'], errors='ignore')
    results_df.sort_values(by="NDCG@20_mean", ascending=False, inplace=True)

    _ensure_dir(result_path)
    if os.path.exists(result_path):
        old_results = pd.read_csv(result_path)
        results_df = pd.concat([old_results, results_df]).drop_duplicates(
            subset=["Model Name", "embedding_type", "config_id"], keep="last"
        ).reset_index(drop=True)

    results_df.to_csv(result_path, index=False)
# def train_and_predict_n(model_class, model_config, dataset, n_trials=5, result_path="results.csv"):
#     all_results_list = []
#     all_trials_records = []
#
#     for param_set in model_config:
#         model_for_info = model_class(**param_set)
#         model_name = model_for_info.name()
#         trial_results = []
#         best_ndcg_20 = -np.inf
#         best_trial_result = {}
#
#         for trial_num in range(n_trials):
#             model = model_class(**param_set)
#             model.train(dataset.get_train_data())
#             model_predictions = model.predict(dataset.get_test_prompts(), top_k=max(TOP_Ks))
#
#             dependencies = {
#                 metrics.MetricDependency.NUM_ITEMS: dataset.get_unique_item_count(),
#                 metrics.MetricDependency.ITEM_COUNT: dataset.get_item_counts(),
#                 metrics.MetricDependency.SAMPLE_COUNT: dataset.get_sample_counts(),
#             }
#
#             model_report = None
#             for top_k in TOP_Ks:
#                 report: EvaluationReport = Evaluation.eval(
#                     predictions=model_predictions,
#                     ground_truths=dataset.get_test_ground_truths(),
#                     model_name=model_name,
#                     top_k=top_k,
#                     metrics_per_sample=True,
#                     dependencies=dependencies,
#                     cores=1,
#                 )
#                 if model_report is None:
#                     model_report = report
#                 else:
#                     model_report.results.update(report.results)
#
#             trial_results.append(model_report.results)
#
#             trial_record = {
#                 "trial_id": trial_num + 1,
#                 "Model Name": model_name,
#             }
#             trial_record.update(model_report.results)
#             all_trials_records.append(trial_record)
#
#             if model_report.results['NDCG@20'] > best_ndcg_20:
#                 best_ndcg_20 = model_report.results['NDCG@20']
#                 best_trial_result = model_report.results.copy()
#
#         metrics_summary = {}
#         for metric in trial_results[0]:
#             metric_values = [trial[metric] for trial in trial_results]
#             metrics_summary[f'{metric}_mean'] = np.mean(metric_values)
#             metrics_summary[f'{metric}_std'] = np.std(metric_values)
#
#         metrics_summary['Best_NDCG@20'] = best_ndcg_20
#         param_set_filtered = {k: v for k, v in param_set.items() if k not in ['cores', 'is_verbose']}
#
#         result_entry = {
#             "Model Name": model_name,
#             "embedding_type": param_set.get("embedding_type", "N/A"),
#             **param_set_filtered,
#             **{f"Best_{k}": best_trial_result[k] for k in sorted(best_trial_result)},
#             **{k: metrics_summary[k] for k in sorted(metrics_summary)}
#         }
#         all_results_list.append(result_entry)
#
#     trial_result_path = result_path.replace(".csv", "_trials.csv")
#     pd.DataFrame(all_trials_records).to_csv(trial_result_path, index=False)
#
#     results_df = pd.DataFrame(all_results_list)
#     metric_order = [col for metric in ['NDCG@10', 'HitRate@10', 'MRR@10', 'Catalog coverage@10', 'Serendipity@10', 'Novelty@10',
#                                        'NDCG@20', 'HitRate@20', 'MRR@20', 'Catalog coverage@20', 'Serendipity@20', 'Novelty@20']
#                     for col in [f'Best_{metric}', f'{metric}_mean', f'{metric}_std'] if col in results_df.columns]
#     columns_order = ['Model Name', 'embedding_type'] + [col for col in results_df.columns if col not in ['Model Name', 'embedding_type', 'cores', 'is_verbose'] + metric_order] + metric_order
#
#     results_df = results_df[columns_order].drop(columns=['cores', 'is_verbose'], errors='ignore')
#     results_df.sort_values(by="NDCG@20_mean", ascending=False, inplace=True)
#
#     os.makedirs(os.path.dirname(result_path), exist_ok=True)
#     if os.path.exists(result_path):
#         old_results = pd.read_csv(result_path)
#         results_df = pd.concat([old_results, results_df]).drop_duplicates(subset=["Model Name", "embedding_type"], keep="last").reset_index(drop=True)
#
#     results_df.to_csv(result_path, index=False)

# Unified execution loop
def run_all_experiments(config):
    for model_name in INCLUDE:
        if model_name not in BEST_CONFIGS:
            continue
        model_class = MODEL_CLASS_MAPPING[model_name]

        for embedding_key, filename in EMBEDDING_PATHS.items():
            if embedding_key not in BEST_CONFIGS[model_name]:
                continue

            dataset_used = dataset

            # Construct full file path using config
            OUTPUT_PATH = os.path.join(config["paths"]["representation"],
                                       filename)
            # Check if file exists
            if not os.path.exists(OUTPUT_PATH):
                print(
                    f"⚠️ Embedding file '{filename}' not found. Generating it...")
                modality2embedding(config, filename)
            else:
                print(f"✅ Embedding file already exists: {OUTPUT_PATH}")

            config_list = BEST_CONFIGS[model_name][embedding_key]
            enriched_configs = []
            for config in config_list:
                cfg = config.copy()
                cfg.update({
                    "product_embeddings_location": OUTPUT_PATH,
                    "cores": CORES,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "is_verbose": IS_VERBOSE,
                    "pred_batch_size": PRED_BATCH_SIZE,
                    "pred_seen": PRED_SEEN,
                    "train_val_fraction": TRAIN_VAL_FRACTION,
                    "red_method": "PCA",
                    "red_params": {},
                })
                enriched_configs.append(cfg)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            result_file = os.path.join(EXPERIMENTS_FOLDER, f"results_{model_name}_{embedding_key}_{timestamp}.csv")
            train_and_predict_n(model_class, enriched_configs, dataset_used,
                                10, result_path=result_file)

# Run experiments
run_all_experiments(config)
