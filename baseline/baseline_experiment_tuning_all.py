import os
import yaml
import pickle
import datetime
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from backbone.data.session_dataset import SessionDataset
from backbone.eval.evaluation import Evaluation, EvaluationReport, metrics
from backbone.eval.metrics.ndcg import NormalizedDiscountedCumulativeGain
from backbone.grurec.grurec_with_embeddings import GRURecWithEmbeddings
from backbone.transformer.sasrec.sasrec_with_embeddings import SASRecWithEmbeddings
from backbone.transformer.bert.bert_with_embeddings import BERTWithEmbeddings

from representation.utils_embedding import modality2embedding
# reproducibility
import numpy as np
import random
import torch
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Config Load ===
# CONFIG_PATH = "../configs/representation_config.yaml"
CONFIG_PATH = "../configs/representation_config_ins.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

INCLUDE = {
    # "LLM2GRURec",
    # "LLM2SASRec",
    "LLM2BERT4Rec",
}

# DATASET_FILENAME = "../yelp/dataset/dataset_training.pickle"
DATASET_FILENAME = "../instagram/dataset/dataset_training.pickle"

EMBEDDING_PATHS = {
    # "baseline": "embedding_meta-review-photo_768_2048.csv.gz"
    # "baseline": "embedding_meta-review-photo_raw_4096_2048_ins.csv.gz"
    # "baseline": "embedding_flava_text3_meta-review-photo_raw_4096_2048.csv.gz"
    # "baseline": "embedding_comm_2048.csv.gz"
    # "baseline": "embedding_reweighted_concat_meta-review-photo_fixedpca_4096_2048.csv.gz"
    # "lumir": "lumir_epoch15.csv.gz"
    # "lumir": "lumir_epoch15_comm.csv.gz"
    "lumir": "lumir_epoch15_ins.csv.gz"
}

CORES = 20
EARLY_STOPPING_PATIENCE = 10
IS_VERBOSE = True
FILTER_PROMPT_ITEMS = True
MAX_SESSION_LENGTH_FOR_DECAY_PRECOMPUTATION = 400
PRED_BATCH_SIZE = 1000
PRED_SEEN = False
TRAIN_VAL_FRACTION = 0.1
TOP_Ks = [10, 20]

model_classes = []
if "LLM2GRURec" in INCLUDE:
    model_classes.append(GRURecWithEmbeddings)
if "LLM2SASRec" in INCLUDE:
    model_classes.append(SASRecWithEmbeddings)
if "LLM2BERT4Rec" in INCLUDE:
    model_classes.append(BERTWithEmbeddings)

def objective(trial):
    if model_class.__name__ == "GRURecWithEmbeddings":
        emb_dim = trial.suggest_int("emb_dim", 512, 1024, step=32)
        hidden_dim = trial.suggest_int("hidden_dim", emb_dim, emb_dim * 2,
                                       step=32)
        llm2grurec_config = {
            "N": 25,
            # "N": 30,
            "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "fit_batch_size": trial.suggest_int("fit_batch_size", 256, 1024,
                                                step=32),
            "optimizer_kwargs": {
                "lr": trial.suggest_float("learning_rate", 0.001, 0.01,
                                          log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.0005,
                                                    0.003),
            },
            "cores": CORES,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "is_verbose": IS_VERBOSE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "pred_seen": PRED_SEEN,
            "train_val_fraction": TRAIN_VAL_FRACTION,
            "product_embeddings_location": OUTPUT_PATH,
            "red_method": "PCA",
            "red_params": {},
        }
        model = model_class(**llm2grurec_config)

    elif model_class.__name__ == "SASRecWithEmbeddings":
        head_dim = trial.suggest_categorical('head_dim', list(range(64, 128,
                                                                    8)))
        h = trial.suggest_int('h', 4, 12)
        llm2sasrec_config = {
            "N": 25,
            "L": trial.suggest_int("L", 4, 12),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
            # "drop_rate": trial.suggest_float("drop_rate", 0.2, 0.6),
            "drop_rate": trial.suggest_float("drop_rate", 0.05, 0.3),
            "emb_dim": h * head_dim,
            "fit_batch_size": trial.suggest_int("fit_batch_size", 16, 128, step=16),
            "h": h,
            "optimizer_kwargs": {
                "lr": trial.suggest_float("learning_rate", 0.00003, 0.003, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.001, log=True),
            },
            "clipnorm": trial.suggest_int("clipnorm", 5, 10),
            "transformer_layer_kwargs": {"layout": "NFDR"},
            "cores": CORES,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "is_verbose": IS_VERBOSE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "pred_seen": PRED_SEEN,
            "train_val_fraction": TRAIN_VAL_FRACTION,
            "product_embeddings_location": OUTPUT_PATH,
            "red_method": "PCA",
            "red_params": {},
        }
        model = model_class(**llm2sasrec_config)

    elif model_class.__name__ == "BERTWithEmbeddings":
        # head_dim = trial.suggest_categorical('head_dim', list(range(16, 192, 16)))
        head_dim = trial.suggest_categorical('head_dim', list(range(32, 128, 8)))
        h = trial.suggest_int('h', 4, 8)
        llm2bert_config = {
            # "N": 25,
            "N": 30,
            "L": trial.suggest_int("L", 4, 8),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
            "drop_rate": trial.suggest_float("drop_rate", 0.2, 0.5),
            "emb_dim": h * head_dim,
            "fit_batch_size": trial.suggest_int("fit_batch_size", 64, 128, step=16),
            "h": h,
            "mask_prob": trial.suggest_float("mask_prob", 0.15, 0.3),
            "optimizer_kwargs": {
                "lr": trial.suggest_float("learning_rate",  0.000005, 0.00001, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.000005, 0.0005, log=True),
                # "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.05, ),
            },
            "clipnorm": trial.suggest_int("clipnorm", 10, 50),
            "transformer_layer_kwargs": {"layout": "FDRN"},
            "cores": CORES,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "is_verbose": IS_VERBOSE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "pred_seen": PRED_SEEN,
            "train_val_fraction": TRAIN_VAL_FRACTION,
            "product_embeddings_location": OUTPUT_PATH,
            "red_method": "PCA",
            "red_params": {},
        }

        model = model_class(**llm2bert_config)

    model.train(dataset_train.get_train_data())
    predictions = model.predict(dataset_train.get_test_prompts(), top_k=max(TOP_Ks))

    report = Evaluation.eval(
        predictions=predictions,
        ground_truths=dataset_train.get_test_ground_truths(),
        model_name=model.name(),
        top_k=max(TOP_Ks),
        metrics=[NormalizedDiscountedCumulativeGain()],
        dependencies={
            metrics.MetricDependency.NUM_ITEMS: dataset_train.get_unique_item_count(),
        },
        metrics_per_sample=False,
    )
    trial_df = report.to_df()
    ndcg = trial_df.at[model.name(), "NDCG@20"]

    return ndcg

if __name__ == "__main__":
    for model_class in model_classes:
        for embedding_key, filename in EMBEDDING_PATHS.items():
            # Construct full file path using config
            OUTPUT_PATH = os.path.join( config["paths"]["representation"], filename)

            # Check if file exists
            if not os.path.exists(OUTPUT_PATH):
                print(f"⚠️ Embedding file '{filename}' not found. Generating it...")
                modality2embedding(config, filename)
            else:
                print(f"✅ Embedding file already exists: {OUTPUT_PATH}")


            EXPERIMENTS_FOLDER = os.path.join("../results", model_class.__name__)
            os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)

            dataset_train = SessionDataset.from_pickle(DATASET_FILENAME)

            OPTUNA_STUDY_FILE = os.path.join(EXPERIMENTS_FOLDER, f"optuna_study_{model_class.__name__}.pkl")

            if os.path.exists(OPTUNA_STUDY_FILE):
                os.remove(OPTUNA_STUDY_FILE)

            study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))

            def save_study_callback(study, trial):
                if trial.number % 5 == 0:
                    with open(OPTUNA_STUDY_FILE, "wb") as f:
                        pickle.dump(study, f)
                    print(f"[{model_class.__name__}-{embedding_key}] Trial {trial.number} saved.")

            study.optimize(objective, n_trials=50, callbacks=[save_study_callback])

            with open(OPTUNA_STUDY_FILE, "wb") as f:
                pickle.dump(study, f)

            sorted_trials = sorted(study.trials, key=lambda x: x.value, reverse=True)
            best_trials_data = [
                {"Rank": i + 1, "Value": t.value, "Params": str(t.params)}
                for i, t in enumerate(sorted_trials[:3])
            ]

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            csv_file_path = os.path.join(
                EXPERIMENTS_FOLDER,
                f"{model_class.__name__}-{embedding_key}_best_3_{timestamp}.csv"
            )
            pd.DataFrame(best_trials_data).to_csv(csv_file_path, index=False)
            print(f"[{model_class.__name__}-{embedding_key}] Best 3 results saved to {csv_file_path}")

            if os.path.exists(OPTUNA_STUDY_FILE):
                os.remove(OPTUNA_STUDY_FILE)
                print("Optuna study file has been deleted.")