# LUMIR: Decoupled Multimodal Representation Learning via LLM-Based Unification for Next-POI Recommendation
We propose LUMIR, a decoupled multimodal representation learning framework for point-of-interest (POI) recommendation.
Instead of tightly coupling multimodal fusion with downstream recommendation models, LUMIR learns generalizable POI embeddings independently.

Heterogeneous modalities (metadata, reviews, photos) are first transformed 
into a unified embedding space via LLMs, followed by an interaction-aware 
fusion mechanism trained with contrastive objectives.

The learned representations are evaluated on next-POI recommendation tasks 
using sequential backbones such as BERT4Rec, SASRec and GRU4Rec.

The downstream evaluation pipeline is built upon the codebase from
[LLM-Sequential-Recommendation](https://github.com/dh-r/LLM-Sequential-Recommendation.git).


## Dataset

All experiments are based on the [Multimodal Yelp Dataset](https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation), extended with multimodal summaries:

| File                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `business.csv`         | Metadata (e.g., name, category, address, coordinates)                      |
| `checkin.csv`          | User-business interaction sequences                                         |
| `review.csv`           | Full user reviews                                                           |
| `photo.csv`            | Yelp photo metadata (labels, image IDs)                                     |
| `review_summary.csv`   | DeepSeek-R1 generated summaries of reviews                                  |
| `photo_summary.csv`    | GPT-4o generated summaries of business images                               |

After downloading the dataset, place all CSV files into: `yelp/csv/`
## Environment Setup

**Base Image**  
We recommend using Docker (base image):

```
nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
```

**Python version**: `3.10`

**Install Dependencies & Activate**

```
poetry install
```

Now you can activate the environment:

```
source $(poetry env info --path)/bin/activate
```

## Data Processing

The dataset is organized in a session-based format, where each user’s interaction history is treated as a single session. Each session is a chronologically ordered sequence of user check-ins.

### Temporal Splitting

We apply a two-stage temporal splitting strategy to ensure a realistic evaluation setting.

First, a global cutoff timestamp (**2019-12-25**) is used to separate users:
- Users with all interactions before this date are assigned to the training and validation set  
- Users with any interaction after this date are assigned to the test set, and their full sequences are used for evaluation  

Within the training set, an earlier timestamp (**2019-12-15**) is used for further splitting:
- Interactions before this date are used for training  
- Interactions after this date are used for validation  

### Rationale

This setup enforces strict temporal ordering, ensuring that the model is trained only on past interactions and evaluated on future behavior, thus avoiding information leakage.

The resulting split ratio is approximately **8 : 1 : 1** (train : validation : test).

### Preprocessing

Run the following script to generate processed datasets:

`preprocessing/preparation.py`

The processed files will be saved in:

`yelp/dataset/`

## Representation Learning

We learn POI representations in a **decoupled manner**, independent of downstream recommendation models. Each POI is described by multiple textual modalities (e.g., metadata, review summaries, photo summaries), which are encoded and fused into a unified embedding space.

### Multi-View Encoding

Each POI is associated with multiple textual sources, encoded using a shared text encoder:

- Metadata  
- Review summaries  
- Photo summaries  

The encoder produces modality-specific embeddings, which are then passed to a fusion module.

Run `train_representation_tuning.py` to perform hyperparameter search and obtain the best embeddings.

### Hyperparameter Search

We tune key components including:

- Loss weights  
- Temperature parameters  
- Positive sampling strategy (next or window)  
- Window size  

Each configuration is trained independently and evaluated using intrinsic metrics.

### Intrinsic Evaluation

We adopt **nearest-neighbor evaluation**:

- For each POI, retrieve top-K nearest neighbors  
- Positives are defined via session transitions  

Metrics:

- **Hit@K**: whether any positive appears in top-K  
- **MRR@K**: reciprocal rank of the first positive  

Model selection is based on **MRR@K (primary)** and **Hit@K (secondary)**.

Best embeddings are stored as compressed `.csv.gz` files and saved to `yelp/embeddings/`

### Downstream Evaluation

To evaluate the learned embeddings on the next-POI recommendation task, we use:

evaluation/baseline_experiment_tuning_all.py

This script performs **model-specific hyperparameter tuning** using Optuna for different sequential backbones, including:

- GRU4Rec  
- SASRec  
- BERT4Rec  

The learned POI embeddings are loaded and injected into these models as fixed item representations.

### Evaluation Pipeline

For each backbone model:

1. Load the processed session dataset  
2. Load the pretrained POI embeddings  
3. Train the sequential model on session data  
4. Generate top-K predictions for each test session  
5. Evaluate predictions using ranking metrics  

The primary metric used during tuning is:

- **NDCG@20**

### Hyperparameter Optimization

We use Optuna with a TPE sampler to optimize model-specific parameters such as:

- Embedding dimension  
- Number of layers and heads  
- Dropout rate  
- Learning rate and weight decay  
- Batch size  

Each trial:

- Trains the model  
- Evaluates on validation/test split  
- Returns NDCG@20 as the objective  

The search runs for a fixed number of trials (e.g., 50 per model).

### Final Evaluation

After hyperparameter tuning, the best configurations are selected and used for final evaluation via:

evaluation/baseline_experiment_modality.py

This script reports the final performance of the embeddings across different models and settings.

### Output

- Top-performing hyperparameter configurations are saved  
- Results are exported as `.csv` files under `results/`  
- Each file records the best trials and corresponding performance  

## Supplementary Material

Additional experimental results are provided in:

supplementary_material/appendix.pdf

This document reports the **beyond-accuracy performance** of the learned embeddings on the downstream next-POI recommendation task, including:

- Coverage  
- Gini Index  
- Novelty  
- Serendipity  

## 📖 Citation
If you use this repository or the associated dataset, please cite our paper:
```bibtex

