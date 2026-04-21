# LLUMIR: Decoupled Multimodal Representation Learning via LLM-Based Unification for Next-POI Recommendation
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


## Environment Setup

**Base Image**  
We recommend using Docker (base image):

```
nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
```

**Python version**: `3.10`

If you're setting up locally:

```
# Install Python 3.10 (Ubuntu)
apt update -y
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update -y
apt install -y python3.10 python3.10-venv python3.11-distutils
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
update-alternatives --config python3
```

---

**Poetry Installation**

We recommend using [`pipx`](https://pypa.github.io/pipx/) to install [Poetry](https://python-poetry.org):

```
# Install pipx (if not yet installed)
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade --force-reinstall pip
python3 -m pip install --user pipx
python3 -m pipx ensurepath
source ~/.bashrc

# Install poetry
pipx install poetry
```

---

**Install Dependencies & Activate**

```
poetry install
```

Now you can activate the environment:

```
source $(poetry env info --path)/bin/activate
```


## Data Processing


## 📖 Citation
If you use this repository or the associated dataset, please cite our paper:
```bibtex

