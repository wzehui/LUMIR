import ast
import inspect
import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection

from backbone.abstract_model import Model
from backbone.data.side_information import (
    create_side_information,
    SideInformation,
)
from backbone.grurec.grurec import GRURec
from backbone.utils.side_encoder import SideEncoder

import torch


def _maybe_eval(v):
    """Safely parse string-y params (e.g., '\"{\\\"epochs\\\": 10}\"') but keep dicts as-is."""
    return ast.literal_eval(v) if isinstance(v, str) else v


class GRURecWithEmbeddings(GRURec):
    """GRU4Rec that injects external item embeddings (after optional dimensionality reduction)
    into the model's nn.Embedding, strictly mirroring the TF version's semantics.
    """
    product_embeddings = None
    product_index_to_embedding = None
    product_index_to_id = None
    product_id_to_index = None
    product_index_to_embedding_red = None

    def __init__(
        self,
        product_embeddings_location: str,
        red_method: Literal["PCA", "RANDOM", "AE", "LDA"],
        red_params: dict,
        **grurec_config,
    ) -> None:
        """Init with external embeddings and a reducer, keep the rest identical to GRURec."""
        self.product_embeddings_location = product_embeddings_location
        self.red_method = red_method
        # 2) only eval strings, not dicts
        self.red_params = {k: _maybe_eval(v) for k, v in red_params.items()}
        self.grurec_config = grurec_config
        super().__init__(**grurec_config)

        # Lazy-load external embeddings once (class-level cache)
        if GRURecWithEmbeddings.product_embeddings is None:
            df = pd.read_csv(product_embeddings_location)
            GRURecWithEmbeddings.product_embeddings = df

            # Build ItemId -> embedding vector mapping (JSON-encoded list -> np.array)
            mp = (
                df[["ItemId", "embedding"]]
                .set_index("ItemId")
                .to_dict()["embedding"]
            )
            mp = {k: json.loads(v) for k, v in mp.items()}
            mat = np.array(list(mp.values()), dtype=np.float32)  # 1) enforce float32 like TF
            GRURecWithEmbeddings.product_index_to_embedding = mat

            ids = df["ItemId"].tolist()
            GRURecWithEmbeddings.product_index_to_id = ids
            GRURecWithEmbeddings.product_id_to_index = {iid: idx for idx, iid in enumerate(ids)}

    def train(self, train_data: Any) -> None:
        """
        Inject reduced external item embeddings into the model's nn.Embedding
        in the exact vocabulary order (reduced ids 0..V-1), then delegate to
        the normal training pipeline.

        Steps:
          1) Build a temporary GRURec (epochs=0) to initialize id_reducer and model.
          2) Reduce external embeddings to (E) dims via PCA/RANDOM/AE/LDA.
          3) Reorder reduced embeddings to match the model's reduced-id order.
          4) Copy (V, E) into embedding[:V], keep last row (mask id) untouched.
          5) Train as usual (super().train).
        """
        # If using precomputed weights, just call the base class.
        if self.filepath_weights is not None:
            super().train(train_data)
            return

        # --- 1) Build temp GRURec to set up id_reducer and model shapes (no real training) ---
        temp_config = self.grurec_config.copy()
        temp_config["num_epochs"] = 0

        # LDA constraint: n_components <= min(input_dim, num_classes-1)
        if self.red_method == "LDA":
            max_reduced_dim_size = min(
                GRURecWithEmbeddings.product_index_to_embedding.shape[1],
                len(np.unique(
                    GRURecWithEmbeddings.product_embeddings["class"])) - 1,
            )
            if self.emb_dim > max_reduced_dim_size:
                temp_config["emb_dim"] = max_reduced_dim_size

        self.temp_model = GRURec(**temp_config)
        # This "train" only builds preprocessors and the torch model, epochs=0.
        self.temp_model.train(train_data)

        # --- 2) Dimensionality reduction to target emb_dim ---
        if self.red_method == "PCA":
            pca = PCA(n_components=self.emb_dim)
            red = pca.fit_transform(
                GRURecWithEmbeddings.product_index_to_embedding)

        elif self.red_method == "RANDOM":
            grp = GaussianRandomProjection(n_components=self.emb_dim)
            red = grp.fit_transform(
                GRURecWithEmbeddings.product_index_to_embedding)

        elif self.red_method == "AE":
            # Build side-information wrapper, then pretrain SideEncoder to get encodings.
            side_information: SideInformation = create_side_information(
                GRURecWithEmbeddings.product_index_to_embedding, []
            )

            # Filter kwargs for SideEncoder.__init__
            se_init_names = {
                p.name for p in
                inspect.signature(SideEncoder.__init__).parameters.values()
                if p.name != "self"
            }
            se_init = {k: v for k, v in self.red_params.items() if
                       k in se_init_names}

            side_encoder: SideEncoder = SideEncoder(
                side_information=side_information,
                encoder_dimension=self.emb_dim,
                **se_init,
            )

            # Filter kwargs for SideEncoder.pretrain
            se_pretrain_names = {
                p.name for p in
                inspect.signature(SideEncoder.pretrain).parameters.values()
                if p.name != "self"
            }
            se_pretrain = {k: v for k, v in self.red_params.items() if
                           k in se_pretrain_names}
            side_encoder.pretrain(**se_pretrain)

            red = side_encoder.get_encodings(
                GRURecWithEmbeddings.product_index_to_embedding
            )

        elif self.red_method == "LDA":
            class_labels = GRURecWithEmbeddings.product_embeddings["class"]
            max_reduced_dim_size = min(
                GRURecWithEmbeddings.product_index_to_embedding.shape[1],
                len(np.unique(class_labels)) - 1,
            )
            n_components = min(self.emb_dim, max_reduced_dim_size)
            lda = LinearDiscriminantAnalysis(n_components=n_components,
                                             **self.red_params)
            red = lda.fit_transform(
                GRURecWithEmbeddings.product_index_to_embedding, class_labels
            )

        else:
            raise ValueError(
                f"Unknown reduce method for embeddings: {self.red_method}")

        # Ensure FP32 like TF / torch default dtype
        red = np.asarray(red,
                         dtype=np.float32)  # shape: (num_external_items, E)

        # --- 3) Reorder to match model's reduced-id order (0..V-1) ---
        # === Build a robust reduced-id -> embedding-row mapping ===
        # We cannot assume reduced ids are perfectly contiguous 0..V-1.
        # We sort by reduced id and then fill rows 0..V-1; any holes or overflows fallback.

        id_lookup = self.temp_model.id_reducer.id_lookup  # {orig_id: reduced_id}
        emb_layer = self.temp_model.model.embedding_layer
        device = emb_layer.weight.device
        dtype = emb_layer.weight.dtype

        # V_model = number of "real" items (the last row is reserved for mask id)
        Vp1, E = emb_layer.weight.shape
        V_model = Vp1 - 1

        # Sort pairs by reduced id
        pairs = sorted(id_lookup.items(), key=lambda kv: kv[1])  # [(orig_id, red_id), ...] ascending by red_id

        # Prepare a (V_model, E) numpy matrix to fill, initialized from current weights for fallback
        with torch.no_grad():
            cur_w = emb_layer.weight.detach().clone()  # (V+1, E)
            base_rows = cur_w[:V_model, :].cpu().numpy()  # (V, E) current weights as safe fallback
            mask_row  = cur_w[V_model:Vp1, :]            # (1, E)

        # Reduced external embedding matrix (already computed above)
        red_mat = red.astype(np.float32)  # (num_ext_items, E) from your reducer
        # Mapping from original ItemId -> row in external embedding file
        ext_index = GRURecWithEmbeddings.product_id_to_index

        # We'll fill row 0..V_model-1; if we cannot find a matching external vector,
        # we keep the existing row (fallback). This avoids IndexError and keeps dims valid.
        filled = 0
        for orig_id, red_id in pairs:
            # Skip padding/special id if present in mapping
            if orig_id == 0:
                continue
            # Only fill rows that actually exist in the model's table
            if not (0 <= red_id < V_model):
                # Out-of-range reduced id → ignore (model doesn't have capacity for it)
                continue

            # Find external row; if missing, fallback to current weight (already set in base_rows)
            ext_row = ext_index.get(orig_id, None)
            if ext_row is None:
                # Missing external embedding: leave base_rows[red_id] as-is
                continue

            # Safety: ext_row must be within red_mat bounds
            if not (0 <= ext_row < red_mat.shape[0]):
                # Corrupted index; fallback
                continue

            base_rows[red_id] = red_mat[ext_row]
            filled += 1

        # Join with mask row to get (V+1, E)
        new_w_np = np.concatenate([base_rows, mask_row.cpu().numpy()], axis=0).astype(np.float32)
        new_w = torch.as_tensor(new_w_np, dtype=dtype, device=device)

        with torch.no_grad():
            emb_layer.weight.copy_(new_w)
        emb_layer.requires_grad_(True)

        # --- 5) Continue with normal training on the full data ---
        super().train(train_data)

    def get_model(self, data: pd.DataFrame) -> Model:
        """Return the underlying model (delegates to temp_model when no filepath_weights)."""
        if self.filepath_weights is not None:
            return super().get_model(data)
        return self.temp_model.model

    def predict(self, predict_data: Any, top_k: int = 10) -> dict[int, np.ndarray]:
        """Delegate predict to temp_model, same as TF path."""
        if self.filepath_weights is not None:
            return super().predict(predict_data, top_k)
        return self.temp_model.predict(predict_data, top_k)

    def name(self) -> str:
        return "LLM2GRU4Rec"