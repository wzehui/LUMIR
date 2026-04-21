import ast
import inspect
import json
from typing import Any, Literal
import torch
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
from backbone.transformer.sasrec.sasrec import SASRec
from backbone.utils.side_encoder import SideEncoder


class SASRecWithEmbeddings(SASRec):
    product_embeddings = None

    def __init__(
        self,
        product_embeddings_location,
        red_method: Literal["PCA", "RANDOM", "AE", "LDA"],
        red_params: dict,
        **sasrec_config,
    ) -> None:
        """Inits an instance of SASRecWithEmbeddings.

        Args:
            product_embeddings_location: A string indicating the path to the embedding
                file.
            red_method: A string indicating the dimensionality reduction method to use.
            red_params: A dictionary containing the configuration of the reduction.
            **sasrec_config: The rest of the parameters that belong to the SASRec model.
        """
        self.product_embeddings_location = product_embeddings_location
        self.red_method = red_method
        self.red_params = {k: ast.literal_eval(v) for k, v in red_params.items()}
        self.sasrec_config: dict = sasrec_config
        super().__init__(**sasrec_config)

        if SASRecWithEmbeddings.product_embeddings is None:
            SASRecWithEmbeddings.product_embeddings = pd.read_csv(
                product_embeddings_location, compression="gzip"
            )
            SASRecWithEmbeddings.product_index_to_embedding = (
                SASRecWithEmbeddings.product_embeddings[["ItemId", "embedding"]]
                .set_index("ItemId")
                .to_dict()["embedding"]
            )
            SASRecWithEmbeddings.product_index_to_embedding = {
                k: json.loads(v)
                for k, v in SASRecWithEmbeddings.product_index_to_embedding.items()
            }
            SASRecWithEmbeddings.product_index_to_embedding = np.array(
                list(SASRecWithEmbeddings.product_index_to_embedding.values())
            )

        SASRecWithEmbeddings.product_index_to_id = (
            SASRecWithEmbeddings.product_embeddings["ItemId"].tolist()
        )
        SASRecWithEmbeddings.product_id_to_index = {
            id: index
            for index, id in enumerate(SASRecWithEmbeddings.product_index_to_id)
        }

    def train(self, train_data: Any) -> None:
        # If we use pre-computed weights, fall back to base implementation
        if self.filepath_weights is not None:
            super().train(train_data)
            return

        # 1) Build a temp SASRec to initialize id_reducer and model shapes (epochs=0)
        temp_config = self.sasrec_config.copy()
        temp_config["num_epochs"] = 0

        # Respect LDA upper bound (same semantics as TF)
        if self.red_method == "LDA":
            max_reduced_dim_size = min(
                SASRecWithEmbeddings.product_index_to_embedding.shape[1],
                len(np.unique(
                    SASRecWithEmbeddings.product_embeddings["class"])) - 1,
            )
            if self.emb_dim > max_reduced_dim_size:
                temp_config["emb_dim"] = max_reduced_dim_size

        self.temp_model = SASRec(**temp_config)
        # This "train" call (with num_epochs=0) should only build id_reducer/model
        self.temp_model.train(train_data)

        # 2) Dimensionality reduction (PCA / RANDOM / AE / LDA), identical to TF semantics
        if self.red_method == "PCA":
            pca = PCA(n_components=self.emb_dim)
            red = pca.fit_transform(
                SASRecWithEmbeddings.product_index_to_embedding)
        elif self.red_method == "RANDOM":
            grp = GaussianRandomProjection(n_components=self.emb_dim)
            red = grp.fit_transform(
                SASRecWithEmbeddings.product_index_to_embedding)
        elif self.red_method == "AE":
            side_information: SideInformation = create_side_information(
                SASRecWithEmbeddings.product_index_to_embedding, []
            )
            # Filter params for SideEncoder init/pretrain
            init_names = {
                p.name for p in
                inspect.signature(SideEncoder.__init__).parameters.values()
                if p.name != "self"
            }
            init_kwargs = {k: v for k, v in self.red_params.items() if
                           k in init_names}
            encoder = SideEncoder(
                side_information=side_information,
                encoder_dimension=self.emb_dim,
                **init_kwargs,
            )
            pretrain_names = {
                p.name for p in
                inspect.signature(SideEncoder.pretrain).parameters.values()
                if p.name != "self"
            }
            pretrain_kwargs = {k: v for k, v in self.red_params.items() if
                               k in pretrain_names}
            encoder.pretrain(**pretrain_kwargs)
            red = encoder.get_encodings(
                SASRecWithEmbeddings.product_index_to_embedding)
        elif self.red_method == "LDA":
            class_labels = SASRecWithEmbeddings.product_embeddings["class"]
            max_reduced_dim_size = min(
                SASRecWithEmbeddings.product_index_to_embedding.shape[1],
                len(np.unique(class_labels)) - 1,
            )
            n_components = min(self.emb_dim, max_reduced_dim_size)
            lda = LinearDiscriminantAnalysis(n_components=n_components,
                                             **self.red_params)
            red = lda.fit_transform(
                SASRecWithEmbeddings.product_index_to_embedding, class_labels)
        else:
            raise ValueError(
                f"Unknown reduce method for embeddings: {self.red_method}")

        # Ensure float32 to match nn.Embedding default dtype
        red = np.asarray(red, dtype=np.float32)

        # 3) Reorder reduced embeddings by the temp model's *vocab order*
        #    Step A: get "reduced id -> original ItemId" in vocab order
        #    id_lookup: {reduced_id:int -> original_id:Any}
        id_lookup = self.temp_model.id_reducer.id_lookup
        vocab_order_orig_ids = [orig for _, orig in
                                sorted(id_lookup.items(), key=lambda kv: kv[0])]

        #    Step B: map original ItemId to external embedding row
        row_index = np.empty(len(vocab_order_orig_ids), dtype=np.int64)
        for rid, orig_id in enumerate(vocab_order_orig_ids):
            try:
                ext_row = SASRecWithEmbeddings.product_id_to_index[orig_id]
            except KeyError as e:
                raise KeyError(
                    f"[Embedding inject] Original ItemId {orig_id} not found in external embeddings. "
                    f"Ensure the embedding file covers all items in training data."
                ) from e
            row_index[rid] = ext_row

        #    Step C: gather rows -> shape (V, E). We'll append the mask row to make (V+1, E)
        emb_mat_np = red[row_index]  # (V, E)

        # 4) Copy into PyTorch embedding weight, appending the mask row at the end
        #    Locate the item embedding: your SASRec uses an EmbeddingLayer with .item_emb (nn.Embedding)
        emb_layer = self.temp_model.model.embedding_layer.item_emb  # nn.Embedding
        device = emb_layer.weight.device
        dtype = emb_layer.weight.dtype

        Vp1, E = emb_layer.weight.shape  # expected (V+1, E)
        V = Vp1 - 1
        if emb_mat_np.shape != (V, E):
            raise ValueError(
                f"[Embedding inject] reduced matrix shape {emb_mat_np.shape} != expected ({V}, {E}). "
                f"Check that reducer output dim == emb_dim ({self.emb_dim})."
            )

        # Build (V+1, E): keep the last row (mask) from existing weight to match TF behavior
        with torch.no_grad():
            cur = emb_layer.weight.detach()
            mask_row = cur[V:Vp1, :]  # (1, E)
            new_w = torch.cat(
                [
                    torch.as_tensor(emb_mat_np, dtype=dtype, device=device),
                    # (V, E)
                    mask_row  # (1, E)
                ],
                dim=0
            )  # (V+1, E)
            emb_layer.weight.copy_(new_w)

        # Ensure the embedding stays trainable like TF's .trainable = True
        for p in emb_layer.parameters():
            p.requires_grad_(True)

        # 5) Proceed with normal training using the injected weights
        super().train(train_data)

    def get_model(self, data: pd.DataFrame) -> Model:
        # If we use pre-computed weights, we do not have to do anything special
        # in this class, and can just use the functionality from SASRec.
        if self.filepath_weights is not None:
            return super().get_model(data)

        return self.temp_model.model

    def predict(self, predict_data: Any, top_k: int = 10) -> dict[int, np.ndarray]:
        # If we use pre-computed weights, we do not have to do anything special
        # in this class, and can just use the functionality from SASRec.
        if self.filepath_weights is not None:
            return super().predict(predict_data, top_k)

        return self.temp_model.predict(predict_data, top_k)

    def name(self) -> str:
        return "LLM2SASRec"
