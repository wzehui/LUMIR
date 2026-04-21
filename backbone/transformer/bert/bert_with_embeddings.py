import ast
import inspect
import json
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection

from backbone.transformer.bert.bert import BERT
from backbone.utils.side_encoder import SideEncoder
from backbone.data.side_information import create_side_information, SideInformation


class BERTWithEmbeddings(BERT):
    product_embeddings = None

    def __init__(
        self,
        product_embeddings_location,
        red_method: Literal["PCA", "RANDOM", "AE", "LDA"],
        red_params: dict,
        **bert_config,
    ) -> None:
        self.product_embeddings_location = product_embeddings_location
        self.red_method = red_method
        self.red_params = {k: ast.literal_eval(v) for k, v in red_params.items()}
        self.bert_config = bert_config
        super().__init__(**bert_config)

        if BERTWithEmbeddings.product_embeddings is None:
            df = pd.read_csv(product_embeddings_location)
            BERTWithEmbeddings.product_embeddings = df
            BERTWithEmbeddings.product_index_to_embedding = df["embedding"].apply(json.loads).tolist()
            BERTWithEmbeddings.product_index_to_embedding = np.array(BERTWithEmbeddings.product_index_to_embedding)

        BERTWithEmbeddings.product_index_to_id = BERTWithEmbeddings.product_embeddings["ItemId"].tolist()
        BERTWithEmbeddings.product_id_to_index = {
            id: idx for idx, id in enumerate(BERTWithEmbeddings.product_index_to_id)
        }

    def train(self, train_data: Any) -> None:
        if self.filepath_weights is not None:
            super().train(train_data)
            return

        temp_config = self.bert_config.copy()
        temp_config["num_epochs"] = 0

        if self.red_method == "LDA":
            max_reduced_dim_size = min(
                BERTWithEmbeddings.product_index_to_embedding.shape[1],
                len(np.unique(BERTWithEmbeddings.product_embeddings["class"])) - 1,
            )
            if self.emb_dim > max_reduced_dim_size:
                temp_config["emb_dim"] = max_reduced_dim_size

        self.temp_model = BERT(**temp_config)
        self.temp_model.train(train_data)

        if self.red_method == "PCA":
            pca = PCA(n_components=self.emb_dim)
            BERTWithEmbeddings.product_index_to_embedding_red = pca.fit_transform(
                BERTWithEmbeddings.product_index_to_embedding
            )
        elif self.red_method == "RANDOM":
            grp = GaussianRandomProjection(n_components=self.emb_dim)
            BERTWithEmbeddings.product_index_to_embedding_red = grp.fit_transform(
                BERTWithEmbeddings.product_index_to_embedding
            )
        elif self.red_method == "AE":
            side_information: SideInformation = create_side_information(
                BERTWithEmbeddings.product_index_to_embedding, []
            )

            side_encoder_param_names: list = [
                param.name
                for param in inspect.signature(SideEncoder.__init__).parameters.values()
                if param.name != "self"
            ]
            side_encoder_params: dict = {
                k: v
                for k, v in self.red_params.items()
                if k in side_encoder_param_names
            }
            side_encoder: SideEncoder = SideEncoder(
                side_information=side_information,
                encoder_dimension=self.emb_dim,
                **side_encoder_params,
            )

            pretrain_param_names: list = [
                param.name
                for param in inspect.signature(SideEncoder.pretrain).parameters.values()
                if param.name != "self"
            ]
            pretrain_params: dict = {
                k: v for k, v in self.red_params.items() if k in pretrain_param_names
            }
            side_encoder.pretrain(**pretrain_params)

            BERTWithEmbeddings.product_index_to_embedding_red = (
                side_encoder.get_encodings(
                    BERTWithEmbeddings.product_index_to_embedding
                )
            )
        elif self.red_method == "LDA":
            class_labels = BERTWithEmbeddings.product_embeddings["class"]
            if self.emb_dim > (
                max_reduced_dim_size := min(
                    BERTWithEmbeddings.product_index_to_embedding.shape[1],
                    len(np.unique(class_labels)) - 1,
                )
            ):
                n_components: int = max_reduced_dim_size
            else:
                n_components: int = self.emb_dim
            lda: LinearDiscriminantAnalysis = LinearDiscriminantAnalysis(
                n_components=n_components, **self.red_params
            )
            BERTWithEmbeddings.product_index_to_embedding_red = lda.fit_transform(
                BERTWithEmbeddings.product_index_to_embedding, class_labels
            )
        else:
            raise Exception(
                f"Unknown reduce method for embeddings. Got {self.red_method}"
            )

        ordering = list(dict(sorted(self.temp_model.id_reducer.id_lookup.items())).values())
        ordering = [BERTWithEmbeddings.product_id_to_index[i] for i in ordering]
        # reduced_embeddings = red[ordering]
        reduced_embeddings = np.array(
            [BERTWithEmbeddings.product_index_to_embedding_red[ordering]]
        )

        # mask_embedding = np.array([
        #     self.temp_model.model.embedding_layer.item_emb.weight.data[-1].cpu().numpy()
        # ])
        # final_embeddings = np.vstack([reduced_embeddings, mask_embedding])
        mask_embedding = np.array(
            [[self.temp_model.model.embedding_layer.item_emb.weight[-1].detach().cpu().numpy()]]
        )
        reduced_embeddings = np.concatenate(
            [reduced_embeddings, mask_embedding], axis=1
        ).squeeze(0)

        with torch.no_grad():
            self.temp_model.model.embedding_layer.item_emb.weight.copy_(
                torch.tensor(reduced_embeddings, dtype=torch.float32).to(
                    self.temp_model.device))
        self.temp_model.model.embedding_layer.item_emb.weight.requires_grad = True

        # self.temp_model.model.compile_model()

        super().train(train_data)

    def get_model(self, data: pd.DataFrame):
        if self.filepath_weights is not None:
            return super().get_model(data)
        return self.temp_model.model

    def predict(self, predict_data: Any, top_k: int = 10) -> dict[int, np.ndarray]:
        if self.filepath_weights is not None:
            return super().predict(predict_data, top_k)
        return self.temp_model.predict(predict_data, top_k)

    def name(self) -> str:
        return "LLM2BERT4Rec"