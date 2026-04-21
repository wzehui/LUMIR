import pandas as pd
import torch.nn as nn

from backbone.transformer.bert.bert_model import BERTModel
from backbone.transformer.bert.custom_generators.test_generator import TestGenerator
from backbone.transformer.bert.custom_generators.train_generator import TrainGenerator
from backbone.transformer.transformer import Transformer


class BERT(Transformer):
    """BERT4REC implementation for sequential recommendation."""

    def __init__(
        self,
        mask_prob: float = 0.4,
        **transformer_kwargs,
    ) -> None:
        self.mask_prob = mask_prob
        transformer_kwargs["trans_dim_scale"] = 4
        super().__init__(**transformer_kwargs)

    def get_model(self, data: pd.DataFrame) -> nn.Module:
        return BERTModel(
            N=self.N,
            L=self.L,
            h=self.h,
            emb_dim=self.emb_dim,
            trans_dim_scale=self.trans_dim_scale,
            drop_rate=self.drop_rate,
            activation=self.activation,
            optimizer_kwargs=self.optimizer_kwargs,
            transformer_layer_kwargs=self.transformer_layer_kwargs,
            num_items=self.data_description["num_items"],
        )

    def get_train_generator(self, data: pd.DataFrame, batch_size: int):
        return TrainGenerator(
            data,
            self.N,
            batch_size,
            self.mask_prob,
            self.data_description,
        )

    def get_test_generator(self, data: pd.DataFrame, for_prediction: bool, batch_size: int):
        return TestGenerator(
            data, self.N, self.data_description, for_prediction=for_prediction
        )

    def name(self) -> str:
        return "BERT4REC"