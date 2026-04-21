import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model

class MultisourceTextFusionEncoder(nn.Module):
    def __init__(self, encoder_name="Linq-AI-Research/Linq-Embed-Mistral", max_length=256, local_dir="../models/LinqEmbedMistral", freeze_encoder=True, used_modalities=None):
        super().__init__()
        self.encoder_name = encoder_name
        self.local_dir = local_dir
        self.max_length = max_length
        self.freeze_encoder = freeze_encoder

        # === Add used_modalities ===
        if used_modalities is None:
            used_modalities = [
                'category',
                'location',
                'review_summary',
                'photo_summary',
                'rating_score',
                'attributes',
                'neighbor'
            ]
        self.used_modalities = used_modalities

        # === Step 1: Auto-download model if not exists ===
        if not os.path.exists(local_dir) or not os.path.isdir(local_dir):
            print(f"🔍 Local model directory '{local_dir}' not found. Downloading from HuggingFace Hub...")
            snapshot_download(
                repo_id=self.encoder_name,
                local_dir=local_dir,
            )
            print(f"✅ Download complete. Model saved to: {local_dir}")

        # === Step 2: Load encoder & tokenizer ===
        self.encoder = AutoModel.from_pretrained(local_dir, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)

        # === Step 3: Apply LoRA (optional fine-tuning hooks) ===
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.encoder = get_peft_model(self.encoder, lora_config)

        # === Step 4: Freeze encoder if needed ===
        if self.freeze_encoder:
            self._freeze_encoder()
            print("🔒 Encoder parameters frozen (no fine-tuning).")
        else:
            for name, param in self.encoder.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("🔓 Encoder: Only LoRA adapter is trainable.")

        # === Step 5: Projection layers (set dynamically)
        self.projection_layers = nn.ModuleDict()

        # Auto-init projection dims for used_modalities (default 768)
        self.set_projection_dims({modality: 768 for modality in self.used_modalities})

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def set_projection_dims(self, dim_map: dict):
        """
        Dynamically initialize projection layers for each modality.
        Args:
            dim_map (dict): modality -> target_dim
        """
        hidden_size = self.encoder.config.hidden_size
        for modality, dim in dim_map.items():
            self.projection_layers[modality] = nn.Linear(hidden_size, dim)

    def forward(self, text_dict):
        """
        Args:
            text_dict (dict): modality -> list[str]
        Returns:
            dict: modality -> [B, D_target]
        """
        output = {}
        for key, texts in text_dict.items():
            try:
                cleaned_texts = [str(t) if t is not None else "" for t in texts]
                tokenized = self.tokenizer(
                    cleaned_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.encoder.device)

                encoded = self.encoder(**tokenized)
                pooled = encoded.last_hidden_state[:, 0, :].float()
                if key not in self.projection_layers:
                    raise ValueError(
                        f"Missing projection layer for modality: {key}")
                projection_layer = self.projection_layers[key].to(
                    pooled.dtype).to(pooled.device)
                projected = projection_layer(pooled)
                output[key] = projected

            except Exception as e:
                print(f"❌ Error processing modality '{key}': {e}")
                raise e

        return output