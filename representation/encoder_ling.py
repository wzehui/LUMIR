import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torch.amp import autocast

class MultisourceTextFusionEncoder(nn.Module):
    def __init__(self, config, used_modalities=None):
        super().__init__()

        encoder_name = config.get("encoder_name", "Linq-AI-Research/Linq-Embed-Mistral")
        local_dir = config.get("encoder_local_dir", "../models/LinqEmbedMistral")
        self.max_length = config.get("max_length", 256)

        self.freeze_encoder = config.get("freeze_encoder", True)
        self.train_lora_only = config.get("train_lora_only", False)

        self.static_embedding_cache = None
        self.static_embedding_dir = config['paths']['embedding']

        if used_modalities is None:
            used_modalities = config["used_modalities"]
        self.used_modalities = used_modalities

        # Projection after pooled token
        hidden_size = AutoModel.from_pretrained(local_dir).config.hidden_size
        self.projection_layer = nn.Linear(hidden_size, config["model"]["embedding_dim"])
        print(f"✅ Shared projection layer: {hidden_size} → {config['model']['embedding_dim']}")

        # Auto-download if missing
        if not os.path.exists(local_dir) or not os.path.isdir(local_dir):
            print(f"🔍 Local model '{local_dir}' not found. Downloading...")
            snapshot_download(repo_id=encoder_name, local_dir=local_dir)
            print(f"✅ Downloaded to {local_dir}")

        self.encoder = AutoModel.from_pretrained(local_dir, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)

        # Apply LoRA
        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION"
        )
        self.encoder = get_peft_model(self.encoder, lora_config)

        # Trainability
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("🔒 Encoder frozen.")
        else:
            # LoRA-only or full finetune
            if self.train_lora_only:
                for name, p in self.encoder.named_parameters():
                    p.requires_grad = ("lora_" in name)
                print("🔓 Encoder: LoRA-only trainable.")
            else:
                for p in self.encoder.parameters():
                    p.requires_grad = True
                print("🔓 Encoder: fully trainable.")

            # Stability for training
            if hasattr(self.encoder, "config"):
                self.encoder.config.use_cache = False
            if hasattr(self.encoder, "gradient_checkpointing_disable"):
                self.encoder.gradient_checkpointing_disable()

    def _encode_in_chunks(self, tokenized, chunk_size=2):
        """
        Enable-grad micro-batch forward to reduce VRAM peak for LoRA training.
        Returns pooled CLS embeddings in float32.
        """
        outs = []
        total = tokenized["input_ids"].size(0)
        with torch.enable_grad():
            for i in range(0, total, chunk_size):
                sl = slice(i, i + chunk_size)
                mini = {k: v[sl] for k, v in tokenized.items()}
                with autocast("cuda", dtype=torch.float16):
                    encoded = self.encoder(**mini).last_hidden_state[:, 0, :]
                outs.append(encoded.float())
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0)

    def forward(self, text_dict, item_ids=None, use_precomputed_embedding=True):
        # Static cache path when frozen
        if self.freeze_encoder and use_precomputed_embedding:
            if self.static_embedding_cache is None:
                print(f"🔄 Loading static embeddings from {self.static_embedding_dir}...")
                self.static_embedding_cache = {}
                for modality in self.used_modalities:
                    modality_path = os.path.join(self.static_embedding_dir, f"{modality}_raw.pt")
                    print(f"→ Loading {modality} from {modality_path}")
                    data = torch.load(modality_path, map_location="cpu")
                    for iid, vec in zip(data["item_id"], data["embedding"]):
                        self.static_embedding_cache.setdefault(iid, {})[modality] = vec.numpy()

            batch_embeddings = {}
            for modality in self.used_modalities:
                emb_list = []
                for iid in item_ids:
                    emb_tensor = torch.tensor(self.static_embedding_cache[iid][modality]).to(self.encoder.device)
                    projected = self.projection_layer(emb_tensor.unsqueeze(0)).squeeze(0)
                    projected = F.normalize(projected, dim=-1)
                    emb_list.append(projected)
                batch_embeddings[modality] = torch.stack(emb_list)
            return batch_embeddings

        # Online encoding branch (LoRA training)
        output = {}
        for key, texts in text_dict.items():
            cleaned_texts = [str(t) if t is not None else "" for t in texts]
            tokenized = self.tokenizer(
                cleaned_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            ).to(self.encoder.device)
            try:
                pooled = self._encode_in_chunks(tokenized, chunk_size=2)  # adjust chunk_size if needed
                projected = self.projection_layer(pooled)
                projected = F.normalize(projected, dim=-1)
                output[key] = projected
            except torch.cuda.OutOfMemoryError:
                print(f"❌ OOM during forward for modality '{key}' with batch size {len(texts)}")
                torch.cuda.empty_cache()
                raise
        return output

    def save_static_embedding(self, dataloader, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for modality in self.used_modalities:
            save_path = os.path.join(save_dir, f"{modality}_raw.pt")
            if os.path.exists(save_path):
                print(f"🟢 {modality} already exists at {save_path}, skipping.")
                continue

            print(f"Saving {modality} raw embedding to {save_path}...")
            all_item_ids, all_embeddings = [], []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"[{modality}] Encoding (raw)"):
                    batch = [item for item in batch if item is not None]
                    item_ids = [item["ItemId"] for item in batch]
                    text_dict = {k: [item[k] for item in batch] for k in self.used_modalities}
                    cleaned_texts = [str(t) if t is not None else "" for t in text_dict[modality]]
                    tokenized = self.tokenizer(
                        cleaned_texts, padding=True, truncation=True,
                        max_length=self.max_length, return_tensors="pt"
                    ).to(self.encoder.device)
                    encoded = self.encoder(**tokenized).last_hidden_state[:, 0, :].float()
                    all_item_ids.extend(item_ids)
                    all_embeddings.append(encoded.cpu())
            all_embeddings = torch.cat(all_embeddings, dim=0)
            assert len(all_item_ids) == all_embeddings.shape[0], "Mismatch in ItemId count!"
            torch.save({"item_id": all_item_ids, "embedding": all_embeddings}, save_path)
            print(f"✅ Saved {modality} raw embedding: {all_embeddings.shape} → {save_path}")

    def get_trainable_parameters(self):
        params = list(self.projection_layer.parameters())
        if not self.freeze_encoder:
            params += list(self.encoder.parameters())
        return params