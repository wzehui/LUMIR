from torch.utils.data import Dataset
from torch.utils.data import Sampler
import pandas as pd
import random
import json
from tqdm import tqdm
from utils import (generate_h3_features, process_attributes,
                   combine_attributes, safe_eval, combine_normalized_attributes)
from backbone.data.session_dataset import SessionDataset

class POITextDataset(Dataset):
    def __init__(self, product_csv_path, review_csv_path, photo_csv_path, itemid_map_path):
        self.texts = {}
        self.ids = []

        with open(itemid_map_path, 'r') as f:
            itemid_map = json.load(f)

        # --- Load product info ---
        product = pd.read_csv(product_csv_path, usecols=[
            'business_id', 'name', 'categories', 'address', 'city', 'state',
            'postal_code', 'latitude', 'longitude', 'attributes', 'stars'])

        product = product[product['business_id'].isin(itemid_map.keys())].dropna(subset=['name'])
        product['ItemId'] = product['business_id'].map(itemid_map)
        product = product.dropna(subset=['ItemId'])

        # --- Generate location text ---
        product["location_info"] = product.apply(
            lambda row: " | ".join(filter(None, [
                f"Address: {row['address']}" if pd.notna(row['address']) else "",
                f"City: {row['city']}" if pd.notna(row['city']) else "",
                f"State: {row['state']}" if pd.notna(row['state']) else "",
                f"Coordinates: ({row['latitude']:.6f}, {row['longitude']:.6f})"
                if pd.notna(row['latitude']) and pd.notna(row['longitude']) else ""
            ])), axis=1
        )

        product["attributes"] = product["attributes"].apply(lambda x: safe_eval(x) if pd.notna(x) else {})
        attribute_normalized = product["attributes"].apply(process_attributes).apply(pd.Series, dtype=object)
        product["attributes_text"] = attribute_normalized.apply(combine_normalized_attributes, axis=1)

        # --- Generate H3 neighborhood text ---
        product["h3_codes"] = product.apply(lambda row: generate_h3_features(row['latitude'], row['longitude']), axis=1)

        # --- Merge with review summary ---
        review = pd.read_csv(review_csv_path, usecols=['business_id','summary', 'keywords', 'themes', 'sentiment_score', 'sentiment_confidence'], encoding="utf-8-sig")
        product = product.merge(review, on="business_id", how="left")
        product["review_summary"] = product.apply(
            lambda row: f"Summary: {row['summary']} | Keywords: {row['keywords']} | Themes: {row['themes']}", axis=1)
        product["rating_score"] = product.apply(
            lambda row: f"Sentiment Score: {row['sentiment_score']:.2f} (Confidence: {row['sentiment_confidence']:.0%}) | Rating: {row['stars']} / 5", axis=1)

        # --- Merge with photo attributes ---
        photo = pd.read_csv(photo_csv_path, usecols=['business_id', 'summary', 'keywords', 'indoor_color_tone', 'venue_style',
                       'food_style', 'drink_style', 'target_audience', 'special_features'], encoding="utf-8-sig")
        product = product.merge(photo, on="business_id", how="left")
        product["photo_summary"] = product.apply(combine_attributes, axis=1)

        # --- Final Assembly ---
        for _, row in tqdm(product.iterrows(), desc="Assembling atomic features", total=len(product)):
            bid = row['business_id']
            if bid not in itemid_map:
                continue
            item_id = itemid_map[bid]
            self.ids.append(item_id)
            # self.texts[item_id] = {
            #     'name': row.get('name', ''),
            #     'category': row.get('categories', ''),
            #     'location': row.get('location_info', ''),
            #     'neighbor': row.get('h3_codes', ''),
            #     'review_summary': row.get('review_summary', ''),
            #     'rating_score': row.get('rating_score', ''),
            #     'photo_summary': row.get('photo_summary', ''),
            #     'attributes': row.get('attributes_text', '')
            # }
            self.texts[item_id] = {
                'meta': " ; ".join(filter(None, [
                    f"name: {row.get('name', '')}",
                    f"category: {row.get('categories', '')}",
                    f"location: {row.get('location_info', '')}",
                    f"neighbor: {row.get('h3_codes', '')}"
                ])),
                'review': " ; ".join(filter(None, [
                    row.get('review_summary', ''),
                    row.get('rating_score', '')
                ])),
                'photo': row.get('photo_summary', '')
            }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        return {'ItemId': id, **self.texts[id]}

class MultiViewPOIDataset(Dataset):
    def __init__(self, session_df_path, product_csv_path, review_csv_path,
                 photo_csv_path, itemid_map_path, subset_ratio=None,
                 subset_size=None, seed=2025):
        # 1. Load textual modality representations for all POIs (each POI is a dictionary)
        self.poi_dataset = POITextDataset(product_csv_path, review_csv_path, photo_csv_path, itemid_map_path)
        self.poi_texts = {poi["ItemId"]: poi for poi in self.poi_dataset}

        # 2. Load session data structure
        session_df = SessionDataset.from_pickle(session_df_path).get_train_data()
        self.session_dict = session_df.groupby("SessionId")["ItemId"].apply(list).to_dict()

        # 3. Build inverted index: map each POI to the sessions it appears in
        self.poi_to_sessions = {}
        for sid, items in self.session_dict.items():
            for idx, iid in enumerate(items):
                if iid not in self.poi_to_sessions:
                    self.poi_to_sessions[iid] = []
                self.poi_to_sessions[iid].append((sid, idx))

        # 4. Collect valid POI IDs and optionally reduce
        all_ids = list(self.poi_texts.keys())
        random.seed(seed)
        if subset_ratio is not None:
            k = int(len(all_ids) * subset_ratio)
            self.all_ids = random.sample(all_ids, k)
        elif subset_size is not None:
            self.all_ids = random.sample(all_ids, min(subset_size, len(all_ids)))
        else:
            self.all_ids = all_ids

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        anchor_id = self.all_ids[idx]
        anchor_poi = self.poi_texts[anchor_id]

        # === Modality dropout: Anchor is the original multimodal input ===
        view1 = anchor_poi  # Used as input for dropout-based augmentation

        # === Sequence-based positives: collect all valid neighbors from sessions ===
        seq_pos = []
        for (sid, i) in self.poi_to_sessions.get(anchor_id, []):
            seq = self.session_dict[sid]
            if i > 0:
                prev = seq[i - 1]
                if prev in self.poi_texts:
                    seq_pos.append(self.poi_texts[prev])
            if i < len(seq) - 1:
                nxt = seq[i + 1]
                if nxt in self.poi_texts:
                    seq_pos.append(self.poi_texts[nxt])

        # fallback: if no neighbors found, include anchor itself
        if not seq_pos:
            seq_pos.append(anchor_poi)

        return {
            "ItemId": anchor_id,
            **anchor_poi
        }


class SessionBatchSampler(Sampler):
    def __init__(self, session_dict, batch_size):
        self.session_dict = session_dict
        self.batch_size = batch_size
        self.session_ids = list(self.session_dict.keys())

        self.num_batches = sum(len(pois) for pois in session_dict.values()) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            sid = random.choice(self.session_ids)
            poi_seq = self.session_dict[sid]

            if len(poi_seq) >= self.batch_size:
                sampled_pois = random.sample(poi_seq, self.batch_size)
            else:
                sampled_pois = random.choices(poi_seq, k=self.batch_size)

            yield sampled_pois

    def __len__(self):
        return self.num_batches