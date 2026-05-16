
import os
import csv
import pickle
import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from collections import Counter
import pandas as pd
import random
from clinical_feature_CN import get_clinical_processor


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set: {seed}")


class SubtypeLoader:
    
    SUBTYPE_MAP = {
        'HR+HER2+': 0,
        'HR+HER2-': 1,
        'HR-HER2+': 2,
        'HR-HER2-': 3,
        'TNBC': 3,
    }
    
    def __init__(self, clinical_csv_path, num_subtypes=4):
        self.num_subtypes = num_subtypes
        self.slide_to_subtype = {}
        self.slide_to_label = {}
        
        if clinical_csv_path and os.path.exists(clinical_csv_path):
            self._load_from_csv(clinical_csv_path)
        else:
            print(f"Clinical CSV not found: {clinical_csv_path}")
    
    def _load_from_csv(self, csv_path):
        print(f"\nLoading clinical info: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            slide_id = row['slide_id']
            
            if 'subtype' in row:
                subtype_str = row['subtype']
                if subtype_str in self.SUBTYPE_MAP:
                    subtype_idx = self.SUBTYPE_MAP[subtype_str]
                elif 'subtype_label' in row:
                    subtype_idx = int(row['subtype_label'])
                else:
                    subtype_idx = 0
            elif 'subtype_label' in row:
                subtype_idx = int(row['subtype_label'])
            else:
                subtype_idx = 0
            
            self.slide_to_subtype[slide_id] = subtype_idx
            
            if 'recurrence' in row:
                self.slide_to_label[slide_id] = int(row['recurrence'])
        
        print(f"   Loaded {len(self.slide_to_subtype)} samples with subtype info")
        
        subtype_counts = Counter(self.slide_to_subtype.values())
        print(f"   Subtype distribution:")
        for subtype_idx, count in sorted(subtype_counts.items()):
            subtype_name = [k for k, v in self.SUBTYPE_MAP.items() if v == subtype_idx]
            subtype_name = subtype_name[0] if subtype_name else f"Type{subtype_idx}"
            print(f"      {subtype_name}: {count}")
    
    def get_subtype_onehot(self, slide_id):
        subtype_idx = self.slide_to_subtype.get(slide_id, 0)
        onehot = np.zeros(self.num_subtypes, dtype=np.float32)
        onehot[subtype_idx] = 1.0
        return onehot
    
    def get_subtype_idx(self, slide_id):
        return self.slide_to_subtype.get(slide_id, 0)


_subtype_loader = None

def get_subtype_loader(clinical_csv_path=None, num_subtypes=4):
    global _subtype_loader
    if _subtype_loader is None and clinical_csv_path:
        _subtype_loader = SubtypeLoader(clinical_csv_path, num_subtypes)
    return _subtype_loader


class PseudoBagAugmentation:
    def __init__(self, 
                 target_ratio=3.0,
                 patch_sample_ratio=0.7,
                 mix_prob=0.3,
                 noise_std=0.02,
                 k_neighbors=8,
                 min_patches=2):
        self.target_ratio = target_ratio
        self.patch_sample_ratio = patch_sample_ratio
        self.mix_prob = mix_prob
        self.noise_std = noise_std
        self.k_neighbors = k_neighbors
        self.min_patches = min_patches
    
    def generate_pseudo_bag_from_single(self, features, edge_index, coords, 
                                       patch_names, add_noise=True):
        if isinstance(features, dict):
            pseudo_features = {}
            pseudo_edge_indices = {}
            pseudo_coords = {}
            pseudo_patch_names = {}
            
            for mag in features.keys():
                feat = features[mag]
                n_patches = feat.size(0)
                
                n_sample = max(int(n_patches * self.patch_sample_ratio), self.min_patches)
                n_sample = min(n_sample, n_patches)
                
                if n_patches < self.min_patches:
                    indices = torch.randint(0, n_patches, (self.min_patches,))
                else:
                    indices = torch.randperm(n_patches)[:n_sample]
                
                sampled_feat = feat[indices]
                
                if add_noise and self.noise_std > 0:
                    noise = torch.randn_like(sampled_feat) * self.noise_std
                    sampled_feat = sampled_feat + noise
                
                pseudo_features[mag] = sampled_feat
                
                if coords and mag in coords:
                    sampled_coords = coords[mag][indices]
                    pseudo_coords[mag] = sampled_coords
                    coords_np = sampled_coords.numpy()
                    pseudo_edge_indices[mag] = build_knn_graph_cpu(coords_np, self.k_neighbors)
                
                if patch_names and mag in patch_names:
                    original_names = patch_names[mag]
                    pseudo_patch_names[mag] = [
                        f"{original_names[i % len(original_names)]}_pseudo" 
                        for i in indices.tolist()
                    ]
            
            return pseudo_features, pseudo_edge_indices, pseudo_coords, pseudo_patch_names
        
        else:
            n_patches = features.size(0)
            n_sample = max(int(n_patches * self.patch_sample_ratio), self.min_patches)
            n_sample = min(n_sample, n_patches)
            
            if n_patches < self.min_patches:
                indices = torch.randint(0, n_patches, (self.min_patches,))
            else:
                indices = torch.randperm(n_patches)[:n_sample]
            
            sampled_feat = features[indices]
            
            if add_noise and self.noise_std > 0:
                noise = torch.randn_like(sampled_feat) * self.noise_std
                sampled_feat = sampled_feat + noise
            
            sampled_coords = coords[indices] if coords is not None else None
            pseudo_edge = build_knn_graph_cpu(sampled_coords.numpy(), self.k_neighbors) \
                         if sampled_coords is not None else None
            
            return sampled_feat, pseudo_edge, sampled_coords, None
    
    def generate_pseudo_bag_from_mix(self, features1, features2, 
                                    coords1, coords2, 
                                    patch_names1, patch_names2):
        """混合两个袋生成伪袋"""
        if isinstance(features1, dict):
            pseudo_features = {}
            pseudo_edge_indices = {}
            pseudo_coords = {}
            pseudo_patch_names = {}
            
            for mag in features1.keys():
                if mag not in features2:
                    continue
                
                feat1 = features1[mag]
                feat2 = features2[mag]
                
                n1 = feat1.size(0)
                n2 = feat2.size(0)
                
                target_total = max(int((n1 + n2) * self.patch_sample_ratio * 0.5), self.min_patches)
                n_sample1 = max(target_total // 2, 1)
                n_sample2 = max(target_total - n_sample1, 1)
                
                if n1 < n_sample1:
                    indices1 = torch.randint(0, n1, (n_sample1,))
                else:
                    indices1 = torch.randperm(n1)[:n_sample1]
                
                if n2 < n_sample2:
                    indices2 = torch.randint(0, n2, (n_sample2,))
                else:
                    indices2 = torch.randperm(n2)[:n_sample2]
                
                sampled_feat1 = feat1[indices1]
                sampled_feat2 = feat2[indices2]
                
                mixed_feat = torch.cat([sampled_feat1, sampled_feat2], dim=0)
                
                if self.noise_std > 0:
                    noise = torch.randn_like(mixed_feat) * self.noise_std
                    mixed_feat = mixed_feat + noise
                
                pseudo_features[mag] = mixed_feat
                
                if coords1 and mag in coords1 and coords2 and mag in coords2:
                    sampled_coords1 = coords1[mag][indices1]
                    sampled_coords2 = coords2[mag][indices2]
                    mixed_coords = torch.cat([sampled_coords1, sampled_coords2], dim=0)
                    pseudo_coords[mag] = mixed_coords
                    coords_np = mixed_coords.numpy()
                    pseudo_edge_indices[mag] = build_knn_graph_cpu(coords_np, self.k_neighbors)
            
            return pseudo_features, pseudo_edge_indices, pseudo_coords, pseudo_patch_names
        
        return features1, None, coords1, None
    
    def augment_minority_class(self, slide_names, features_list, edge_indices_list,
                               coords_list, patch_names_list, labels, subtypes,
                               clinical_features_list,
                               minority_label=1):
        """扩充少数类数据"""
        label_list = [l.item() if isinstance(l, torch.Tensor) else l for l in labels]
        class_counts = Counter(label_list)
        
        if len(class_counts) < 2:
            print("   Only one class, no augmentation needed")
            return (slide_names, features_list, edge_indices_list,
                   coords_list, patch_names_list, labels, subtypes, clinical_features_list)
        
        n_majority = class_counts[1 - minority_label]
        n_minority = class_counts[minority_label]
        
        if n_minority == 0:
            print("   Minority class is empty, cannot augment")
            return (slide_names, features_list, edge_indices_list,
                   coords_list, patch_names_list, labels, subtypes, clinical_features_list)
        
        current_ratio = n_majority / (n_minority + 1e-6)
        

        
        target_minority = int(n_majority / self.target_ratio)
        n_pseudo = max(target_minority - n_minority, 0)
        
        
        if n_pseudo == 0:
            return (slide_names, features_list, edge_indices_list,
                   coords_list, patch_names_list, labels, subtypes, clinical_features_list)
        
        minority_indices = [i for i, l in enumerate(label_list) if l == minority_label]
        
        new_slide_names = []
        new_features = []
        new_edge_indices = []
        new_coords = []
        new_patch_names = []
        new_labels = []
        new_subtypes = []
        new_clinical_features = []
        
        for i in tqdm(range(n_pseudo), desc="Generating pseudo-bags"):
            if np.random.rand() < self.mix_prob and len(minority_indices) >= 2:
                idx1, idx2 = np.random.choice(minority_indices, 2, replace=False)
                
                pseudo_feat, pseudo_edge, pseudo_coord, pseudo_names = \
                    self.generate_pseudo_bag_from_mix(
                        features_list[idx1], features_list[idx2],
                        coords_list[idx1] if coords_list else None,
                        coords_list[idx2] if coords_list else None,
                        patch_names_list[idx1] if patch_names_list else None,
                        patch_names_list[idx2] if patch_names_list else None
                    )
                
                slide_name = f"{slide_names[idx1]}_pseudo_mix_{i}"
                subtype = subtypes[idx1]
                clinical_feat = clinical_features_list[idx1]
            else:
                idx = np.random.choice(minority_indices)
                
                pseudo_feat, pseudo_edge, pseudo_coord, pseudo_names = \
                    self.generate_pseudo_bag_from_single(
                        features_list[idx], 
                        edge_indices_list[idx] if edge_indices_list else None,
                        coords_list[idx] if coords_list else None,
                        patch_names_list[idx] if patch_names_list else None
                    )
                
                slide_name = f"{slide_names[idx]}_pseudo_{i}"
                subtype = subtypes[idx]
                clinical_feat = clinical_features_list[idx]
            
            new_slide_names.append(slide_name)
            new_features.append(pseudo_feat)
            new_edge_indices.append(pseudo_edge)
            new_coords.append(pseudo_coord)
            new_patch_names.append(pseudo_names)
            new_labels.append(torch.tensor(minority_label, dtype=torch.long))
            new_subtypes.append(subtype)
            new_clinical_features.append(clinical_feat)
        
        augmented_slide_names = slide_names + new_slide_names
        augmented_features = features_list + new_features
        augmented_edge_indices = edge_indices_list + new_edge_indices
        augmented_coords = (coords_list + new_coords) if coords_list else None
        augmented_patch_names = (patch_names_list + new_patch_names) if patch_names_list else None
        augmented_labels = labels + new_labels
        augmented_subtypes = subtypes + new_subtypes
        augmented_clinical_features = clinical_features_list + new_clinical_features
        
        final_label_list = [l.item() if isinstance(l, torch.Tensor) else l 
                           for l in augmented_labels]
        final_counts = Counter(final_label_list)
        final_ratio = final_counts[1 - minority_label] / (final_counts[minority_label] + 1e-6)
        

        
        return (augmented_slide_names, augmented_features, augmented_edge_indices,
                augmented_coords, augmented_patch_names, augmented_labels, augmented_subtypes,
                augmented_clinical_features)


def build_knn_graph_cpu(coords_np, k_neighbors=8):
    num_patches = len(coords_np)
    if num_patches == 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    k = min(k_neighbors, num_patches - 1)
    if k <= 0:
        return torch.tensor([[0], [0]], dtype=torch.long)
    adj = kneighbors_graph(coords_np, k, mode='connectivity', include_self=False)
    edge_index = torch.from_numpy(np.array(adj.nonzero())).long()
    return edge_index


def load_single_slide_features_v2(slide_name, features_dir, magnifications, 
                                   subtype_loader=None):
    slide_data = {}
    
    try:
        for mag in magnifications:
            pt_file = os.path.join(features_dir, mag, f'{slide_name}.pt')
            pkl_file = os.path.join(features_dir, mag, f'{slide_name}.pkl')
            
            if os.path.exists(pt_file):
                features = torch.load(pt_file, map_location='cpu')
                num_patches = features.size(0)
                
                grid_size = int(np.ceil(np.sqrt(num_patches)))
                coords = []
                for i in range(num_patches):
                    x = (i % grid_size) * 256
                    y = (i // grid_size) * 256
                    coords.append([x, y])
                coords = np.array(coords, dtype=np.float32)
                
                slide_data[mag] = {
                    'features': [
                        {
                            'feature': features[i].numpy(),
                            'coords': coords[i],
                            'file_name': f'patch_{i}.jpg'
                        }
                        for i in range(num_patches)
                    ],
                    'num_patches': num_patches,
                    'feature_dim': features.size(1)
                }
                
            elif os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as f:
                    mag_data = pickle.load(f)
                slide_data[mag] = mag_data
        
        if subtype_loader and slide_data:
            for mag in slide_data:
                slide_data[mag]['subtype_onehot'] = subtype_loader.get_subtype_onehot(slide_name)
        
        return slide_name, slide_data, len(slide_data) > 0
        
    except Exception as e:
        print(f"Error loading {slide_name}: {e}")
        return slide_name, {}, False


def load_multiscale_features_threaded_v2(slide_names, features_dir, magnifications, 
                                          n_workers=0, clinical_csv=None, num_subtypes=4):
    mDATA = {mag: {} for mag in magnifications}
    
    subtype_loader = get_subtype_loader(clinical_csv, num_subtypes)
    
    if len(slide_names) == 0:
        return mDATA, subtype_loader
    
    slide_names_sorted = sorted(list(slide_names))
    
    print(f"Using {n_workers} threads loading {len(slide_names_sorted)} slides...")
    
    load_func = partial(load_single_slide_features_v2,
                       features_dir=features_dir,
                       magnifications=magnifications,
                       subtype_loader=subtype_loader)
    
    successful_loads = 0
    failed_loads = 0
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {slide_name: executor.submit(load_func, slide_name)
                  for slide_name in slide_names_sorted}
        
        with tqdm(total=len(slide_names_sorted), desc="Loading features", ncols=100) as pbar:
            for slide_name in slide_names_sorted:
                future = futures[slide_name]
                _, slide_data, success = future.result()
                if success and slide_data:
                    for mag, mag_data in slide_data.items():
                        mDATA[mag][slide_name] = mag_data
                    successful_loads += 1
                else:
                    failed_loads += 1
                pbar.update(1)
    
    print(f"Loading complete: success {successful_loads}, failed {failed_loads}")
    return mDATA, subtype_loader


def organize_multiscale_data_sequential_v2(multiscale_mDATA, slide_to_label, 
                                            magnifications, k_neighbors=8,
                                            subtype_loader=None, num_subtypes=4,
                                            use_clinical=False, clinical_processor=None):
    SlideNames = []
    MultiscaleFeatures = []
    MultiscaleCoords = []
    MultiscaleEdgeIndices = []
    MultiscalePatchNames = []
    Labels = []
    Subtypes = []
    ClinicalFeatures = []
    
    all_slides = set()
    for mag_data in multiscale_mDATA.values():
        all_slides.update(mag_data.keys())
    
    all_slides_sorted = sorted(list(all_slides))
    
    print(f"Organizing {len(all_slides_sorted)} slides...")
    for slide_name in tqdm(all_slides_sorted, desc="Organizing", ncols=100):
        if slide_name not in slide_to_label:
            continue
        
        has_data = False
        for mag in magnifications:
            if slide_name in multiscale_mDATA[mag]:
                has_data = True
                break
        
        if not has_data:
            continue
        
        slide_features = {}
        slide_coords = {}
        slide_edge_indices = {}
        slide_patch_names = {}
        for mag in magnifications:
            if slide_name in multiscale_mDATA[mag]:
                slide_data = multiscale_mDATA[mag][slide_name]
                
                feat_list = []
                coord_list = []
                name_list = []
                
                for patch_data in slide_data['features']:
                    feature = patch_data['feature']
                    if isinstance(feature, np.ndarray):
                        feat_list.append(torch.from_numpy(feature).float())
                    else:
                        feat_list.append(torch.tensor(feature, dtype=torch.float32))
                    
                    coords = patch_data['coords']
                    if isinstance(coords, np.ndarray):
                        coord_list.append(torch.from_numpy(coords).float())
                    else:
                        coord_list.append(torch.tensor(coords, dtype=torch.float32))
                    
                    name_list.append(patch_data.get('file_name', f'patch_{len(name_list)}'))
                
                if len(feat_list) > 0:
                    slide_features[mag] = torch.stack(feat_list)
                    slide_coords[mag] = torch.stack(coord_list)
                    slide_patch_names[mag] = name_list
                    coords_np = slide_coords[mag].numpy()
                    slide_edge_indices[mag] = build_knn_graph_cpu(coords_np, k_neighbors)
        
        clinical_feat = None
        if use_clinical and clinical_processor is not None:
            clinical_vec = clinical_processor.get_clinical_features(slide_name)
            clinical_feat = clinical_vec.unsqueeze(0)
        
        if len(slide_features) > 0:
            SlideNames.append(slide_name)
            MultiscaleFeatures.append(slide_features)
            MultiscaleCoords.append(slide_coords)
            MultiscaleEdgeIndices.append(slide_edge_indices)
            MultiscalePatchNames.append(slide_patch_names)
            Labels.append(torch.tensor(slide_to_label[slide_name], dtype=torch.long))
            
            if subtype_loader:
                subtype_onehot = subtype_loader.get_subtype_onehot(slide_name)
                Subtypes.append(torch.from_numpy(subtype_onehot).float())
            else:
                Subtypes.append(torch.zeros(num_subtypes))
            
            ClinicalFeatures.append(clinical_feat)
    
    print(f"Organized: {len(SlideNames)} slides")
    if use_clinical:
        valid_clinical = sum(1 for c in ClinicalFeatures if c is not None)
        print(f"   Clinical features: {valid_clinical}/{len(ClinicalFeatures)} valid")
    
    return SlideNames, MultiscaleFeatures, MultiscaleCoords, MultiscaleEdgeIndices, \
           MultiscalePatchNames, Labels, Subtypes, ClinicalFeatures


def load_fold_data(csv_path):
    train_slide_to_label = {}
    val_slide_to_label = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_slide = row['train'].strip()
            train_label_str = row['train_label'].strip()
            if train_slide and train_label_str:
                train_label = int(train_label_str)
                train_slide_to_label[train_slide] = train_label
            
            val_slide = row['val'].strip()
            val_label_str = row['val_label'].strip()
            if val_slide and val_label_str:
                val_label = int(val_label_str)
                val_slide_to_label[val_slide] = val_label
    
    return train_slide_to_label, val_slide_to_label


def load_test_data(csv_path):
    test_slide_to_label = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        print(f"Loading test CSV: {csv_path}")
        print(f"   Columns: {reader.fieldnames}")
        
        for row in reader:
            if 'test' in row and 'test_label' in row:
                slide_id = row['test'].strip()
                label_str = row['test_label'].strip()
            elif 'slide' in row:
                slide_id = row['slide'].strip()
                label_str = row.get('label', '0').strip()
            elif 'slide_id' in row:
                slide_id = row['slide_id'].strip()
                label_str = row.get('label', '0').strip()
            else:
                keys = list(row.keys())
                if len(keys) >= 2:
                    slide_id = row[keys[0]].strip()
                    label_str = row[keys[1]].strip()
                else:
                    continue
            
            if slide_id and label_str:
                try:
                    label = int(label_str)
                    test_slide_to_label[slide_id] = label
                except ValueError:
                    print(f"Cannot parse label: {slide_id}={label_str}")
    
    print(f"   Loaded {len(test_slide_to_label)} test samples")
    return test_slide_to_label


def load_all_folds_combined(csv_dir, n_folds):
    combined_slide_to_label = {}
    
    for fold_idx in range(n_folds):
        csv_path = os.path.join(csv_dir, f'fold{fold_idx}.csv')
        if os.path.exists(csv_path):
            train_data, val_data = load_fold_data(csv_path)
            combined_slide_to_label.update(train_data)
            combined_slide_to_label.update(val_data)
    
    print(f"Merging {n_folds} folds, total {len(combined_slide_to_label)} samples")
    return combined_slide_to_label


def load_and_organize_data_with_cache(args, fold_idx, slide_to_label, phase):
    cache_path = None
    if hasattr(args, 'use_cache') and args.use_cache:
        cache_path = os.path.join(args.cache_dir, f'fold_{fold_idx}_{phase}_cache.pkl')
        if os.path.exists(cache_path) and not getattr(args, 'force_reload', False):
            print(f"Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    
    print(f"\nLoading {phase} data...")
    
    clinical_csv = getattr(args, 'clinical_csv', 
                          '/home/yujy/CAMS_data/pca/CAMS_PCA_result.csv')

    clinical_processor = None
    if getattr(args, 'use_clinical', False):
        clinical_processor = get_clinical_processor(
            clinical_csv, 
            cache_dir=getattr(args, 'cache_dir', None)
        )
    
    sorted_slide_names = sorted(list(slide_to_label.keys()))
    
    mDATA, subtype_loader = load_multiscale_features_threaded_v2(
        sorted_slide_names,
        args.features_dir,
        args.magnifications,
        n_workers=getattr(args, 'n_workers', 0),
        clinical_csv=clinical_csv,
        num_subtypes=getattr(args, 'num_subtypes', 4)
    )
    
    print(f"\nOrganizing {phase} data...")
    
    (slide_names, features, coords, edge_indices, 
     patch_names, labels, subtypes, clinical_features) = organize_multiscale_data_sequential_v2(
        mDATA,
        slide_to_label,
        args.magnifications,
        k_neighbors=getattr(args, 'k_neighbors', 8),
        subtype_loader=subtype_loader,
        num_subtypes=getattr(args, 'num_subtypes', 4),
        use_clinical=getattr(args, 'use_clinical', False),
        clinical_processor=clinical_processor
    )

    if phase == 'train' and getattr(args, 'use_pseudo_bag_aug', False) and len(labels) > 0:

        
        aug_seed = getattr(args, 'seed', 42)
        random.seed(aug_seed)
        np.random.seed(aug_seed)
        torch.manual_seed(aug_seed)
        
        augmenter = PseudoBagAugmentation(
            target_ratio=getattr(args, 'pseudo_bag_ratio', 3.0),
            patch_sample_ratio=getattr(args, 'pseudo_bag_sample_ratio', 0.7),
            mix_prob=getattr(args, 'pseudo_bag_mix_prob', 0.3),
            noise_std=getattr(args, 'pseudo_bag_noise_std', 0.02),
            k_neighbors=getattr(args, 'k_neighbors', 8),
            min_patches=getattr(args, 'pseudo_bag_min_patches', 2)
        )
        
        (slide_names, features, edge_indices, coords, patch_names, 
         labels, subtypes, clinical_features) = augmenter.augment_minority_class(
            slide_names, features, edge_indices, coords, patch_names,
            labels, subtypes, clinical_features, minority_label=1
        )
    
    cache_data = {
        'slide_names': slide_names,
        'features': features,
        'coords': coords,
        'edge_indices': edge_indices,
        'patch_names': patch_names,
        'labels': labels,
        'subtypes': subtypes,
        'clinical_features': clinical_features
    }
    
    if cache_path and hasattr(args, 'use_cache') and args.use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved: {cache_path}")
    
    return cache_data


def load_and_organize_test_data(args, test_slide_to_label):
    clinical_csv = getattr(args, 'clinical_csv',
                          '/home/yujy/CAMS_data/pca/CAMS_PCA_result.csv')
    
    clinical_processor = None
    if getattr(args, 'use_clinical', False):
        clinical_processor = get_clinical_processor(
            clinical_csv,
            cache_dir=getattr(args, 'cache_dir', None)
        )
    
    sorted_slide_names = sorted(list(test_slide_to_label.keys()))
    
    mDATA, subtype_loader = load_multiscale_features_threaded_v2(
        sorted_slide_names,
        args.features_dir,
        args.magnifications,
        n_workers=getattr(args, 'n_workers', 0),
        clinical_csv=clinical_csv,
        num_subtypes=getattr(args, 'num_subtypes', 4)
    )
    
    (slide_names, features, coords, edge_indices,
     patch_names, labels, subtypes, clinical_features) = organize_multiscale_data_sequential_v2(
        mDATA,
        test_slide_to_label,
        args.magnifications,
        k_neighbors=getattr(args, 'k_neighbors', 8),
        subtype_loader=subtype_loader,
        num_subtypes=getattr(args, 'num_subtypes', 4),
        use_clinical=getattr(args, 'use_clinical', False),
        clinical_processor=clinical_processor
    )
    
    return {
        'slide_names': slide_names,
        'features': features,
        'edge_indices': edge_indices,
        'labels': labels,
        'subtypes': subtypes,
        'clinical_features': clinical_features
    }

def custom_collate(batch):

    slide_names = [item[0] for item in batch]
    multiscale_features = [item[1] for item in batch]
    multiscale_edge_indices = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    subtypes = [item[4] for item in batch]
    clinical_features = [item[5] for item in batch]
    return slide_names, multiscale_features, multiscale_edge_indices, labels, subtypes, clinical_features


class MultiscaleDataset(torch.utils.data.Dataset):
    def __init__(self, slide_names, multiscale_features, multiscale_edge_indices,
                 labels, subtypes, clinical_features=None, augmentation=None, mode='train'):
        self.slide_names = slide_names
        self.multiscale_features = multiscale_features
        self.multiscale_edge_indices = multiscale_edge_indices
        self.labels = labels
        self.subtypes = subtypes
        self.clinical_features = clinical_features if clinical_features else [None] * len(slide_names)
        self.augmentation = augmentation
        self.mode = mode
    
    def __len__(self):
        return len(self.slide_names)
    
    def __getitem__(self, idx):
        features = self.multiscale_features[idx]
        label = self.labels[idx]
        
        if self.augmentation is not None and self.mode == 'train':
            features, label, _ = self.augmentation(features, label, mode=self.mode)
        
        return (
            self.slide_names[idx],
            features,
            self.multiscale_edge_indices[idx],
            label,
            self.subtypes[idx],
            self.clinical_features[idx]
        )


def calculate_class_weights(labels, num_classes, strategy='balanced_smooth',
                           beta=0.9999, temperature=0.5, smooth_factor=0.3):
    if isinstance(labels, list) and len(labels) > 0:
        if isinstance(labels[0], torch.Tensor):
            labels = [label.item() for label in labels]
    
    if len(labels) == 0:
        print("Warning: no valid samples, using uniform weights")
        return torch.ones(num_classes), np.zeros(num_classes)
    
    class_counts = Counter(labels)
    counts = np.array([class_counts.get(i, 0) for i in range(num_classes)])
    total_samples = sum(counts)
    
    if total_samples == 0:
        print("Warning: no valid samples, using uniform weights")
        return torch.ones(num_classes), counts
    
    counts = np.maximum(counts, 1)
    
    imbalance_ratio = counts.max() / counts.min()
    
    if strategy == 'effective':
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    elif strategy == 'inverse':
        weights = total_samples / (num_classes * counts + 1e-6)
    elif strategy == 'balanced':
        weights = total_samples / (counts + 1e-6)
        weights = weights / weights.min()
    elif strategy == 'balanced_smooth':
        weights = total_samples / (counts + 1e-6)
        weights = weights / weights.min()
        uniform_weights = np.ones(num_classes)
        weights = (1 - smooth_factor) * weights + smooth_factor * uniform_weights
    elif strategy == 'temperature':
        inverse_freq = total_samples / (counts + 1e-6)
        log_weights = np.log(inverse_freq + 1e-6)
        weights = np.exp(log_weights / temperature)
    else:
        weights = np.ones(num_classes)
    
    weights = weights / weights.sum() * num_classes
    
    return torch.FloatTensor(weights), counts


def calculate_dynamic_sample_weights(labels, class_weights, epoch, warmup_epochs=10,
                                    temperature=0.5, smooth_factor=0.3):
    if isinstance(labels, list) and len(labels) > 0:
        if isinstance(labels[0], torch.Tensor):
            labels = [label.item() for label in labels]
    
    base_sample_weights = [class_weights[label].item() for label in labels]
    base_sample_weights = np.array(base_sample_weights)
    
    log_weights = np.log(base_sample_weights + 1e-6)
    tempered_weights = np.exp(log_weights / temperature)
    
    uniform_weights = np.ones_like(tempered_weights)
    smoothed_weights = (1 - smooth_factor) * tempered_weights + smooth_factor * uniform_weights
    
    if epoch < warmup_epochs:
        alpha = 0.5 * (1 - np.cos(np.pi * epoch / warmup_epochs))
        final_weights = (1 - alpha) * uniform_weights + alpha * smoothed_weights
    else:
        final_weights = smoothed_weights
   
    final_weights = final_weights / final_weights.mean()
    
    return final_weights.tolist()


class BalancedUnderSampler:
    
    def __init__(self, target_ratio=3.0, minority_aug_ratio=1.5):
        self.target_ratio = target_ratio
        self.minority_aug_ratio = minority_aug_ratio
    
    def balance_dataset(self, slide_names, features, edge_indices,
                       coords, patch_names, labels, subtypes, clinical_features,
                       minority_label=1):
        majority_indices = [i for i, l in enumerate(labels) 
                          if (l.item() if isinstance(l, torch.Tensor) else l) != minority_label]
        minority_indices = [i for i, l in enumerate(labels)
                          if (l.item() if isinstance(l, torch.Tensor) else l) == minority_label]
        
        n_minority = len(minority_indices)
        n_majority = len(majority_indices)
        
        if n_minority == 0:
            print("Warning: minority class is empty, cannot balance")
            return (slide_names, features, edge_indices, coords, patch_names, labels, subtypes, clinical_features)
        
        print(f"\nBalanced sampling:")
        print(f"   Original majority: {n_majority}")
        print(f"   Original minority: {n_minority}")
        
        n_minority_target = int(n_minority * self.minority_aug_ratio)
        n_majority_target = int(n_minority_target * self.target_ratio)
        n_majority_target = min(n_majority_target, n_majority)
        
        sampled_majority_indices = random.sample(majority_indices, n_majority_target)
        final_indices = sampled_majority_indices + minority_indices
        
        new_slide_names = [slide_names[i] for i in final_indices]
        new_features = [features[i] for i in final_indices]
        new_edge_indices = [edge_indices[i] for i in final_indices]
        new_coords = [coords[i] for i in final_indices] if coords else None
        new_patch_names = [patch_names[i] for i in final_indices] if patch_names else None
        new_labels = [labels[i] for i in final_indices]
        new_subtypes = [subtypes[i] for i in final_indices]
        new_clinical_features = [clinical_features[i] for i in final_indices]
        
        print(f"   Balanced! Total samples: {len(final_indices)}")
        
        return (new_slide_names, new_features, new_edge_indices,
                new_coords, new_patch_names, new_labels, new_subtypes, new_clinical_features)