import pandas as pd
import numpy as np
import torch
import pickle
import os


class ClinicalFeatureProcessor:

    PCA_FEATURES = [
        'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 
        'PCA_5', 'PCA_6', 'PCA_7'
    ]
    
    NUMERIC_FEATURES = [
        'pathology',
        'neoadjuvant',
        'eccentricity',
        'orientation',
        'age',
    ]
    
    ONEHOT_FEATURES = [
        'T_stage_T3',
        'T_stage_T4',
        'N_stage_N0',
        'N_stage_N1',
        'N_stage_N2',
        'N_stage_N3',
        'N_stage_NX',
        'TNM_stage_I',
        'TNM_stage_II',
        'TNM_stage_III',
        'age_group',
        'surgery_type',
    ]
    
    ALL_FEATURES = PCA_FEATURES + NUMERIC_FEATURES + ONEHOT_FEATURES
    
    RETAIN_COLS = ['slide_id', 'slide_path', 'subtype']
    
    def __init__(self, clinical_csv_path, cache_dir=None):
        self.clinical_csv_path = clinical_csv_path
        self.cache_dir = cache_dir
        
        self.df = pd.read_csv(clinical_csv_path)
        
        self._validate_columns()
        
        self._handle_missing_values()
        
        self.feature_dim = len(self.ALL_FEATURES)
    
    def _validate_columns(self):

        missing_cols = []
        for col in self.ALL_FEATURES:
            if col not in self.df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"❌ missing_cols: {missing_cols}")
        
    
    def _handle_missing_values(self):
        for col in self.ALL_FEATURES:
            if self.df[col].isnull().any():
                missing_count = self.df[col].isnull().sum()
                self.df[col].fillna(0.0, inplace=True)
    
    def get_clinical_features(self, slide_id):

        row = self.df[self.df['slide_id'] == slide_id]
        
        if len(row) == 0:
            print(f"Warning: slide_id={slide_id} not found, returning zero vector")
            return torch.zeros(self.feature_dim, dtype=torch.float32)
        
        row = row.iloc[0]
        features = [float(row[col]) for col in self.ALL_FEATURES]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_batch_clinical_features(self, slide_ids):
        features = [self.get_clinical_features(sid) for sid in slide_ids]
        return torch.stack(features)
    
    def get_feature_names(self):
        return self.ALL_FEATURES.copy()
    
    def get_feature_info(self):
        info = {
            'total_features': self.feature_dim,
            'pca_features': {
                'count': len(self.PCA_FEATURES),
                'names': self.PCA_FEATURES
            },
            'numeric_features': {
                'count': len(self.NUMERIC_FEATURES),
                'names': self.NUMERIC_FEATURES
            },
            'onehot_features': {
                'count': len(self.ONEHOT_FEATURES),
                'names': self.ONEHOT_FEATURES
            }
        }
        return info
    
    def get_statistics(self):
        stats = {}
        for col in self.ALL_FEATURES:
            stats[col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'missing_count': int(self.df[col].isnull().sum())
            }
        return stats
    
    def save_cache(self, cache_path):
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, cache_path)
        else:
            cache_file = cache_path
        
        cache_data = {
            'df': self.df,
            'feature_dim': self.feature_dim,
            'all_features': self.ALL_FEATURES,
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Clinical feature cache saved: {cache_file}")
    
    @classmethod
    def load_from_cache(cls, cache_path):
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        processor = cls.__new__(cls)
        processor.df = cache_data['df']
        processor.feature_dim = cache_data['feature_dim']
        processor.ALL_FEATURES = cache_data['all_features']
        
        print(f"Loading clinical features from cache: {cache_path}")
        print(f"   Feature dim: {processor.feature_dim}")
        return processor


_clinical_processor = None


def get_clinical_processor(clinical_csv_path=None, cache_dir=None):
    global _clinical_processor
    
    if _clinical_processor is None and clinical_csv_path:
        _clinical_processor = ClinicalFeatureProcessor(clinical_csv_path, cache_dir)
    
    return _clinical_processor


def create_clinical_features_for_slide(slide_id, clinical_processor):

    clinical_vec = clinical_processor.get_clinical_features(slide_id)
    

    features = clinical_vec.unsqueeze(0)
    
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return features, edge_index