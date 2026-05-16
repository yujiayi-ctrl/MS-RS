import os
import numpy as np
import cv2
import PIL.Image as Image

from config import (
    enable_stain_normalization,
    stain_norm_reference_image,
    stain_norm_method
)


class StainNormalizer:
    """
    Stain Normalization Class - Supports Reinhard and Macenko Methods
    
    Reinhard Method: Statistical matching based on LAB color space, simple and efficient
    Macenko Method: Chromatic matrix decomposition based on OD space, more accurate but computationally intensive
    """
    
    def __init__(self, method='reinhard'):
        """
        Initialize the Stain Normalizer
        
        Parameters:
            method: Normalization method ('reinhard' recommended, or 'macenko')
        """
        self.method = method.lower()
        self.is_fitted = False
        
        # Reinhard method's target statistics
        self.target_mean_lab = None
        self.target_std_lab = None
        
        # Macenko method's parameters
        self.Io = 240  # Background light intensity
        self.alpha = 1  # Percentile parameter
        self.beta = 0.15  # OD threshold
        self.target_stain_matrix = None
        self.target_max_concentrations = None
    
    def _reinhard_fit(self, target_img):
        """Reinhard method: Calculate the LAB statistics of the target image"""
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        self.target_mean_lab = target_lab.mean(axis=(0, 1))
        self.target_std_lab = target_lab.std(axis=(0, 1))
        
    def _reinhard_transform(self, source_img):
        """Reinhard method: Match the color distribution of the source image to the target"""
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        src_mean = source_lab.mean(axis=(0, 1))
        src_std = source_lab.std(axis=(0, 1))
        src_std[src_std == 0] = 1
        
        result_lab = (source_lab - src_mean) * (self.target_std_lab / src_std) + self.target_mean_lab
        
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], 0, 255)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], 0, 255)
        
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return result
    
    def _get_stain_matrix(self, img):
        """Macenko method: Extract stain matrix"""
        img = img.astype(np.float64)
        img = np.maximum(img, 1)
        OD = -np.log10(img / self.Io)
        OD_flat = OD.reshape(-1, 3)
        
        # Filter background
        mask = np.all(OD_flat > self.beta, axis=1)
        OD_hat = OD_flat[mask]
        
        if len(OD_hat) < 100:
            return np.array([[0.650, 0.704, 0.286],
                           [0.072, 0.990, 0.105]]).T
        
        # PCA
        OD_centered = OD_hat - OD_hat.mean(axis=0)
        cov = np.cov(OD_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx[:2]]
        
        proj = np.dot(OD_hat, eigvecs)
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        
        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100 - self.alpha)
        
        vec1 = np.dot(eigvecs, np.array([np.cos(min_angle), np.sin(min_angle)]))
        vec2 = np.dot(eigvecs, np.array([np.cos(max_angle), np.sin(max_angle)]))
        
        vec1 = vec1 if vec1.sum() > 0 else -vec1
        vec2 = vec2 if vec2.sum() > 0 else -vec2
        
        if vec1[0] > vec2[0]:
            HE = np.array([vec1, vec2]).T
        else:
            HE = np.array([vec2, vec1]).T
        
        HE = HE / np.linalg.norm(HE, axis=0, keepdims=True)
        return HE
    
    def _get_concentrations(self, img, HE):
        """Macenko method: Calculate stain concentrations"""
        img = np.maximum(img.astype(np.float64), 1)
        OD = -np.log10(img / self.Io)
        OD_flat = OD.reshape(-1, 3)
        C = np.linalg.lstsq(HE, OD_flat.T, rcond=None)[0].T
        return C
    
    def _macenko_fit(self, target_img):
        """Macenko method: Fit the target image"""
        self.target_stain_matrix = self._get_stain_matrix(target_img)
        C_target = self._get_concentrations(target_img, self.target_stain_matrix)
        self.target_max_concentrations = np.percentile(C_target, 99, axis=0)
    
    def _macenko_transform(self, source_img):
        """Macenko method: Normalize the source image"""
        original_shape = source_img.shape
        HE_source = self._get_stain_matrix(source_img)
        C_source = self._get_concentrations(source_img, HE_source)
        
        maxC_source = np.percentile(C_source, 99, axis=0)
        maxC_source = np.maximum(maxC_source, 1e-6)
        
        C_normalized = C_source * (self.target_max_concentrations / maxC_source)
        OD_result = np.dot(C_normalized, self.target_stain_matrix.T)
        result = self.Io * np.power(10, -OD_result)
        result = np.clip(result, 0, 255).reshape(original_shape).astype(np.uint8)
        return result
    
    def fit(self, target_img):

        if isinstance(target_img, Image.Image):
            target_img = np.array(target_img)
        
        if target_img.shape[-1] == 4:  # RGBA
            target_img = target_img[:, :, :3]
        
        if self.method == 'reinhard':
            self._reinhard_fit(target_img)
            print(f"  ✓ Stain normalizer fitted (Reinhard method)")
            print(f"    Target LAB means: L={self.target_mean_lab[0]:.1f}, A={self.target_mean_lab[1]:.1f}, B={self.target_mean_lab[2]:.1f}")        else:
            self._macenko_fit(target_img)
            print(f"  ✓ Stain normalizer fitted (Macenko method)")
            print(f"    Target stain matrix:\n    H: {self.target_stain_matrix[:, 0]}")
            print(f"    E: {self.target_stain_matrix[:, 1]}")
        
        self.is_fitted = True
    
    def transform(self, source_img):

        if not self.is_fitted:
            raise RuntimeError("Please call the fit() method to fit the normalizer first")
        
        return_pil = isinstance(source_img, Image.Image)
        if return_pil:
            source_img = np.array(source_img)
        
        if source_img.shape[-1] == 4:  # RGBA
            source_img = source_img[:, :, :3]
        
        if self.method == 'reinhard':
            result = self._reinhard_transform(source_img)
        else:
            result = self._macenko_transform(source_img)
        
        if return_pil:
            return Image.fromarray(result)
        return result


_stain_normalizer = None


def initialize_stain_normalizer():
    """
    Initialize the global stain normalizer
    """
    global _stain_normalizer
    
    if not enable_stain_normalization:
        print("Stain normalization: disabled")
        return None
    
    if not os.path.exists(stain_norm_reference_image):
        print(f"⚠ Warning: Reference image for stain normalization does not exist: {stain_norm_reference_image}")
        print("  Stain normalization will be disabled")
        return None
    
    print(f"\nInitializing stain normalizer...")
    print(f"  Reference image: {stain_norm_reference_image}")
    print(f"  Normalization method: {stain_norm_method}")
    
    try:
        _stain_normalizer = StainNormalizer(method=stain_norm_method)
        reference_img = Image.open(stain_norm_reference_image).convert('RGB')
        _stain_normalizer.fit(np.array(reference_img))
        print(f"  ✓ Stain normalizer initialized successfully\n")
        return _stain_normalizer
    except Exception as e:
        print(f"  ✗ Stain normalizer initialization failed: {e}")
        print("  Stain normalization will be disabled\n")
        return None


def apply_stain_normalization(img):

    global _stain_normalizer
    
    if _stain_normalizer is None:
        return img
    
    try:
        return _stain_normalizer.transform(img)
    except Exception as e:
        return img


def get_stain_normalizer():
    global _stain_normalizer
    return _stain_normalizer
