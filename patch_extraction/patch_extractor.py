import os
import numpy as np

from config import resize_interpolation, jpeg_quality
from stain_normalizer import apply_stain_normalization, get_stain_normalizer


def extract_patches_from_annotations(tslide, tissue_mask, slide_save_dir, selected_levels, 
                                     mask_level, threshold=0.8, jpeg_quality=95):
    """
    Extracting patches from annotated regions (with stain normalization)
    """
    mask_sH, mask_sW = tissue_mask.shape
    slide_name = os.path.basename(slide_save_dir)
    
    print(f"  Annotated region mask shape: {tissue_mask.shape}")
    
    # 检查染色标准化状态
    _stain_normalizer = get_stain_normalizer()
    stain_norm_enabled = _stain_normalizer is not None
    if stain_norm_enabled:
        print(f"  ✓ Stain normalization: Enabled")
    else:
        print(f"  ⚠ Stain normalization: Disabled")
    
    patch_counts = {mag_name: 0 for mag_name in selected_levels.keys()}
    error_counts = {mag_name: 0 for mag_name in selected_levels.keys()}
    resize_counts = {mag_name: 0 for mag_name in selected_levels.keys()}
    stain_norm_counts = {mag_name: 0 for mag_name in selected_levels.keys()}
    
    mask_downsample = tslide.level_downsamples[mask_level]
    
    for mag_name, level_info in selected_levels.items():
        print(f"  Processing {mag_name} magnification (target={level_info['target_mag']:.1f}x, actual={level_info['actual_mag']:.2f}x)...")
        
        patch_level = level_info['level']
        patch_size = level_info['patch_size']
        read_size = level_info['read_patch_size']
        read_stride = level_info['read_stride']
        needs_resize = level_info['needs_resize']
        
        patch_downsample = tslide.level_downsamples[patch_level]
        
        scale_factor = mask_downsample / patch_downsample
        mask_read_size = int(read_size / scale_factor)
        mask_stride = int(read_stride / scale_factor)
        
        mask_stride = max(1, mask_stride)
        mask_read_size = max(1, mask_read_size)
        
        mask_patch_size_square = mask_read_size ** 2
        
        if needs_resize:
            print(f"    Level: {patch_level} (downsample={patch_downsample:.2f})")
            print(f"    Read size: {read_size}×{read_size} → resize to {patch_size}×{patch_size}")
        else:
            print(f"    Level: {patch_level} (downsample={patch_downsample:.2f})")
            print(f"    Read size: {read_size}×{read_size} (no resize needed)")
        print(f"    Mask: size={mask_read_size}, stride={mask_stride}")
        
        mag_save_dir = os.path.join(slide_save_dir, mag_name)
        os.makedirs(mag_save_dir, exist_ok=True)
        
        # Iterate through mask to extract patches (only within annotated regions)
        for iw in range((mask_sW - mask_read_size) // mask_stride + 1):
            for ih in range((mask_sH - mask_read_size) // mask_stride + 1):
                ww = iw * mask_stride
                hh = ih * mask_stride
                
                if (ww + mask_read_size) > mask_sW or (hh + mask_read_size) > mask_sH:
                    continue
                
                # Check the tissue ratio in the mask patch
                tmask = tissue_mask[hh:hh+mask_read_size, ww:ww+mask_read_size]
                mRatio = float(np.sum(tmask > 0)) / mask_patch_size_square
                
                if mRatio > threshold:
                    try:
                        level0_w = int(ww * mask_downsample)
                        level0_h = int(hh * mask_downsample)
                        
                        tpatch = tslide.read_region(
                            (level0_w, level0_h),
                            patch_level,
                            (read_size, read_size)
                        )
                        
                        tpatch_RGB = tpatch.convert('RGB')
                        
                        if needs_resize and read_size != patch_size:
                            tpatch_RGB = tpatch_RGB.resize(
                                (patch_size, patch_size),
                                resize_interpolation
                            )
                            resize_counts[mag_name] += 1
                        
                        # ============== Apply stain normalization ==============
                        if stain_norm_enabled:
                            tpatch_RGB = apply_stain_normalization(tpatch_RGB)
                            stain_norm_counts[mag_name] += 1
                        # ============================================
                        
                        patch_name = f"{slide_name}_{mag_name}_x{level0_w}_y{level0_h}.jpg"
                        save_path = os.path.join(mag_save_dir, patch_name)
                        
                        tpatch_RGB.save(save_path, 'JPEG', quality=jpeg_quality, optimize=True)
                        patch_counts[mag_name] += 1
                        
                    except Exception as e:
                        error_counts[mag_name] += 1
                        if error_counts[mag_name] <= 3:
                            print(f"      Error: {str(e)}")
        
        resize_info = f", {resize_counts[mag_name]} resized" if resize_counts[mag_name] > 0 else ""
        stain_info = f", {stain_norm_counts[mag_name]} stain_normed" if stain_norm_counts[mag_name] > 0 else ""
        print(f"    ✓ Completed {mag_name}: {patch_counts[mag_name]} patches{resize_info}{stain_info}, {error_counts[mag_name]} errors")
    
    return patch_counts, error_counts, resize_counts
