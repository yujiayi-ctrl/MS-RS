import os
import time
import openslide

from config import (
    skip_existing, 
    magnifications, 
    min_patches_threshold,
    mask_dimension_level,
    tissue_mask_threshold,
    jpeg_quality,
    use_xml_annotations,
    xml_annotation_dir,
    xml_suffix
)
from stain_normalizer import get_stain_normalizer
from slide_analyzer import analyze_slide_levels, check_slide_processed_optimized
from tissue_segmentation import get_roi_bounds_with_annotations
from patch_extractor import extract_patches_from_annotations
import numpy as np


def process_single_slide_with_annotations(args):
    """
    Process a single slide - extract tumor regions using XML annotations
    """
    slide_path, slide_id, slide_save_dir, recurrence, subtype = args
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Processing: {slide_id}")
    print(f"  Recurrence: {recurrence}, Subtype: {subtype}")
    print(f"  WSI Path: {slide_path}")
    
    if skip_existing and os.path.exists(slide_save_dir):
        is_processed, patch_counts, reason = check_slide_processed_optimized(
            slide_save_dir, 
            magnifications, 
            min_patches=min_patches_threshold
        )
        
        if is_processed:
            total_patches = sum(patch_counts.values())
            print(f"✓ Skipped {slide_id}: Completed ({total_patches} patches)")
            return {
                'slide_id': slide_id,
                'status': 'skipped',
                'patch_counts': patch_counts,
                'recurrence': recurrence,
                'subtype': subtype,
                'processing_time': time.time() - start_time
            }
        else:
            print(f"→ Re-processing {slide_id}: {reason}")
    
    try:
        print(f"  Opening WSI file...")
        tslide = openslide.open_slide(slide_path)
        
        selected_levels = analyze_slide_levels(tslide, slide_id)
        
        max_level = len(tslide.level_dimensions) - 1
        adjusted_mask_level = min(mask_dimension_level, max_level)
        
        # Search for corresponding XML annotation file
        xml_path = None
        if use_xml_annotations:
            xml_path = os.path.join(xml_annotation_dir, f"{slide_id}{xml_suffix}")
            if not os.path.exists(xml_path):
                print(f"  ⚠ Warning: XML annotation file not found: {xml_path}")
                xml_path = None
        
        print(f"  Executing tumor region segmentation...")
        tissue_mask, bounding_boxes, actual_mask_level = get_roi_bounds_with_annotations(
            tslide, 
            xml_path=xml_path,
            mask_level=adjusted_mask_level
        )
        
        if isinstance(tissue_mask, float) and np.isnan(tissue_mask):
            raise ValueError("File may be corrupted or not a valid WSI.")
        
        print(f"  Extracting multi-scale patches from annotated regions...")
        patch_counts, error_counts, resize_counts = extract_patches_from_annotations(
            tslide, 
            tissue_mask, 
            slide_save_dir,
            selected_levels,
            actual_mask_level,
            threshold=tissue_mask_threshold,
            jpeg_quality=jpeg_quality
        )
        
        tslide.close()
        
        total_patches = sum(patch_counts.values())
        total_errors = sum(error_counts.values())
        total_resized = sum(resize_counts.values())
        processing_time = time.time() - start_time
        
        base_mag = selected_levels['5x']['base_mag']
        has_xml = xml_path is not None and os.path.exists(xml_path)
        
        _stain_normalizer = get_stain_normalizer()
        
        status_str = f"✓ Completed {slide_id} [{base_mag}x scanning]"
        status_str += ", XML annotations✓" if has_xml else ", Traditional segmentation"
        status_str += ", Stain normalization✓" if _stain_normalizer is not None else ""
        status_str += "]: "
        for mag_name, count in patch_counts.items():
            status_str += f"{mag_name}={count} "
        status_str += f"({processing_time:.1f}s)"
        if total_resized > 0:
            status_str += f" [resized:{total_resized}]"
        if total_errors > 0:
            status_str += f" [errors:{total_errors}]"
        print(status_str)
        
        return {
            'slide_id': slide_id,
            'status': 'processed',
            'patch_counts': patch_counts,
            'error_counts': error_counts,
            'resize_counts': resize_counts,
            'base_mag': base_mag,
            'has_xml_annotation': has_xml,
            'stain_normalized': _stain_normalizer is not None,
            'recurrence': recurrence,
            'subtype': subtype,
            'processing_time': processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"✗ Failed {slide_id}: {str(e)} ({processing_time:.1f}s)")
        return {
            'slide_id': slide_id,
            'status': 'failed',
            'patch_counts': {},
            'error_counts': {},
            'reason': str(e),
            'recurrence': recurrence,
            'subtype': subtype,
            'processing_time': processing_time
        }
