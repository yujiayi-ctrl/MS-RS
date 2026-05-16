import os
import time
import pandas as pd
from multiprocessing import Pool

from config import (
    num_thread,
    dataset_csv,
    save_folder_dir,
    xml_annotation_dir,
    use_xml_annotations,
    enable_stain_normalization,
    stain_norm_reference_image,
    stain_norm_method,
    jpeg_quality
)
from stain_normalizer import initialize_stain_normalizer, get_stain_normalizer
from slide_processor import process_single_slide_with_annotations


def main():
    print(f"{'='*80}")
    print(f"TCGA Dataset Multi-scale Patch Extraction (XML Annotation-based Version + Stain Normalization)")
    print(f"{'='*80}")
    
    if not os.path.exists(dataset_csv):
        print(f"✗ Error: Dataset CSV does not exist: {dataset_csv}")
        exit(1)
    
    df = pd.read_csv(dataset_csv)
    print(f"Dataset: {len(df)} samples")
    
    print(f"\nConfiguration:")
    print(f"  Output Directory: {save_folder_dir}")
    print(f"  XML Annotation Directory: {xml_annotation_dir}")
    print(f"  Use XML Annotations: {'✓ Enabled' if use_xml_annotations else '✗ Disabled'}")
    print(f"  Stain Normalization: {'✓ Enabled' if enable_stain_normalization else '✗ Disabled'}")
    if enable_stain_normalization:
        print(f"    Reference Image: {stain_norm_reference_image}")
        print(f"    Method: {stain_norm_method}")
    print(f"  Target Magnifications: 5x, 10x, 20x")
    print(f"  JPEG Quality: {jpeg_quality}")
    print(f"  Parallel Processes: {num_thread}")
    
    print(f"\n✨ Core Features:")
    print(f"  - Accurate Tumor Region Extraction Based on XML Annotations")
    print(f"  - Automatic Scan Magnification Detection and Intelligent Resize")
    print(f"  - Automatic Fallback to Traditional Tissue Segmentation when XML Not Found")
    print(f"  - Macenko Stain Normalization (Optional)")
    print(f"{'='*80}\n")
    
    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        print("✗ Error: shapely library is required")
        print("Please run: pip install shapely")
        exit(1)
    
    initialize_stain_normalizer()
    
    arg_list = []
    for idx, row in df.iterrows():
        slide_id = row['slide_id']
        slide_path = row['slide_path']
        recurrence = row['recurrence']
        subtype = row['subtype']
        
        if not os.path.exists(slide_path):
            print(f"⚠ Warning: Skipping {slide_id}: WSI file does not exist")
            continue
        
        slide_save_dir = os.path.join(save_folder_dir, slide_id)
        arg_list.append([slide_path, slide_id, slide_save_dir, recurrence, subtype])
    
    print(f"准备处理 {len(arg_list)}/{len(df)} slides\n")
    
    start_time = time.time()
    
    _stain_normalizer = get_stain_normalizer()
    
    # Note: Since the stain normalizer is a global variable, each process needs to re-initialize it when using multiprocessing
    # Here we use single-process processing to ensure stain normalization works properly
    # For multiprocessing, you can pass the normalizer parameters to child processes
    if enable_stain_normalization and _stain_normalizer is not None:
        print("⚠ Warning: When stain normalization is enabled, single-process handling is used to ensure consistency")
        results = []
        for args in arg_list:
            result = process_single_slide_with_annotations(args)
            results.append(result)
    else:
        pool = Pool(processes=num_thread)
        results = pool.map(process_single_slide_with_annotations, arg_list)
        pool.close()
        pool.join()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Processing Completion Summary")
    print(f"{'='*80}")
    
    processed = [r for r in results if r and r['status'] == 'processed']
    
    if processed:
        with_xml = [r for r in processed if r.get('has_xml_annotation', False)]
        without_xml = [r for r in processed if not r.get('has_xml_annotation', False)]
        with_stain_norm = [r for r in processed if r.get('stain_normalized', False)]
        
        print(f"\nXML Annotation Statistics:")
        print(f"  Using XML annotations: {len(with_xml)} slides")
        print(f"  Traditional segmentation: {len(without_xml)} slides")

        print(f"\nStain Normalization Statistics:")
        print(f"  Normalized: {len(with_stain_norm)} slides")
        
        total_patches_xml = sum(sum(r['patch_counts'].values()) for r in with_xml)
        total_patches_traditional = sum(sum(r['patch_counts'].values()) for r in without_xml)
        
        print(f"\nPatch Count Statistics:")
        print(f"  Extracted from XML annotations: {total_patches_xml:,} patches")
        print(f"  Extracted from traditional segmentation: {total_patches_traditional:,} patches")
    
    log_path = os.path.join(save_folder_dir, 'processing_log_xml_annotation_stain_norm.csv')
    log_df = pd.DataFrame([r for r in results if r is not None])
    log_df.to_csv(log_path, index=False)
    print(f"\nProcessing log saved: {log_path}")
    
    print(f"\n{'='*80}")
    print(f"Patch extraction completed based on XML annotations! (With stain normalization)")
    print(f"Save location: {save_folder_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
