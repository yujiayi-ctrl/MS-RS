import os
import numpy as np
import cv2
from skimage.filters import threshold_otsu

from annotation_parser import parse_xml_annotations, create_annotation_mask


def get_roi_bounds_with_annotations(tslide, xml_path=None, mask_level=5):
    """
    Use XML annotations to obtain tumor region boundaries
    If no XML annotations are available, fall back to traditional tissue segmentation methods
    """
    max_level = len(tslide.level_dimensions) - 1
    if mask_level > max_level:
        mask_level = max_level
        print(f"    Adjusting mask_level: {mask_level} (maximum level={max_level})")
    
    # 尝试使用XML标注
    if xml_path and os.path.exists(xml_path):
        print(f"    Using XML annotation file: {os.path.basename(xml_path)}")
        polygons = parse_xml_annotations(xml_path)
        
        if polygons:
            print(f"    Found {len(polygons)} tumor annotation regions")
            
            annotation_mask = create_annotation_mask(
                tslide.dimensions,
                polygons,
                mask_level,
                tslide.level_downsamples
            )
            
            if annotation_mask is not None:
                contours, _ = cv2.findContours(
                    annotation_mask, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                boundingBox = [cv2.boundingRect(c) for c in contours]
                
                # Filter out small bounding boxes that are unlikely to contain meaningful tissue
                boundingBox = [box for box in boundingBox 
                             if box[2] > 50 and box[3] > 50]
                
                print(f"    Found {len(boundingBox)} valid regions from XML annotations")
                
                return annotation_mask // 255, boundingBox, mask_level
    
    print(f"    Not using XML annotations, executing traditional tissue segmentation")
    return get_roi_bounds_traditional(tslide, mask_level)


def get_roi_bounds_traditional(tslide, mask_level=5):
    """
    Traditional tissue segmentation method based on color thresholding
    """
    subSlide = tslide.read_region((0, 0), mask_level, tslide.level_dimensions[mask_level])
    subSlide_np = np.array(subSlide)

    hsv = cv2.cvtColor(subSlide_np, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    try:
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)
    except:
        return np.NaN, np.NaN, mask_level

    minhsv = np.array([hthresh, sthresh, 70], np.uint8)
    maxhsv = np.array([180, 255, vthresh], np.uint8)
    thresh = [minhsv, maxhsv]

    mask = cv2.inRange(hsv, thresh[0], thresh[1])

    close_kernel = np.ones((50, 50), dtype=np.uint8)
    image_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    
    open_kernel = np.ones((30, 30), dtype=np.uint8)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, open_kernel)
    
    image_open = cv2.medianBlur(image_open, 5)

    contours, _ = cv2.findContours(image_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBox = [cv2.boundingRect(c) for c in contours]
    boundingBox = [box for box in boundingBox if box[2] > 150 and box[3] > 150]

    print(f"    Traditional method found valid tissue regions: {len(boundingBox)}")

    return image_open // 255, boundingBox, mask_level
