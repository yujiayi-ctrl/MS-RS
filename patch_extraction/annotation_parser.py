import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon


def parse_xml_annotations(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        polygons = []
        for annotation in root.findall('.//Annotation'):
            coordinates = []
            for coord in annotation.findall('.//Coordinate'):
                x = float(coord.get('X'))
                y = float(coord.get('Y'))
                coordinates.append((x, y))
            
            if len(coordinates) >= 3:
                poly = Polygon(coordinates)
                polygons.append(poly)
        
        return polygons
    except Exception as e:
        print(f"    ⚠ Warning: Failed to parse XML file: {e}")
        return []


def create_annotation_mask(slide_dimensions, polygons, mask_level, level_downsamples):
    """
    Create tumor region mask based on XML annotations
    
    Args:
        slide_dimensions: slide dimensions at level 0 (width, height)
        polygons: List of Shapely Polygon objects
        mask_level: mask level
        level_downsamples: downsampling factor for each level
    
    Returns:
        annotation_mask: binary mask (255=tumor region, 0=background)
    """
    if not polygons:
        return None
    
    mask_downsample = level_downsamples[mask_level]
    mask_width = int(slide_dimensions[0] / mask_downsample)
    mask_height = int(slide_dimensions[1] / mask_downsample)
    
    # create empty mask
    annotation_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    
    # scale polygon coordinates to mask level and fill the polygon area
    for poly in polygons:
        scaled_coords = [(x / mask_downsample, y / mask_downsample) 
                        for x, y in poly.exterior.coords]
        scaled_coords = np.array(scaled_coords, dtype=np.int32)
        
        # fill the polygon area in the mask
        cv2.fillPoly(annotation_mask, [scaled_coords], 255)
    
    return annotation_mask
