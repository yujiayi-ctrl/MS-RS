import PIL.Image as Image

####====================================== User Configuration
num_thread = 4

# magnification levels and their corresponding patch extraction parameters
magnifications = {
    '5x': {'patch_size': 256, 'stride': 256},
    '10x': {'patch_size': 256, 'stride': 256},
    '20x': {'patch_size': 256, 'stride': 256},
}

tissue_mask_threshold = 0.5
mask_dimension_level = 5

# Dataset and file paths
dataset_csv = '/home/yujy/CAMS_data/CAMS_clinical_processed762849-4.csv'
save_folder_dir = '/252_node_user_storage/yujy/CAMS_data/patches'
xml_annotation_dir = '/252_node_user_storage/yujy/CAMS_data/Raw'  # XML annotations directory

skip_existing = False
min_patches_threshold = 10
jpeg_quality = 95

# Resize configuration
enable_resize = True
resize_interpolation = Image.LANCZOS
max_deviation_for_resize = 30

# XML annotation configuration
use_xml_annotations = True  # If True, use XML annotations to filter patches; if False, extract patches from the entire tissue region
xml_suffix = '.xml'  # XML file suffix

# ============== Standardization Configuration ==============
enable_stain_normalization = True
stain_norm_reference_image = '/data/home/scxj642/run/yujy/MS-RS/patch_extraction/reference__deagram.jpg'
stain_norm_method = 'reinhard' 
# ============================================

####======================================
