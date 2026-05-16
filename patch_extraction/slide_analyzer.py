import os

from config import magnifications


def get_base_magnification(slide):
    """Retrieve the base scanning magnification of the slide"""
    props = slide.properties
    
    mag_keys = [
        'aperio.AppMag',
        'openslide.objective-power',
        'hamamatsu.SourceLens',
        'leica.objective',
    ]
    
    for key in mag_keys:
        if key in props:
            try:
                mag = float(props[key])
                if 10 <= mag <= 100:
                    return mag
            except (ValueError, TypeError):
                continue
    
    try:
        mpp_x = float(props.get('openslide.mpp-x', 0))
        if mpp_x > 0:
            if mpp_x < 0.35:
                return 40.0
            elif mpp_x < 0.7:
                return 20.0
            else:
                return 10.0
    except (ValueError, TypeError, KeyError):
        pass
    
    return None


def select_best_level_for_magnification(slide, target_downsample, mag_name):
    """Choose the optimal level according to the target downsampling factor"""
    level_downsamples = slide.level_downsamples
    
    deviations = []
    for level, downsample in enumerate(level_downsamples):
        deviation = abs(downsample - target_downsample) / target_downsample
        deviations.append((level, downsample, deviation))
    
    best_level, actual_downsample, min_deviation = min(deviations, key=lambda x: x[2])
    deviation_percent = min_deviation * 100
    
    return best_level, actual_downsample, deviation_percent


def analyze_slide_levels(slide, slide_id):
    """Analyze the level information of the slide"""
    level_count = len(slide.level_dimensions)
    level_downsamples = slide.level_downsamples
    
    print(f"  Slide level analysis:")
    print(f"    Total levels: {level_count}")
    print(f"    Downsampling multiples for each level: {[f'{d:.2f}' for d in level_downsamples]}")
    
    base_mag = get_base_magnification(slide)
    
    if base_mag is None:
        print(f"    ⚠ Warning: Unable to retrieve scanning magnification, assuming 40x")
        base_mag = 40.0
    else:
        print(f"    ✓ Detected scanning magnification: {base_mag}x")
    
    selected_levels = {}
    target_mags = {'5x': 5.0, '10x': 10.0, '20x': 20.0}
    
    for mag_name in ['5x', '10x', '20x']:
        target_mag = target_mags[mag_name]
        target_downsample = base_mag / target_mag
        
        base_patch_size = magnifications[mag_name]['patch_size']
        base_stride = magnifications[mag_name]['stride']
        
        best_level, actual_downsample, deviation = select_best_level_for_magnification(
            slide, target_downsample, mag_name
        )
        
        actual_mag = base_mag / actual_downsample
        
        if actual_mag > target_mag * 1.3:
            needs_resize = True
            resize_mode = 'downsample'
            mag_ratio = actual_mag / target_mag
            read_patch_size = int(base_patch_size * mag_ratio + 0.5)
            if read_patch_size % 2 != 0:
                read_patch_size += 1
            read_stride = int(base_stride * mag_ratio + 0.5)
            if read_stride % 2 != 0:
                read_stride += 1
        elif actual_mag < target_mag * 0.7:
            needs_resize = True
            resize_mode = 'upsample'
            mag_ratio = actual_mag / target_mag
            read_patch_size = int(base_patch_size * mag_ratio + 0.5)
            if read_patch_size % 2 != 0:
                read_patch_size += 1
            read_stride = int(base_stride * mag_ratio + 0.5)
            if read_stride % 2 != 0:
                read_stride += 1
            print(f"    ⚠ {mag_name} Warning: Need to upsample, which may lead to quality loss")
        else:
            needs_resize = False
            resize_mode = 'none'
            read_patch_size = base_patch_size
            read_stride = base_stride
        
        selected_levels[mag_name] = {
            'level': best_level,
            'target_downsample': target_downsample,
            'actual_downsample': actual_downsample,
            'patch_size': base_patch_size,
            'read_patch_size': read_patch_size,
            'stride': base_stride,
            'read_stride': read_stride,
            'target_mag': target_mag,
            'actual_mag': actual_mag,
            'base_mag': base_mag,
            'needs_resize': needs_resize,
            'resize_mode': resize_mode,
            'deviation_percent': deviation,
        }
        
        status = "→needs resize" if needs_resize else ""
        print(f"    {mag_name}: Level {best_level} "
              f"(target={target_mag}x, actual={actual_mag:.2f}x, deviation={deviation:.1f}%) {status}")
    
    return selected_levels


def check_slide_processed_optimized(slide_save_dir, magnifications, min_patches=10):
    """Check if the slide has already been processed"""
    try:
        patch_counts = {}
        
        for mag_name in magnifications.keys():
            mag_dir = os.path.join(slide_save_dir, mag_name)
            
            if not os.path.exists(mag_dir):
                return False, {}, f"{mag_name} directory does not exist"
            
            patch_files = []
            for f in os.listdir(mag_dir):
                if f.endswith('.jpg'):
                    file_path = os.path.join(mag_dir, f)
                    if os.path.isfile(file_path) or (os.path.islink(file_path) and os.path.exists(file_path)):
                        patch_files.append(f)
            
            num_patches = len(patch_files)
            patch_counts[mag_name] = num_patches
            
            if num_patches < min_patches:
                return False, patch_counts, f"{mag_name} patch数量不足({num_patches}<{min_patches})"
        
        return True, patch_counts, "Complete"
        
    except Exception as e:
        return False, {}, f"Check error: {str(e)}"
