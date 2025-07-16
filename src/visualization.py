import itk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_full_analysis():
    print("Start analisis full")
    
    from src.notebook_executor import run_notebook_segmentation, verify_segmentation_quality, preview_notebook_results
    
    seg1, seg2, orig1, orig2, reg = run_notebook_segmentation()
    
    if seg1 is None or seg2 is None:
        return False
    
    is_good, message = verify_segmentation_quality(seg1, seg2)
    print(f"Segmentation quality: {message}")
    
    volume_metrics = calculate_volume_change(seg1, seg2, orig1)
    intensity_metrics = calculate_intensity_changes(seg1, seg2, orig1, reg)
    spatial_metrics = calculate_spatial_changes(seg1, seg2, orig1)
    
    generate_analysis_report(volume_metrics, intensity_metrics, spatial_metrics)
    
    
    response = input("Show notebook style preview? (y/n): ")
    if response.lower() == 'y':
        preview_notebook_results(seg1, seg2, orig1, reg)
    
    create_comparison_plots(seg1, seg2, orig1, reg, volume_metrics, intensity_metrics, spatial_metrics)
    
    response = input("Show 3D vizualization? (y/n): ")
    if response.lower() == 'y':
        create_3d_visualization(seg1, seg2)
    
    return True

def calculate_volume_change(seg1, seg2, orig1):
    print("Calculating volumes changes")
    
    array1 = itk.GetArrayFromImage(seg1)
    array2 = itk.GetArrayFromImage(seg2)
    
    volume1 = np.sum(array1 > 0)
    volume2 = np.sum(array2 > 0)
    
    spacing = orig1.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    physical_volume1 = volume1 * voxel_volume
    physical_volume2 = volume2 * voxel_volume
    
    volume_change_absolute = physical_volume2 - physical_volume1
    volume_change_relative = (volume_change_absolute / physical_volume1) * 100 if physical_volume1 > 0 else 0
    
    return {
        'volume1': physical_volume1,
        'volume2': physical_volume2,
        'absolute_change': volume_change_absolute,
        'relative_change': volume_change_relative,
        'volume1_voxels': volume1,
        'volume2_voxels': volume2
    }

def calculate_intensity_changes(seg1, seg2, orig1, reg):
    print("Calculating intensity changes")
    
    orig_array = itk.GetArrayFromImage(orig1)
    reg_array = itk.GetArrayFromImage(reg)
    mask1 = itk.GetArrayFromImage(seg1) > 0
    mask2 = itk.GetArrayFromImage(seg2) > 0
    
    intensities1 = orig_array[mask1]
    intensities2 = reg_array[mask2]
    
    mean1 = np.mean(intensities1) if len(intensities1) > 0 else 0
    mean2 = np.mean(intensities2) if len(intensities2) > 0 else 0
    
    change_relative = ((mean2 - mean1) / mean1 * 100) if mean1 > 0 else 0
    
    return {
        'mean_intensity1': mean1,
        'mean_intensity2': mean2,
        'std_intensity1': np.std(intensities1) if len(intensities1) > 0 else 0,
        'std_intensity2': np.std(intensities2) if len(intensities2) > 0 else 0,
        'mean_change_relative': change_relative
    }

def calculate_spatial_changes(seg1, seg2, orig1):
    print("Calculating spatial changes")
    
    array1 = itk.GetArrayFromImage(seg1) > 0
    array2 = itk.GetArrayFromImage(seg2) > 0
    
    centroid1 = calculate_centroid(array1)
    centroid2 = calculate_centroid(array2)
    
    shift = np.array(centroid2) - np.array(centroid1)
    shift_magnitude = np.linalg.norm(shift)
    
    spacing = orig1.GetSpacing()
    physical_shift = shift * np.array(spacing)
    physical_shift_magnitude = np.linalg.norm(physical_shift)
    
    intersection = np.sum(array1 & array2)
    union = np.sum(array1) + np.sum(array2)
    overlap_coefficient = (2.0 * intersection / union) if union > 0 else 0
    
    return {
        'centroid1': centroid1,
        'centroid2': centroid2,
        'shift_magnitude_physical': physical_shift_magnitude,
        'overlap_coefficient': overlap_coefficient
    }

def calculate_centroid(binary_array):

    indices = np.where(binary_array)
    if len(indices[0]) == 0:
        return [0, 0, 0]
    
    centroid = [np.mean(indices[i]) for i in range(3)]
    return centroid

def generate_analysis_report(volume_metrics, intensity_metrics, spatial_metrics):
    print("=" * 60)
    print("REPORT :")
    print("=" * 60)
    
    print("\nsegmentation parameter :")
    print("-" * 30)
    print("Method: Connected Threshold")
    print("Lower threshold: 500.0")
    print("Upper threshold: 800.0")
    print("Seed point: (90, 70, 51)")
    print("Smoothing: CurvatureFlow, 5 iterations")
    
    print("\nVolume analisis : ")
    print("-" * 30)
    print(f"Initial volume: {volume_metrics['volume1']:.2f} mm³")
    print(f"Final volume: {volume_metrics['volume2']:.2f} mm³")
    print(f"Absolute change: {volume_metrics['absolute_change']:.2f} mm³")
    print(f"Relative change: {volume_metrics['relative_change']:.1f}%")
    
    print("\nIntensity analisis :")
    print("-" * 30)
    print(f"Initial mean intensity: {intensity_metrics['mean_intensity1']:.2f}")
    print(f"Final mean intensity: {intensity_metrics['mean_intensity2']:.2f}")
    print(f"Intensity change: {intensity_metrics['mean_change_relative']:.1f}%")
    
    print("\nspacial analisis :")
    print("-" * 30)
    print(f"Centroid shift: {spatial_metrics['shift_magnitude_physical']:.2f} mm")
    print(f"Overlap coefficient: {spatial_metrics['overlap_coefficient']:.3f}")
    
    print("\ninterpretation :")
    print("-" * 30)
    vol_change = volume_metrics['relative_change']
    if vol_change > 20:
        print("Tumor growth detected")
    elif vol_change < -20:
        print("tumor reduction detected")
    else:
        print("Stable tumor size")
    
    shift = spatial_metrics['shift_magnitude_physical']
    if shift > 5:
        print("Significant tumor displacement")
    else:
        print("Tumor position stable")
    
    overlap = spatial_metrics['overlap_coefficient']
    if overlap > 0.7:
        print("Good spatial correspondence")
    elif overlap > 0.5:
        print("Moderate good spatial correspondence")
    else:
        print("Bad spatial correspondence")
    
    print("=" * 60)

def create_comparison_plots(seg1, seg2, orig1, reg, volume_metrics, intensity_metrics, spatial_metrics):
    print("Creating plots")
    
    orig_array = itk.GetArrayFromImage(orig1)
    reg_array = itk.GetArrayFromImage(reg)
    mask1 = itk.GetArrayFromImage(seg1) > 0
    mask2 = itk.GetArrayFromImage(seg2) > 0
    
    slice_index = 55
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Tumor Evolution Analysis', fontsize=16)
    
    axes[0, 0].imshow(orig_array[slice_index], cmap='gray')
    axes[0, 0].set_title('Original Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(reg_array[slice_index], cmap='gray')
    axes[0, 1].set_title('Registered Image 2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(orig_array[slice_index], cmap='gray')
    axes[0, 2].imshow(mask1[slice_index], cmap='Reds', alpha=0.5)
    axes[0, 2].set_title('Segmented Tumor 1')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(reg_array[slice_index], cmap='gray')
    axes[0, 3].imshow(mask2[slice_index], cmap='Blues', alpha=0.5)
    axes[0, 3].set_title('Segmented Tumor 2')
    axes[0, 3].axis('off')
    
    overlay = np.zeros((*orig_array[slice_index].shape, 3))
    overlay[mask1[slice_index], 0] = 1  # Red pour T1
    overlay[mask2[slice_index], 2] = 1  # Blue pour T2
    overlap = mask1[slice_index] & mask2[slice_index]
    overlay[overlap] = [1, 0, 1]  # Magenta overlap
    
    axes[1, 0].imshow(orig_array[slice_index], cmap='gray')
    axes[1, 0].imshow(overlay, alpha=0.5)
    axes[1, 0].set_title('Tumor Overlay\n(Red: T1, Blue: T2, Magenta: Overlap)')
    axes[1, 0].axis('off')
    
    diff = reg_array[slice_index] - orig_array[slice_index]
    im = axes[1, 1].imshow(diff, cmap='RdBu_r')
    axes[1, 1].set_title('Intensity Difference (T2 - T1)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    volumes = [volume_metrics['volume1'], volume_metrics['volume2']]
    axes[1, 2].bar(['Time 1', 'Time 2'], volumes, color=['crimson', 'royalblue'], alpha=0.7)
    axes[1, 2].set_title('Volume Comparison')
    axes[1, 2].set_ylabel('Volume (mm³)')
    
    vol_change_text = f"Change: {volume_metrics['relative_change']:.1f}%"
    axes[1, 2].text(0.5, max(volumes) * 0.8, vol_change_text, 
                   ha='center', fontsize=12, fontweight='bold')
    
    stats_text = f"""Volume Change: {volume_metrics['relative_change']:.1f}%
Intensity Change: {intensity_metrics['mean_change_relative']:.1f}%
Centroid Shift: {spatial_metrics['shift_magnitude_physical']:.2f} mm
Overlap Coefficient: {spatial_metrics['overlap_coefficient']:.3f}

Notebook Parameters:
- Threshold: 500-800
- Seed: (90, 70, 51)
- Smoothing: 5 iterations"""
    
    axes[1, 3].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    axes[1, 3].set_title('Analysis Summary')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_3d_visualization(seg1, seg2):

    print("3D visualisation")
    
    try:
        array1 = itk.GetArrayFromImage(seg1) > 0
        array2 = itk.GetArrayFromImage(seg2) > 0
        
        array1_ds = array1[::2, ::2, ::2]
        array2_ds = array2[::2, ::2, ::2]
        
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        plot_3d_tumor(ax1, array1_ds, 'red', 'Tumor Time 1')
        
        ax2 = fig.add_subplot(132, projection='3d')
        plot_3d_tumor(ax2, array2_ds, 'blue', 'Tumor Time 2')
        
        ax3 = fig.add_subplot(133, projection='3d')
        plot_3d_overlay(ax3, array1_ds, array2_ds)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"FAIL (pas assez de mémoire ? essaye dataset plus petit):  {e}")

def plot_3d_tumor(ax, array, color, title):
    z, y, x = np.where(array)
    
    step = max(1, len(x) // 1000)
    if len(x) > 0:
        x_sub = x[::step]
        y_sub = y[::step] 
        z_sub = z[::step]
        ax.scatter(x_sub, y_sub, z_sub, c=color, alpha=0.6, s=1)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_3d_overlay(ax, array1, array2):
    z1, y1, x1 = np.where(array1)
    z2, y2, x2 = np.where(array2)
    
    step1 = max(1, len(x1) // 500)
    step2 = max(1, len(x2) // 500)
    
    if len(x1) > 0:
        ax.scatter(x1[::step1], y1[::step1], z1[::step1], 
                  c='red', alpha=0.6, s=1, label='Time 1')
    if len(x2) > 0:
        ax.scatter(x2[::step2], y2[::step2], z2[::step2], 
                  c='blue', alpha=0.6, s=1, label='Time 2')
    
    ax.set_title('Tumor Overlay')
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.legend()

if __name__ == "__main__":
    run_full_analysis()