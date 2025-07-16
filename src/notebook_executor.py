import itk
import numpy as np
import os

def run_notebook_segmentation():
    print("Run segmentation")
    
    try:
        required_files = [
            "data/case6_gre1.nrrd",
            "data/case6_gre2.nrrd", 
            "translation.nrrd"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                print(f"Error: Required file {file} not found!")
                return None, None, None, None, None
        print("Load")
        PixelType = itk.F
        images_1 = itk.imread("data/case6_gre1.nrrd", PixelType)
        images_2 = itk.imread("data/case6_gre2.nrrd", PixelType)
        recalage = itk.imread("translation.nrrd", PixelType)
        
        print("Process")
        images_2_resampled = recalage.astype(float)
        images_2_resampled = itk.image_from_array(images_2_resampled)
        cast_filter = itk.CastImageFilter[images_2_resampled, itk.Image[itk.F, 3]].New()
        cast_filter.SetInput(images_2_resampled)
        cast_filter.Update()
        images_2_resampled = cast_filter.GetOutput()
        
        print("Smooth")
        smoothing = itk.CurvatureFlowImageFilter.New(Input=images_1)
        smoothing.SetNumberOfIterations(5)
        smoothing.SetTimeStep(0.125)
        smoothing.Update()
        images_1_smoothed = smoothing.GetOutput()
        smoothing = itk.CurvatureFlowImageFilter.New(Input=images_2_resampled)
        smoothing.SetNumberOfIterations(5)
        smoothing.SetTimeStep(0.125)
        smoothing.Update()
        images_2_smoothed = smoothing.GetOutput()

        print("Segmenting")
        ConnectedFilterType = itk.ConnectedThresholdImageFilter[itk.Image[itk.F,3], itk.Image[itk.F,3]]
        connectedThreshold = ConnectedFilterType.New()

        lower_threshold = 500.
        upper_threshold = 800.
        index = (90, 70, 51)

        connectedThreshold.SetInput(images_1_smoothed)
        connectedThreshold.SetLower(lower_threshold)
        connectedThreshold.SetUpper(upper_threshold)
        connectedThreshold.SetReplaceValue(255)
        connectedThreshold.SetSeed(index)
        connectedThreshold.Update()
        
        images_1_segmented = connectedThreshold.GetOutput()

        connectedThreshold2 = ConnectedFilterType.New()
        connectedThreshold2.SetInput(images_2_smoothed)
        connectedThreshold2.SetLower(lower_threshold)
        connectedThreshold2.SetUpper(upper_threshold)
        connectedThreshold2.SetReplaceValue(255)
        connectedThreshold2.SetSeed(index)
        connectedThreshold2.Update()
        
        images_2_segmented = connectedThreshold2.GetOutput()
        
        print("SUCESS")
        
        try:
            itk.imwrite(images_1_segmented, "debug_segmented1.nrrd")
            itk.imwrite(images_2_segmented, "debug_segmented2.nrrd")
            print("Fichier de debug sauvegarder")
        except Exception as e:
            print(f"FAIL : {e}")
        
        return images_1_segmented, images_2_segmented, images_1, images_2, recalage
        
    except Exception as e:
        print(f"FAIL : {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def verify_segmentation_quality(segmented1, segmented2):
    if segmented1 is None or segmented2 is None:
        return False, "FAIL"
    
    array1 = itk.GetArrayFromImage(segmented1)
    array2 = itk.GetArrayFromImage(segmented2)
    
    vol1 = np.sum(array1 > 0)
    vol2 = np.sum(array2 > 0)
    
    min_volume = 100
    max_volume = 100000
    
    if vol1 < min_volume or vol1 > max_volume:
        return False, f"Tumor 1 volume ({vol1} voxels) outside reasonable range"
    
    if vol2 < min_volume or vol2 > max_volume:
        return False, f"Tumor 2 volume ({vol2} voxels) outside reasonable range"
    
    return True, f"Segmentation quality OK (Vol1: {vol1}, Vol2: {vol2} voxels)"

def preview_notebook_results(segmented1, segmented2, original1, registered):
    if segmented1 is None or segmented2 is None:
        print("No segmentation")
        return
    
    import matplotlib.pyplot as plt
    
    orig1_array = itk.GetArrayFromImage(original1)
    reg_array = itk.GetArrayFromImage(registered)
    seg1_array = itk.GetArrayFromImage(segmented1)
    seg2_array = itk.GetArrayFromImage(segmented2)
    
    index = (90, 70, 51)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Notebook Segmentation Results', fontsize=16)
    
    axes[0, 0].imshow(orig1_array[index[0] - 35, :, :], cmap='gray')
    axes[0, 0].set_title('Original Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(seg1_array[index[0] - 35, :, :], cmap='gray')
    axes[0, 1].set_title('Segmented Tumor 1')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(orig1_array[:, index[1], :], cmap='gray')
    axes[1, 0].set_title('Original Image 1 (Coronal)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(seg1_array[:, index[1], :], cmap='gray')
    axes[1, 1].set_title('Segmented Tumor 1 (Coronal)')
    axes[1, 1].axis('off')
    
    axes[2, 0].imshow(orig1_array[:, :, index[2] + 30], cmap='gray')
    axes[2, 0].set_title('Original Image 1 (Axial)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(seg1_array[:, :, index[2] + 30], cmap='gray')
    axes[2, 1].set_title('Segmented Tumor 1 (Axial)')
    axes[2, 1].axis('off')
    
    axes[0, 2].imshow(reg_array[index[0] - 35, :, :], cmap='gray')
    axes[0, 2].set_title('Registered Image 2')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(seg2_array[index[0] - 35, :, :], cmap='gray')
    axes[0, 3].set_title('Segmented Tumor 2')
    axes[0, 3].axis('off')
    
    axes[1, 2].imshow(reg_array[:, index[1], :], cmap='gray')
    axes[1, 2].set_title('Registered Image 2 (Coronal)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(seg2_array[:, index[1], :], cmap='gray')
    axes[1, 3].set_title('Segmented Tumor 2 (Coronal)')
    axes[1, 3].axis('off')
    
    axes[2, 2].imshow(reg_array[:, :, index[2] + 30], cmap='gray')
    axes[2, 2].set_title('Registered Image 2 (Axial)')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(seg2_array[:, :, index[2] + 30], cmap='gray')
    axes[2, 3].set_title('Segmented Tumor 2 (Axial)')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    vol1 = np.sum(seg1_array > 0)
    vol2 = np.sum(seg2_array > 0)
    spacing = original1.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    print(f"\nSegmentation Stats:")
    print(f"Tumor 1 volume: ({vol1 * voxel_volume:.2f} mm³)")
    print(f"Tumor 2 volume: ({vol2 * voxel_volume:.2f} mm³)")
    if vol1 > 0:
        print(f"Volume change: {((vol2 - vol1) / vol1 * 100):.1f}%")

def test_notebook_executor():
    print("Test")
    
    seg1, seg2, orig1, orig2, reg = run_notebook_segmentation()
    
    if seg1 is not None and seg2 is not None:
        print("Sucess")
        
        is_good, message = verify_segmentation_quality(seg1, seg2)
        print(f"Quality check: {message}")
        
        if is_good:
            preview_notebook_results(seg1, seg2, orig1, reg)
        
        return seg1, seg2, orig1, orig2, reg
    else:
        print("FAIL")
        return None, None, None, None, None

if __name__ == "__main__":
    test_notebook_executor()