import itk
import numpy as np
import vtk
from vtk.util import numpy_support
from vtk.util.colors import *

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
    
    create_image_grid(orig_array, reg_array, mask1, mask2, slice_index, volume_metrics, intensity_metrics, spatial_metrics)

def create_image_grid(orig_array, reg_array, mask1, mask2, slice_index, volume_metrics, intensity_metrics, spatial_metrics):
    main_renderer = vtk.vtkRenderer()
    main_window = vtk.vtkRenderWindow()
    main_window.AddRenderer(main_renderer)
    main_window.SetSize(1600, 800)
    main_window.SetWindowName("Tumor Evolution Analysis")
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(main_window)
    
    viewports = [
        [0.0, 0.5, 0.25, 1.0],  
        [0.25, 0.5, 0.5, 1.0],    
        [0.5, 0.5, 0.75, 1.0],  
        [0.75, 0.5, 1.0, 1.0],  
        [0.0, 0.0, 0.25, 0.5],  
        [0.25, 0.0, 0.5, 0.5],  
        [0.5, 0.0, 0.75, 0.5],  
        [0.75, 0.0, 1.0, 0.5]   
    ]
    
    titles = [
        "Original Image 1",
        "Registered Image 2", 
        "Segmented Tumor 1",
        "Segmented Tumor 2",
        "Tumor Overlay\n(Red: T1, Blue: T2, Magenta: Overlap)",
        "Intensity Difference (T2 - T1)",
        "Volume Comparison", 
        "Analysis Summary"
    ]
    
    renderers = []
    for i, viewport in enumerate(viewports):
        renderer = vtk.vtkRenderer()
        renderer.SetViewport(viewport)
        renderer.SetBackground(0.1, 0.1, 0.1)
        main_window.AddRenderer(renderer)
        renderers.append(renderer)
        
        title_actor = vtk.vtkTextActor()
        title_actor.SetInput(titles[i])
        title_actor.SetPosition(10, viewport[3] * 800 - 30 - viewport[1] * 800)
        title_actor.GetTextProperty().SetFontSize(12)
        title_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        title_actor.GetTextProperty().SetBold(True)
        renderer.AddActor2D(title_actor)
    
    create_image_slice(renderers[0], orig_array[slice_index], grayscale=True)
    
    create_image_slice(renderers[1], reg_array[slice_index], grayscale=True)
    
    create_image_with_overlay(renderers[2], orig_array[slice_index], mask1[slice_index], color=[1.0, 0.0, 0.0])
    
    create_image_with_overlay(renderers[3], reg_array[slice_index], mask2[slice_index], color=[0.0, 0.0, 1.0])
    
    create_overlay_visualization(renderers[4], orig_array[slice_index], mask1[slice_index], mask2[slice_index])
    
    diff = reg_array[slice_index] - orig_array[slice_index]
    create_image_slice(renderers[5], diff, colormap=True)
    
    create_volume_bar_chart(renderers[6], volume_metrics)
    
    create_summary_text(renderers[7], volume_metrics, intensity_metrics, spatial_metrics)
    
    for renderer in renderers:
        camera = renderer.GetActiveCamera()
        camera.SetPosition(0, 0, 1)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
        camera.ParallelProjectionOn()
        renderer.ResetCamera()
        renderer.InteractiveOff()
    
    main_window.Render()
    
    interactor.SetInteractorStyle(None)
    
    class NoInteractionStyle(vtk.vtkInteractorStyle):
        def __init__(self):
            self.AddObserver("LeftButtonPressEvent", self.do_nothing)
            self.AddObserver("RightButtonPressEvent", self.do_nothing)
            self.AddObserver("MiddleButtonPressEvent", self.do_nothing)
            self.AddObserver("MouseMoveEvent", self.do_nothing)
            self.AddObserver("MouseWheelForwardEvent", self.do_nothing)
            self.AddObserver("MouseWheelBackwardEvent", self.do_nothing)
            self.AddObserver("KeyPressEvent", self.handle_key)
        
        def do_nothing(self, obj, event):
            pass
        
        def handle_key(self, obj, event):
            key = obj.GetKeySym()
            if key == 'q' or key == 'Escape':
                obj.GetRenderWindow().Finalize()
                obj.TerminateApp()
    
    no_interaction = NoInteractionStyle()
    interactor.SetInteractorStyle(no_interaction)
    
    interactor.Start()

def create_image_slice(renderer, image_data, grayscale=True, colormap=False):
    
    if len(image_data.shape) == 3: 
        image_flipped = image_data
    else:  
        image_flipped = np.flipud(image_data)
    
    if grayscale and len(image_flipped.shape) == 2:
        img_norm = ((image_flipped - image_flipped.min()) / (image_flipped.max() - image_flipped.min()) * 255).astype(np.uint8)
        img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    elif colormap:
        img_norm = (image_flipped - image_flipped.min()) / (image_flipped.max() - image_flipped.min())
        img_rgb = apply_rdbu_colormap(img_norm)
    else:
        img_rgb = image_flipped
    
    vtk_image = numpy_to_vtk_image(img_rgb)
    
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(0, 0, 0)
    plane.SetPoint1(img_rgb.shape[1], 0, 0)
    plane.SetPoint2(0, img_rgb.shape[0], 0)
    
    texture = vtk.vtkTexture()
    texture.SetInputData(vtk_image)
    texture.InterpolateOn()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(texture)
    
    renderer.AddActor(actor)
    renderer.ResetCamera()

def create_image_with_overlay(renderer, image_data, mask, color):
    
    image_flipped = np.flipud(image_data)
    mask_flipped = np.flipud(mask)
    
    img_norm = ((image_flipped - image_flipped.min()) / (image_flipped.max() - image_flipped.min()) * 255).astype(np.uint8)
    img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    
    img_rgb[mask_flipped, 0] = np.minimum(255, img_rgb[mask_flipped, 0] + int(color[0] * 128))
    img_rgb[mask_flipped, 1] = np.minimum(255, img_rgb[mask_flipped, 1] + int(color[1] * 128))
    img_rgb[mask_flipped, 2] = np.minimum(255, img_rgb[mask_flipped, 2] + int(color[2] * 128))
    
    create_image_slice(renderer, img_rgb, grayscale=False)

def create_overlay_visualization(renderer, image_data, mask1, mask2):
    
    image_flipped = np.flipud(image_data)
    mask1_flipped = np.flipud(mask1)
    mask2_flipped = np.flipud(mask2)
    
    img_norm = ((image_flipped - image_flipped.min()) / (image_flipped.max() - image_flipped.min()) * 255).astype(np.uint8)
    img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    
    img_rgb[mask1_flipped, 0] = np.minimum(255, img_rgb[mask1_flipped, 0] + 128)
    
    img_rgb[mask2_flipped, 2] = np.minimum(255, img_rgb[mask2_flipped, 2] + 128)
    
    overlap = mask1_flipped & mask2_flipped
    img_rgb[overlap, 0] = np.minimum(255, img_rgb[overlap, 0] + 128)
    img_rgb[overlap, 2] = np.minimum(255, img_rgb[overlap, 2] + 128)
    
    create_image_slice(renderer, img_rgb, grayscale=False)

def create_volume_bar_chart(renderer, volume_metrics):
    
    volumes = [volume_metrics['volume1'], volume_metrics['volume2']]
    max_volume = max(volumes) if max(volumes) > 0 else 1
    
    bg_plane = vtk.vtkPlaneSource()
    bg_plane.SetOrigin(-0.6, -0.3, 0)
    bg_plane.SetPoint1(0.6, -0.3, 0)
    bg_plane.SetPoint2(-0.6, 0.9, 0)
    
    bg_mapper = vtk.vtkPolyDataMapper()
    bg_mapper.SetInputConnection(bg_plane.GetOutputPort())
    
    bg_actor = vtk.vtkActor()
    bg_actor.SetMapper(bg_mapper)
    bg_actor.GetProperty().SetColor(1.0, 1.0, 1.0)  
    
    renderer.AddActor(bg_actor)
    
    colors = [[0.8, 0.2, 0.2], [0.2, 0.2, 0.8]]  
    labels = ["Time 1", "Time 2"]
    
    bar_width = 0.2
    bar_spacing = 0.3
    
    for i, volume in enumerate(volumes):
        bar_height = (volume / max_volume) * 0.6
        
        x_pos = (i - 0.5) * bar_spacing
        
        cube = vtk.vtkCubeSource()
        cube.SetXLength(bar_width)
        cube.SetYLength(bar_height)
        cube.SetZLength(0.01) 
        cube.SetCenter(x_pos, bar_height/2, 0.01)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors[i][0], colors[i][1], colors[i][2])
        
        renderer.AddActor(actor)
        
        label_text = vtk.vtkVectorText()
        label_text.SetText(labels[i])
        
        label_mapper = vtk.vtkPolyDataMapper()
        label_mapper.SetInputConnection(label_text.GetOutputPort())
        
        label_actor = vtk.vtkActor()
        label_actor.SetMapper(label_mapper)
        label_actor.SetPosition(x_pos - 0.08, -0.25, 0.02)
        label_actor.SetScale(0.05, 0.05, 0.05)
        label_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        
        renderer.AddActor(label_actor)
        
        value_text = vtk.vtkVectorText()
        value_text.SetText(f"{volume:.0f}")
        
        value_mapper = vtk.vtkPolyDataMapper()
        value_mapper.SetInputConnection(value_text.GetOutputPort())
        
        value_actor = vtk.vtkActor()
        value_actor.SetMapper(value_mapper)
        value_actor.SetPosition(x_pos - 0.05, bar_height + 0.05, 0.02)
        value_actor.SetScale(0.03, 0.03, 0.03)
        value_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        
        renderer.AddActor(value_actor)
    
    change_text = vtk.vtkVectorText()
    change_text.SetText(f"Change: {volume_metrics['relative_change']:.1f}%")
    
    change_mapper = vtk.vtkPolyDataMapper()
    change_mapper.SetInputConnection(change_text.GetOutputPort())
    
    change_actor = vtk.vtkActor()
    change_actor.SetMapper(change_mapper)
    change_actor.SetPosition(-0.15, 0.7, 0.02)
    change_actor.SetScale(0.04, 0.04, 0.04)
    change_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
    
    renderer.AddActor(change_actor)
    
    title_text = vtk.vtkVectorText()
    title_text.SetText("Volume (mm³)")
    
    title_mapper = vtk.vtkPolyDataMapper()
    title_mapper.SetInputConnection(title_text.GetOutputPort())
    
    title_actor = vtk.vtkActor()
    title_actor.SetMapper(title_mapper)
    title_actor.SetPosition(-0.1, 0.8, 0.02)
    title_actor.SetScale(0.03, 0.03, 0.03)
    title_actor.GetProperty().SetColor(0.0, 0.0, 0.0) 
    
    renderer.AddActor(title_actor)
    
    line_points = vtk.vtkPoints()
    line_points.InsertNextPoint(-0.4, 0, 0.01)
    line_points.InsertNextPoint(-0.4, 0.6, 0.01)
    
    line_cells = vtk.vtkCellArray()
    line_cells.InsertNextCell(2)
    line_cells.InsertCellPoint(0)
    line_cells.InsertCellPoint(1)
    
    line_polydata = vtk.vtkPolyData()
    line_polydata.SetPoints(line_points)
    line_polydata.SetLines(line_cells)
    
    line_mapper = vtk.vtkPolyDataMapper()
    line_mapper.SetInputData(line_polydata)
    
    line_actor = vtk.vtkActor()
    line_actor.SetMapper(line_mapper)
    line_actor.GetProperty().SetColor(0.0, 0.0, 0.0) 
    line_actor.GetProperty().SetLineWidth(2)
    
    renderer.AddActor(line_actor)
    
    hline_points = vtk.vtkPoints()
    hline_points.InsertNextPoint(-0.4, 0, 0.01)
    hline_points.InsertNextPoint(0.4, 0, 0.01)
    
    hline_cells = vtk.vtkCellArray()
    hline_cells.InsertNextCell(2)
    hline_cells.InsertCellPoint(0)
    hline_cells.InsertCellPoint(1)
    
    hline_polydata = vtk.vtkPolyData()
    hline_polydata.SetPoints(hline_points)
    hline_polydata.SetLines(hline_cells)
    
    hline_mapper = vtk.vtkPolyDataMapper()
    hline_mapper.SetInputData(hline_polydata)
    
    hline_actor = vtk.vtkActor()
    hline_actor.SetMapper(hline_mapper)
    hline_actor.GetProperty().SetColor(0.0, 0.0, 0.0) 
    hline_actor.GetProperty().SetLineWidth(2)
    
    renderer.AddActor(hline_actor)
    
    renderer.ResetCamera()

def create_summary_text(renderer, volume_metrics, intensity_metrics, spatial_metrics):
    
    stats_text = f"""Volume Change: {volume_metrics['relative_change']:.1f}%
Intensity Change: {intensity_metrics['mean_change_relative']:.1f}%
Centroid Shift: {spatial_metrics['shift_magnitude_physical']:.2f} mm
Overlap Coefficient: {spatial_metrics['overlap_coefficient']:.3f}

Notebook Parameters:
- Threshold: 500-800
- Seed: (90, 70, 51)
- Smoothing: 5 iterations"""
    
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(stats_text)
    text_actor.SetPosition(10, 200) 
    text_actor.GetTextProperty().SetFontSize(12)  
    text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    renderer.AddActor2D(text_actor)

def apply_rdbu_colormap(normalized_data):
    h, w = normalized_data.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            val = normalized_data[i, j]
            if val < 0.5:
                rgb_image[i, j, 0] = int(255 * (1 - val * 2))
                rgb_image[i, j, 1] = int(255 * val * 2)
                rgb_image[i, j, 2] = int(255 * val * 2)
            else:
                val_adj = (val - 0.5) * 2
                rgb_image[i, j, 0] = int(255 * (1 - val_adj))
                rgb_image[i, j, 1] = int(255 * (1 - val_adj))
                rgb_image[i, j, 2] = 255
    
    return rgb_image

def numpy_to_vtk_image(numpy_array):
    if len(numpy_array.shape) == 3: 
        h, w, c = numpy_array.shape
        vtk_data = numpy_support.numpy_to_vtk(numpy_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_data.SetNumberOfComponents(c)
    else:  
        h, w = numpy_array.shape
        vtk_data = numpy_support.numpy_to_vtk(numpy_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_data.SetNumberOfComponents(1)
    
    img = vtk.vtkImageData()
    img.SetDimensions(w, h, 1)
    img.GetPointData().SetScalars(vtk_data)
    
    return img

def create_3d_visualization(seg1, seg2):
    print("3D visualisation")
    
    try:
        array1 = itk.GetArrayFromImage(seg1) > 0
        array2 = itk.GetArrayFromImage(seg2) > 0
        
        array1_ds = array1[::2, ::2, ::2]
        array2_ds = array2[::2, ::2, ::2]
        
        create_3d_tumor_window(array1_ds, "Tumor Time 1", [0.8, 0.2, 0.2])
        create_3d_tumor_window(array2_ds, "Tumor Time 2", [0.2, 0.2, 0.8])
        create_3d_overlay_window(array1_ds, array2_ds)
        
    except Exception as e:
        print(f"FAIL (pas assez de mémoire ? essaye dataset plus petit):  {e}")

def create_3d_tumor_window(array, title, color):
    
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(400, 400)
    render_window.SetWindowName(title)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    z, y, x = np.where(array)
    
    if len(x) > 0:
        step = max(1, len(x) // 1000)
        x_sub = x[::step]
        y_sub = y[::step]
        z_sub = z[::step]
        
        points = vtk.vtkPoints()
        for i in range(len(x_sub)):
            points.InsertNextPoint(x_sub[i], y_sub[i], z_sub[i])
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        vertices = vtk.vtkCellArray()
        for i in range(points.GetNumberOfPoints()):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
        polydata.SetVerts(vertices)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        actor.GetProperty().SetPointSize(2)
        
        renderer.AddActor(actor)
    
    renderer.SetBackground(0.1, 0.1, 0.1)
    renderer.ResetCamera()
    
    title_actor = vtk.vtkTextActor()
    title_actor.SetInput(title)
    title_actor.SetPosition(10, 370)
    title_actor.GetTextProperty().SetFontSize(14)
    title_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    renderer.AddActor2D(title_actor)
    
    render_window.Render()
    interactor.Start()

def create_3d_overlay_window(array1, array2):
    
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(400, 400)
    render_window.SetWindowName("Tumor Overlay")
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    z1, y1, x1 = np.where(array1)
    if len(x1) > 0:
        step1 = max(1, len(x1) // 500)
        
        points1 = vtk.vtkPoints()
        for i in range(0, len(x1), step1):
            points1.InsertNextPoint(x1[i], y1[i], z1[i])
        
        polydata1 = vtk.vtkPolyData()
        polydata1.SetPoints(points1)
        
        vertices1 = vtk.vtkCellArray()
        for i in range(points1.GetNumberOfPoints()):
            vertices1.InsertNextCell(1)
            vertices1.InsertCellPoint(i)
        polydata1.SetVerts(vertices1)
        
        mapper1 = vtk.vtkPolyDataMapper()
        mapper1.SetInputData(polydata1)
        
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper1)
        actor1.GetProperty().SetColor(0.8, 0.2, 0.2)
        actor1.GetProperty().SetPointSize(2)
        
        renderer.AddActor(actor1)
    
    z2, y2, x2 = np.where(array2)
    if len(x2) > 0:
        step2 = max(1, len(x2) // 500)
        
        points2 = vtk.vtkPoints()
        for i in range(0, len(x2), step2):
            points2.InsertNextPoint(x2[i], y2[i], z2[i])
        
        polydata2 = vtk.vtkPolyData()
        polydata2.SetPoints(points2)
        
        vertices2 = vtk.vtkCellArray()
        for i in range(points2.GetNumberOfPoints()):
            vertices2.InsertNextCell(1)
            vertices2.InsertCellPoint(i)
        polydata2.SetVerts(vertices2)
        
        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputData(polydata2)
        
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)
        actor2.GetProperty().SetColor(0.2, 0.2, 0.8)
        actor2.GetProperty().SetPointSize(2)
        
        renderer.AddActor(actor2)
    
    renderer.SetBackground(0.1, 0.1, 0.1)
    renderer.ResetCamera()
    
    title_actor = vtk.vtkTextActor()
    title_actor.SetInput("Tumor Overlay")
    title_actor.SetPosition(10, 370)
    title_actor.GetTextProperty().SetFontSize(14)
    title_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    renderer.AddActor2D(title_actor)
    
    render_window.Render()
    interactor.Start()

def plot_3d_tumor(ax, array, color, title):
    create_3d_tumor_window(array, title, color)

def plot_3d_overlay(ax, array1, array2):
    create_3d_overlay_window(array1, array2)

if __name__ == "__main__":
    run_full_analysis()