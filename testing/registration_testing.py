import src.registration as ImageRegistration

def registration_testing():
    fixed_image_path = "data/case6_gre1.nrrd"
    moving_image_path = "data/case6_gre2.nrrd"

    # Step 1: Register
    print("Starting image registration...")

    # Create registration object
    registration = ImageRegistration(fixed_image_path, moving_image_path)
    
    # Load images
    registration.load_images()
    
    # Try different registration methods
    print("\n" + "="*50)
    print("TESTING DIFFERENT REGISTRATION METHODS")
    print("="*50)
    
    # 1. Rigid registration
    print("\n1. RIGID REGISTRATION")
    print("-" * 30)
    if registration.rigid_registration():
        registered_image = registration.apply_transform("registered_rigid.nrrd")
        metrics = registration.get_registration_quality_metrics()
        print(f"Quality metrics: {metrics}")
    
    # 2. Affine registration
    print("\n2. AFFINE REGISTRATION")
    print("-" * 30)
    if registration.affine_registration():
        registered_image = registration.apply_transform("registered_affine.nrrd")
        metrics = registration.get_registration_quality_metrics()
        print(f"Quality metrics: {metrics}")
    
    # 3. B-spline registration (commented out as it's computationally intensive)
    print("\n3. B-SPLINE REGISTRATION")
    print("-" * 30)
    if registration.bspline_registration():
        registered_image = registration.apply_transform("registered_bspline.nrrd")
        metrics = registration.get_registration_quality_metrics()
        print(f"Quality metrics: {metrics}")