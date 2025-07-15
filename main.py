from src.registration import ImageRegistration


def main():
    fixed_image_path = "data/case6_gre1.nrrd"
    moving_image_path = "data/case6_gre2.nrrd"

    # Step 1: Register
    print("Starting image registration...")

    registration = ImageRegistration(fixed_image_path, moving_image_path)
    registered_image = None
    
    registration.load_images()
    if registration.rigid_registration():
        registered_image = registration.apply_transform()
    
    print("\nRegistration completed! Check the output files for results.")

    # Step 2: Segment
    print("Starting tumor segmentation...")

    # Step 3: Visualize
    print("Displaying comparison...")


if __name__ == "__main__":
    main()
