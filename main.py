import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package.split('>=')[0])
        return True
    except ImportError:
        return False

packages = [
    'itk>=5.3.0',
    'vtk>=9.2.0', 
    'numpy>=1.21.0',
    'matplotlib>=3.5.0',
    'scipy>=1.7.0',
    'nbformat>=5.0.0',
    'nbconvert>=6.0.0'
]

missing = [pkg for pkg in packages if not install_if_missing(pkg)]

if missing:
    print(f"Missing packages: {', '.join(missing)}")
    response = input("Do you want to install them with pip (else go check requirement.txt)? (y/n): ")
    if response.lower() == 'y':
        for pkg in missing:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    else:
        print("Fail")
        sys.exit(1)

from src.registration import ImageRegistration


def main():
    print("Brain Tumor Analisis")
    print("=" * 40)
    fixed_image_path = "data/case6_gre1.nrrd"
    moving_image_path = "data/case6_gre2.nrrd"
    registered_output_path = "translation.nrrd"

    print("\nStep 1: registration")
    print("-" * 30)
    
    registration = ImageRegistration(fixed_image_path, moving_image_path)
    registration.load_images()
    
    if registration.translation_registration():
        registered_image = registration.apply_transform(registered_output_path)
        print("Success")
    else:
        print("FAIL")
        return

    print("\nStep 2: segmentation")
    print("-" * 50)
    
    try:
        from src.visualization import run_full_analysis
        
        if run_full_analysis():
            print("Success")
        else:
            print("FAIL")            
    except Exception as e:
        
        try:
            from src.notebook_executor import test_notebook_executor
            test_notebook_executor()
        except Exception as e2:
            print("Basic test failed")

def test_segmentation_only():

    print("Test Notebook")
    print("=" * 40)
    
    try:
        from src.notebook_executor import test_notebook_executor
        
        result = test_notebook_executor()
        if result[0] is not None:  # seg1
            print("Sucess")
        else:
            print("FAIL")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

def check_files():
    import os
    
    print("check file required")
    print("=" * 30)
    
    required_files = [
        "data/case6_gre1.nrrd",
        "data/case6_gre2.nrrd",
        "src/registration.py",
        "src/notebook_executor.py",
        "src/visualization.py"
    ]
    
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            print(f"{file}")
        else:
            print(f"{file} - not Found")
            all_good = False
    
    if os.path.exists("translation.nrrd"):
        print("Success")
    else:
        print("Fail")
    
    if all_good:
        print("Success all")
    else:
        print("FAIL")
    
    return all_good

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_segmentation_only()
        elif sys.argv[1] == "check":
            check_files()
        else:
            print("Unknown option")
    else:
        print("Starting")
        main()