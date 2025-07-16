import itk
import numpy as np

def test_itk_installation():
    print("Testing itk")
    
    try:
        print("1.read image")
        image1 = itk.imread("data/case6_gre1.nrrd", itk.F)
        image2 = itk.imread("data/case6_gre2.nrrd", itk.F)
        print(f"Image 1 size: {itk.size(image1)}")
        print(f"Image 2 size: {itk.size(image2)}")
        
        print("2.transform création")
        TransformType = itk.TranslationTransform[itk.D, 3]
        transform = TransformType.New()
        print("Transform created successfully")
        
        print("3. optimizer creation")
        optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
        print("Optimizer created successfully")
        
        print("4.metric creation")
        ImageType = itk.Image[itk.F, 3]
        metric = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType].New()
        print("Metric createdsuccessfully")
        
        print("\nSUccess All")
        
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_itk_installation()