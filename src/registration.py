import itk
import numpy as np
import os

class ImageRegistration:
    def __init__(self, fixed_image_path, moving_image_path):
        """
        Initialize the image registration class
        
        Args:
            fixed_image_path: Path to the reference image (case6_gre1.nrrd)
            moving_image_path: Path to the image to be registered (case6_gre2.nrrd)
        """
        self.fixed_image_path = fixed_image_path
        self.moving_image_path = moving_image_path
        self.fixed_image = None
        self.moving_image = None
        self.registered_image = None
        self.transform = None
        
    def load_images(self):
        """Load the NRRD images using ITK"""
        print("Loading images...")
        
        # Define the image type (assuming 3D float images)
        ImageType = itk.Image[itk.F, 3]
        
        # Read the fixed image
        fixed_reader = itk.ImageFileReader[ImageType].New()
        fixed_reader.SetFileName(self.fixed_image_path)
        fixed_reader.Update()
        self.fixed_image = fixed_reader.GetOutput()
        
        # Read the moving image
        moving_reader = itk.ImageFileReader[ImageType].New()
        moving_reader.SetFileName(self.moving_image_path)
        moving_reader.Update()
        self.moving_image = moving_reader.GetOutput()
        
        print(f"Fixed image size: {itk.size(self.fixed_image)}")
        print(f"Moving image size: {itk.size(self.moving_image)}")
        
    def rigid_registration(self):
        """
        Perform rigid registration (translation + rotation)
        Good for aligning scans of the same patient taken at different times
        Uses VersorRigid3DTransform which is supported in ITK Python
        """
        print("Performing rigid registration...")
        
        # Define types - using supported transforms
        ImageType = itk.Image[itk.F, 3]
        TransformType = itk.VersorRigid3DTransform[itk.D]
        OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
        MetricType = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType]
        RegistrationType = itk.ImageRegistrationMethodv4[ImageType, ImageType]
        
        # Create components
        metric = MetricType.New()
        optimizer = OptimizerType.New()
        registration = RegistrationType.New()
        
        # Set up the registration
        registration.SetMetric(metric)
        registration.SetOptimizer(optimizer)
        registration.SetFixedImage(self.fixed_image)
        registration.SetMovingImage(self.moving_image)
        
        # Initialize transform
        transform = TransformType.New()
        initializer = itk.CenteredTransformInitializer[
            TransformType, ImageType, ImageType
        ].New()
        initializer.SetTransform(transform)
        initializer.SetFixedImage(self.fixed_image)
        initializer.SetMovingImage(self.moving_image)
        initializer.MomentsOn()
        initializer.InitializeTransform()
        
        registration.SetInitialTransform(transform)
        
        # Set up optimizer - VersorRigid3D needs different parameters
        optimizer.SetLearningRate(0.2)
        optimizer.SetMinimumStepLength(0.001)
        optimizer.SetNumberOfIterations(200)
        optimizer.SetReturnBestParametersAndValue(True)
        
        # Set up metric sampling
        registration.SetMetricSamplingPercentage(0.20)
        
        # Set up multi-resolution
        registration.SetNumberOfLevels(3)
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        
        try:
            registration.Update()
            self.transform = registration.GetTransform()
            print("Rigid registration completed successfully!")
            return True
        except Exception as e:
            print(f"Rigid registration failed: {e}")
            return False
    
    def affine_registration(self):
        """
        Perform affine registration (translation + rotation + scaling + shearing)
        More flexible than rigid registration
        Uses MatrixOffsetTransformBase which is supported in ITK Python
        """
        print("Performing affine registration...")
        
        # Define types - using supported transforms
        ImageType = itk.Image[itk.F, 3]
        TransformType = itk.AffineTransform[itk.D, 3]
        OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
        MetricType = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType]
        RegistrationType = itk.ImageRegistrationMethodv4[ImageType, ImageType]
        
        # Create components
        metric = MetricType.New()
        optimizer = OptimizerType.New()
        registration = RegistrationType.New()
        
        # Set up the registration
        registration.SetMetric(metric)
        registration.SetOptimizer(optimizer)
        registration.SetFixedImage(self.fixed_image)
        registration.SetMovingImage(self.moving_image)
        
        # Initialize transform manually since CenteredTransformInitializer may not be available
        transform = TransformType.New()
        
        # Manual initialization - center the transform at the image center
        fixed_center = [0.0, 0.0, 0.0]
        fixed_size = self.fixed_image.GetLargestPossibleRegion().GetSize()
        fixed_spacing = self.fixed_image.GetSpacing()
        fixed_origin = self.fixed_image.GetOrigin()
        
        for i in range(3):
            fixed_center[i] = fixed_origin[i] + fixed_spacing[i] * fixed_size[i] / 2.0
        
        transform.SetCenter(fixed_center)
        transform.SetIdentity()
        
        registration.SetInitialTransform(transform)
        
        # Set up optimizer
        optimizer.SetLearningRate(0.1)
        optimizer.SetMinimumStepLength(0.001)
        optimizer.SetNumberOfIterations(300)
        optimizer.SetReturnBestParametersAndValue(True)
        
        # Set up metric sampling
        registration.SetMetricSamplingPercentage(0.20)
        
        # Set up multi-resolution
        registration.SetNumberOfLevels(3)
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        
        try:
            registration.Update()
            self.transform = registration.GetTransform()
            print("Affine registration completed successfully!")
            return True
        except Exception as e:
            print(f"Affine registration failed: {e}")
            return False
        
    def translation_registration(self):
        dimension = 3
        FixedImageType = type(self.fixed_image)
        MovingImageType = type(self.moving_image)

        TransformType = itk.TranslationTransform[itk.D, dimension]
        initialTransform = TransformType.New()

        optimizer = itk.RegularStepGradientDescentOptimizerv4.New()

        optimizer.SetLearningRate(1)
        optimizer.SetMinimumStepLength(1e-6)
        optimizer.SetNumberOfIterations(100)

        metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()
        fixed_interpolation = itk.LinearInterpolateImageFunction[FixedImageType, itk.D].New()
        metric.SetFixedInterpolator(fixed_interpolation)

        registration = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType].New(FixedImage=self.fixed_image, MovingImage=self.moving_image, Metric=metric,
                                                                                    Optimizer=optimizer, InitialTransform=initialTransform)

        moving_initial_transform = TransformType.New()
        initial_parameters = moving_initial_transform.GetParameters()
        initial_parameters[0] = 0
        initial_parameters[1] = 0
        moving_initial_transform.SetParameters(initial_parameters)
        registration.SetMovingInitialTransform(moving_initial_transform)

        identity_transform = TransformType.New()
        identity_transform.SetIdentity()
        registration.SetFixedInitialTransform(identity_transform)

        registration.SetNumberOfLevels(1)

        try:
            registration.Update()
            self.transform = registration.GetTransform()
            print("Affine registration completed successfully!")
            return True
        except Exception as e:
            print(f"Affine registration failed: {e}")
            return False

    
    def bspline_registration(self):
        """
        Perform B-spline registration for non-rigid deformation
        More computationally intensive but can handle local deformations
        """
        print("Performing B-spline registration...")
        
        # Define types
        ImageType = itk.Image[itk.F, 3]
        TransformType = itk.BSplineTransform[itk.D, 3, 3]
        OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]  # Changed optimizer
        MetricType = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType]
        RegistrationType = itk.ImageRegistrationMethodv4[ImageType, ImageType]
        
        # Create components
        metric = MetricType.New()
        optimizer = OptimizerType.New()
        registration = RegistrationType.New()
        
        # Set up the registration
        registration.SetMetric(metric)
        registration.SetOptimizer(optimizer)
        registration.SetFixedImage(self.fixed_image)
        registration.SetMovingImage(self.moving_image)
        
        # Initialize B-spline transform
        transform = TransformType.New()
        fixed_region = self.fixed_image.GetLargestPossibleRegion()
        fixed_spacing = self.fixed_image.GetSpacing()
        fixed_origin = self.fixed_image.GetOrigin()
        fixed_direction = self.fixed_image.GetDirection()
        
        # Set up grid - reduce grid size for stability
        grid_size = [4, 4, 4]  # Further reduced for better stability
        transform.SetTransformDomainOrigin(fixed_origin)
        transform.SetTransformDomainDirection(fixed_direction)
        
        # Calculate physical dimensions
        physical_dimensions = []
        for i in range(3):
            physical_dimensions.append(fixed_spacing[i] * (fixed_region.GetSize()[i] - 1))
        transform.SetTransformDomainPhysicalDimensions(physical_dimensions)
        transform.SetTransformDomainMeshSize(grid_size)
        
        registration.SetInitialTransform(transform)
        
        # Set up optimizer - using RegularStepGradientDescent instead of LBFGS
        optimizer.SetLearningRate(0.01)  # Smaller learning rate for B-spline
        optimizer.SetMinimumStepLength(0.0001)
        optimizer.SetNumberOfIterations(100)
        optimizer.SetReturnBestParametersAndValue(True)
        
        # Set up metric sampling
        registration.SetMetricSamplingPercentage(0.05)  # Reduced for faster execution
        
        # Single resolution level for B-spline
        registration.SetNumberOfLevels(1)
        
        try:
            registration.Update()
            self.transform = registration.GetTransform()
            print("B-spline registration completed successfully!")
            return True
        except Exception as e:
            print(f"B-spline registration failed: {e}")
            return False
    
    def apply_transform(self, output_path=None):
        """
        Apply the computed transform to the moving image
        
        Args:
            output_path: Path to save the registered image
        """
        if self.transform is None:
            print("No transform available. Please run registration first.")
            return None
        
        print("Applying transform to moving image...")
        
        # Define types
        ImageType = itk.Image[itk.F, 3]
        ResampleFilterType = itk.ResampleImageFilter[ImageType, ImageType]
        
        # Create resampler
        resampler = ResampleFilterType.New()
        resampler.SetInput(self.moving_image)
        resampler.SetTransform(self.transform)
        resampler.SetUseReferenceImage(True)
        resampler.SetReferenceImage(self.fixed_image)
        resampler.SetDefaultPixelValue(0)
        
        # Apply transform
        resampler.Update()
        self.registered_image = resampler.GetOutput()
        
        # Save if path provided
        if output_path:
            writer = itk.ImageFileWriter[ImageType].New()
            writer.SetFileName(output_path)
            writer.SetInput(self.registered_image)
            writer.Update()
            print(f"Registered image saved to: {output_path}")
        
        return self.registered_image
    
    def get_registration_quality_metrics(self):
        """
        Calculate quality metrics for the registration
        """
        if self.registered_image is None:
            print("No registered image available.")
            return None
        
        # Convert to numpy arrays for easier computation
        fixed_array = itk.GetArrayFromImage(self.fixed_image)
        registered_array = itk.GetArrayFromImage(self.registered_image)
        
        # Calculate metrics
        mse = np.mean((fixed_array - registered_array) ** 2)
        normalized_cc = np.corrcoef(fixed_array.flatten(), registered_array.flatten())[0, 1]
        
        metrics = {
            'mse': mse,
            'normalized_cross_correlation': normalized_cc,
            'transform_parameters': self.transform.GetParameters()
        }
        
        return metrics