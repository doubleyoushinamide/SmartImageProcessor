#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ANSI color codes for output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def print_status(message):
    print(f"{Colors.GREEN}[STATUS]{Colors.NC} {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def print_error(message):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def check_python_version():
    """Check if Python version is sufficient"""
    print_status("Checking Python version...")
    if sys.version_info < (3, 6):
        print_error("Python 3.6 or higher is required")
        sys.exit(1)
    print_status(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")

def safe_import(module_name, package_name=None):
    """
    Attempt to import a module, with helpful error messages
    Returns (success: bool, module: object or None)
    """
    try:
        if package_name is None:
            package_name = module_name
        __import__(module_name)
        return True, sys.modules[module_name]
    except ImportError:
        print_warning(f"Required package '{package_name}' not found")
        return False, None
    except Exception as e:
        print_error(f"Error importing {module_name}: {str(e)}")
        return False, None

def install_package(package_name):
    """Install a package using pip"""
    print_status(f"Attempting to install {package_name}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print_status(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print_error(f"Failed to install {package_name}")
        return False
    except Exception as e:
        print_error(f"Unexpected error installing {package_name}: {str(e)}")
        return False

def setup_environment():
    """Ensure all dependencies are available"""
    check_python_version()
    
    # List of required packages with their pip names and import names
    required_packages = [
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL'),
        ('numpy', 'numpy'),
        ('scikit-image', 'skimage'),
        ('colorthief-py', 'colorthief')
    ]
    
    # Check each package
    for pip_name, import_name in required_packages:
        success, _ = safe_import(import_name)
        if not success:
            if not install_package(pip_name):
                print_error(f"Critical dependency {pip_name} could not be installed")
                sys.exit(1)
            # Verify the install worked
            success, _ = safe_import(import_name)
            if not success:
                print_error(f"Failed to import {import_name} after installation")
                sys.exit(1)
    
    print_status("All dependencies verified")
    
#------------------- Args Parser ------------------------#
def parse_args():
    parser = argparse.ArgumentParser(description="Smart Photo Enhancer with Logo Placement Options")
    group_pos = parser.add_mutually_exclusive_group()
    group_pos.add_argument('-b', '--bottom', action='store_true', help='Place logo at the bottom (default)')
    group_pos.add_argument('-t', '--top', action='store_true', help='Place logo at the top')

    group_width = parser.add_mutually_exclusive_group()
    group_width.add_argument('-w', '--wide', action='store_true', help='Logo width: wide (full image width, default)')
    group_width.add_argument('-l', '--left', action='store_true', help='Logo width: left')
    group_width.add_argument('-r', '--right', action='store_true', help='Logo width: right')
    group_width.add_argument('-c', '--centre', action='store_true', help='Logo width: centre')

    group_height = parser.add_mutually_exclusive_group()
    group_height.add_argument('--tall', action='store_true', help='Logo height: tall (default)')
    group_height.add_argument('--short', action='store_true', help='Logo height: short')
    parser.add_argument('--noenhance', action='store_true', help='Do not apply photo enhancement, only copy/add logo if enabled')
    parser.add_argument('--nologo', action='store_true', help='Do not add logo to processed image')
    
    return parser.parse_args()

# ------------------- MAIN CODE ------------------------#

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat, ImageOps
from colorthief import ColorThief
from skimage import exposure, metrics
import hashlib
import concurrent.futures
from typing import Tuple, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('photo_enhancer.log')
    ]
)
logger = logging.getLogger(__name__) 

class ImageAnalyzer:
    """Analyzes image properties for smart enhancement"""
    
    @staticmethod
    def get_basic_stats(img: Image.Image) -> Dict:
        stat = ImageStat.Stat(img)
        return {
            'mean': stat.mean,
            'median': stat.median,
            'stddev': stat.stddev,
            'extrema': stat.extrema
        }

    @staticmethod
    def get_color_dominance(img_path: str) -> List[Tuple[int, int, int]]:
        try:
            color_thief = ColorThief(img_path)
            return color_thief.get_palette(color_count=5, quality=10)
        except Exception as e:
            logger.warning(f"Color dominance analysis failed: {str(e)}")
            return [(0, 0, 0)] * 5  # Return neutral colors on failure

    @staticmethod
    def get_histogram_analysis(cv_img: np.ndarray) -> Dict:
        try:
            histograms = {
                'b': cv2.calcHist([cv_img], [0], None, [256], [0, 256]),
                'g': cv2.calcHist([cv_img], [1], None, [256], [0, 256]),
                'r': cv2.calcHist([cv_img], [2], None, [256], [0, 256])
            }
            
            metrics = {}
            for channel, hist in histograms.items():
                metrics[f'{channel}_peak'] = np.argmax(hist)
                hist_nonzero = np.where(hist > 0)[0]
                if len(hist_nonzero) > 0:
                    metrics[f'{channel}_dynamic_range'] = hist_nonzero[-1] - hist_nonzero[0]
                else:
                    metrics[f'{channel}_dynamic_range'] = 0
            
            return {**histograms, **metrics}
        except Exception as e:
            logger.warning(f"Histogram analysis failed: {str(e)}")
            return {}

    @staticmethod
    def get_contrast_metrics(cv_img: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            return np.std(gray)
        except:
            return 50  # Default average contrast

    @staticmethod
    def get_brightness(cv_img: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            return np.mean(gray)
        except:
            return 128  # Default middle brightness

    @staticmethod
    def analyze_image(img_path: str) -> Dict:
        """Comprehensive image analysis with error handling"""
        try:
            pil_img = Image.open(img_path)
            cv_img = cv2.imread(img_path)
            
            if cv_img is None:
                raise ValueError("OpenCV failed to read image")
                
            return {
                'basic_stats': ImageAnalyzer.get_basic_stats(pil_img),
                'color_dominance': ImageAnalyzer.get_color_dominance(img_path),
                'histogram': ImageAnalyzer.get_histogram_analysis(cv_img),
                'contrast': ImageAnalyzer.get_contrast_metrics(cv_img),
                'brightness': ImageAnalyzer.get_brightness(cv_img),
                'sharpness': ImageAnalyzer.estimate_sharpness(cv_img)
            }
        except Exception as e:
            logger.error(f"Image analysis failed for {img_path}: {str(e)}")
            return {}  # Return empty dict on failure

    @staticmethod
    def estimate_sharpness(cv_img: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 100  # Default average sharpness

class SmartEnhancer:
    """Dynamically enhances images based on analysis"""
    
    def __init__(self):
        self.params = {
            'exposure': 0,
            'contrast': 1.0,
            'highlights': 0,
            'shadows': 0,
            'whites': 0,
            'blacks': 0,
            'sharpness': 0,
            'color_temp': 5500,
            'saturation': 1.0
        }
        
    def calculate_enhancements(self, analysis: Dict) -> None:
        """Calculate enhancement parameters with safety checks"""
        if not analysis:  # Empty analysis results
            logger.warning("Using default enhancements due to failed analysis")
            return
            
        try:
            # Adjust exposure based on brightness
            brightness = analysis.get('brightness', 128)
            if brightness < 85:
                self.params['exposure'] = 0.5 + (85 - brightness)/85
            elif brightness > 170:
                self.params['exposure'] = -0.3 - (brightness - 170)/85
            
            # Adjust contrast
            contrast = analysis.get('contrast', 50)
            self.params['contrast'] = max(1.0, min(1.5, 50/contrast))
            
            # Adjust highlights/shadows based on histogram
            hist_peaks = {k: v for k, v in analysis.get('histogram', {}).items() if '_peak' in k}
            if hist_peaks:
                avg_peak = sum(hist_peaks.values()) / len(hist_peaks)
                
                if avg_peak > 180:
                    self.params['highlights'] = -0.3
                    self.params['whites'] = -0.3
                elif avg_peak < 75:
                    self.params['shadows'] = 0.4
                    self.params['blacks'] = 0.4
            
            # Adjust sharpness
            sharpness = analysis.get('sharpness', 100)
            if sharpness < 100:
                self.params['sharpness'] = min(0.5, (100 - sharpness)/200)
            
            # Adjust color temperature
            dominant_colors = analysis.get('color_dominance', [(128, 128, 128)])
            avg_color = np.mean(dominant_colors, axis=0)
            if len(avg_color) == 3:
                blue_ratio = avg_color[2] / (avg_color[0] + 1)
                if blue_ratio > 1.2:
                    self.params['color_temp'] = 6500
                elif blue_ratio < 0.8:
                    self.params['color_temp'] = 4500
        except Exception as e:
            logger.error(f"Enhancement calculation failed: {str(e)}")

class LogoHandler:
    """Handles logo operations"""
    
    def __init__(self, logo_path='./logo/logo.png', position='bottom', width_mode='wide', height_mode='tall'):
        self.logo_path = logo_path
        self.logo = None
        self.position = position
        self.width_mode = width_mode
        self.height_mode = height_mode
        self.load_logo()
    
    def load_logo(self):
        """Load logo image with error handling"""
        try:
            if os.path.exists(self.logo_path):
                self.logo = Image.open(self.logo_path).convert('RGBA')
                logger.info(f"Logo loaded from {self.logo_path}")
            else:
                logger.warning(f"Logo not found at {self.logo_path}")
        except Exception as e:
            logger.error(f"Failed to load logo: {str(e)}")
    
    def add_logo(self, img: Image.Image) -> Image.Image:
        """Add logo to image with safety checks, auto-detecting portrait or landscape and using CLI args"""
        if self.logo is None:
            logger.warning("No logo available - skipping logo addition")
            return img
        try:
            is_landscape = img.width >= img.height
            # Improved logic: Only bottom-wide stretches full width. All other placements are confined.
            if self.position == 'bottom' and self.width_mode == 'wide':
                max_logo_width_ratio = 1.0
                max_logo_height_ratio = 0.22
            else:
                # For non-portrait (landscape/square), keep logo small for all but bottom-wide
                if is_landscape:
                    max_logo_width_ratio = 0.25
                    max_logo_height_ratio = 0.15
                else:
                    # For portrait, allow a bit larger logo
                    max_logo_width_ratio = 0.4
                    max_logo_height_ratio = 0.18
            # Calculate logo size
            logo_width = int(img.width * max_logo_width_ratio)
            aspect = self.logo.width / self.logo.height
            logo_height = int(logo_width / aspect)
            # If logo is too tall, limit by height
            if logo_height > int(img.height * max_logo_height_ratio):
                logo_height = int(img.height * max_logo_height_ratio)
                logo_width = int(logo_height * aspect)
            resized_logo = self.logo.resize((logo_width, logo_height), Image.LANCZOS)
            # Positioning
            if self.position == 'top':
                y = 0
            else:
                y = img.height - logo_height
            if self.width_mode == 'left':
                x = 0
            elif self.width_mode == 'right':
                x = img.width - logo_width
            else:
                x = (img.width - logo_width) // 2
            position = (x, y)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            img.paste(resized_logo, position, resized_logo)
            return img.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to add logo: {str(e)}")
            return img

class PhotoEnhancer:
    """Main photo enhancement pipeline"""
    
    def __init__(self):
        #DependencyManager.check_dependencies()
        self.analyzer = ImageAnalyzer()
        self.enhancer = SmartEnhancer()
        self.logo_handler = LogoHandler()
        
        # Create output directory
        self.output_dir = './processed_pics'
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file contents"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def apply_enhancements(self, img: Image.Image, params: Dict) -> Image.Image:
        """Apply all enhancements to an image"""
        try:
            # Convert to numpy array for processing
            img_array = np.array(img)
            
            # Exposure adjustment
            exposure_factor = 2 ** params['exposure']
            img_array = img_array * exposure_factor
            
            # Contrast adjustment
            mean_value = np.mean(img_array)
            img_array = (img_array - mean_value) * params['contrast'] + mean_value
            
            # Color temperature adjustment
            if params['color_temp'] != 5500:
                img_array = self.adjust_color_temp(img_array, params['color_temp'])
            
            # Clip values and convert back to Image
            img_array = np.clip(img_array, 0, 255)
            img = Image.fromarray(img_array.astype('uint8'))
            
            # Apply sharpening if needed
            if params['sharpness'] > 0:
                img = self.apply_sharpening(img, params['sharpness'])
            
            return img
        except Exception as e:
            logger.error(f"Enhancement application failed: {str(e)}")
            return img  # Return original if enhancement fails
    
    @staticmethod
    def adjust_color_temp(img_array: np.ndarray, temp: float) -> np.ndarray:
        """Adjust color temperature (simplified implementation)"""
        try:
            if temp > 5500:  # Make cooler (more blue)
                img_array[:,:,0] *= 0.9  # Reduce red
                img_array[:,:,2] *= 1.1  # Increase blue
            else:  # Make warmer (more red)
                img_array[:,:,0] *= 1.1  # Increase red
                img_array[:,:,2] *= 0.9  # Reduce blue
            return img_array
        except:
            return img_array
    
    @staticmethod
    def apply_sharpening(img: Image.Image, amount: float) -> Image.Image:
        """Apply controlled sharpening"""
        try:
            radius = 1.0 + amount * 1.0
            percent = int(50 + amount * 150)
            threshold = 3
            return img.filter(ImageFilter.UnsharpMask(
                radius=radius, 
                percent=percent, 
                threshold=threshold
            ))
        except:
            return img
    
    def process_image(self, input_path: str) -> bool:
        """Process a single image with error handling"""
        try:
            # Skip non-image files
            if not input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                return False
            
            # Generate output path
            filename = os.path.basename(input_path)
            output_path = os.path.join(self.output_dir, filename)
            
            # Skip if output exists
            if os.path.exists(output_path):
                logger.info(f"Skipping existing output: {output_path}")
                return False
            
            # Analyze and enhance
            analysis = self.analyzer.analyze_image(input_path)
            self.enhancer.calculate_enhancements(analysis)
            
            with Image.open(input_path) as img:
                # Convert to RGB if needed
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                # Apply enhancements
                enhanced_img = self.apply_enhancements(img, self.enhancer.params)
                
                # Add logo
                final_img = self.logo_handler.add_logo(enhanced_img)
                
                # Save result
                final_img.save(output_path, quality=95)
                logger.info(f"Processed image saved to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to process {input_path}: {str(e)}")
            return False
    
    def process_directory(self, input_dir: str = './photos') -> int:
        """Process all images in a directory"""
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return 0
        
        image_files = [
            os.path.join(input_dir, f) 
            for f in os.listdir(input_dir) 
            if f.lower().endswith((
                '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
            and not f.startswith('._')
        ]
        
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return 0
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images in parallel with error handling
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_image, img_path) for img_path in image_files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
        
        logger.info(f"Successfully processed {success_count}/{len(image_files)} images")
        return success_count

def main():
    """Main entry point"""
    try:
        enhancer = PhotoEnhancer()
        
        # Check if logo exists
        if enhancer.logo_handler.logo is None:
            logger.warning("Processing will continue without logo")
        
        # Process all images in photos directory
        success_count = enhancer.process_directory()
        
        if success_count > 0:
            logger.info(f"Done! Processed {success_count} images. Output in '{enhancer.output_dir}'")
        else:
            logger.warning("No images were processed. Check logs for details.")
        
        return 0 if success_count > 0 else 1
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    args = parse_args()
    # Determine position
    position = 'bottom'
    if args.top:
        position = 'top'
    # Determine width mode
    if args.left:
        width_mode = 'left'
    elif args.right:
        width_mode = 'right'
    elif args.centre:
        width_mode = 'centre'
    else:
        width_mode = 'wide'
    # Determine height mode
    height_mode = 'tall'
    if args.short:
        height_mode = 'short'

    class CustomPhotoEnhancer(PhotoEnhancer):
        def __init__(self, do_enhance=True, do_logo=True):
            self.analyzer = ImageAnalyzer()
            self.enhancer = SmartEnhancer()
            self.logo_handler = LogoHandler(position=position, width_mode=width_mode, height_mode=height_mode) if do_logo else None
            self.output_dir = './processed_pics'
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self.do_enhance = do_enhance
            self.do_logo = do_logo

        def apply_enhancements(self, img: Image.Image, params: Dict) -> Image.Image:
            if not self.do_enhance:
                return img
            return super().apply_enhancements(img, params)

        def process_image(self, input_path: str) -> bool:
            try:
                if not input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    return False
                filename = os.path.basename(input_path)
                output_path = os.path.join(self.output_dir, filename)
                if os.path.exists(output_path):
                    logger.info(f"Skipping existing output: {output_path}")
                    return False
                analysis = self.analyzer.analyze_image(input_path) if self.do_enhance else None
                if self.do_enhance:
                    self.enhancer.calculate_enhancements(analysis)
                with Image.open(input_path) as img:
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    if self.do_enhance:
                        img = self.apply_enhancements(img, self.enhancer.params)
                    if self.do_logo and self.logo_handler is not None:
                        img = self.logo_handler.add_logo(img)
                    img.save(output_path, quality=95)
                    logger.info(f"Processed image saved to {output_path}")
                    return True
            except Exception as e:
                logger.error(f"Failed to process {input_path}: {str(e)}")
                return False

    if __name__ == "__main__":
        args = parse_args()
        # ...existing code for position, width_mode, height_mode...
        do_enhance = not args.noenhance
        do_logo = not args.nologo

        def main():
            try:
                enhancer = CustomPhotoEnhancer(do_enhance=do_enhance, do_logo=do_logo)
                if do_logo and (enhancer.logo_handler is None or enhancer.logo_handler.logo is None):
                    logger.warning("Processing will continue without logo")
                success_count = enhancer.process_directory()
                if success_count > 0:
                    logger.info(f"Done! Processed {success_count} images. Output in '{enhancer.output_dir}'")
                else:
                    logger.warning("No images were processed. Check logs for details.")
                return 0 if success_count > 0 else 1
            except Exception as e:
                logger.critical(f"Fatal error: {str(e)}")
                return 1

        sys.exit(main())