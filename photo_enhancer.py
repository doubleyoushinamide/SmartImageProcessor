import os
import sys
import subprocess
import logging
from pathlib import Path
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

class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'

def print_status(message):
    print(f"{Colors.GREEN}[STATUS]{Colors.NC} {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def print_error(message):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def check_python_version():
    print_status("Checking Python version...")
    if sys.version_info < (3, 6):
        print_error("Python 3.6 or higher is required")
        sys.exit(1)
    print_status(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")

def safe_import(module_name, package_name=None):
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
    check_python_version()
    required_packages = [
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL'),
        ('numpy', 'numpy'),
        ('scikit-image', 'skimage'),
        ('colorthief-py', 'colorthief')
    ]
    for pip_name, import_name in required_packages:
        success, _ = safe_import(import_name)
        if not success:
            if not install_package(pip_name):
                print_error(f"Critical dependency {pip_name} could not be installed")
                sys.exit(1)
            success, _ = safe_import(import_name)
            if not success:
                print_error(f"Failed to import {import_name} after installation")
                sys.exit(1)
    print_status("All dependencies verified")

class ImageAnalyzer:
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
            return [(0, 0, 0)] * 5

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
            return 50

    @staticmethod
    def get_brightness(cv_img: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            return np.mean(gray)
        except:
            return 128

    @staticmethod
    def analyze_image(img_path: str) -> Dict:
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
            return {}

    @staticmethod
    def estimate_sharpness(cv_img: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 100

class SmartEnhancer:
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
        if not analysis:
            logger.warning("Using default enhancements due to failed analysis")
            return
        try:
            brightness = analysis.get('brightness', 128)
            if brightness < 85:
                self.params['exposure'] = 0.5 + (85 - brightness)/85
            elif brightness > 170:
                self.params['exposure'] = -0.3 - (brightness - 170)/85
            contrast = analysis.get('contrast', 50)
            self.params['contrast'] = max(1.0, min(1.5, 50/contrast))
            hist_peaks = {k: v for k, v in analysis.get('histogram', {}).items() if '_peak' in k}
            if hist_peaks:
                avg_peak = sum(hist_peaks.values()) / len(hist_peaks)
                if avg_peak > 180:
                    self.params['highlights'] = -0.3
                    self.params['whites'] = -0.3
                elif avg_peak < 75:
                    self.params['shadows'] = 0.4
                    self.params['blacks'] = 0.4
            sharpness = analysis.get('sharpness', 100)
            if sharpness < 100:
                self.params['sharpness'] = min(0.5, (100 - sharpness)/200)
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
    def __init__(self, logo_path=None, position='bottom', width_mode='wide', height_mode='tall'):
        self.logo_path = logo_path or './logo/logo.png'
        self.position = position
        self.width_mode = width_mode
        self.height_mode = height_mode
        self.logo = None
        self.load_logo()

    def load_logo(self):
        try:
            if os.path.exists(self.logo_path):
                self.logo = Image.open(self.logo_path).convert('RGBA')
                logger.info(f"Logo loaded from {self.logo_path}")
            else:
                logger.warning(f"Logo not found at {self.logo_path}")
        except Exception as e:
            logger.error(f"Failed to load logo: {str(e)}")

    def add_logo(self, img: Image.Image) -> Image.Image:
        if self.logo is None:
            logger.warning("No logo available - skipping logo addition")
            return img
        try:
            is_landscape = img.width >= img.height
            if self.width_mode == 'wide':
                logo_width = int(img.width)
            elif self.width_mode in ('left', 'right'):
                logo_width = int(img.width * 0.5)
            elif self.width_mode == 'centre':
                logo_width = int(img.width * 0.7)
            else:
                logo_width = int(img.width)
            if self.height_mode == 'tall':
                if is_landscape:
                    logo_height = int(img.height * 0.22)
                else:
                    logo_height = int(self.logo.height * (logo_width / self.logo.width))
            elif self.height_mode == 'short':
                if is_landscape:
                    logo_height = int(img.height * 0.12)
                else:
                    logo_height = int(self.logo.height * (logo_width / self.logo.width) * 0.5)
            else:
                if is_landscape:
                    logo_height = int(img.height * 0.22)
                else:
                    logo_height = int(self.logo.height * (logo_width / self.logo.width))
            resized_logo = self.logo.resize((logo_width, logo_height), Image.LANCZOS)
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
    def __init__(self, output_dir='./processed_pics'):
        setup_environment()
        self.analyzer = ImageAnalyzer()
        self.enhancer = SmartEnhancer()
        self.logo_handler = LogoHandler()
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def apply_enhancements(self, img: Image.Image, params: Dict) -> Image.Image:
        try:
            img_array = np.array(img)
            exposure_factor = 2 ** params['exposure']
            img_array = img_array * exposure_factor
            mean_value = np.mean(img_array)
            img_array = (img_array - mean_value) * params['contrast'] + mean_value
            if params['color_temp'] != 5500:
                img_array = self.adjust_color_temp(img_array, params['color_temp'])
            img_array = np.clip(img_array, 0, 255)
            img = Image.fromarray(img_array.astype('uint8'))
            if params['sharpness'] > 0:
                img = self.apply_sharpening(img, params['sharpness'])
            return img
        except Exception as e:
            logger.error(f"Enhancement application failed: {str(e)}")
            return img

    @staticmethod
    def adjust_color_temp(img_array: np.ndarray, temp: float) -> np.ndarray:
        try:
            if temp > 5500:
                img_array[:,:,0] *= 0.9
                img_array[:,:,2] *= 1.1
            else:
                img_array[:,:,0] *= 1.1
                img_array[:,:,2] *= 0.9
            return img_array
        except:
            return img_array

    @staticmethod
    def apply_sharpening(img: Image.Image, amount: float) -> Image.Image:
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
        try:
            if not input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                return False
            filename = os.path.basename(input_path)
            output_path = os.path.join(self.output_dir, filename)
            if os.path.exists(output_path):
                logger.info(f"Skipping existing output: {output_path}")
                return False
            analysis = self.analyzer.analyze_image(input_path)
            self.enhancer.calculate_enhancements(analysis)
            with Image.open(input_path) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                enhanced_img = self.apply_enhancements(img, self.enhancer.params)
                final_img = self.logo_handler.add_logo(enhanced_img)
                final_img.save(output_path, quality=95)
                logger.info(f"Processed image saved to {output_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to process {input_path}: {str(e)}")
            return False

    def process_directory(self, input_dir: str = './photos') -> int:
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return 0
        image_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
            and not f.startswith('._')
        ]
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return 0
        logger.info(f"Found {len(image_files)} images to process")
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

class CustomPhotoEnhancer(PhotoEnhancer):
    def __init__(self, do_enhance=True, do_logo=True, logo_path=None, output_dir='./processed_pics'):
        self.analyzer = ImageAnalyzer()
        self.enhancer = SmartEnhancer()
        self.logo_handler = LogoHandler(logo_path=logo_path) if do_logo else None
        self.output_dir = output_dir
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