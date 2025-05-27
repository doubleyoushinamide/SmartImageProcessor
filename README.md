# SmartImageProcessor CLI
## Directory Structure

```
cli_smart_PROCESSOR/
├── photos/           # Place your input images here (jpg, jpeg, png, bmp, tif, tiff)
├── logo/
│   └── logo.png      # Your logo image (PNG with transparency recommended)
└── smart_enhancer.py # Main CLI script
```

## Features
- **Batch photo enhancement** with smart auto-adjustments (exposure, contrast, sharpness, color temperature, etc.)
- **Logo placement** with flexible options (position, width, height)
- **Parallel processing** for fast batch jobs
- **Automatic dependency installation** (if missing)
- **Safe for repeated runs** (skips already processed images)

## Requirements
- Python 3.6+
- The script will attempt to install any missing dependencies automatically.

## Dependencies
- opencv-python
- Pillow
- numpy
- scikit-image
- colorthief-py

## Setup
1. **Place your images** to be processed in the `photos/` directory.
2. **Place your logo** (PNG recommended) as `logo/logo.png`.
3. Open a terminal and navigate to the `cli_smart_PROCESSOR` directory:
   ```bash
   cd cli_smart_PROCESSOR
   ```
4. Run the CLI script (see usage below).

## Usage
Run the script from the `cli_smart_PROCESSOR` directory:

```bash
python3 smart_enhancer.py [OPTIONS]
```

### Options
- **Logo Position (choose one):**
  - `-b`, `--bottom`   Place logo at the bottom (default)
  - `-t`, `--top`      Place logo at the top

- **Logo Width Mode (choose one):**
  - `-w`, `--wide`     Logo width: wide (full image width, default)
  - `-l`, `--left`     Logo width: left
  - `-r`, `--right`    Logo width: right
  - `-c`, `--centre`   Logo width: centre

- **Logo Height Mode (choose one):**
  - `--tall`           Logo height: tall (default)
  - `--short`          Logo height: short

- **Other Options:**
  - `--noenhance`      Do not apply photo enhancement, only add logo (if enabled)
  - `--nologo`         Do not add logo to processed image

### Example Commands

- **Default (enhance and add logo, bottom-wide):**
  ```bash
  python3 smart_enhancer.py
  ```
- **Top-right, short logo, no enhancement:**
  ```bash
  python3 smart_enhancer.py -t -r --short --noenhance
  ```
- **Enhance only, no logo:**
  ```bash
  python3 smart_enhancer.py --nologo
  ```

## Output
- Processed images are saved in `processed_pics/` (created automatically).
- Logs are written to `photo_enhancer.log`.

## Notes
- Only images with extensions `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` in the `photos/` directory will be processed.
- The script skips files that have already been processed.
- If the logo is missing, processing will continue without adding a logo.
- For best results, use a transparent PNG for your logo.

## Troubleshooting
- If you encounter missing dependencies, the script will attempt to install them automatically. If installation fails, install them manually:
  ```bash
  pip install opencv-python Pillow numpy scikit-image colorthief-py
  ```
- For permission errors, try running with `python3` and ensure you have write access to the directory.

---

**SmartImageProcessor CLI** — Fast, flexible, and smart batch photo enhancement for your projects!
