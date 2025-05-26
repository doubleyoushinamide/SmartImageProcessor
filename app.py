from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from photo_enhancer import CustomPhotoEnhancer  # Import the modified class

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload and processed folders if they don't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['PROCESSED_FOLDER']):
    os.makedirs(app.config['PROCESSED_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'images' not in request.files:
        return 'No images uploaded', 400
    images = request.files.getlist('images')
    logo_file = request.files.get('logo')
    enhance = 'enhance' in request.form
    add_logo = 'add_logo' in request.form and logo_file
    position = request.form.get('position', 'bottom')
    width_mode = request.form.get('width_mode', 'wide')
    height_mode = request.form.get('height_mode', 'tall')

    # Save logo if uploaded
    logo_path = None
    if add_logo and logo_file and allowed_file(logo_file.filename):
        filename = secure_filename(logo_file.filename)
        logo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logo_file.save(logo_path)

    processed_files = []
    for image_file in images:
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(input_path)

            # Process image
            enhancer = CustomPhotoEnhancer(
                do_enhance=enhance,
                do_logo=add_logo,
                logo_path=logo_path,
                output_dir=app.config['PROCESSED_FOLDER']
            )
            # Update logo_handler with position, width_mode, height_mode
            if add_logo and enhancer.logo_handler:
                enhancer.logo_handler.position = position
                enhancer.logo_handler.width_mode = width_mode
                enhancer.logo_handler.height_mode = height_mode
                enhancer.logo_handler.load_logo()  # Reload logo with new settings
            success = enhancer.process_image(input_path)
            if success:
                processed_files.append(filename)

    return render_template('results.html', files=processed_files)

@app.route('/processed/<filename>')
def downloaded_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)