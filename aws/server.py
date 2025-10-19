from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import base64
from datetime import datetime
from vision_pipeline import VisionPipeline

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

vision_pipeline = VisionPipeline()

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Endpoint to handle image and depth file uploads.
    Accepts multipart/form-data with 'image' field and optional 'depth' field.
    """
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        depth_file = request.files.get('depth')  # Optional depth file
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if file and allowed_file(file.filename):
            # Get file info without saving
            original_filename = secure_filename(file.filename)
            file_size = len(file.read())
            file.seek(0)  # Reset file pointer to beginning
            
            # Process image through vision pipeline directly from memory
            vision_results = vision_pipeline.process_image(file, depth_file)
            
            # Extract processed image if available
            processed_image_data = None
            if 'processed_image' in vision_results and vision_results['processed_image']['data']:
                # Encode the processed image as base64 for JSON transmission
                processed_image_data = base64.b64encode(vision_results['processed_image']['data']).decode('utf-8')
                # Remove the binary data from vision_results to avoid JSON serialization issues
                vision_results['processed_image']['data'] = 'base64_encoded'
            
            return jsonify({
                'success': True,
                'message': 'Image processed successfully',
                'data': {
                    'original_filename': original_filename,
                    'file_size': file_size,
                    'upload_time': datetime.utcnow().isoformat(),
                    'vision_processing': vision_results,
                    'processed_image_base64': processed_image_data
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information."""
    return jsonify({
        'message': 'Image Upload API',
        'endpoints': {
            'POST /upload': 'Upload and process an image file',
            'GET /health': 'Health check',
            'GET /vision/stats': 'Get vision pipeline statistics',
            'GET /': 'This information'
        },
        'allowed_file_types': list(ALLOWED_EXTENSIONS),
        'max_file_size': f"{MAX_CONTENT_LENGTH // (1024 * 1024)}MB"
    }), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"Max file size: {MAX_CONTENT_LENGTH // (1024 * 1024)}MB")
    print("Vision Pipeline running in headless mode - no GUI windows will be displayed")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=80, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        vision_pipeline.cleanup()
        print("Server stopped.")