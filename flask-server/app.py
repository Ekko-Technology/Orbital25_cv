from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import numpy as np
import random
import logging
from datetime import datetime, timezone
# for securing user password
from werkzeug.security import generate_password_hash, check_password_hash 
# import libraries to initialize cloudinary for image storage
import cloudinary
import cloudinary.uploader as cloud_upload
from image_processing import apply_changes


app = Flask(__name__)
# Allows Flask as a backend to be accessed from React which is ran on another domain
CORS(app) 

# Configure Cloudinary for image storage
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_NAME"), 
    api_key = os.getenv("CLOUDINARY_API_KEY"), 
    api_secret = os.getenv("CLOUDINARY_API_SECRET"), # Click 'View API Keys' above to copy your API secret
    secure=True
)



# Postgresql configuration with render
db_url = os.getenv('DATABASE_URL') # Render Supports this internally
if db_url:
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url.replace("postgres://", "postgresql://", 1)  # Render fix
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///local.db'  # fallback for local dev

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)  # Initialize the DB


# User Table for storing user's particulars
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)  # Hashed password

    # JOINS with GameRecord Table 
    game_records = db.relationship('GameRecord', backref='user', lazy=True)

    # Standard helper methods for generating password
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    

# Table to track User's past images, scores and history
class GameRecord(db.Model): 
    id = db.Column(db.Integer, primary_key=True)
    # JOINS with User table via User.id value
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    original_image_path = db.Column(db.String(255), nullable=False)
    modified_image_path = db.Column(db.String(255), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    total_differences = db.Column(db.Integer, nullable=False)
    time_taken = db.Column(db.Float, nullable=False)
    played_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))



@app.route('/')
def index():
    return "Flask backend is running on render"


# ----- User Logins ------
# Backend for handling user data when registering new user
@app.route('/register', methods=['POST'])
def register_user():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 409

        user = User(username=username)
        user.set_password(password)  # Hash password
        db.session.add(user)
        db.session.commit()

        return jsonify({'message': 'User created', 'user_id': user.id})
    
    except Exception as e:
        app.logger.error(f"[REGISTER ERROR] {e}")
        return jsonify({'error': 'Server error at registration'}), 500



# Backend of user login page
@app.route('/login', methods=['POST'])
def login_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return jsonify({'message': 'Login successful', 'user_id': user.id})
    return jsonify({'error': 'Invalid username or password'}), 401


# Backend for 
@app.route('/save-game', methods=['POST'])
def save_game():
    data = request.json
    user_id = data.get('user_id')
    original_path = data.get('original_image')
    modified_path = data.get('modified_image')
    score = data.get('score')
    total = data.get('total')
    time_taken = data.get('time_taken')

    if not all([user_id, original_path, modified_path, score, total, time_taken]):
        return jsonify({'error': 'Missing fields'}), 400

    game = GameRecord(
        user_id=user_id,
        original_image_path=original_path,
        modified_image_path=modified_path,
        score=score,
        total_differences=total,
        time_taken=time_taken
    )
    db.session.add(game)
    db.session.commit()

    return jsonify({'message': 'Game saved', 'record_id': game.id}), 201



@app.route('/user/<int:user_id>/history')
def game_history(user_id):
    user = User.query.get_or_404(user_id)
    games = [{
        'original_image': g.original_image_path,
        'modified_image': g.modified_image_path,
        'score': g.score,
        'total': g.total_differences,
        'time_taken': g.time_taken,
        'played_at': g.played_at.isoformat()
    } for g in user.game_records]

    return jsonify({'username': user.username, 'games': games})




# ----- Image modification Backend Logic ------
UPLOAD_FOLDER = 'uploads' # Directory to save uploaded and processed images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Creates an upload folder directory
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")


objects_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'objects')
if not os.path.exists(objects_path):
    print(f"Warning: 'objects' folder not found at '{objects_path}'. Object addition will not work.")


# function to check if file is of valid type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/upload-and-process', methods=['POST'])
def upload_and_process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # obtain image type (e.g. jpg, jpeg)
            original_extension = file.filename.rsplit('.', 1)[1].lower()
            original_filename = f"original_{timestamp}.{original_extension}"
            modified_filename = f"modified_{timestamp}.{original_extension}" # Keep same extension for modified

            original_filepath = os.path.join(UPLOAD_FOLDER, original_filename)
            modified_filepath = os.path.join(UPLOAD_FOLDER, modified_filename)

            # Save original image
            file.save(original_filepath)
            print(f"Original image saved to: {original_filepath}")

            # Load image with OpenCV for processing
            original_img_array = cv2.imread(original_filepath)
            if original_img_array is None:
                return jsonify({'error': 'Could not read original image file (OpenCV failed to load)'}), 500

            # resize the original image to an approriate size for preprocessing
            fixed_width = 640
            h, w = original_img_array.shape[:2]
            aspect_ratio = h / w
            MIN_ASPECT_RATIO = 0.5
            MAX_ASPECT_RATIO = 2.0
            if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                return jsonify({'error': f"Image aspect ratio is too extreme. Please upload an image that it is more square-ish"}), 400 # Code 400 for invalid inputs error
            
            new_height = int(fixed_width * aspect_ratio)
            original_img_array = cv2.resize(original_img_array, (fixed_width, new_height))

            # Save the resized original image
            cv2.imwrite(original_filepath, original_img_array)


            # apply image modifications
            num_changes = 4

            modified_img_array, differences = apply_changes(original_img_array, num_changes)
            logging.info(f"Backend: Differences generated: {len(differences)}")

            logging.info(differences)


            if modified_img_array is None:
                print("Backend: Image manipulation returned None, using original image.")
                modified_img_array = original_img_array.copy()
                differences = [] # No changes if manipulation failed

            # Save modified image from a numpy array to file before uploading into cloudinary
            cv2.imwrite(modified_filepath, modified_img_array)
            
            # OPTION 1: Upload the images within Cloudinary
            cloudinary_original = cloud_upload.upload(original_filepath)
            cloudinary_modified = cloud_upload.upload(modified_filepath)

            # URL from cloudinary to access images
            original_url = cloudinary_original['secure_url']
            modified_url = cloudinary_modified['secure_url']

            # To save space, delete the images stored internally once uploads are complete
            try:
                os.remove(original_filepath)
                os.remove(modified_filepath)
            except Exception as e:
                print(f"Error removing locally stored files after cloudinary uploads: {e}")

            # return cloudinary filepaths in JSON format
            return jsonify({
                'originalImageUrl': original_url,
                'modifiedImageUrl':modified_url,
                'rawDifferencesForFrontendDemo': differences # Send the bounding box differences'
            }), 200


            # # OPTION 2: Save the modified image internally (failsafe)
            # cv2.imwrite(modified_filepath, modified_img_array)
            # print(f"Modified image saved to: {modified_filepath}")

            # # If loaded internally, return filepaths of both original and modifed images in JSON format 
            # return jsonify({
            #     'originalImageUrl': f'/{UPLOAD_FOLDER}/{original_filename}',
            #     'modifiedImageUrl': f'/{UPLOAD_FOLDER}/{modified_filename}',
            #     'rawDifferencesForFrontendDemo': differences # Send the bounding box differences
            # }), 200 # 200 is the status code for successful request

        except Exception as e:
            print(f"Server error during processing: {e}")

    else:
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400 # Code 400 for invalid inputs error

# Route to serve the uploaded/modified files (only needed if serving files from own internal server storage)
# @app.route(f'/{UPLOAD_FOLDER}/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)



# Created database tables are created at startup
with app.app_context(): 
    db.create_all()


if __name__ == '__main__':
    if not os.path.exists(objects_path):
        os.makedirs(objects_path)

    app.run(debug=True, port=5000)


