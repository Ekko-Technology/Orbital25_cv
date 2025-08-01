import sys
from dotenv import load_dotenv
load_dotenv()

# flask imports
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session

import os
import cv2
import numpy as np
import random

import urllib.parse, requests, tempfile
from datetime import datetime, timezone, timedelta
# for securing user password
from werkzeug.security import generate_password_hash, check_password_hash 

# import libraries to initialize cloudinary for image storage
import cloudinary
import cloudinary.uploader as cloud_upload
import cloudinary.api as cloud_api
from image_processing import apply_changes
from functools import wraps

import redis

app = Flask(__name__)
CORS(app, supports_credentials=True)  
# initialize secret key for session management and CSRF protection
app.secret_key = os.getenv("FLASK_SECRET_KEY")
# Allows Flask as a backend to be accessed from React which is ran on another domain


IS_LOCAL_DEV_FLAG = os.getenv('IS_LOCAL_DEV', 'False').lower() == 'true'

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True

if os.getenv("REDIS_URL"):
    app.config["SESSION_TYPE"] = "redis"
    app.config["SESSION_REDIS"] = redis.from_url(os.getenv("REDIS_URL"))
    print("SESSION_TYPE set to: redis (using REDIS_URL)")
else:
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = os.path.join(os.getcwd(), 'flask_session_data')
    if not os.path.exists(app.config["SESSION_FILE_DIR"]):
        os.makedirs(app.config["SESSION_FILE_DIR"])
    print("SESSION_TYPE set to: filesystem (local fallback)")

app.config['SESSION_COOKIE_SECURE'] = os.getenv('FLASK_SESSION_SECURE_COOKIE', 'True').lower() == 'true'
app.config['SESSION_COOKIE_SAMESITE'] = "None"

app.config['SESSION_COOKIE_DOMAIN'] = os.getenv('SESSION_COOKIE_DOMAIN', None) 

print(f"SESSION_COOKIE_DOMAIN set to: {app.config['SESSION_COOKIE_DOMAIN']}")
print(f"SESSION_COOKIE_SECURE set to: {app.config['SESSION_COOKIE_SECURE']} (from FLASK_SESSION_SECURE_COOKIE env var)")
print(f"IS_LOCAL_DEV_FLAG (for reference): {IS_LOCAL_DEV_FLAG}")

# Initialize Flask-Session
Session(app)

# Allows Flask as a backend to be accessed from React which is ran on another domain
allowed_origins_env = os.environ.get(
    'CORS_ORIGINS',
    'http://localhost:5173,https://localhost:5173'
)
ALLOWED_ORIGINS_LIST = [o.strip() for o in allowed_origins_env.split(',') if o.strip()]



CORS(app, supports_credentials=True, origins=["http://localhost:5173"]) 

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



# ----- User Logins ------

# Helper function to check if user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized: Please log in."}), 401
        return f(*args, **kwargs)
    return decorated_function

def extract_public_id_from_url(url):
    if not url:
        return None
    try:
        parsed_url = urllib.parse.urlparse(url)
        path_segments = parsed_url.path.split('/')

        if 'upload' in path_segments:
            upload_index = path_segments.index('upload')

            if upload_index + 1 < len(path_segments) and path_segments[upload_index + 1].startswith('v'):
                public_id_parts = path_segments[upload_index + 2:]
            else:
                public_id_parts = path_segments[upload_index + 1:]

            full_public_id = '/'.join(public_id_parts)
            if '.' in full_public_id:
                return full_public_id.rsplit('.', 1)[0]
            return full_public_id
    except Exception as e:
        print(f"Error extracting public ID from URL {url}: {e}")
        return None
    return None

def delete_cloudinary_assets_and_folders(public_ids_list, user_identifier="unknown"):
    deleted_assets_count = 0
    deleted_folders_count = 0
    errors = []
    folder_paths_to_delete = set()

    for public_id in public_ids_list:
        try:
            response = cloud_upload.destroy(public_id)
            if response['result'] == 'ok':
                deleted_assets_count += 1
                print(f"Deleted Cloudinary asset for {user_identifier}: {public_id}")
                folder = extract_folder_from_public_id(public_id)
                if folder:
                    folder_paths_to_delete.add(folder)
            else:
                print(f"Failed to delete Cloudinary asset {public_id} for {user_identifier}: {response}")
                errors.append(f"Failed to delete {public_id}: {response.get('error', {}).get('message', 'Unknown error')}")
        except Exception as e:
            print(f"Error deleting Cloudinary asset {public_id} for {user_identifier}: {e}")
            errors.append(f"Error deleting {public_id}: {str(e)}")

    for folder_path in folder_paths_to_delete:
        try:
            folder_response = cloud_api.delete_folder(folder_path)
            print(f"DEBUG: Cloudinary delete_folder response for {folder_path}: {folder_response}")
            if 'deleted' in folder_response and folder_path in folder_response['deleted']:
                deleted_folders_count += 1
                print(f"Deleted Cloudinary folder for {user_identifier}: {folder_path}")
            elif 'error' in folder_response:
                print(f"Failed to delete Cloudinary folder {folder_path} for {user_identifier}: {folder_response.get('error', 'Unknown error')}")
                errors.append(f"Failed to delete folder {folder_path}: {folder_response.get('error', 'Unknown error')}")
            else:
                print(f"Unexpected Cloudinary delete_folder response for {folder_path}: {folder_response}")
                errors.append(f"Unexpected response for folder {folder_path}: {folder_response}")
        except Exception as e:
            print(f"Error deleting Cloudinary folder {folder_path} for {user_identifier}: {e}")
            errors.append(f"Error deleting folder {folder_path}: {str(e)}")
    
    return deleted_assets_count, deleted_folders_count, errors



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
        # set session
        session["user_id"] = user.id
        session.modified = True
        return jsonify({'message': 'Login successful', 
                        'user_id': user.id,
                        'username': user.username,
                        'games_played': user.games_played,
                        'games_won': user.games_won,
                        'total_differences_found': user.total_differences_found
                        }), 200
    return jsonify({'error': 'Invalid username or password'}), 401


# logging out of session
@app.route('/logout', methods=['POST'])
def logout():
    session.pop("user_id", None)
    session.pop("current_game_temp_files", None)
    session.modified = True
    return jsonify({"Message": "Logged Out"}), 200

# rendering user stats
@app.route('/user_stats', methods=['GET'])
@login_required
def user_stats():
    user_id = session.get("user_id")
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        'user_id': user.id,
        'username': user.username,
        'games_played': user.games_played,
        'games_won': user.games_won,
        'total_differences_found': user.total_differences_found
    }), 200

# update user's stats after a game
@app.route('/update_stats', methods=['POST'])
@login_required
def update_stats():
    user_id = session.get("user_id")
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.json
    differences_found = data.get('differencesFound', 0)
    game_won = data.get('gameWon', False)

    try:
        user.games_played += 1
        user.total_differences_found += differences_found
        if game_won:
            user.games_won += 1
        
        db.session.commit()
        return jsonify({'message': 'Stats updated successfully!'}), 200
    except Exception as e:
        db.session.rollback()
        print(f"Error updating stats for user {user_id}: {e}")
        return jsonify({'error': 'Failed to update stats.'}), 500


# Backend for saving Game Records after game is over
@app.route('/save-game', methods=['POST'])
@login_required
def save_game():
    data = request.json
    user_id = session.get('user_id')

    print(f"[/save-game] User ID from session: {user_id}")
    print(f"[/save-game] Full session: {session}") # NEW: Print entire session
    print(f"[/save-game] Raw request data: {request.json}") # NEW: Print raw request data

    original_image_cloudinary_url = data.get('original_image_cloudinary_url')
    modified_image_cloudinary_url = data.get('modified_image_cloudinary_url')
    score = data.get('score')
    total = data.get('total')
    time_taken = data.get('time_taken')

    print(f"[/save-game] User ID from session: {user_id}")
    print(f"[/save-game] Received original_image_local_path: {'original_image_cloudinary_url'}")
    print(f"[/save-game] Received modified_image_local_path: {'modified_image_cloudinary_url'}")
    print(f"[/save-game] Received score: {score}")
    print(f"[/save-game] Received total: {total}")
    print(f"[/save-game] Received time_taken: {time_taken}")

    missing_fields = []
    if not original_image_cloudinary_url:
        missing_fields.append('original_image_cloudinary_url')
    if not modified_image_cloudinary_url:
        missing_fields.append('modified_image_cloudinary_url')
    if score is None:
        missing_fields.append('score')
    if total is None:
        missing_fields.append('total')
    if time_taken is None:
        missing_fields.append('time_taken')

    if missing_fields:
        print(f"[/save-game] Missing fields detected: {', '.join(missing_fields)}")
        print(f"[/save-game] Received data: {data}")
        return jsonify({'error': f"Missing fields: {', '.join(missing_fields)}"}), 400

    user = User.query.get(user_id)
    if not user:
        print(f"[/save-game] Error: User not found for ID: {user_id}")
        return jsonify({'error': 'User not logged in'}), 404
    
    session_temp_files = session.get("current_game_temp_files")

    # Extract public IDs from the incoming Cloudinary URLs
    request_original_public_id = extract_public_id_from_url(original_image_cloudinary_url)
    request_modified_public_id = extract_public_id_from_url(modified_image_cloudinary_url)

    print(f"[/save-game] DEBUG: session_temp_files: {session_temp_files} (Type: {type(session_temp_files)})")
    print(f"[/save-game] DEBUG: Request original_image_cloudinary_url: {original_image_cloudinary_url} (Type: {type(original_image_cloudinary_url)})")
    print(f"[/save-game] DEBUG: Request modified_image_cloudinary_url: {modified_image_cloudinary_url} (Type: {type(modified_image_cloudinary_url)})")
    print(f"[/save-game] DEBUG: Extracted request original_public_id: {request_original_public_id}")
    print(f"[/save-game] DEBUG: Extracted request modified_public_id: {request_modified_public_id}")

    session_original_public_id = session_temp_files.get('original_public_id') if session_temp_files else None
    session_modified_public_id = session_temp_files.get('modified_public_id') if session_temp_files else None

    # Check for session mismatch with Cloudinary public IDs
    if not session_temp_files:
        print(f"SECURITY ALERT: session_temp_files is None or empty for user {user_id}. Cannot verify image origin.")
        return jsonify({'error': 'Invalid image URLs provided. Session data missing.'}), 400
    elif session_original_public_id != request_original_public_id or \
       session_modified_public_id != request_modified_public_id:
        print(f"SECURITY ALERT: Mismatch in Cloudinary Public IDs for user {user_id}.")
        print(f"Session data: Original Public ID={session_original_public_id}, Modified Public ID={session_modified_public_id}")
        print(f"Request data: Original Public ID={request_original_public_id}, Modified Public ID={request_modified_public_id}")
        return jsonify({'error': 'Invalid image URLs provided. Session mismatch.'}), 400
    
    try:
        # No need to upload again, URLs are already on Cloudinary
        original_url = original_image_cloudinary_url
        modified_url = modified_image_cloudinary_url
        
        print(f"[/save-game] Cloudinary original URL (from request): {original_url}")
        print(f"[/save-game] Cloudinary modified URL (from request): {modified_url}")


        session.pop("current_game_temp_files", None)
        session.modified = True

        current_utc_time = datetime.now(timezone.utc)
        print(f"DEBUG: Setting played_at for GameRecord to: {current_utc_time}")

        game = GameRecord(
            user_id=user_id,
            original_image_path=original_url,
            modified_image_path=modified_url,
            score=score,
            total_differences=total,
            time_taken=time_taken,
            played_at=current_utc_time
        )
        db.session.add(game)
        db.session.commit()

        return jsonify({'message': 'Game and Images saved to Game History!', 'record_id': game.id}), 201

    except Exception as e:
        db.session.rollback()
        print(f"[SAVE-GAME ERROR] {e}")
        return jsonify({'error': 'Failed to save game or upload images'}), 500



@app.route('/user/<int:user_id>/history')
@login_required
def game_history(user_id):
    if user_id != session.get("user_id"):
        return jsonify({"error": "Unauthorized: Cannot view another user's history."}), 403
    
    user = User.query.get_or_404(user_id)
     # Order by played_at in descending order (latest first)
    games = GameRecord.query.filter_by(user_id=user.id).order_by(GameRecord.played_at.desc()).all() # NEW: Ordering
    games_data = [{ # Renamed to avoid conflict with 'games' variable
        'original_image': g.original_image_path,
        'modified_image': g.modified_image_path,
        'score': g.score,
        'total': g.total_differences,
        'time_taken': g.time_taken,
        'played_at': g.played_at.isoformat() # Convert datetime to ISO format string
    } for g in games]

    return jsonify({'username': user.username, 'games': games_data}), 200


@app.route('/test-session', methods=['GET'])
def test_session():
    user_id = session.get("user_id")
    print(f"[/test-session] User ID from session: {user_id}")

    if user_id:
        return jsonify({"message": "Session active", "user_id": user_id}), 200
    else:
        return jsonify({"message": "Session not active", "user_id": None}), 200


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
    # Check sessions to see if user is logged in
    user_id = session.get("user_id")
    print(f"[/upload-and-process] User ID from session: {user_id}")

    if user_id:
        previous_temp_files = session.pop("current_game_temp_files", None)
        if previous_temp_files:
            public_ids_to_delete = [
                previous_temp_files.get('original_public_id'), 
                previous_temp_files.get('modified_public_id')
            ]
            public_ids_to_delete = [pid for pid in public_ids_to_delete if pid]

            if public_ids_to_delete:
                print(f"Proactively cleaning up previous temp files for user {user_id} before new upload.")
                deleted_assets, deleted_folders, errors = delete_cloudinary_assets_and_folders(public_ids_to_delete, f"user {user_id} (proactive)")
                if errors:
                    print(f"Proactive cleanup errors for user {user_id}: {errors}")
            session.modified = True # Ensure session change is saved
    else:
        previous_guest_public_ids = session.pop("guest_cloudinary_public_ids", [])
        if previous_guest_public_ids:
            print(f"Proactively cleaning up previous guest temp files before new upload.")
            deleted_assets, deleted_folders, errors = delete_cloudinary_assets_and_folders(previous_guest_public_ids, "guest (proactive)")
            if errors:
                print(f"Proactive cleanup errors for guest: {errors}")
        session.modified = True # Ensure session change is saved

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:

            filestr = file.read()
            np_img = np.frombuffer(filestr, np.uint8)
            original_img_array = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if original_img_array is None:
                return jsonify({'error': 'Could not read original image file (OpenCV failed to load from memory)'}), 500
            


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

            # apply image modifications
            num_changes = 4

            modified_img_array, differences = apply_changes(original_img_array, num_changes)
            print(f"Backend: Differences generated: {len(differences)}")
            print(differences)


            if modified_img_array is None:
                print("Backend: Image manipulation returned None, using original image.")
                modified_img_array = original_img_array.copy()
                differences = [] # No changes if manipulation failed


            # Upload images to Cloudinary directly from memory
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

            temp_folder_name = f"temp_spotthedifference/{user_id or 'guest'}_{timestamp}"

            # Encode images to bytes for Cloudinary upload
            _, original_buffer = cv2.imencode('.png', original_img_array)
            original_bytes = original_buffer.tobytes()

            _, modified_buffer = cv2.imencode('.png', modified_img_array)
            modified_bytes = modified_buffer.tobytes()

            print(f"Uploading original image to Cloudinary folder: {temp_folder_name}")
            cloudinary_original_response = cloud_upload.upload(original_bytes, folder=temp_folder_name, resource_type="image")
            print(f"Uploading modified image to Cloudinary folder: {temp_folder_name}")
            cloudinary_modified_response = cloud_upload.upload(modified_bytes, folder=temp_folder_name, resource_type="image")

            original_url = cloudinary_original_response['secure_url']
            modified_url = cloudinary_modified_response['secure_url']

            # Store Cloudinary public IDs in session for logged-in users for later deletion
            if user_id:
                session["current_game_temp_files"] = {
                    "original_public_id": cloudinary_original_response['public_id'],
                    "modified_public_id": cloudinary_modified_response['public_id']
                }
                session.modified = True 
                print(f"Stored Cloudinary public IDs in session for user {user_id}: {session['current_game_temp_files']}")
                print(f"Session explicitly marked as modified: {session.modified}") 
            else:
                session.setdefault("guest_cloudinary_public_ids", []).extend([
                    cloudinary_original_response['public_id'], 
                    cloudinary_modified_response['public_id']
                ])
                session.modified = True
                print(f"Stored guest Cloudinary public IDs: {session['guest_cloudinary_public_ids']}")

            print(f"Backend returning originalImageUrl: {original_url}")
            print(f"Backend returning modifiedImageUrl: {modified_url}")

            return jsonify({
                'originalImageUrl': original_url,
                'modifiedImageUrl': modified_url,
                'original_image_cloudinary_url': original_url,
                'modified_image_cloudinary_url': modified_url,
                'original_public_id': cloudinary_original_response['public_id'],
                'modified_public_id': cloudinary_modified_response['public_id'],
                'rawDifferencesForFrontendDemo': differences
            }), 200            

        except Exception as e:
            print(f"Server error during processing: {e}")
            return jsonify({'error': 'Server error during processing'}), 500


    return jsonify({'error': 'Invalid file type'}), 400


# Route to serve the uploaded/modified files (~only needed if serving files from own internal server storage~ edit:now deprecated as images are served directly from Cloudinary)
@app.route(f'/{UPLOAD_FOLDER}/<filename>')
def uploaded_file(filename):
    # return send_from_directory(UPLOAD_FOLDER, filename)
    return jsonify({'error': 'Local file serving deprecated. Images served from Cloudinary.'}), 404

def extract_folder_from_public_id(public_id):
    if not public_id:
        return None
    if '/' in public_id:
        return public_id.rsplit('/', 1)[0]
    return None

# Route for deleting guest images from Cloudinary
@app.route("/cleanup-guest-files", methods=["POST"])
def cleanup_guest_files():

    # Public IDs can come from session (if not explicitly sent by frontend) or request body
    public_ids_from_request = request.json.get('public_ids', [])
    public_ids_from_session = session.pop("guest_cloudinary_public_ids", [])

    public_ids_to_delete = list(set(public_ids_from_request + public_ids_from_session))

    if public_ids_to_delete:
        deleted_assets, deleted_folders, errors = delete_cloudinary_assets_and_folders(public_ids_to_delete, "guest (explicit)")
        if errors:
            return jsonify({"message": f"Cleaned up {deleted_assets} guest Cloudinary assets and {deleted_folders} folders with errors: {errors}"}), 200
        return jsonify({"message": f"Successfully cleaned up {deleted_assets} guest Cloudinary assets and {deleted_folders} folders."}), 200
    return jsonify({"message": "No guest images to clean up."}), 200


@app.route("/delete-user-temp-images", methods=["POST"])
@login_required
def delete_user_temp_images():
    user_id = session.get('user_id')
    print(f"[/delete-user-temp-images] User ID from session: {user_id}")

     # Public IDs can come from session (if not explicitly sent by frontend) or request body
    public_ids_from_request = request.json.get('public_ids', [])
    previous_temp_files_from_session = session.pop("current_game_temp_files", None)

    public_ids_from_session = []
    if previous_temp_files_from_session:
        public_ids_from_session.append(previous_temp_files_from_session.get('original_public_id'))
        public_ids_from_session.append(previous_temp_files_from_session.get('modified_public_id'))
    public_ids_from_session = [pid for pid in public_ids_from_session if pid] # Filter out None

    public_ids_to_delete = list(set(public_ids_from_request + public_ids_from_session))

    if not public_ids_to_delete:
        print(f"[/delete-user-temp-images] No public IDs provided for user {user_id}.")
        return jsonify({"message": "No images to delete."}), 200
    
    deleted_assets, deleted_folders, errors = delete_cloudinary_assets_and_folders(public_ids_to_delete, f"user {user_id} (explicit)")

    session.pop("current_game_temp_files", None)
    session.modified = True # Ensure session change is saved

    if errors:
        return jsonify({"message": f"Deleted {deleted_assets} images and {deleted_folders} folders with errors: {errors}"}), 200 # Still 200 if some deleted
    return jsonify({"message": f"Successfully deleted {deleted_assets} temporary Cloudinary assets and {deleted_folders} folders."}), 200



# Prompts Pollinate.ai to generate an image for us
@app.route("/ai-generate-image", methods=["POST"])
def generate_ai_image():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"Error": "No Prompt Given"}), 400
    # Prompt AI to only give cartoon images
    prompt = f"cartoon style {prompt}"
    encoded_prompt = urllib.parse.quote(prompt)

    pollinations_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    pollination_request = requests.get(
        pollinations_url,
        params={
            "width": 640,
            "height": 640,
            # "model": "gptimage"
        },
        timeout=300
    )

    if pollination_request.status_code != 200:
        print(f"Pollinations error {pollination_request.status_code}: {pollination_request.text[:200]}")
        return jsonify({"error": "Pollination AI generation failed"}), 502
    

    return jsonify({"imageUrl": pollinations_url}), 200



# Created database tables are created at startup
with app.app_context(): 
    db.create_all()

if __name__ == '__main__':
    if not os.path.exists(objects_path):
        os.makedirs(objects_path)

    app.run(debug=True, port=5000)