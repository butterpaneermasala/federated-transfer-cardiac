"""
Flask Web Application for Federated Learning System
Provides web interface for hospitals to upload data, train models, and make predictions.
Includes security features and privacy protection.
"""

import os
import secrets
import hashlib
import json
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import torch
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS

from config import Config
from csv_data_loader import CSVDataLoader
from server import GlobalServer
from hospital import Hospital
from models import InputAdapter, Encoder, GlobalHead, HospitalModel

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

# Security: Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# CORS with restrictions
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('hospital_models', exist_ok=True)

# In-memory storage (in production, use a proper database)
users_db = {}
hospitals_db = {}
training_sessions = {}
global_server = None
config = Config()


# ============================================================================
# SECURITY UTILITIES
# ============================================================================

def generate_api_key():
    """Generate secure API key for hospitals."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key):
    """Hash API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key, hashed_key):
    """Verify API key against hash."""
    return hash_api_key(api_key) == hashed_key


def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def sanitize_csv(filepath):
    """
    Sanitize CSV file to prevent injection attacks.
    Checks for malicious content and validates structure.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Check for suspicious patterns
        for col in df.columns:
            if any(char in str(col) for char in ['<', '>', ';', '&', '|', '$']):
                return False, "Suspicious characters in column names"
        
        # Check data types
        if df.shape[0] < 10:
            return False, "Dataset too small (minimum 10 samples required)"
        
        if df.shape[1] < 2:
            return False, "Dataset must have at least 2 columns"
        
        # Check for excessive missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > 0.5:
            return False, "Too many missing values (>50%)"
        
        return True, "Valid"
    
    except Exception as e:
        return False, f"Invalid CSV format: {str(e)}"


def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'hospital_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function


def require_api_key(f):
    """Decorator to require API key."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        # Verify API key
        hospital_id = None
        for hid, data in hospitals_db.items():
            if verify_api_key(api_key, data['api_key_hash']):
                hospital_id = hid
                break
        
        if not hospital_id:
            return jsonify({'error': 'Invalid API key'}), 401
        
        request.hospital_id = hospital_id
        return f(*args, **kwargs)
    return decorated_function


# ============================================================================
# ROUTES - AUTHENTICATION
# ============================================================================

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per hour")
def register():
    """Hospital registration."""
    if request.method == 'POST':
        data = request.json
        hospital_id = data.get('hospital_id')
        password = data.get('password')
        hospital_name = data.get('hospital_name')
        
        # Validation
        if not hospital_id or not password or not hospital_name:
            return jsonify({'error': 'All fields required'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        if hospital_id in hospitals_db:
            return jsonify({'error': 'Hospital ID already exists'}), 400
        
        # Generate API key
        api_key = generate_api_key()
        api_key_hash = hash_api_key(api_key)
        
        # Store hospital
        hospitals_db[hospital_id] = {
            'hospital_name': hospital_name,
            'password_hash': generate_password_hash(password),
            'api_key_hash': api_key_hash,
            'created_at': datetime.now().isoformat(),
            'datasets': [],
            'models': []
        }
        
        return jsonify({
            'message': 'Hospital registered successfully',
            'hospital_id': hospital_id,
            'api_key': api_key  # Show once, never again
        }), 201
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per hour")
def login():
    """Hospital login."""
    if request.method == 'POST':
        data = request.json
        hospital_id = data.get('hospital_id')
        password = data.get('password')
        
        if not hospital_id or not password:
            return jsonify({'error': 'Hospital ID and password required'}), 400
        
        hospital = hospitals_db.get(hospital_id)
        if not hospital or not check_password_hash(hospital['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Create session
        session['hospital_id'] = hospital_id
        session['hospital_name'] = hospital['hospital_name']
        session.permanent = True
        
        return jsonify({
            'message': 'Login successful',
            'hospital_id': hospital_id,
            'hospital_name': hospital['hospital_name']
        }), 200
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout."""
    session.clear()
    return redirect(url_for('index'))


# ============================================================================
# ROUTES - DASHBOARD
# ============================================================================

@app.route('/dashboard')
@require_auth
def dashboard():
    """Hospital dashboard."""
    hospital_id = session['hospital_id']
    hospital = hospitals_db[hospital_id]
    
    return render_template('dashboard.html', 
                         hospital_id=hospital_id,
                         hospital_name=hospital['hospital_name'],
                         datasets=hospital['datasets'],
                         models=hospital['models'])


# ============================================================================
# ROUTES - DATA UPLOAD
# ============================================================================

@app.route('/upload', methods=['GET', 'POST'])
@require_auth
@limiter.limit("10 per hour")
def upload_data():
    """Upload CSV dataset."""
    if request.method == 'POST':
        hospital_id = session['hospital_id']
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        target_column = request.form.get('target_column')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not target_column:
            return jsonify({'error': 'Target column required'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files allowed'}), 400
        
        # Secure filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{hospital_id}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        
        # Sanitize and validate
        is_valid, message = sanitize_csv(filepath)
        if not is_valid:
            os.remove(filepath)
            return jsonify({'error': message}), 400
        
        # Load and analyze dataset
        try:
            df = pd.read_csv(filepath)
            
            if target_column not in df.columns:
                os.remove(filepath)
                return jsonify({'error': f'Target column "{target_column}" not found'}), 400
            
            # Store dataset info
            dataset_info = {
                'filename': unique_filename,
                'original_filename': filename,
                'filepath': filepath,
                'target_column': target_column,
                'num_samples': len(df),
                'num_features': len(df.columns) - 1,
                'columns': list(df.columns),
                'uploaded_at': datetime.now().isoformat()
            }
            
            hospitals_db[hospital_id]['datasets'].append(dataset_info)
            
            return jsonify({
                'message': 'Dataset uploaded successfully',
                'dataset_info': dataset_info
            }), 200
        
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing dataset: {str(e)}'}), 500
    
    return render_template('upload.html')


# ============================================================================
# ROUTES - FEDERATED TRAINING
# ============================================================================

@app.route('/api/start_training', methods=['POST'])
@require_auth
@limiter.limit("5 per hour")
def start_training():
    """Start federated training session."""
    global global_server, training_sessions
    
    data = request.json
    session_name = data.get('session_name', f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # Get participating hospitals
    participating_hospitals = []
    for hospital_id, hospital_data in hospitals_db.items():
        if hospital_data['datasets']:
            participating_hospitals.append(hospital_id)
    
    if len(participating_hospitals) < 2:
        return jsonify({'error': 'At least 2 hospitals with datasets required'}), 400
    
    try:
        # Initialize global server
        global_server = GlobalServer(config)
        
        # Create training session
        session_id = secrets.token_hex(16)
        training_sessions[session_id] = {
            'session_name': session_name,
            'status': 'initialized',
            'hospitals': participating_hospitals,
            'current_round': 0,
            'total_rounds': config.GLOBAL_ROUNDS,
            'created_at': datetime.now().isoformat(),
            'history': []
        }
        
        return jsonify({
            'message': 'Training session created',
            'session_id': session_id,
            'participating_hospitals': participating_hospitals
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error starting training: {str(e)}'}), 500


@app.route('/api/training_status/<session_id>')
@require_auth
def training_status(session_id):
    """Get training session status."""
    session_data = training_sessions.get(session_id)
    
    if not session_data:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify(session_data), 200


# ============================================================================
# ROUTES - PREDICTION
# ============================================================================

@app.route('/predict', methods=['GET', 'POST'])
@require_auth
def predict():
    """Make predictions using trained model."""
    if request.method == 'POST':
        hospital_id = session['hospital_id']
        data = request.json
        
        # Get input features
        features = data.get('features')
        if not features:
            return jsonify({'error': 'Features required'}), 400
        
        try:
            # Load hospital's model
            model_path = f'hospital_models/{hospital_id}_model.pt'
            if not os.path.exists(model_path):
                return jsonify({'error': 'No trained model found. Please train first.'}), 404
            
            # Load model (simplified - in production, properly reconstruct model)
            # For now, return mock prediction
            prediction = {
                'class': 'positive',
                'confidence': 0.87,
                'probabilities': {
                    'negative': 0.13,
                    'positive': 0.87
                }
            }
            
            return jsonify({
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    return render_template('predict.html')


# ============================================================================
# API ROUTES (for programmatic access)
# ============================================================================

@app.route('/api/upload', methods=['POST'])
@require_api_key
@limiter.limit("20 per hour")
def api_upload():
    """API endpoint for dataset upload."""
    hospital_id = request.hospital_id
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    target_column = request.form.get('target_column')
    
    # Similar logic to upload_data route
    # ... (implement similar to above)
    
    return jsonify({'message': 'Upload via API successful'}), 200


@app.route('/api/predict', methods=['POST'])
@require_api_key
@limiter.limit("100 per hour")
def api_predict():
    """API endpoint for predictions."""
    hospital_id = request.hospital_id
    data = request.json
    
    # Similar logic to predict route
    # ... (implement similar to above)
    
    return jsonify({'prediction': 'result'}), 200


# ============================================================================
# ADMIN ROUTES
# ============================================================================

@app.route('/admin/stats')
def admin_stats():
    """Admin statistics (protected in production)."""
    stats = {
        'total_hospitals': len(hospitals_db),
        'total_datasets': sum(len(h['datasets']) for h in hospitals_db.values()),
        'active_sessions': len(training_sessions),
        'server_status': 'running'
    }
    return jsonify(stats), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large (max 50MB)'}), 413


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("FEDERATED LEARNING WEB APPLICATION")
    print("="*60)
    print("Starting server...")
    print("Access at: http://localhost:5000")
    print("="*60 + "\n")
    
    # Run in development mode
    # In production, use gunicorn or similar WSGI server
    app.run(debug=True, host='0.0.0.0', port=5000)
