#!/usr/bin/env python3
"""
Startup script for Federated Learning Web Application
Handles initialization, dependency checking, and server startup.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'flask', 'torch', 'numpy', 'pandas', 'scikit-learn', 
        'matplotlib', 'flask_limiter', 'flask_cors', 'werkzeug'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
        else:
            print(f"✅ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("\n📦 Installing missing packages...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("   Please run: pip install -r requirements.txt")
            return False
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        'uploads',
        'checkpoints', 
        'hospital_models',
        'static/css',
        'static/js',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory: {directory}")

def check_files():
    """Check if required files exist."""
    required_files = [
        'app.py',
        'config.py',
        'models.py',
        'server.py',
        'hospital.py',
        'csv_data_loader.py',
        'templates/base.html',
        'static/css/style.css',
        'static/js/main.js'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def start_server():
    """Start the Flask development server."""
    print("\n" + "="*60)
    print("🚀 STARTING FEDERATED LEARNING WEB APPLICATION")
    print("="*60)
    print("📍 URL: http://localhost:5000")
    print("🔐 Security: Rate limiting enabled")
    print("🛡️  Privacy: Data stays local")
    print("="*60)
    print("\n🌐 Open your browser and navigate to: http://localhost:5000")
    print("📝 Register your hospital to get started")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("-"*60 + "\n")
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    print("🏥 Federated Learning Platform - Web Application Startup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\n📁 Creating directories...")
    create_directories()
    
    print("\n📄 Checking required files...")
    if not check_files():
        print("\n❌ Some required files are missing.")
        print("   Please ensure all files are in the correct location.")
        sys.exit(1)
    
    print("\n✅ All checks passed!")
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()
