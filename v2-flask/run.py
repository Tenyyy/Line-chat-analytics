#!/usr/bin/env python3
"""
LINE Chat Analytics v2.0 - Flask Web Application
Easy startup script for the Flask application
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """Check if required packages are installed"""
    # Map package names to their import names
    required_packages = {
        'flask': 'flask',
        'pandas': 'pandas', 
        'plotly': 'plotly',
        'wordcloud': 'wordcloud',
        'pythainlp': 'pythainlp',
        'pillow': 'PIL'  # Pillow is imported as PIL
    }
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} - not installed")
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    print("\n🎉 All dependencies are installed!")
    return True

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting LINE Chat Analytics v2.0...")
    print("📱 Open http://localhost:8888 in your browser")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    try:
        # Import and run the Flask app
        from flask_app import app
        app.run(host='0.0.0.0', port=8888, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 50)
    print("🎯 LINE Chat Analytics v2.0 - Flask Edition")
    print("=" * 50)
    
    # Check system requirements
    check_python_version()
    
    # Check dependencies
    if not check_dependencies():
        print("\n🔧 Please install the required packages first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Start the application
    start_application()
