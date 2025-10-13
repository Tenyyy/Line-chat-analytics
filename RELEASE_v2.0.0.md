# 🚀 Release v2.0.0 - LINE Chat Analytics

## 📋 Release Summary

**Version 2.0.0** introduces a completely rewritten Flask-based web application with enhanced features, better stability, and full Thai language support. This major release maintains backward compatibility by preserving the original Streamlit version as v1.

## 📁 Project Structure

```
line-chat-analytics/
├── README.md                 # Main project documentation
├── CHANGELOG.md              # Version history and changes
├── .gitignore               # Git ignore rules
├── .github/workflows/       # CI/CD automation
├── assets/                  # Shared assets (images, fonts)
│
├── v1-streamlit/            # 📱 Original Streamlit Version
│   ├── README.md
│   ├── main.py              # Streamlit application
│   ├── requirements.txt     # Streamlit dependencies
│   ├── model.sav           # TensorFlow model
│   └── ...
│
├── v2-flask/               # 🆕 New Flask Version (Recommended)
│   ├── README.md
│   ├── flask_app.py        # Flask application
│   ├── run.py             # Easy startup script
│   ├── version.py         # Version information
│   ├── requirements.txt   # Flask dependencies
│   ├── templates/         # HTML templates
│   └── assets/           # Static files
│
└── Legacy files (will be cleaned up)
```

## 🎯 Quick Start Commands

### For v2.0 (Flask - Recommended)
```bash
# Clone the repository
git clone https://github.com/pannapann/line-chat-analytics.git
cd line-chat-analytics/v2-flask

# Install dependencies
pip install -r requirements.txt

# Start the application (Option 1: Direct)
python flask_app.py

# Start the application (Option 2: With checks)
python run.py

# Open browser
open http://localhost:8888
```

### For v1.0 (Streamlit - Legacy)
```bash
cd line-chat-analytics/v1-streamlit
pip install -r requirements.txt
python download_spacy_models.py  # If needed
streamlit run main.py
```

## 🆕 Major Changes in v2.0

### ✨ New Features
- **Modern Web Interface**: Bootstrap 5 + custom LINE-themed design
- **Full Thai Support**: pythainlp integration for proper text processing  
- **Interactive Charts**: 7+ chart types with Plotly.js
- **Advanced Analytics**: Response patterns, emoji analysis, conversation insights
- **Large File Support**: Handle chat files up to 100MB
- **Mobile Responsive**: Works perfectly on all device sizes
- **Smart Prediction**: Lightweight author prediction without heavy ML

### 🔧 Technical Improvements
- **Eliminated TensorFlow**: Reduced dependencies from 20+ to 6 core packages
- **Better Performance**: Optimized processing for large chat files
- **Improved Error Handling**: User-friendly error messages and validation
- **RESTful Design**: Clean API endpoints for each feature
- **Modular Architecture**: Organized code structure with LineAnalyzer class

### 🌏 Language Support
- **Thai Text Processing**: Native support with proper tokenization
- **Mixed Content**: Handles Thai-English mixed messages
- **Thai Fonts**: Beautiful Thai typography in word clouds
- **Stopwords**: Thai language stopwords filtering

## 📊 Feature Comparison

| Feature | v1.0 Streamlit | v2.0 Flask |
|---------|----------------|------------|
| **Setup Difficulty** | Hard (TensorFlow) | Easy (6 packages) |
| **File Size Limit** | ~16MB | 100MB |
| **Thai Language** | Basic | Full Support |
| **Mobile Friendly** | No | Yes |
| **Chart Types** | 3-4 basic | 7+ interactive |
| **Dependencies** | 20+ packages | 6 core packages |
| **Startup Time** | Slow | Fast |
| **Memory Usage** | High (TensorFlow) | Low |
| **Maintenance** | Complex | Simple |

## 🚀 Deployment Options

### Local Development
```bash
cd v2-flask
python flask_app.py  # Development server
```

### Production Deployment
```bash
# Using Gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8888 flask_app:app

# Using Docker (create Dockerfile)
# Using cloud platforms (Heroku, Railway, etc.)
```

## 🧪 Testing

The project includes automated testing with GitHub Actions:
- **Python Compatibility**: Tests on Python 3.8-3.11
- **Flask Server**: Validates server startup and responses
- **Thai Language**: Checks pythainlp integration
- **Import Tests**: Ensures all dependencies work correctly

## 📝 Migration Guide

### From v1 to v2
1. **Backup your data**: Export any important analysis results
2. **Switch directories**: Use `v2-flask/` instead of root
3. **Install new deps**: `pip install -r v2-flask/requirements.txt`
4. **Update bookmarks**: Change port from 8501 to 8888
5. **Enjoy new features**: Explore the enhanced analytics!

### Staying with v1
- All v1 files are preserved in `v1-streamlit/`
- Original functionality remains unchanged
- Use for TensorFlow-based ML features if needed

## 🎉 Release Checklist

- [x] ✅ Flask application fully functional
- [x] ✅ Thai language support implemented
- [x] ✅ All 7 chart types working
- [x] ✅ Advanced analytics features added
- [x] ✅ Mobile responsive design completed
- [x] ✅ Error handling improved
- [x] ✅ Large file support (100MB) tested
- [x] ✅ Documentation updated
- [x] ✅ Version directories organized
- [x] ✅ CI/CD workflows added
- [x] ✅ Migration guide created

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Features**: Request new features via GitHub Discussions
- **Documentation**: Check README files in each version directory
- **Community**: Share your analytics results!

---

**🎊 Ready for release! Users can now enjoy the enhanced v2.0 experience while keeping access to the original v1.0 functionality.**
