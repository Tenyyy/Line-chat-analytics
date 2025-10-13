# Changelog

All notable changes to LINE Chat Analytics will be documented in this file.

## [Version 2.0.0] - 2025-10-13

### 🎉 Major Release - Flask Web Application

#### ✨ Added
- **Complete Flask-based web application** replacing Streamlit
- **Modern responsive UI** with LINE-themed design
- **Full Thai language support** with pythainlp integration
- **7 interactive chart types** using Plotly.js:
  - Messages over time (line chart)
  - Daily message distribution (bar chart) 
  - Hourly activity patterns (heatmap-style bar chart)
  - User message counts (bar chart)
  - User activity percentage (pie chart)
  - Message length distribution (histogram)
  - Response time patterns (box plot)
- **Advanced statistics**:
  - Message length analysis (average, longest, by user)
  - Response pattern analysis (average response time, quick responses)
  - Conversation starter detection
  - Emoji usage statistics by user
- **Intelligent text prediction** without heavy ML dependencies
- **Large file support** up to 100MB chat exports
- **Drag & drop file upload** with progress indicators
- **Mobile-friendly responsive design**
- **Better error handling** with user-friendly messages

#### 🌏 Internationalization
- **Native Thai text processing** with pythainlp word_tokenize
- **Thai font support** in word clouds (THSarabunNew.ttf)
- **Mixed language handling** for Thai-English content
- **Thai stopwords** filtering for better analysis

#### 🎨 UI/UX Improvements
- **LINE app-inspired color scheme** (#00d084, #06c755)
- **Bootstrap 5** with custom styling and animations
- **Font Awesome icons** throughout the interface
- **Gradient backgrounds** and smooth transitions
- **Card-based layouts** for better content organization
- **Interactive elements** with hover effects and loading states

#### 🔧 Technical Improvements
- **Eliminated TensorFlow dependency** for better compatibility
- **Lighter requirements** with core packages only
- **Better memory management** for large file processing
- **Improved error handling** with detailed logging
- **Modular code structure** with LineAnalyzer class
- **RESTful API endpoints** for different analysis components

#### 📊 Analytics Enhancements
- **Word frequency analysis** with user-specific patterns
- **Emoji detection and counting** with Unicode support
- **Time-based analysis** with hourly and daily patterns
- **User behavior insights** including response times
- **Conversation flow analysis** with starter detection
- **Statistical summaries** with formatted number displays

### 🛠️ Changed
- **Architecture**: Migrated from Streamlit to Flask
- **Dependencies**: Reduced from 20+ to 6 core packages
- **File size limit**: Increased from ~16MB to 100MB
- **Processing speed**: Optimized for better performance
- **Code organization**: Restructured into version directories

### 🐛 Fixed
- **Protobuf version conflicts** that caused startup errors
- **Thai text encoding issues** in word cloud generation
- **File upload size limitations** with proper error messages
- **Memory usage** with large chat files
- **Cross-platform compatibility** issues

---

## [Version 1.0.0] - Original Release

### Features
- **Streamlit-based web interface** for chat analysis
- **TensorFlow text prediction model** with Keras layers
- **Basic chat statistics** and visualizations
- **Word cloud generation** with limited Thai support
- **Message frequency analysis** and user patterns
- **spaCy integration** for text processing

### Technical Stack
- Streamlit 1.12.0 for web interface
- TensorFlow 2.x for machine learning
- pandas for data processing  
- matplotlib/plotly for basic visualizations
- wordcloud for text visualization

### Limitations
- Heavy dependencies with TensorFlow
- Limited file size support
- Protobuf version conflicts
- Basic Thai language handling
- Non-responsive design
- Complex setup requirements

---

## Migration Guide: v1 → v2

### For Users
1. **Use v2-flask/** directory instead of root
2. **Install lighter requirements**: `pip install -r requirements.txt` 
3. **Run Flask app**: `python flask_app.py`
4. **Access on port 8888**: http://localhost:8888 (not 8501)

### For Developers  
1. **Architecture change**: Streamlit → Flask + Bootstrap
2. **ML approach**: TensorFlow → frequency-based prediction
3. **Thai processing**: Basic → pythainlp integration
4. **File structure**: Monolithic → modular with classes
5. **API design**: Streamlit widgets → RESTful endpoints

### Breaking Changes
- **Port change**: 8501 → 8888
- **URL structure**: Single page → multiple endpoints  
- **Dependencies**: TensorFlow removed, pythainlp added
- **File location**: Root directory → v2-flask/
- **Configuration**: Streamlit config → Flask config

---

## Compatibility

### Version 2 (Flask)
- **Python**: 3.8+ (recommended 3.9+)
- **OS**: Windows, macOS, Linux
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Memory**: 1GB+ recommended for large files
- **Disk Space**: 500MB+ for dependencies

### Version 1 (Streamlit)  
- **Python**: 3.8-3.10 (due to TensorFlow constraints)
- **OS**: Windows, macOS, Linux (with compatible TensorFlow build)
- **Memory**: 2GB+ required for TensorFlow
- **Disk Space**: 2GB+ for all dependencies
