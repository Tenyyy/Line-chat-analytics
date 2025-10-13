# LINE Chat Analytics 💬📊

<p align="center">
  <img src="assets/Jun-19-2565 22-38-26.gif" alt="Example usage" width="400"/>
  <img src="assets/Jun-19-2565 22-37-26.gif" alt="Example usage" width="400"/>
</p>

<p align="center">
  <strong>Analyze your LINE chat conversations with beautiful visualizations and insights</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-2.3+-green?style=flat-square&logo=flask">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.12+-red?style=flat-square&logo=streamlit">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square">
</p>

---

## 🚀 Choose Your Version

This project offers **two versions** to suit different needs:

### 🆕 **Version 2 - Flask Web App** (Recommended)
- ✨ **Modern responsive web interface**
- 🌏 **Full Thai language support** 
- 📊 **7+ interactive visualizations**
- 🎯 **Advanced analytics** (response patterns, emoji usage, text prediction)
- 🚀 **Better stability** and performance
- 📱 **Mobile-friendly design**
- 💾 **Large file support** (up to 100MB)

**[➡️ Go to v2-flask/](./v2-flask/)**

### 📱 **Version 1 - Streamlit** (Original)
- 🤖 **TensorFlow-based ML predictions**
- 📈 **Classic Streamlit interface** 
- 🎯 **Core analytics features**
- 📊 **Basic visualizations**

**[➡️ Go to v1-streamlit/](./v1-streamlit/)**

---

## ⚡ Quick Start (v2 - Recommended)

```bash
# Clone the repository
git clone https://github.com/pannapann/line-chat-analytics.git

# Navigate to Flask version
cd line-chat-analytics/v2-flask

# Install dependencies  
pip install -r requirements.txt

# Run the application
python flask_app.py

# Open http://localhost:8888 in your browser
```

## 📱 How to Export LINE Chat Data

1. **Open LINE app** on your mobile device
2. **Select the chat** you want to analyze
3. **Tap settings** (≡ menu → Settings)  
4. **Choose "Export chat history"**
5. **Select "Without media"** for faster processing
6. **Save the .txt file** and upload to the app

## ✨ Features Overview

| Feature | v1 (Streamlit) | v2 (Flask) |
|---------|----------------|------------|
| 📊 **Basic Analytics** | ✅ | ✅ Enhanced |
| 🎨 **Interactive Charts** | ✅ Basic | ✅ 7+ Chart Types |
| 🌏 **Thai Language** | ⚠️ Limited | ✅ Full Support |
| 📱 **Mobile Friendly** | ❌ | ✅ Responsive |
| 🤖 **Text Prediction** | ✅ TensorFlow | ✅ Lightweight |
| 💾 **Large Files** | ❌ Limited | ✅ Up to 100MB |
| 📈 **Advanced Stats** | ❌ | ✅ Response patterns, emoji analysis |
| 🎯 **Easy Setup** | ⚠️ Dependencies | ✅ Simple |

## 📊 Analytics Capabilities

- **📈 Message Volume**: Daily/hourly patterns, trends over time
- **👥 User Activity**: Individual statistics, most active users
- **💬 Response Patterns**: Average response times, conversation starters
- **📝 Text Analysis**: Word frequency, message length analysis  
- **😊 Emoji Usage**: Most used emojis by user
- **🔮 Text Prediction**: Predict message author from text content
- **🌈 Word Clouds**: Beautiful visualizations (Thai + English fonts)
- **📊 Interactive Charts**: Zoom, filter, and explore your data

## 🛠️ Technical Details

### Version 2 (Flask) - Recommended
- **Backend**: Flask 2.3+, pandas, plotly
- **UI**: Bootstrap 5, Font Awesome, custom CSS
- **Thai Support**: pythainlp for proper text processing
- **Charts**: Plotly.js for interactive visualizations
- **File Handling**: Up to 100MB chat exports

### Version 1 (Streamlit) - Original  
- **Backend**: Streamlit 1.12, TensorFlow 2.x
- **ML**: TensorFlow/Keras for text prediction
- **Limitations**: Dependency conflicts, smaller file limits

## 🤝 Contributing

Found a bug or want to add features? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes  
4. Submit a pull request

## 📄 License

This project is open source under the MIT License.

## 🆕 What's New in v2?

- 🎨 **Complete UI redesign** with LINE-themed colors
- 🌏 **Native Thai language support** with proper tokenization
- 📊 **7 new chart types** including user comparisons and time patterns
- 🔮 **Improved text prediction** without heavy ML dependencies
- 📱 **Mobile-responsive design** that works on all devices
- ⚡ **Better performance** with optimized data processing
- 📈 **Advanced analytics** including response patterns and emoji usage
- 🛠️ **Easier deployment** with simplified dependencies

---

<p align="center">
  <strong>Start exploring your LINE chat data today! 🚀</strong>
</p>
