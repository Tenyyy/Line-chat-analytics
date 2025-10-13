# LINE Chat Analytics - Version 2.1 (Flask Web App)

This is the enhanced Flask-based web version of the LINE Chat Analytics application with advanced relationship analysis and comprehensive behavioral insights.

## ✨ Features

### 📊 Comprehensive Analytics
- **Chat Overview**: Total messages, participants, date range, most active user
- **Interactive Visualizations**: 10+ different chart types with Plotly
- **Cumulative View Toggle**: Switch between daily and cumulative message counts
- **Message Patterns**: Hourly and daily activity analysis with heatmaps
- **User Statistics**: Individual user metrics and comparisons

### 🤝 Relationship Dynamics (NEW in v2.1)
- **Balance Score (0-100)**: Measure conversation equilibrium between users
- **Connection Strength**: Track relationship quality (Strong 💪, Good 👍, Moderate 🤝)
- **Mutual Responsiveness**: See who responds to whom most
- **Conversation Initiation**: Identify who starts conversations

### 💬 Conversation Flow Analysis (NEW in v2.1)
- **Message Streaks**: Track longest consecutive message sequences
- **Conversation Length**: Average messages per conversation session
- **Peak Activity Hours**: Identify busiest chat times
- **Quiet Hours**: Detect low-activity periods

### ⏰ Temporal Pattern Detection (NEW in v2.1)
- **Weekend vs Weekday**: Analyze messaging preferences
- **Monthly Trends**: Track growth/decline patterns (📈📉📊)
- **Seasonal Analysis**: Discover seasonal chat preferences
- **Activity Breakdown**: Complete temporal insights

### 🏆 Engagement Scoring System (NEW in v2.1)
- **0-100 Engagement Score**: Multi-factor participation analysis
  - Message Frequency (30 points)
  - Media Sharing (20 points)
  - Message Diversity (20 points)
  - Response Activity (30 points)
- **Medal Rankings**: Top participants (🏆 🥈 🥉)
- **Engagement Levels**: Super Engaged 🔥, Highly Engaged ⭐, Moderately Engaged 👌, Casual Chatter 💬

### 🌏 Multi-Language Support  
- **Thai Language**: Full support for Thai text analysis with pythainlp
- **Smart Tokenization**: Handles mixed Thai-English content
- **Thai Font Wordcloud**: Beautiful Thai typography in word clouds

### 🎯 Advanced Features
- **Word Prediction**: Predict message authors based on text patterns
- **Advanced Statistics**: Message length analysis, response patterns, conversation starters
- **Emoji Analysis**: Track emoji usage by user
- **Word Frequency**: Top words and user-specific vocabulary
- **Personalized Fun Facts**: Unique insights for each participant

### 🎨 Modern UI
- **Responsive Design**: Works on desktop and mobile
- **LINE-Themed**: Authentic LINE app color scheme
- **Interactive Charts**: Zoom, hover, and filter capabilities
- **Drag & Drop Upload**: Easy file handling
- **Progress Bars & Badges**: Visual metric displays
- **Color-Coded Indicators**: Quick status identification

## 🚀 Quick Start

### Requirements
- Python 3.8+
- 100MB+ free disk space for large chat files

### Installation

1. **Clone and navigate to v2-flask:**
```bash
cd v2-flask
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python flask_app.py
```

4. **Open your browser:**
   - Go to http://localhost:8888
   - Upload your LINE chat export (.txt file)
   - Explore your chat analytics!

## 📱 Export LINE Chat Data

1. Open LINE app on your mobile device
2. Go to the chat you want to analyze  
3. Tap the chat settings (≡ menu → Settings)
4. Select **"Export chat history"**
5. Choose **"Without media"** for faster processing
6. Save the .txt file to your device
7. Upload it to the web application

## 🔧 Configuration

The application supports files up to **100MB** in size. For larger files, you can modify the limit in `flask_app.py`:

```python
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
```

## 📊 Supported Analytics

| Feature | Description |
|---------|-------------|
| **Message Volume** | Total messages, daily/hourly patterns |
| **User Activity** | Individual user statistics and rankings |
| **Word Analysis** | Frequency analysis, word clouds (Thai + English) |
| **Response Patterns** | Average response times, quick responses |
| **Conversation Flow** | Who starts conversations, message lengths |
| **Emoji Usage** | Most used emojis by user |
| **Text Prediction** | ML-style author prediction from text |

## 🌟 Improvements over V1

- ✅ **No TensorFlow dependency** - Lighter and more stable
- ✅ **Better Thai support** - Native Thai text processing
- ✅ **Modern web UI** - Responsive and mobile-friendly
- ✅ **Larger file support** - Handle chat files up to 100MB
- ✅ **More visualizations** - 7 interactive chart types
- ✅ **Advanced analytics** - Response patterns, emoji analysis
- ✅ **Better error handling** - Clear error messages and validation
- ✅ **Easy deployment** - Simple Flask server, no complex setup

## 🐛 Troubleshooting

### File Upload Issues
- **File too large**: Check file size (max 100MB)
- **Invalid format**: Ensure you're uploading a LINE .txt export
- **Upload fails**: Check console for detailed error messages

### Thai Text Issues  
- **Garbled text**: Ensure your chat export uses UTF-8 encoding
- **Missing words**: Thai text processing requires pythainlp library

### Performance
- **Slow loading**: Large files (>50MB) may take 30+ seconds to process
- **Memory usage**: Very large chats may require 1GB+ RAM

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

## 🤝 Contributing

Found a bug or want to add a feature? Feel free to open an issue or submit a pull request!

---

**Enjoy analyzing your LINE chats! 💬✨**
