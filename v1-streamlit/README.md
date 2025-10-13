# LINE Chat Analytics - Version 1 (Streamlit)

This is the original Streamlit-based version of the LINE Chat Analytics application.

## Features

- 📊 Basic chat statistics and visualizations
- 🤖 TensorFlow-based text prediction model
- 📈 Message frequency analysis
- 👥 User activity patterns
- 🎯 Word prediction functionality

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit 1.12.0

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy models:
```bash
python download_spacy_models.py
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run main.py
```

2. Open your browser and go to the displayed local URL (usually http://localhost:8501)

3. Upload your LINE chat export (.txt file) using the file uploader

## Notes

- This version requires TensorFlow for machine learning features
- May have dependency conflicts with newer Python versions
- For a more stable experience, consider using Version 2 (Flask)

## Export LINE Chat Data

1. Open LINE app on your mobile device
2. Go to the chat you want to analyze
3. Tap on the chat settings (usually top-right corner)
4. Select "Export chat history"
5. Choose "Without media" for faster processing
6. Save the .txt file and upload it to the application
