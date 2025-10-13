from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import re
from io import StringIO, BytesIO
import numpy as np
from wordcloud import WordCloud
import json
import base64
from collections import Counter
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class LineAnalyzer:
    def __init__(self):
        self.df = None
        self.chat_day = None
        self.call_time = None
        
    def deEmojify(self, text):
        """removes emoji, so we can feed the data to pandas"""
        regrex_pattern = re.compile(pattern="["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
        return regrex_pattern.sub(r'', text)
    
    def create_date(self, date_list):
        """create date column"""
        result = []
        latest_date = ""
        for i in date_list:
            if i.find("BE") != -1:
                latest_date = str(i)
                result.append(str(i))
            else:
                result.append(latest_date)
        return result
    
    def extract_day_date(self, df_, date_list):
        """extract date feature from string and create is_weekend feature"""
        date_series = pd.Series(date_list)
        day_series = pd.Series(date_list)
        
        date_split = date_series.str.split(' ', n=2, expand=True)
        day_split = day_series.str.split(',', n=1, expand=True)
        
        try:
            df_['date'] = date_split[1]
            df_['dow'] = day_split[0]
        except:
            df_['date'] = date_split[0] if len(date_split.columns) > 0 else date_series
            df_['dow'] = day_split[0] if len(day_split.columns) > 0 else day_series
        
        df_['is_weekday'] = df_['dow'].apply(lambda x: 1 if str(x) in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] else 0)
        return df_
    
    def create_datetime(self, df):
        """change Buddhist year to Christian year and change the column type to be pd.datetime"""
        df = df[(df.time.str.len() < 10)].copy()
        if len(df) == 0:
            return df
            
        date_list = df['date'].str.split('/', n=2, expand=True)
        df['day'] = date_list[0]
        df['month'] = date_list[1]
        df['year'] = date_list[2]
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Convert Buddhist year to Christian year
        df.loc[df['year'] > 2500, 'year'] = df.loc[df['year'] > 2500, 'year'] - 543
        df['year'] = df['year'].astype('str')
        df['datetime'] = df['day'] + '/' + df['month'] + '/' + df['year'] + ' ' + df['time']
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['hour'] = df['datetime'].dt.hour
        df['time'] = df['time'].astype('str')
        return df
    
    def tokenize_thai(self, text):
        """Tokenize Thai text using PyThaiNLP"""
        return word_tokenize(text, engine='newmm')
    
    def process_line_chat(self, file_content):
        """Process LINE chat file and return analyzed data"""
        try:
            # Clean emoji and read as CSV
            text = StringIO(self.deEmojify(file_content))
            df = pd.read_csv(text, sep="\t", header=None, names=["time", "name", "chat"])
            
            if len(df) < 3:
                raise ValueError("File too short or invalid format")
            
            df = df.iloc[2:]  # Skip first 2 rows
            df.reset_index(inplace=True)
            df['date'] = self.create_date(df['time'].values)
            df = self.extract_day_date(df, df['date'].values)
            
            # Create chat in a day feature
            chat_day = df[(df.isnull().any(axis=1)) & (df.time.str.len() > 10)].copy()
            df = self.create_datetime(df)
            
            # Process chat_day if it exists
            if len(chat_day) > 1:
                chat_day.drop(columns=['name', 'chat'], inplace=True, errors='ignore')
                temp = chat_day['index'].iloc[1:].values
                chat_day = chat_day[:-1].copy()
                chat_day['diff_chat'] = temp
                chat_day['chat in a day'] = chat_day['diff_chat'] - chat_day['index']
                chat_day = chat_day.set_index('time')
                chat_day = chat_day.sort_values(by='chat in a day', ascending=False)
                chat_day = self.extract_day_date(chat_day, chat_day.index)
            
            # Process call time data
            call_time = None
            try:
                df2 = df.dropna().copy()
                call_time = df2[df2["chat"].str.contains("☎ Call time", na=False)]
                if len(call_time) > 0:
                    call_time = call_time[call_time['chat'].str.len() < 30]
                    new = call_time["chat"].str.split("e", n=1, expand=True)
                    if len(new.columns) > 1:
                        call_time["call time"] = new[1]
                        call_time['call second'] = call_time['call time'].apply(self.fix_call)
                        call_time = call_time.sort_values(by='call second', ascending=False)
                        call_time.reset_index(inplace=True)
            except:
                call_time = None
            
            self.df = df
            self.chat_day = chat_day
            self.call_time = call_time
            
            return {
                'success': True,
                'message': f'Successfully processed {len(df)} messages',
                'participants': list(df['name'].unique()),
                'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}" if 'datetime' in df.columns else "Unknown"
            }
        
        except Exception as e:
            return {'success': False, 'message': f'Error processing file: {str(e)}'}
    
    def fix_call(self, x):
        """create a call time feature in second unit"""
        try:
            if len(x) <= 6:
                split_string = x.split(":", 1)
                substring1 = split_string[0]
                substring2 = split_string[1]
                second = int(substring1) * 60 + int(substring2)
            else:
                split_string = x.split(":", 2)
                substring1 = split_string[0]
                substring2 = split_string[1]
                substring3 = split_string[2]
                second = int(substring1) * 60 * 60 + int(substring2) * 60 + int(substring3)
            return second
        except:
            return 0
    
    def get_message_stats(self):
        """Get basic message statistics"""
        if self.df is None:
            return {}
        
        stats = {
            'total_messages': len(self.df),
            'participants': len(self.df['name'].unique()),
            'date_range': f"{self.df['datetime'].min().strftime('%Y-%m-%d')} to {self.df['datetime'].max().strftime('%Y-%m-%d')}" if 'datetime' in self.df.columns else "Unknown"
        }
        
        # Message count by user
        user_counts = self.df['name'].value_counts().to_dict()
        stats['user_message_counts'] = user_counts
        
        # Chat types
        chat_types = ['[Photo]', '[Sticker]', '[Video]', '☎ Missed call', '☎ Canceled call', '[File]', '[Voice message]']
        type_counts = {}
        for chat_type in chat_types:
            type_counts[chat_type] = len(self.df[self.df['chat'] == chat_type])
        stats['chat_type_counts'] = type_counts
        
        return stats
    
    def create_visualizations(self):
        """Create all visualizations and return as JSON"""
        if self.df is None:
            return {}
        
        plots = {}
        
        # 1. Top messages histogram
        top_messages = self.df['chat'].value_counts().head(20)
        fig1 = px.bar(x=top_messages.index, y=top_messages.values, title='Top 20 Messages')
        fig1.update_layout(template='plotly_white', xaxis_title='Message', yaxis_title='Count')
        plots['top_messages'] = fig1.to_json()
        
        # 2. Messages per user
        user_counts = self.df['name'].value_counts()
        fig2 = px.bar(x=user_counts.index, y=user_counts.values, title='Messages per User')
        fig2.update_layout(template='plotly_white', xaxis_title='User', yaxis_title='Message Count')
        plots['messages_per_user'] = fig2.to_json()
        
        # 3. Messages by hour (if datetime available)
        if 'hour' in self.df.columns:
            hourly_counts = self.df.groupby('hour').size()
            fig3 = px.bar(x=hourly_counts.index, y=hourly_counts.values, title='Messages by Hour of Day')
            fig3.update_layout(template='plotly_white', xaxis_title='Hour', yaxis_title='Message Count')
            plots['messages_by_hour'] = fig3.to_json()
        
        # 4. Messages by day of week
        if 'dow' in self.df.columns:
            dow_counts = self.df['dow'].value_counts()
            fig4 = px.pie(values=dow_counts.values, names=dow_counts.index, title='Messages by Day of Week')
            plots['messages_by_dow'] = fig4.to_json()
        
        # 5. Chat types distribution
        chat_types = ['[Photo]', '[Sticker]', '[Video]', '[File]', '[Voice message]']
        type_counts = [len(self.df[self.df['chat'] == ct]) for ct in chat_types]
        fig5 = px.bar(x=chat_types, y=type_counts, title='Chat Types Distribution')
        fig5.update_layout(template='plotly_white', xaxis_title='Chat Type', yaxis_title='Count')
        plots['chat_types'] = fig5.to_json()
        
        return plots
    
    def create_wordcloud(self):
        """Create wordcloud from messages"""
        if self.df is None:
            return None
        
        try:
            # Get all text messages (exclude special message types)
            text_messages = self.df[~self.df['chat'].str.contains(r'\[.*\]|☎', na=False)]['chat']
            
            if len(text_messages) == 0:
                return None
            
            # Tokenize and clean Thai text
            all_words = []
            stopwords = thai_stopwords()
            
            for message in text_messages:
                tokens = self.tokenize_thai(str(message))
                words = [word for word in tokens if word not in stopwords and len(word) > 1]
                all_words.extend(words)
            
            if len(all_words) == 0:
                return None
            
            # Create wordcloud
            text = ' '.join(all_words[:1000])  # Limit words for performance
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='Set2',
                max_words=100
            ).generate(text)
            
            # Convert to base64
            img_buffer = BytesIO()
            wordcloud.to_image().save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            return img_base64
        except Exception as e:
            print(f"Wordcloud error: {e}")
            return None

# Initialize analyzer
analyzer = LineAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and file.filename.endswith('.txt'):
        try:
            file_content = file.read().decode('utf-8')
            result = analyzer.process_line_chat(file_content)
            return jsonify(result)
        except UnicodeDecodeError:
            return jsonify({'success': False, 'message': 'File encoding error. Please use UTF-8 encoded text file.'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Processing error: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Please upload a .txt file'})

@app.route('/stats')
def get_stats():
    stats = analyzer.get_message_stats()
    return jsonify(stats)

@app.route('/visualizations')
def get_visualizations():
    plots = analyzer.create_visualizations()
    return jsonify(plots)

@app.route('/wordcloud')
def get_wordcloud():
    wordcloud_img = analyzer.create_wordcloud()
    return jsonify({'wordcloud': wordcloud_img})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
