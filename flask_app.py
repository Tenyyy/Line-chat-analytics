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
from datetime import datetime
import pythainlp

# Thai text processing
try:
    from pythainlp import word_tokenize
    from pythainlp.corpus import thai_stopwords
    THAI_SUPPORT = True
    print("Thai language support enabled")
except ImportError:
    THAI_SUPPORT = False
    print("Thai language support disabled - install pythainlp for Thai text processing")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Handle file too large error
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False, 
        'message': f'File too large. Maximum size allowed is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'
    }), 413

class LineAnalyzer:
    def __init__(self):
        self.df = None
        self.chat_day = None
        self.call_time = None
        self.user_word_patterns = {}  # For word prediction
        self.emoji_patterns = {}      # For emoji analysis
        
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
            if str(i).find("BE") != -1:
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
            df_['date'] = date_split[1] if len(date_split.columns) > 1 else date_split[0]
            df_['dow'] = day_split[0]
        except:
            df_['date'] = date_series
            df_['dow'] = day_series
        
        df_['is_weekday'] = df_['dow'].apply(lambda x: 1 if str(x) in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] else 0)
        return df_
    
    def create_datetime(self, df):
        """change Buddhist year to Christian year and change the column type to be pd.datetime"""
        if 'time' not in df.columns:
            return df
            
        df_filtered = df[df.time.str.len() < 10].copy()
        if len(df_filtered) == 0:
            return df
            
        date_pattern = r'(\d{2})/(\d{2})/(\d{4})'
        date_parts = df_filtered['date'].str.extract(date_pattern)
        if date_parts.empty or date_parts.shape[1] < 3:
        # If extraction failed for all filtered rows, return the original df
            return df
            
        df_filtered['day'] = date_parts[0]
        df_filtered['month'] = date_parts[1]
        # Use errors='coerce' to turn non-numeric years (where regex failed) into NaN
        df_filtered['year'] = pd.to_numeric(date_parts[2], errors='coerce')
        df_filtered.dropna(subset=['year'], inplace=True)
        if df_filtered.empty:
            return df
        
        # Convert Buddhist year to Christian year
        mask = df_filtered['year'] > 2500
        df_filtered.loc[mask, 'year'] = df_filtered.loc[mask, 'year'] - 543
        df_filtered['year'] = df_filtered['year'].astype('int').astype('str')
        df_filtered['datetime'] = pd.to_datetime(
        df_filtered['day'] + '/' + df_filtered['month'] + '/' + df_filtered['year'] + ' ' + df_filtered['time'],
        format="%d/%m/%Y %H:%M",
        errors='coerce'
    )

        df_filtered['hour'] = df_filtered['datetime'].dt.hour
        
        return df_filtered
    
    def tokenize_text(self, text):
        """Enhanced tokenization for wordcloud supporting Thai and English"""
        if not text or pd.isna(text):
            return []
        
        text = str(text).strip()
        if not text:
            return []
        
        # Check if text contains Thai characters
        thai_pattern = re.compile(r'[\u0E00-\u0E7F]')
        has_thai = bool(thai_pattern.search(text))
        
        if has_thai and THAI_SUPPORT:
            # Thai text processing
            try:
                # Tokenize Thai text
                tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
                
                # Get Thai stopwords
                thai_stops = thai_stopwords()
                
                # Clean and filter tokens
                cleaned_tokens = []
                for token in tokens:
                    token = token.strip()
                    # Skip empty tokens, single characters, numbers, and stopwords
                    if (len(token) > 1 and 
                        not token.isdigit() and 
                        token not in thai_stops and
                        not re.match(r'^[^\u0E00-\u0E7Fa-zA-Z]+$', token)):  # Skip pure punctuation/symbols
                        cleaned_tokens.append(token)
                
                return cleaned_tokens
            except Exception as e:
                print(f"Thai tokenization error: {e}")
                # Fall back to basic tokenization
                return self.basic_tokenize(text)
        else:
            # English/other language processing
            return self.basic_tokenize(text)
    
    def basic_tokenize(self, text):
        """Basic tokenization for non-Thai text"""
        # Remove special characters but keep Thai characters
        text = re.sub(r'[^\w\s\u0E00-\u0E7F]', ' ', str(text))
        words = text.split()
        
        # English stopwords
        english_stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'a', 'an', 'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should',
            'have', 'has', 'had', 'do', 'does', 'did', 'can', 'may', 'might', 'must',
            'shall', 'should', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'her', 'its', 'our', 'their', 'am', 'be', 'been', 'being', 'not', 'no'
        }
        
        return [word for word in words 
                if len(word) > 1 and 
                word.lower() not in english_stopwords and 
                not word.isdigit()]
    
    def process_line_chat(self, file_content):
        """Process LINE chat file and return analyzed data"""
        try:
            # Clean emoji and read as CSV
            text = StringIO(self.deEmojify(file_content))
            df = pd.read_csv(text, sep="$", header=None, names=["time", "name", "chat"])
            
            if len(df) < 3:
                raise ValueError("File too short or invalid format")
            
            df = df.iloc[2:]  # Skip first 2 rows
            df.reset_index(drop=True, inplace=True)
            df['original_index'] = df.index
            
            # Create date column
            df['date'] = self.create_date(df['time'].values)
            df = self.extract_day_date(df, df['date'].values)
            
            # Create datetime
            df = self.create_datetime(df)
            
            # Process chat_day if needed
            chat_day = df[(df.isnull().any(axis=1)) & (df['time'].str.len() > 10)].copy()
            
            # Process call time data
            call_time = None
            try:
                df_calls = df.dropna().copy()
                call_time_mask = df_calls["chat"].str.contains("☎ Call time", na=False)
                call_time = df_calls[call_time_mask]
                
                if len(call_time) > 0:
                    call_time = call_time[call_time['chat'].str.len() < 30]
                    call_split = call_time["chat"].str.split("e", n=1, expand=True)
                    if len(call_split.columns) > 1:
                        call_time["call_duration"] = call_split[1]
                        call_time = call_time.sort_values(by='call_duration', ascending=False)
                        call_time.reset_index(inplace=True)
            except Exception as e:
                print(f"Call time processing error: {e}")
                call_time = None
            
            self.df = df
            self.chat_day = chat_day
            self.call_time = call_time
            
            # Build word patterns for prediction
            self._build_user_patterns()
            
            participants = [p for p in df['name'].unique() if pd.notna(p)]
            
            return {
                'success': True,
                'message': f'Successfully processed {len(df)} messages',
                'participants': participants,
                'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}" if 'datetime' in df.columns and not df['datetime'].isna().all() else "Date parsing failed"
            }
        
        except Exception as e:
            return {'success': False, 'message': f'Error processing file: {str(e)}'}
    
    def get_message_stats(self):
        """Get basic message statistics"""
        if self.df is None:
            return {}
        
        # Filter out NaN names
        df_clean = self.df[self.df['name'].notna()].copy()
        
        stats = {
            'total_messages': len(df_clean),
            'participants': len(df_clean['name'].unique()),
        }
        
        # Add date range if available
        if 'datetime' in df_clean.columns and not df_clean['datetime'].isna().all():
            min_date = df_clean['datetime'].min()
            max_date = df_clean['datetime'].max()
            stats['date_range'] = f"{min_date.strftime('%d/%m/%Y')} to {max_date.strftime('%d/%m/%Y')}"
        else:
            stats['date_range'] = "Date information unavailable"
        
        # Message count by user
        user_counts = df_clean['name'].value_counts().to_dict()
        stats['user_message_counts'] = user_counts
        
        # Chat types
        chat_types = ['[Photo]', '[Sticker]', '[Video]', '☎ Missed call', '☎ Canceled call', '[File]', '[Voice message]']
        type_counts = {}
        for chat_type in chat_types:
            type_counts[chat_type] = len(df_clean[df_clean['chat'] == chat_type])
        stats['chat_type_counts'] = type_counts
        
        return stats
    
    def create_visualizations(self):
        """Create all visualizations and return as JSON"""
        if self.df is None:
            return {}
        
        plots = {}
        df_clean = self.df[self.df['name'].notna()].copy()
        
        # Color scheme for consistency
        colors = ['#00d084', '#06c755', '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', '#eb4d4b', '#6c5ce7']
        
        try:
            # 1. Top messages histogram
            top_messages = df_clean['chat'].value_counts().head(15)
            fig1 = px.bar(
                x=top_messages.values, 
                y=[str(msg)[:40] + ('...' if len(str(msg)) > 40 else '') for msg in top_messages.index], 
                orientation='h',
                color=top_messages.values,
                color_continuous_scale='Viridis'
            )
            fig1.update_layout(
                template='plotly_white', 
                xaxis_title='Count', 
                yaxis_title='Message',
                height=500,
                showlegend=False,
                title_font_size=16
            )
            plots['top_messages'] = fig1.to_json()
        except Exception as e:
            print(f"Top messages chart error: {e}")
        
        try:
            # 2. Messages per user - enhanced with colors
            user_counts = df_clean['name'].value_counts()
            custom_labels = (
                user_counts.index + 
                ' (' + 
                user_counts.values.astype(str) + 
                ')'
            )
            fig2 = px.pie(
                values=user_counts.values, 
                names=custom_labels,
                color_discrete_sequence=colors
            )
            fig2.update_layout(
                template='plotly_white',
                height=400  
            )
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            plots['messages_per_user'] = fig2.to_json()
        except Exception as e:
            print(f"Messages per user chart error: {e}")
        
        try:
            # 3. Messages by hour (if datetime available)
            if 'hour' in df_clean.columns and not df_clean['hour'].isna().all():
                hourly_counts = df_clean.groupby('hour').size().reindex(range(24), fill_value=0)
                fig3 = px.line(
                    x=hourly_counts.index, 
                    y=hourly_counts.values,
                    markers=True
                )
                fig3.update_traces(line_color='#00d084', line_width=3, marker_size=8)
                fig3.update_layout(
                    template='plotly_white', 
                    xaxis_title='Hour of Day', 
                    yaxis_title='Number of Messages',
                    height=400,
                    xaxis=dict(tickmode='linear', tick0=0, dtick=2)
                )
                plots['messages_by_hour'] = fig3.to_json()
        except Exception as e:
            print(f"Messages by hour chart error: {e}")
        
        try:
            # 4. Messages by day of week
            if 'dow' in df_clean.columns:
                # Order days properly
                day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                dow_counts = df_clean['dow'].value_counts().reindex(day_order, fill_value=0)
                
                fig4 = px.bar(
                    x=dow_counts.index, 
                    y=dow_counts.values,
                    color=dow_counts.values,
                    color_continuous_scale='Sunsetdark',
                    text_auto=True
                )
                fig4.update_layout(
                    template='plotly_white',
                    xaxis_title='Day of Week', 
                    yaxis_title='Number of Messages',
                    height=400,
                    showlegend=False
                )
                plots['messages_by_dow'] = fig4.to_json()
        except Exception as e:
            print(f"Messages by day of week chart error: {e}")
        
        try:
            # 5. Chat types distribution
            chat_types = ['[Photo]', '[Sticker]', '[Video]', '[File]', '[Voice message]']
            type_counts = [len(df_clean[df_clean['chat'] == ct]) for ct in chat_types]
            
            # Filter out zero counts
            non_zero_data = [(ct.replace('[', '').replace(']', ''), count) for ct, count in zip(chat_types, type_counts) if count > 0]
            
            if non_zero_data:
                types, counts = zip(*non_zero_data)
                fig5 = px.bar(
                    x=list(types), 
                    y=list(counts),
                    color=list(counts),
                    color_continuous_scale='Sunsetdark',
                    text_auto=True
                )
                fig5.update_layout(
                    template='plotly_white', 
                    xaxis_title='Content Type', 
                    yaxis_title='Count',
                    height=400,
                    showlegend=False
                )
                plots['chat_types'] = fig5.to_json()
        except Exception as e:
            print(f"Chat types chart error: {e}")
        
        try:
            # 6. Messages over time (if datetime available)
            if 'datetime' in df_clean.columns and not df_clean['datetime'].isna().all():
                daily_counts = df_clean.groupby(df_clean['datetime'].dt.date).size()
                fig6 = px.line(
                    x=daily_counts.index, 
                    y=daily_counts.values
                )
                fig6.update_traces(line_color='#06c755', line_width=2)
                fig6.update_layout(
                    template='plotly_white',
                    xaxis_title='Date', 
                    yaxis_title='Messages per Day',
                    height=400
                )
                plots['messages_over_time'] = fig6.to_json()
        except Exception as e:
            print(f"Messages over time chart error: {e}")
        
        try:
            # 7. Activity heatmap (hour vs day of week)
            if 'hour' in df_clean.columns and 'dow' in df_clean.columns:
                # Create heatmap data
                
                heatmap_data = df_clean.groupby(['dow', 'hour']).size().unstack(fill_value=0)
                day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                heatmap_data = heatmap_data.reindex(day_order, fill_value=0)
                fig7 = px.imshow(
                    heatmap_data.values,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
                    x=list(range(24)),
                    y=day_order,
                    color_continuous_scale='YlOrRd'
                )
                fig7.update_layout(
                    template='plotly_white',
                    height=300
                )
                plots['activity_heatmap'] = fig7.to_json()
        except Exception as e:
            print(f"Activity heatmap chart error: {e}")
        
        return plots
    
    def create_wordcloud(self):
        """Create wordcloud from messages supporting Thai text"""
        if self.df is None:
            return None
        
        try:
            df_clean = self.df[self.df['name'].notna()].copy()
            
            # Get all text messages (exclude special message types)
            text_mask = ~df_clean['chat'].str.contains(r'\[.*\]|☎', na=False)
            text_messages = df_clean[text_mask]['chat'].dropna()
            
            if len(text_messages) == 0:
                return None
            
            print(f"Processing {len(text_messages)} messages for wordcloud...")
            
            # Enhanced tokenization and word collection
            all_words = []
            thai_words = []
            english_words = []
            
            for message in text_messages:
                words = self.tokenize_text(str(message))
                all_words.extend(words)
                
                # Separate Thai and English words for better processing
                for word in words:
                    if re.search(r'[\u0E00-\u0E7F]', word):
                        thai_words.append(word)
                    else:
                        english_words.append(word)
            
            if len(all_words) == 0:
                print("No words found for wordcloud")
                return None
            
            print(f"Found {len(all_words)} total words ({len(thai_words)} Thai, {len(english_words)} English)")
            
            # Count word frequencies
            word_freq = Counter(all_words)
            
            # Remove very common but not meaningful words
            common_words_to_remove = {'55555', 'haha', 'lol', 'ok', 'okay', 'yes', 'no', 'hi', 'hello', 'bye'}
            for word in common_words_to_remove:
                if word in word_freq:
                    del word_freq[word]
            
            # Keep most frequent words
            top_words = dict(word_freq.most_common(150))
            
            if not top_words:
                print("No meaningful words found after filtering")
                return None
            
            # Create wordcloud with Thai font support
            font_path = None
            if THAI_SUPPORT:
                # Try to find Thai font in the assets folder
                thai_font_path = r'C:\Users\EkpitiKawtummachai\OneDrive - Anypay Company Limited\Desktop\chatana\Line-chat-analytics\assets\THSarabunNew.ttf'
                if os.path.exists(thai_font_path):
                    font_path = thai_font_path
                    print("Using Thai font for wordcloud")
            
            # Create wordcloud
            wordcloud_params = {
                'width': 1000,
                'height': 500,
                'background_color': 'white',
                'colormap': 'viridis',
                'max_words': 100,
                'relative_scaling': 0.3,
                'min_font_size': 12,
                'max_font_size': 80,
                'prefer_horizontal': 0.7,
                'regexp': r"[\u0E00-\u0E7Fa-zA-Z0-9']+",  # Support Thai characters
                'collocations': False
            }
            
            if font_path:
                wordcloud_params['font_path'] = font_path
            
            wordcloud = WordCloud(**wordcloud_params)
            
            # Generate from frequencies for better control
            wordcloud.generate_from_frequencies(top_words)
            
            # Convert to base64
            img_buffer = BytesIO()
            wordcloud.to_image().save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            print("Wordcloud generated successfully")
            return img_base64
            
        except Exception as e:
            print(f"Wordcloud error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_user_patterns(self):
        """Build word patterns for each user for prediction"""
        if self.df is None:
            return
        
        try:
            df_clean = self.df[self.df['name'].notna()].copy()
            text_mask = ~df_clean['chat'].str.contains(r'\[.*\]|☎', na=False)
            df_text = df_clean[text_mask]
            
            self.user_word_patterns = {}
            
            for user in df_text['name'].unique():
                user_messages = df_text[df_text['name'] == user]['chat']
                user_words = []
                
                for message in user_messages:
                    words = self.tokenize_text(str(message))
                    user_words.extend(words)
                
                # Count word frequencies for this user
                word_freq = Counter(user_words)
                self.user_word_patterns[user] = dict(word_freq)
                
                print(f"Built word patterns for {user}: {len(word_freq)} unique words")
        
        except Exception as e:
            print(f"Error building user patterns: {e}")
    
    
    def get_advanced_stats(self):
        """Get advanced statistics and insights"""
        if self.df is None:
            return {}
        
        try:
            df_clean = self.df[self.df['name'].notna()].copy()
            text_mask = ~df_clean['chat'].str.contains(r'\[.*\]|☎', na=False)
            df_text = df_clean[text_mask]
            
            stats = {}
            
            # Message length analysis
            df_text['message_length'] = df_text['chat'].str.len()
            stats['message_length'] = {
                'average': round(df_text['message_length'].mean(), 1),
                'median': round(df_text['message_length'].median(), 1),
                'longest': int(df_text['message_length'].max()),
                'shortest': int(df_text['message_length'].min()),
                'by_user': df_text.groupby('name')['message_length'].mean().round(1).to_dict()
            }
            
            # Activity patterns
            if 'hour' in df_clean.columns:
                peak_hour = df_clean['hour'].mode().iloc[0] if not df_clean['hour'].empty else 0
                stats['activity_patterns'] = {
                    'peak_hour': int(peak_hour),
                    'hourly_distribution': df_clean.groupby('hour').size().to_dict(),
                }
            
            # Conversation starters (who sends first message after long gaps)
            if 'datetime' in df_clean.columns:
                df_sorted = df_clean.sort_values('datetime')
                df_sorted['time_diff'] = df_sorted['datetime'].diff()
                # Consider messages after 1+ hour gap as conversation starters
                conversation_starters = df_sorted[df_sorted['time_diff'] > pd.Timedelta(hours=1)]
                if len(conversation_starters) > 0:
                    starter_counts = conversation_starters['name'].value_counts()
                    stats['conversation_starters'] = starter_counts.to_dict()
            
            # Emoji usage analysis
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
            
            user_emojis = {}
            for user in df_text['name'].unique():
                user_messages = df_text[df_text['name'] == user]['chat']
                all_emojis = []
                for message in user_messages:
                    emojis = emoji_pattern.findall(str(message))
                    all_emojis.extend(emojis)
                
                emoji_count = Counter(all_emojis)
                user_emojis[user] = dict(emoji_count.most_common(10))
            
            stats['emoji_usage'] = user_emojis
            
            # Response time patterns (simplified)
            if 'datetime' in df_clean.columns:
                df_sorted = df_clean.sort_values('datetime')
                response_times = []
                
                for i in range(1, min(len(df_sorted), 1000)):  # Limit for performance
                    prev_msg = df_sorted.iloc[i-1]
                    curr_msg = df_sorted.iloc[i]
                    
                    # Check if it's a response (different user, within reasonable time)
                    if (prev_msg['name'] != curr_msg['name']):
                        time_diff = curr_msg['datetime'] - prev_msg['datetime']
                        if time_diff < pd.Timedelta(hours=2):  # Within 2 hours
                            response_times.append(time_diff.total_seconds() / 60)  # Convert to minutes
                
                if response_times:
                    stats['response_patterns'] = {
                        'average_response_time_minutes': round(np.mean(response_times), 1),
                        'median_response_time_minutes': round(np.median(response_times), 1),
                        'quick_responses_under_1min': len([t for t in response_times if t < 1])
                    }
            
            return stats
            
        except Exception as e:
            print(f"Advanced stats error: {e}")
            return {}

# Initialize analyzer
analyzer = LineAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return jsonify({'message': 'Flask is working!'})

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload request received")
    print("Files in request:", request.files.keys())
    
    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    
    # Check file size more efficiently
    file.seek(0, 2)  # Seek to end of file
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    print(f"File received: {file.filename}, size: {file_size} bytes ({file_size / (1024*1024):.1f}MB)")
    
    if file.filename == '':
        print("Empty filename")
        return jsonify({'success': False, 'message': 'No file selected'})
    
    # Additional size check (this should be caught by Flask's MAX_CONTENT_LENGTH but let's be safe)
    max_size = app.config.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024)
    if file_size > max_size:
        return jsonify({
            'success': False, 
            'message': f'File too large ({file_size / (1024*1024):.1f}MB). Maximum size allowed is {max_size // (1024*1024)}MB'
        })
    
    if file and file.filename.endswith('.txt'):
        try:
            print("Processing file...")
            file_content = file.read().decode('utf-8')
            print(f"File content length: {len(file_content)}")
            
            result = analyzer.process_line_chat(file_content)
            print(f"Processing result: {result}")
            return jsonify(result)
        except UnicodeDecodeError:
            try:
                print("Trying UTF-8-sig encoding...")
                file.seek(0)
                file_content = file.read().decode('utf-8-sig')  # Try with BOM
                result = analyzer.process_line_chat(file_content)
                return jsonify(result)
            except Exception as e:
                print(f"Encoding error: {e}")
                return jsonify({'success': False, 'message': 'File encoding error. Please use UTF-8 encoded text file.'})
        except Exception as e:
            print(f"Processing error: {e}")
            return jsonify({'success': False, 'message': f'Processing error: {str(e)}'})
    
    print("Invalid file type")
    return jsonify({'success': False, 'message': 'Please upload a .txt file'})

@app.route('/stats')
def get_stats():
    try:
        stats = analyzer.get_message_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/visualizations')
def get_visualizations():
    try:
        plots = analyzer.create_visualizations()
        return jsonify(plots)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/wordcloud')
def get_wordcloud():
    try:
        wordcloud_img = analyzer.create_wordcloud()
        return jsonify({'wordcloud': wordcloud_img})
    except Exception as e:
        return jsonify({'error': str(e), 'wordcloud': None})

@app.route('/word_frequency')
def get_word_frequency():
    try:
        if analyzer.df is None:
            return jsonify({'error': 'No data available'})
        
        df_clean = analyzer.df[analyzer.df['name'].notna()].copy()
        text_mask = ~df_clean['chat'].str.contains(r'\[.*\]|☎', na=False)
        text_messages = df_clean[text_mask]['chat'].dropna()
        
        all_words = []
        for message in text_messages:
            words = analyzer.tokenize_text(str(message))
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        top_15_words = dict(word_freq.most_common(15))
        
        return jsonify({
            'word_frequencies': top_15_words,
            'total_unique_words': len(word_freq),
            'total_words': sum(word_freq.values())
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/advanced_stats')
def get_advanced_stats():
    try:
        stats = analyzer.get_advanced_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888, use_reloader=False)
