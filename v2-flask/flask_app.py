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
            
        date_list = df_filtered['date'].str.split('/', n=2, expand=True)
        if len(date_list.columns) < 3:
            return df_filtered
            
        df_filtered['day'] = date_list[0]
        df_filtered['month'] = date_list[1]
        df_filtered['year'] = pd.to_numeric(date_list[2], errors='coerce')
        
        # Convert Buddhist year to Christian year
        mask = df_filtered['year'] > 2500
        df_filtered.loc[mask, 'year'] = df_filtered.loc[mask, 'year'] - 543
        
        df_filtered['year'] = df_filtered['year'].astype('str')
        df_filtered['datetime'] = pd.to_datetime(
            df_filtered['day'] + '/' + df_filtered['month'] + '/' + df_filtered['year'] + ' ' + df_filtered['time'], 
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
            df = pd.read_csv(text, sep="\t", header=None, names=["time", "name", "chat"])
            
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
            stats['date_range'] = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
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
            fig2 = px.pie(
                values=user_counts.values, 
                names=user_counts.index,
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
                    color_continuous_scale='Blues'
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
                    color_continuous_scale='Plasma'
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
            # 6. Messages over time (with cumulative option)
            if 'datetime' in df_clean.columns and not df_clean['datetime'].isna().all():
                daily_counts = df_clean.groupby(df_clean['datetime'].dt.date).size()
                
                # Regular daily messages
                fig6 = px.line(
                    x=daily_counts.index, 
                    y=daily_counts.values
                )
                fig6.update_traces(line_color='#06c755', line_width=2, name='Daily Messages')
                fig6.update_layout(
                    template='plotly_white',
                    xaxis_title='Date', 
                    yaxis_title='Messages per Day',
                    height=400,
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="left",
                            buttons=list([
                                dict(
                                    args=[{"visible": [True, False]}],
                                    label="Daily",
                                    method="update"
                                ),
                                dict(
                                    args=[{"visible": [False, True]}],
                                    label="Cumulative",
                                    method="update"
                                )
                            ]),
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.11,
                            xanchor="left",
                            y=1.15,
                            yanchor="top"
                        ),
                    ]
                )
                
                # Add cumulative trace (hidden by default)
                cumulative_counts = daily_counts.cumsum()
                fig6.add_scatter(
                    x=cumulative_counts.index,
                    y=cumulative_counts.values,
                    mode='lines',
                    line=dict(color='#00d084', width=3),
                    name='Cumulative Messages',
                    visible=False
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
                thai_font_path = 'assets/THSarabunNew.ttf'
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
    
    def predict_user_from_text(self, input_text):
        """Predict which user is most likely to write the given text"""
        if not self.user_word_patterns or not input_text:
            return {"prediction": "Unknown", "confidence": 0, "scores": {}}
        
        try:
            # Tokenize input text
            input_words = self.tokenize_text(input_text)
            if not input_words:
                return {"prediction": "Unknown", "confidence": 0, "scores": {}}
            
            user_scores = {}
            
            for user, word_freq in self.user_word_patterns.items():
                score = 0
                total_user_words = sum(word_freq.values())
                
                if total_user_words == 0:
                    continue
                
                # Calculate score based on word frequency
                for word in input_words:
                    word_count = word_freq.get(word, 0)
                    # Use frequency ratio as probability
                    word_probability = word_count / total_user_words
                    score += word_probability
                
                # Normalize by input length
                user_scores[user] = score / len(input_words) if input_words else 0
            
            if not user_scores:
                return {"prediction": "Unknown", "confidence": 0, "scores": {}}
            
            # Find best prediction
            best_user = max(user_scores, key=user_scores.get)
            max_score = user_scores[best_user]
            total_score = sum(user_scores.values())
            
            # Calculate confidence as percentage
            confidence = (max_score / total_score * 100) if total_score > 0 else 0
            
            # Convert scores to percentages for display
            display_scores = {}
            if total_score > 0:
                for user, score in user_scores.items():
                    display_scores[user] = round((score / total_score * 100), 1)
            
            return {
                "prediction": best_user,
                "confidence": round(confidence, 1),
                "scores": display_scores
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"prediction": "Unknown", "confidence": 0, "scores": {}}
    
    def get_advanced_stats(self):
        """Get detailed individual participant insights and advanced analytics"""
        if self.df is None:
            return {}
        
        try:
            df_clean = self.df[self.df['name'].notna()].copy()
            text_mask = ~df_clean['chat'].str.contains(r'\[.*\]|☎', na=False)
            df_text = df_clean[text_mask]
            
            stats = {}
            
            # Individual participant detailed analysis
            participant_insights = {}
            
            for user in df_text['name'].unique():
                user_data = df_clean[df_clean['name'] == user]
                user_text = df_text[df_text['name'] == user]
                
                # Message type analysis for each user
                photos_sent = len(user_data[user_data['chat'].str.contains(r'\[Photo\]', na=False)])
                stickers_sent = len(user_data[user_data['chat'].str.contains(r'\[Sticker\]', na=False)])
                voice_messages = len(user_data[user_data['chat'].str.contains(r'\[Voice message\]', na=False)])
                videos_sent = len(user_data[user_data['chat'].str.contains(r'\[Video\]', na=False)])
                files_sent = len(user_data[user_data['chat'].str.contains(r'\[File\]', na=False)])
                calls_made = len(user_data[user_data['chat'].str.contains(r'☎', na=False)])
                
                # Text message analysis
                text_messages = len(user_text)
                if text_messages > 0:
                    avg_message_length = round(user_text['chat'].str.len().mean(), 1)
                    longest_message = int(user_text['chat'].str.len().max())
                    
                    # Emoji analysis
                    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
                    
                    user_emojis = []
                    for message in user_text['chat']:
                        emojis = emoji_pattern.findall(str(message))
                        user_emojis.extend(emojis)
                    
                    emoji_count = Counter(user_emojis)
                    top_emojis = dict(emoji_count.most_common(5))
                    total_emojis = sum(emoji_count.values())
                else:
                    avg_message_length = 0
                    longest_message = 0
                    top_emojis = {}
                    total_emojis = 0
                
                # Activity time analysis
                activity_by_hour = {}
                if 'hour' in user_data.columns:
                    activity_by_hour = user_data['hour'].value_counts().to_dict()
                    most_active_hour = user_data['hour'].mode().iloc[0] if not user_data['hour'].empty else 0
                else:
                    most_active_hour = 0
                
                participant_insights[user] = {
                    'total_messages': len(user_data),
                    'text_messages': text_messages,
                    'photos_sent': photos_sent,
                    'stickers_sent': stickers_sent,
                    'voice_messages': voice_messages,
                    'videos_sent': videos_sent,
                    'files_sent': files_sent,
                    'calls_made': calls_made,
                    'avg_message_length': avg_message_length,
                    'longest_message': longest_message,
                    'total_emojis_used': total_emojis,
                    'top_emojis': top_emojis,
                    'most_active_hour': int(most_active_hour),
                    'activity_by_hour': activity_by_hour,
                    'activity_percentage': round((len(user_data) / len(df_clean)) * 100, 1)
                }
            
            stats['participant_insights'] = participant_insights
            
            # Individual response patterns with detailed analysis
            if 'datetime' in df_clean.columns:
                df_sorted = df_clean.sort_values('datetime')
                user_response_patterns = {}
                
                for user in df_text['name'].unique():
                    response_times = []
                    conversations_started = 0
                    responses_given = 0
                    total_possible_responses = 0
                    night_owl_messages = 0  # Messages sent between 10PM-6AM
                    early_bird_messages = 0  # Messages sent between 5AM-8AM
                    
                    user_messages = df_sorted[df_sorted['name'] == user]
                    
                    # Analyze time-based patterns
                    if 'hour' in user_messages.columns:
                        night_owl_messages = len(user_messages[user_messages['hour'].between(22, 23) | user_messages['hour'].between(0, 5)])
                        early_bird_messages = len(user_messages[user_messages['hour'].between(5, 7)])
                    
                    for i in range(1, len(df_sorted)):
                        prev_msg = df_sorted.iloc[i-1]
                        curr_msg = df_sorted.iloc[i]
                        
                        if curr_msg['name'] == user:
                            # Check if this is a response to someone else
                            if prev_msg['name'] != user:
                                total_possible_responses += 1
                                time_diff = curr_msg['datetime'] - prev_msg['datetime']
                                if time_diff < pd.Timedelta(hours=2):  # Within 2 hours
                                    response_times.append(time_diff.total_seconds() / 60)
                                    responses_given += 1
                            
                            # Check if this is a conversation starter (after long gap)
                            time_diff = curr_msg['datetime'] - prev_msg['datetime']
                            if time_diff > pd.Timedelta(hours=1):
                                conversations_started += 1
                    
                    # Calculate response statistics
                    if response_times:
                        avg_response = round(np.mean(response_times), 1)
                        median_response = round(np.median(response_times), 1)
                        quick_responses = len([t for t in response_times if t < 1])
                        super_quick_responses = len([t for t in response_times if t < 0.1])  # Under 6 seconds
                        slow_responses = len([t for t in response_times if t > 60])  # Over 1 hour
                        fastest_response = round(min(response_times), 2)
                        slowest_response = round(max(response_times), 1)
                        response_rate = round((responses_given / max(total_possible_responses, 1)) * 100, 1)
                    else:
                        avg_response = 0
                        median_response = 0
                        quick_responses = 0
                        super_quick_responses = 0
                        slow_responses = 0
                        fastest_response = 0
                        slowest_response = 0
                        response_rate = 0
                    
                    # Calculate response speed ranking
                    response_speed_score = 0
                    if response_times:
                        # Lower average time = higher score
                        response_speed_score = max(0, 100 - avg_response)
                    
                    user_response_patterns[user] = {
                        'avg_response_time_minutes': avg_response,
                        'median_response_time_minutes': median_response,
                        'quick_responses_count': quick_responses,
                        'super_quick_responses_count': super_quick_responses,
                        'slow_responses_count': slow_responses,
                        'fastest_response_minutes': fastest_response,
                        'slowest_response_minutes': slowest_response,
                        'conversations_started': conversations_started,
                        'response_rate_percentage': response_rate,
                        'response_speed_score': round(response_speed_score, 1),
                        'total_responses_given': responses_given,
                        'night_owl_messages': night_owl_messages,
                        'early_bird_messages': early_bird_messages,
                        'response_style': 'Lightning Fast ⚡' if avg_response < 1 else 'Quick Responder 🏃' if avg_response < 5 else 'Thoughtful Replier 🤔' if avg_response < 30 else 'Takes Time 🐌'
                    }
                
                stats['individual_response_patterns'] = user_response_patterns
            
            # Communication style analysis
            communication_styles = {}
            for user in df_text['name'].unique():
                user_messages = df_text[df_text['name'] == user]['chat']
                
                if len(user_messages) > 0:
                    # Calculate communication metrics
                    short_messages = len([msg for msg in user_messages if len(str(msg)) < 20])
                    medium_messages = len([msg for msg in user_messages if 20 <= len(str(msg)) < 100])
                    long_messages = len([msg for msg in user_messages if len(str(msg)) >= 100])
                    
                    # Question asking frequency
                    questions = len([msg for msg in user_messages if '?' in str(msg)])
                    
                    # Exclamation usage
                    exclamations = len([msg for msg in user_messages if '!' in str(msg)])
                    
                    communication_styles[user] = {
                        'short_messages_percentage': round((short_messages / len(user_messages)) * 100, 1),
                        'medium_messages_percentage': round((medium_messages / len(user_messages)) * 100, 1),
                        'long_messages_percentage': round((long_messages / len(user_messages)) * 100, 1),
                        'questions_asked': questions,
                        'exclamations_used': exclamations,
                        'question_frequency': round((questions / len(user_messages)) * 100, 1),
                        'exclamation_frequency': round((exclamations / len(user_messages)) * 100, 1)
                    }
            
            stats['communication_styles'] = communication_styles
            
            # Personalized fun facts for each participant
            personalized_fun_facts = {}
            
            for user in df_text['name'].unique():
                user_data = df_clean[df_clean['name'] == user]
                user_text = df_text[df_text['name'] == user]
                fun_facts = []
                
                # Message timing facts
                if 'hour' in user_data.columns and len(user_data) > 0:
                    most_active_hour = user_data['hour'].mode().iloc[0] if not user_data['hour'].empty else 0
                    hour_counts = user_data['hour'].value_counts()
                    
                    if most_active_hour >= 22 or most_active_hour <= 5:
                        fun_facts.append(f"🌙 Night Owl - Most active at {most_active_hour}:00")
                    elif most_active_hour >= 5 and most_active_hour <= 8:
                        fun_facts.append(f"🌅 Early Bird - Most active at {most_active_hour}:00")
                    else:
                        fun_facts.append(f"☀️ Day Person - Most active at {most_active_hour}:00")
                
                # Message length personality
                if len(user_text) > 0:
                    avg_length = user_text['chat'].str.len().mean()
                    if avg_length < 20:
                        fun_facts.append("📝 Short & Sweet - Keeps messages concise")
                    elif avg_length > 100:
                        fun_facts.append("📚 Storyteller - Loves detailed messages")
                    else:
                        fun_facts.append("💬 Balanced Writer - Perfect message length")
                
                # Media sharing personality
                photos = len(user_data[user_data['chat'].str.contains(r'\[Photo\]', na=False)])
                stickers = len(user_data[user_data['chat'].str.contains(r'\[Sticker\]', na=False)])
                voice_msgs = len(user_data[user_data['chat'].str.contains(r'\[Voice message\]', na=False)])
                
                total_media = photos + stickers + voice_msgs
                if total_media > len(user_text) * 0.3:  # More than 30% media
                    if stickers > photos and stickers > voice_msgs:
                        fun_facts.append("🎭 Sticker Master - Expresses with stickers")
                    elif photos > stickers and photos > voice_msgs:
                        fun_facts.append("📸 Photo Enthusiast - Loves sharing moments")
                    elif voice_msgs > photos and voice_msgs > stickers:
                        fun_facts.append("🎤 Voice Message Pro - Prefers speaking")
                    else:
                        fun_facts.append("🎨 Media Mixer - Uses all types of content")
                
                # Emoji personality
                user_emojis = []
                for message in user_text['chat']:
                    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
                    emojis = emoji_pattern.findall(str(message))
                    user_emojis.extend(emojis)
                
                emoji_ratio = len(user_emojis) / max(len(user_text), 1)
                if emoji_ratio > 0.5:
                    fun_facts.append("😍 Emoji Enthusiast - Messages full of emojis")
                elif emoji_ratio < 0.1:
                    fun_facts.append("📖 Text Purist - Minimal emoji usage")
                else:
                    fun_facts.append("😊 Emoji Balanced - Uses emojis just right")
                
                # Response behavior
                if user in stats.get('individual_response_patterns', {}):
                    response_data = stats['individual_response_patterns'][user]
                    avg_response = response_data['avg_response_time_minutes']
                    
                    if avg_response < 1:
                        fun_facts.append("⚡ Lightning Responder - Replies in seconds")
                    elif avg_response < 5:
                        fun_facts.append("🏃 Quick Replier - Fast responses")
                    elif avg_response > 60:
                        fun_facts.append("🤔 Thoughtful Responder - Takes time to reply")
                    
                    if response_data.get('conversations_started', 0) > len(df_text['name'].unique()) * 2:
                        fun_facts.append("🎬 Conversation Starter - Often initiates chats")
                
                # Communication style facts
                if user in communication_styles:
                    style = communication_styles[user]
                    if style['question_frequency'] > 20:
                        fun_facts.append("❓ Curious Mind - Asks lots of questions")
                    if style['exclamation_frequency'] > 15:
                        fun_facts.append("🎉 Enthusiastic - Loves exclamation marks")
                
                # Activity level
                total_messages = len(user_data)
                if len(df_clean['name'].unique()) > 1:
                    activity_percentage = (total_messages / len(df_clean)) * 100
                    if activity_percentage > 60:
                        fun_facts.append("💬 Chat Champion - Dominates conversations")
                    elif activity_percentage < 20:
                        fun_facts.append("🎧 Quiet Observer - Listens more than talks")
                    else:
                        fun_facts.append("⚖️ Balanced Participant - Perfect chat balance")
                
                # Ensure we have at least 3-4 fun facts, add generic ones if needed
                if len(fun_facts) < 3:
                    fun_facts.extend([
                        f"📊 Sent {total_messages} total messages",
                        f"🗓️ Been chatting for {(df_clean['datetime'].max() - df_clean['datetime'].min()).days} days"
                    ])
                
                personalized_fun_facts[user] = fun_facts[:5]  # Limit to 5 fun facts per user
            
            stats['personalized_fun_facts'] = personalized_fun_facts
            
            # ADVANCED RELATIONSHIP METRICS
            relationship_metrics = self._analyze_relationships(df_clean, df_text)
            stats['relationship_metrics'] = relationship_metrics
            
            # CONVERSATION DYNAMICS
            conversation_dynamics = self._analyze_conversation_dynamics(df_clean)
            stats['conversation_dynamics'] = conversation_dynamics
            
            # TEMPORAL PATTERNS
            temporal_patterns = self._analyze_temporal_patterns(df_clean)
            stats['temporal_patterns'] = temporal_patterns
            
            # ENGAGEMENT METRICS
            engagement_metrics = self._analyze_engagement(df_clean, df_text)
            stats['engagement_metrics'] = engagement_metrics
            
            return stats
            
        except Exception as e:
            print(f"Advanced stats error: {e}")
            return {}
    
    def _analyze_relationships(self, df_clean, df_text):
        """Analyze relationship dynamics between participants"""
        relationships = {}
        
        try:
            users = df_text['name'].unique()
            
            if len(users) >= 2:
                for i, user1 in enumerate(users):
                    for user2 in users[i+1:]:
                        # Message exchange ratio
                        user1_msgs = len(df_text[df_text['name'] == user1])
                        user2_msgs = len(df_text[df_text['name'] == user2])
                        
                        # Response rate to each other
                        if 'datetime' in df_clean.columns:
                            df_sorted = df_clean.sort_values('datetime')
                            user1_to_user2 = 0
                            user2_to_user1 = 0
                            
                            for i in range(1, min(len(df_sorted), 5000)):
                                prev_msg = df_sorted.iloc[i-1]
                                curr_msg = df_sorted.iloc[i]
                                
                                if prev_msg['name'] == user1 and curr_msg['name'] == user2:
                                    time_diff = curr_msg['datetime'] - prev_msg['datetime']
                                    if time_diff < pd.Timedelta(hours=2):
                                        user1_to_user2 += 1
                                        
                                elif prev_msg['name'] == user2 and curr_msg['name'] == user1:
                                    time_diff = curr_msg['datetime'] - prev_msg['datetime']
                                    if time_diff < pd.Timedelta(hours=2):
                                        user2_to_user1 += 1
                            
                            # Conversation balance score (0-100, 50 is perfect balance)
                            total_exchanges = user1_to_user2 + user2_to_user1
                            if total_exchanges > 0:
                                balance_score = 50 - abs(50 - (user1_to_user2 / total_exchanges * 100))
                            else:
                                balance_score = 0
                            
                            # Mutual responsiveness
                            mutual_response_rate = (user1_to_user2 + user2_to_user1) / max(user1_msgs + user2_msgs, 1) * 100
                            
                            relationships[f"{user1} ↔ {user2}"] = {
                                'message_ratio': f"{user1_msgs}:{user2_msgs}",
                                'balance_score': round(balance_score, 1),
                                'mutual_responses': user1_to_user2 + user2_to_user1,
                                'response_rate': round(mutual_response_rate, 1),
                                'connection_strength': 'Strong 💪' if mutual_response_rate > 40 else 'Good 👍' if mutual_response_rate > 20 else 'Moderate 🤝',
                                f'{user1}_initiates': user1_to_user2,
                                f'{user2}_initiates': user2_to_user1
                            }
            
        except Exception as e:
            print(f"Relationship analysis error: {e}")
        
        return relationships
    
    def _analyze_conversation_dynamics(self, df_clean):
        """Analyze conversation flow and patterns"""
        dynamics = {}
        
        try:
            if 'datetime' in df_clean.columns:
                df_sorted = df_clean.sort_values('datetime')
                
                # Conversation streak analysis
                max_streak = 0
                current_streak = 1
                streak_holder = None
                max_streak_holder = None
                prev_user = None
                
                for _, row in df_sorted.iterrows():
                    if row['name'] == prev_user:
                        current_streak += 1
                        if current_streak > max_streak:
                            max_streak = current_streak
                            max_streak_holder = row['name']
                    else:
                        current_streak = 1
                        prev_user = row['name']
                
                dynamics['longest_streak'] = {
                    'user': max_streak_holder,
                    'count': max_streak
                }
                
                # Average conversation length (messages before long gap)
                conversation_lengths = []
                conv_length = 0
                
                for i in range(1, len(df_sorted)):
                    time_diff = df_sorted.iloc[i]['datetime'] - df_sorted.iloc[i-1]['datetime']
                    if time_diff < pd.Timedelta(hours=1):
                        conv_length += 1
                    else:
                        if conv_length > 0:
                            conversation_lengths.append(conv_length)
                        conv_length = 0
                
                if conversation_lengths:
                    dynamics['avg_conversation_length'] = round(np.mean(conversation_lengths), 1)
                    dynamics['longest_conversation'] = max(conversation_lengths)
                    dynamics['total_conversations'] = len(conversation_lengths)
                
                # Peak activity analysis
                if 'hour' in df_clean.columns:
                    hourly_activity = df_clean.groupby('hour').size()
                    peak_hour = hourly_activity.idxmax()
                    peak_count = hourly_activity.max()
                    quiet_hour = hourly_activity.idxmin()
                    
                    dynamics['peak_activity'] = {
                        'hour': int(peak_hour),
                        'messages': int(peak_count),
                        'quiet_hour': int(quiet_hour)
                    }
        
        except Exception as e:
            print(f"Conversation dynamics error: {e}")
        
        return dynamics
    
    def _analyze_temporal_patterns(self, df_clean):
        """Analyze time-based patterns and trends"""
        patterns = {}
        
        try:
            if 'datetime' in df_clean.columns:
                # Weekend vs weekday analysis
                df_clean['is_weekend'] = df_clean['datetime'].dt.dayofweek.isin([5, 6])
                weekend_msgs = len(df_clean[df_clean['is_weekend']])
                weekday_msgs = len(df_clean[~df_clean['is_weekend']])
                
                patterns['weekend_vs_weekday'] = {
                    'weekend_messages': weekend_msgs,
                    'weekday_messages': weekday_msgs,
                    'weekend_percentage': round((weekend_msgs / len(df_clean)) * 100, 1),
                    'preference': 'Weekend Chattier 🎉' if weekend_msgs > weekday_msgs else 'Weekday Active 💼'
                }
                
                # Monthly trends
                df_clean['month'] = df_clean['datetime'].dt.to_period('M')
                monthly_counts = df_clean.groupby('month').size()
                
                if len(monthly_counts) > 1:
                    # Calculate growth rate
                    first_month = monthly_counts.iloc[0]
                    last_month = monthly_counts.iloc[-1]
                    growth_rate = ((last_month - first_month) / first_month) * 100 if first_month > 0 else 0
                    
                    patterns['chat_trend'] = {
                        'most_active_month': str(monthly_counts.idxmax()),
                        'peak_messages': int(monthly_counts.max()),
                        'growth_rate': round(growth_rate, 1),
                        'trend': 'Growing 📈' if growth_rate > 10 else 'Declining 📉' if growth_rate < -10 else 'Stable 📊'
                    }
                
                # Season analysis (if data spans multiple seasons)
                df_clean['season'] = df_clean['datetime'].dt.month % 12 // 3
                season_map = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'}
                df_clean['season_name'] = df_clean['season'].map(season_map)
                season_counts = df_clean['season_name'].value_counts()
                
                if len(season_counts) > 0:
                    patterns['seasonal_preference'] = {
                        'favorite_season': season_counts.idxmax(),
                        'messages': int(season_counts.max())
                    }
        
        except Exception as e:
            print(f"Temporal patterns error: {e}")
        
        return patterns
    
    def _analyze_engagement(self, df_clean, df_text):
        """Analyze engagement levels and interaction quality"""
        engagement = {}
        
        try:
            users = df_text['name'].unique()
            
            for user in users:
                user_data = df_clean[df_clean['name'] == user]
                user_text = df_text[df_text['name'] == user]
                
                # Calculate engagement score (0-100)
                score = 0
                factors = []
                
                # Message frequency (max 30 points)
                msg_percentage = (len(user_data) / len(df_clean)) * 100
                freq_score = min(msg_percentage * 0.6, 30)
                score += freq_score
                
                # Media sharing (max 20 points)
                photos = len(user_data[user_data['chat'].str.contains(r'\[Photo\]', na=False)])
                stickers = len(user_data[user_data['chat'].str.contains(r'\[Sticker\]', na=False)])
                media_ratio = (photos + stickers) / max(len(user_data), 1)
                media_score = min(media_ratio * 100, 20)
                score += media_score
                
                # Message length diversity (max 20 points)
                if len(user_text) > 0:
                    lengths = user_text['chat'].str.len()
                    length_std = lengths.std()
                    diversity_score = min(length_std / 10, 20)
                    score += diversity_score
                
                # Response activity (max 30 points)
                if 'datetime' in df_clean.columns:
                    df_sorted = df_clean.sort_values('datetime')
                    responses = 0
                    for i in range(1, len(df_sorted)):
                        if df_sorted.iloc[i]['name'] == user and df_sorted.iloc[i-1]['name'] != user:
                            time_diff = df_sorted.iloc[i]['datetime'] - df_sorted.iloc[i-1]['datetime']
                            if time_diff < pd.Timedelta(hours=2):
                                responses += 1
                    
                    response_rate = (responses / max(len(user_data), 1)) * 100
                    response_score = min(response_rate, 30)
                    score += response_score
                
                # Engagement level classification
                if score >= 70:
                    level = 'Super Engaged 🔥'
                elif score >= 50:
                    level = 'Highly Engaged ⭐'
                elif score >= 30:
                    level = 'Moderately Engaged 👌'
                else:
                    level = 'Casual Chatter 💬'
                
                engagement[user] = {
                    'engagement_score': round(score, 1),
                    'engagement_level': level,
                    'message_contribution': round(msg_percentage, 1),
                    'media_activity': round(media_ratio * 100, 1),
                    'interaction_quality': 'High 💎' if score >= 60 else 'Good ✨' if score >= 40 else 'Fair 👍'
                }
        
        except Exception as e:
            print(f"Engagement analysis error: {e}")
        
        return engagement

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
        top_30_words = dict(word_freq.most_common(30))
        
        return jsonify({
            'word_frequencies': top_30_words,
            'total_unique_words': len(word_freq),
            'total_words': sum(word_freq.values())
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_user', methods=['POST'])
def predict_user():
    try:
        data = request.get_json()
        input_text = data.get('text', '').strip()
        
        if not input_text:
            return jsonify({'error': 'No text provided'})
        
        prediction = analyzer.predict_user_from_text(input_text)
        return jsonify(prediction)
    
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
