import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import re
from io import StringIO
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
from PIL import Image
import pickle
import json
import requests
import deepcut
from textblob import TextBlob
import spacy
from langdetect import detect
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.preprocessing import LabelEncoder
from collections import Counter
# from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Embedding
# from keras.layers import SpatialDropout1D
# from keras.layers import Embedding
import plotly.graph_objects as go
# import download_spacy_models



# Todo


def deEmojify(text):
    """removes emoji, so we can feed the data to pandas"""
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def who_write(array):
    """return the name (string) by selecting the max score from prediction"""
    max_index = np.argmax(array)
    name = df['name'].unique()
    return name[max_index]


# def load_tokenizer_and_model(MAX_WORDS):
#     filename = 'model.sav'
#     with open(filename, 'rb') as file:
#         loaded_model = pickle.load(file)

#     # Load the PyThaiNLP tokenizer's word_index from a JSON file
#     with open('word_index.json', 'r', encoding='utf-8') as f:
#         word_index = json.load(f)

#     # Create a Keras tokenizer with the PyThaiNLP tokenizer's word_index
#     tokenizer = Tokenizer(num_words=MAX_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
#     tokenizer.word_index = word_index

#     return tokenizer, loaded_model


# def get_predict(X_test):
#     """load and predict value of the text classification"""
#     if X_test == '...':
#         return '...'
#     else:
#         MAX_WORDS = 2500
#         MAX_SEQUENCE_LENGTH = 10
#         tokenizer, loaded_model = load_tokenizer_and_model(MAX_WORDS)

#         # Use the custom tokenize_thai function
#         X_test = [' '.join(tokenize_thai(X_test))]
        
#         X_test=tokenizer.texts_to_sequences(X_test)
#         X_test=pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
#         result = loaded_model.predict(X_test)
#         who = who_write(result)
#     return who

def fix_call(x):
    """create a call time feature in second unit"""
    if len(x)<=6:
        split_string = x.split(":", 1)
        substring1 = split_string[0]
        substring2 = split_string[1]
        second = int(substring1)*60 + int(substring2)
    else:
        split_string = x.split(":", 2)
        substring1 = split_string[0]
        substring2 = split_string[1]
        substring3 = split_string[2]
        second = int(substring1)*60*60 + int(substring2)*60 + int(substring3)
    return second

def get_language(text):
    try:
        lang = detect(text)
    except Exception as e:
        lang = "unknown"
        st.warning(f"Error detecting language: {e}")
    return lang

def get_sentiment(text, language):
    if language == "en":
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
    else:
        sentiment = None
    return sentiment

def get_keywords(text, language):
    if language == "en":
        doc = nlp_en(text)
    else:
        doc = nlp_multilingual(text)
    
    keywords = [token.lemma_ for token in doc if token.is_stop == False and token.is_punct == False]
    return keywords

# Tokenize the text using PyThaiNLP's 'newmm' tokenizer
def tokenize_thai(text):
    return word_tokenize(text, engine='newmm')

def create_date(list):
    """create date column"""
    result = []
    latest_date = ""
    for i in list:
        if i.find("BE") != -1:
            latest_date = str(i)
            result.append(str(i))
        else:
            result.append(latest_date)
    return result

def extract_day_date(df_, list):
    """extract date feature from string and create is_weekend feature"""
    date_list = list
    day_list = list
    date_list = date_list.str.split(' ', n=2, expand=True)
    day_list = day_list.str.split(',', n=1, expand=True)
    try:
        df_['date'] = date_list[1]
        df_['dow'] = day_list[0]
    except:
        df_['date'] = date_list
        df_['dow'] = day_list
        df_['date'] = df_['date'].str[1]
        df_['dow'] = df_['dow'].str[0]
    df_['is_weekday'] = df_['dow'].apply(lambda x: 1 if x in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] else 0)
    return df_

def create_datetime(df):
    """change Buddhist year to Christian year and change the column type to be pd.datetime"""
    df = df[(df.time.str.len() < 10)]
    date_list = df['date'].str.split('/', n=2, expand=True)
    df['day'] = date_list[0]
    df['month'] = date_list[1]
    df['year'] = date_list[2]
    df['year'] = df['year'].astype('int')
    if df['year'].max() > 2500:
        df['year'] = df['year'] - 543
    df['year'] = df['year'].astype('str')
    df['datetime'] = df['day'] + '/' + df['month'] + '/' + df['year'] + ' ' + df['time']
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['time'] = df['time'].astype('str')
    return df

def convert_by_to_cy(df):
    """change Buddhist year to Christian year and change the column type to be pd.datetime"""
    df = df[(df.date.str.len() == 10)]
    date_list = df['date'].str.split('/', n=2, expand=True)
    df['day'] = date_list[0]
    df['month'] = date_list[1]
    df['year'] = date_list[2]
    df['year'] = df['year'].astype('int')
    df['year'] = df['year'] - 543
    df['year'] = df['year'].astype('str')
    df['datetime'] = df['day'] + '/' + df['month'] + '/' + df['year']
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    return df

@st.cache
def count_word(df):
    """count number of unique words"""
    result = []
    df_ = df.copy()
    df_['chat'] = df_['chat'].astype('str')
    for i in df_['chat']:
        result.extend(i)
    words = deepcut.tokenize(result)
    df_word = pd.DataFrame(words,columns=['words'])
    return df_word

@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

# start streamlit web application
st.header('Upload line chat file (.txt)')
uploaded_file = st.file_uploader("Upload line chat file (.txt)")
if uploaded_file is None:
    st.stop()
if uploaded_file is not None:
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        st.write('upload complete')
    except:
        print('Upload line chat.txt file only!!')


# personal chat only
text = StringIO(deEmojify(string_data))
df = pd.read_csv(text, sep="$", header=None, names=["time", "name", "chat"])
df = df.iloc[2:] # the first 2 rows of data is unusable
df.reset_index(inplace=True)
df['date'] = create_date(df['time'].values)
df = extract_day_date(df, df['date'])


# create chat in a day feature
chat_day= df[(df.isnull().any(axis=1))&(df.time.str.len() > 10)]
df = create_datetime(df)
st.write(df)

csv = convert_df(df)

st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)

# Sentiment Analysis
st.header('Sentiment Analysis (for English chat only)')
if st.button('Sentiment Analysis'):
    try:
        # Load language models
        nlp_en = spacy.load("en_core_web_sm")
        nlp_multilingual = spacy.load("xx_ent_wiki_sm")

        # Load your DataFrame (replace with your actual DataFrame)
        chat_type = ['[Photo]','[Sticker]', '[Video]', '☎ Missed call', '☎ Canceled call', '[File]', '[Voice message]', '[Location]', '[Contact]', '[GIF]', '[Link]', '[☎ No answer]']
        # Get language, sentiment, and keywords for each message
        df['language'] = df['chat'].apply(get_language)
        df['sentiment'] = df.apply(lambda row: get_sentiment(row['chat'], row['language']), axis=1)
        df['sentiment_cleaned'] = np.where((~df['chat'].isin(chat_type))&(df['language']=='en'), True, False)
        df['keywords'] = df.apply(lambda row: get_keywords(row['chat'], row['language']), axis=1)

        def visualize_sentiment(df):
            # Remove rows with missing sentiment values (i.e., non-English text)
            df_filtered = df.dropna(subset=['sentiment'])
            df_filtered = df_filtered[df_filtered['sentiment_cleaned']==True]

            # Create a histogram of sentiment scores
            fig = px.histogram(df_filtered, x='sentiment', color='name', nbins=50, title='Sentiment Score Distribution')

            # Set the x-axis range to cover the entire sentiment range
            fig.update_xaxes(range=[-1, 1])
            fig.update_layout(barmode='group', xaxis_title='Sentiment Score', yaxis_title='Count')

            return fig

        # Streamlit app
        st.title("Sentiment Analysis Visualization")
        st.write("This part visualizes the sentiment scores of text messages")
        st.write("A value of -1 indicates a highly negative sentiment")
        st.write("A value of 0 indicates a neutral sentiment")
        st.write("A value of 1 indicates a highly positive sentiment")
        

        # Visualize the sentiment distribution
        fig = visualize_sentiment(df)
        st.plotly_chart(fig)

        unique_name = df['name'].unique()
        for i in unique_name:
            i = str(i)
            if i == 'nan':
                pass
            else:
                average_sentiment = round(df[df['name']==i]['sentiment'].mean(),2)
                st.write('Average sentiment score of', i, 'is', average_sentiment)
    except:
        pass
else:
    pass



chat_day.drop(columns=['name','chat'],inplace=True)
temp =  chat_day['index'].iloc[1:].values
chat_day = chat_day[:-1]
chat_day['diff_chat'] = temp
chat_day['chat in a day'] = chat_day['diff_chat'] - chat_day['index']
chat_day = chat_day.set_index('time')
chat_day = chat_day.sort_values(by = 'chat in a day', ascending = False)
chat_day = extract_day_date(chat_day, chat_day.index)

# call time
try:
    df2 = df.copy()
    df2.dropna(inplace=True)
    call_time = df2[df2["chat"].str.contains("☎ Call time")]
    call_time = call_time[call_time['chat'].str.len() < 30]
    df['chat'] = df['chat'].astype('str')
    new = call_time["chat"].str.split("e", n = 1, expand = True)
    call_time["call time"]= new[1]
    call_time['call second'] = call_time['call time'].apply(lambda x: fix_call(x))
    call_time = call_time.sort_values(by='call second', ascending=False)
    call_time.reset_index(inplace=True)

except: # for chat that never call
    pass


# for wordcloud
wordcloud_value = df['chat'].value_counts().index.tolist()


# plot graph
st.header('Top messages')
count= st.slider('top message:', min_value=0, max_value=100, step=1, value=20)
df = df[df['chat'] != 'nan']
df_temp = df[df['chat'].isin((df['chat'].value_counts()[:count].index.tolist()))]
fig1 = px.histogram(df_temp,x='chat',title='Count chat', text_auto='s',color='name').update_xaxes(categoryorder="total descending")
fig1.update_layout(template = 'plotly_white')
st.plotly_chart(fig1)

# Most frequently used words
st.header('Top words')
if st.button('Top words (may take a while depending on the number of chats)'):
    try:
        count= st.slider('top words:', min_value=0, max_value=100, step=1, value=20)
        words = " ".join(df["chat"]).lower()
        tokenized_words = word_tokenize(words, keep_whitespace=False)
        word_count = Counter(tokenized_words)
        most_common_words = word_count.most_common(count)
        word_df = pd.DataFrame(most_common_words, columns=["word", "count"])
        fig2 = px.bar(word_df, x="word", y="count", title="Most Frequently Used Words")
        fig2.update_traces(marker_color='#ff6961')
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2)
    except:
        pass
else:
    pass

st.header('Top time')
count= st.slider('top time:', min_value=0, max_value=100, step=1, value=20)
fig3 = px.bar(df['time'].value_counts()[:count],title='Count time', text_auto='s')
fig3.update_traces(marker_color='#ff6961')
fig3.update_layout(template = 'plotly_white')
st.plotly_chart(fig3)


# st.header('Number of chat by time')
# option = st.selectbox(
#      'Group data by: ',
#      ('hour','time', 'day', 'dow', 'month', 'year'))
# plot_type = st.radio("Average or Cumulative",('Average', 'Cumulative'))
# df_plot_weekday = df[df['is_weekday'] == 1]
# df_plot_weekend = df[df['is_weekday'] == 0]
# grouped_chat_weekday = df_plot_weekday.groupby(option).agg({"name": "count"})
# grouped_chat_weekday[option] = grouped_chat_weekday.index
# if plot_type  == 'Average':
#     grouped_chat_weekday['name'] = grouped_chat_weekday['name']/5
# grouped_chat_weekend = df_plot_weekend.groupby(option).agg({"name": "count"})
# grouped_chat_weekend[option] = grouped_chat_weekend.index
# if plot_type  == 'Average':
#     grouped_chat_weekend['name'] = grouped_chat_weekend['name']/2
# fig4 = go.Figure()


# fig4.add_trace(go.Scatter(y=grouped_chat_weekday['name'].values, x=grouped_chat_weekday[option].values
#                               , mode='lines', name='weekday lines'))
# fig4.add_trace(go.Scatter(y=grouped_chat_weekend['name'].values, x=grouped_chat_weekend[option].values
#                               , mode='lines', name='weekend lines'))
# fig4.update_xaxes(rangeslider_visible=True)
# fig4.update_layout(template='plotly_white')
# st.plotly_chart(fig4)



st.header('Number of chat by time')
option = st.selectbox(
     'Group data by: ',
     ('hour', 'day', 'dow', 'month', 'year'))  # Removed 'time' as it might not be suitable for grouping
plot_type = st.radio("Average or Cumulative", ('Average', 'Cumulative'))

df_plot_weekday = df[df['is_weekday'] == 1]
df_plot_weekend = df[df['is_weekday'] == 0]

# Grouping and calculating counts
grouped_chat_weekday = df_plot_weekday.groupby(option).agg({"name": "count"}).reset_index()
grouped_chat_weekend = df_plot_weekend.groupby(option).agg({"name": "count"}).reset_index()

if plot_type == 'Average':
    # Handling Zero Division
    grouped_chat_weekday['name'] = grouped_chat_weekday['name'] / 5 if len(df_plot_weekday) > 0 else 0
    grouped_chat_weekend['name'] = grouped_chat_weekend['name'] / 2 if len(df_plot_weekend) > 0 else 0
elif plot_type == 'Cumulative':
    # Cumulative sum for 'name' column
    grouped_chat_weekday['name'] = grouped_chat_weekday['name'].cumsum()
    grouped_chat_weekend['name'] = grouped_chat_weekend['name'].cumsum()

fig4 = go.Figure()

# Adding traces for weekday and weekend lines
fig4.add_trace(go.Scatter(y=grouped_chat_weekday['name'], x=grouped_chat_weekday[option],
                          mode='lines', name='weekday lines'))
fig4.add_trace(go.Scatter(y=grouped_chat_weekend['name'], x=grouped_chat_weekend[option],
                          mode='lines', name='weekend lines'))

fig4.update_xaxes(rangeslider_visible=True)
fig4.update_layout(template='plotly_white')
st.plotly_chart(fig4)


bar_type = st.radio("Choose bar type",('stack', 'group', 'overlay', 'relative'))
fig5 = go.Figure(data=[
    go.Bar(name='weekday', y=grouped_chat_weekday['name'].values, x=grouped_chat_weekday[option].values
           , text=grouped_chat_weekday['name'].values),
    go.Bar(name='weekend', y=grouped_chat_weekend['name'].values, x=grouped_chat_weekend[option].values
           , text=grouped_chat_weekend['name'].values)
])
fig5.update_layout(barmode=bar_type)
fig5.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig5.update_layout(template = 'plotly_white')
st.plotly_chart(fig5)


st.header('Chat type by time')
chat_type = ['[Photo]','[Sticker]', '[Video]', '☎ Missed call', '☎ Canceled call', '[File]', '[Voice message]', '[Location]', '[Contact]', '[GIF]', '[Link]', '[☎ No answer]']
fig6 = go.Figure()
for i in chat_type:
    grouped_chat= df[df['chat'] == i]
    grouped_chat= grouped_chat.groupby(option).agg({"name": "count"})
    grouped_chat[option] = grouped_chat.index
    if plot_type == 'Average':
        grouped_chat['name'] = grouped_chat['name'] / 7
    fig6.add_trace(go.Scatter(y=grouped_chat['name'].values, x=grouped_chat[option].values
                              , mode='lines', name=f'{i}'))
fig6.update_xaxes(rangeslider_visible=True)
fig6.update_layout(template='plotly_white')
st.plotly_chart(fig6)


st.header('Top chat in a day count')
day_type = st.radio("Weekday or Weekend",('Weekday', 'Weekend'))
if day_type  == 'Weekday':
     chat_day_plot = chat_day[chat_day['is_weekday'] == 1]
else:
     chat_day_plot = chat_day[chat_day['is_weekday'] == 0]
count= st.slider('top chat in a day count:', min_value=0, max_value=100, step=1, value=20)
fig7 = px.bar(chat_day_plot['chat in a day'][:count],title='Chat in a day', text_auto='s', text=chat_day_plot['chat in a day'][:count])
fig7.update_traces(marker_color='#ff6961')
fig7.update_layout(template = 'plotly_white')
st.plotly_chart(fig7)


df = df[df['dow'].str.len() < 10]
pie_chart = df.groupby('dow').agg({"chat":"count"})
fig8 = go.Figure(
    data=[go.Pie(labels=pie_chart.index, values=pie_chart['chat'], hole=0.4, textinfo='label+percent', insidetextorientation='radial')])
st.plotly_chart(fig8)

chat_day = convert_by_to_cy(chat_day)

chat_day['date'] = pd.to_datetime(chat_day['datetime'], format='%Y-%m-%d')
st.header('Chat in a day')
chat_day = chat_day.sort_values(by='date')
fig14 = px.bar(chat_day, x='date', y='chat in a day', text='chat in a day')
st.write(fig14)

st.header('Cumulative chat in a day')
chat_day['cum chat in a day'] = chat_day['chat in a day'].cumsum()


fig15 = px.bar(chat_day, x='date', y='cum chat in a day', text='cum chat in a day')
st.write(fig15)


st.header('Top chat in a day count')
fig13 = px.bar(df.groupby('dow').agg({"chat":"count"}).sort_values(by='chat', ascending=False).reset_index().head(10), x='dow', y='chat', text='chat')
st.plotly_chart(fig13)

try:
    st.write(call_time)
    st.header('Top call time (second)')
    count= st.slider('top count call time:', min_value=0, max_value=100, step=1, value=20)
    fig9 = px.bar(y=call_time['call second'][:count], x=call_time['date'][:count],title='Count call time', text_auto='s')
    fig9.update_traces(marker_color='#ff6961')
    fig9.update_layout(template = 'plotly_white')
    st.plotly_chart(fig9)

    call_time = convert_by_to_cy(call_time)
    call_time['date'] = pd.to_datetime(call_time['datetime'], format='%Y-%m-%d')
    call_time = call_time.sort_values(by='date')
    st.header('Call time (second)')
    fig16 = px.bar(call_time, x='date', y='call second', text='call second')
    st.write(fig16)
    st.header('Cumulative call time (second)')

    call_time['call second'] = call_time['call second'].astype('int64')
    call_time_agg = call_time.groupby('datetime')['call second'].sum().reset_index()

    call_time_agg['cum call second'] = call_time_agg['call second'].cumsum()
    fig17 = px.bar(call_time_agg, x='datetime', y='cum call second', text='cum call second')
    st.write(fig17) 
    
    # call time วันที่ไม่ตรงกันกับวันที่คอล
    st.header('who initiate call')
    fig18 = px.histogram(call_time, x='name')
    st.write(fig18)
  
except:
    pass




# Distribution of chat messages per user
st.header("Number of Messages per User")
fig10 = px.histogram(df, x="name", title="Number of Messages per User", text_auto="s", nbins=len(df["name"].unique()), color="name")
fig10.update_layout(template="plotly_white")
st.plotly_chart(fig10)


# Average message length per user
st.header("Average Message Length per User")
df["msg_length"] = df["chat"].apply(len)
average_msg_length = df.groupby("name")["msg_length"].mean().reset_index()
fig11 = px.bar(average_msg_length, x="name", y="msg_length", title="Average Message Length per User")
fig11.update_layout(template="plotly_white")
fig11.update_traces(marker_color='#ff6961')
st.plotly_chart(fig11)

# Heatmap of message activity
for name in df["name"].unique():
    if type(name) == float:
        break
    st.header(f"Message Activity Heatmap for {name}")
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day_name()
    heatmap_data = df[df['name']==name].groupby(["day", "hour"]).size().reset_index(name="count")
    heatmap_data = heatmap_data.pivot(index="day", columns="hour", values="count")
    fig12 = px.imshow(heatmap_data, title="Message Activity Heatmap")
    fig12.update_layout(template="plotly_white")
    st.plotly_chart(fig12)


try:
    st.header("Fun fact")
    stopwords = thai_stopwords()

    # Longest message
    df['message_length'] = df['chat'].apply(lambda x: len(x))
    longest_message = df['message_length'].max()

    # Most common words
    all_words = [word for text in df['chat'].apply(tokenize_thai) for word in text if word not in stopwords and not (word.startswith('[') or word.endswith(']') or (word == 'Sticker') or (word == 'Photo') or (word == ' ') or (word == ':'))]
    word_counter = Counter(all_words)
    top_n_words = 1
    most_common_words = word_counter.most_common(top_n_words)

    # Average message length
    average_message_length = df['message_length'].mean()

    col1, col2, col3= st.columns(3)

    col1.metric(label="Total chat", value=str(df['chat'].count()))
    col2.metric(label="Top chat by", value=f"{str(df['name'].value_counts().index[0])} : {str(df['name'].value_counts()[0])} ")
    col3.metric(label="Total photos", value = len(df[df['chat']=='[Photo]']))

    col1.metric(label="Longest message", value=longest_message)
    col2.metric(label="Top {} most common words".format(top_n_words), value=', '.join([f"{word[0]}: {word[1]}" for word in most_common_words]))
    col3.metric(label="Average message length", value="{:.2f}".format(average_message_length))

    col1.metric(label="Total stickers", value = len(df[df['chat']=='[Sticker]']))
    col2.metric(label="Total videos", value=len(df[df['chat'] == '[Video]']))
    col3.metric(label="Total call", value=len(call_time))

    col1.metric(label="Total missed call", value=len(df[df['chat'] == '☎ Missed call']))
    col2.metric(label="Total canceled call", value = len(df[df['chat']=='☎ Canceled call']))
    col3.metric(label="Cumulative call time (second)", value=str(call_time['call second'].sum()))

    col1.metric(label="Total File", value=len(df[df['chat'] == '[File]']))
    col2.metric(label="Total Voice message", value = len(df[df['chat']=='[Voice message]']))
    col3.metric(label="Total Location", value=len(df[df['chat'] == '[Location]']))

    col1.metric(label="Total Contact", value=len(df[df['chat'] == '[Contact]']))
    col2.metric(label="Total GIF", value=len(df[df['chat'] == '[GIF]']))
    col3.metric(label="Total Link", value=len(df[df['chat'] == '[Link]']))

except:
    pass

mostcommon = wordcloud_value[:100]
wordcloud = WordCloud( font_path='assets/THSarabunNew.ttf', width=1600, height=800, background_color='white',regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",colormap='Set2').generate(str(mostcommon))
wordcloud.to_file('wordcloud.png')
image = Image.open('wordcloud.png')
st.header('Wordcloud of top messages')
st.image(image, caption='Wordcloud of top messages')



#train model
# st.header('Train LSTM model to predict word')
# if st.button('Train model'):
#     train_model_state = st.text('Training model...')
#     MAX_WORDS = 2500
#     MAX_SEQUENCE_LENGTH = 10
#     EMBEDDING_DIM = 100

#     tokenizer = Tokenizer(num_words=MAX_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
#     texts = [' '.join(tokenize_thai(text)) for text in df.chat.values]
#     tokenizer.fit_on_texts(texts)
#     word_index = tokenizer.word_index
#     X = tokenizer.texts_to_sequences(texts)

#     X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
#     Y = pd.get_dummies(df['name']).values
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

#     model = Sequential()
#     model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
#     model.add(SpatialDropout1D(0.2))
#     model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#     epochs = 5
#     batch_size = 64

#     history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
#     train_model_state.text('Training model... done!')
#     accr = model.evaluate(X_test, Y_test)
#     st.write('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

#     # Save the PyThaiNLP tokenizer's word_index to a JSON file
#     with open('word_index.json', 'w', encoding='utf-8') as f:
#         json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=4)

#     # save the model to disk
#     filename = 'model.sav'
#     pickle.dump(model, open(filename, 'wb'))
#     st.write('model saved!!')
# else:
#     pass

text = st.text_input('Any words', '...')


# predicted = get_predict(str(text))
# st.write(f"Predicted writer: {predicted}")