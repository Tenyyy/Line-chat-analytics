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
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pickle
import json
import requests
import deepcut




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

def get_predict(X_test):
    """load and predict value of the text classification"""
    if X_test == '...':
        return '...'
    else:
        filename = 'model.sav'
        MAX_WORDS = 2500
        MAX_SEQUENCE_LENGTH = 10
        loaded_model = pickle.load(open(filename, 'rb'))
        tokenizer = Tokenizer(num_words=MAX_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
        tokenizer.fit_on_texts(df.c.values)
        X_test = [X_test]
        X_test=tokenizer.texts_to_sequences(X_test)
        X_test=pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
        result = loaded_model.predict(X_test)
        who = who_write(result)
    return who


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
    date_list = date_list.str.split(' ', 2, expand=True)
    day_list = day_list.str.split(',', 1, expand=True)
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
    date_list = df['date'].str.split('/', 2, expand=True)
    df['day'] = date_list[0]
    df['month'] = date_list[1]
    df['year'] = date_list[2]
    df['year'] = df['year'].astype('int')
    df['year'] = df['year'] - 543
    df['year'] = df['year'].astype('str')
    df['datetime'] = df['day'] + '/' + df['month'] + '/' + df['year'] + ' ' + df['time']
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['time'] = df['time'].astype('str')
    st.write(df)
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
df = pd.read_csv(text, sep="\t", header=None, names=["time", "name", "chat"])
df = df.iloc[2:] # the first 2 rows of data is unusable
df.reset_index(inplace=True)
df['date'] = create_date(df['time'].values)
df = extract_day_date(df, df['date'])


# create chat in a day feature
chat_day= df[(df.isnull().any(axis=1))&(df.time.str.len() > 10)]
df = create_datetime(df)
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
fig1 = px.bar(df['chat'].value_counts()[:count],title='Count chat', text_auto='s')
fig1.update_traces(marker_color='#ff6961')
fig1.update_layout(template = 'plotly_white')
st.plotly_chart(fig1)



st.header('Top words')
if st.button('Top words (take 1-2 minutes)'):
    try:
        count= st.slider('top words:', min_value=0, max_value=100, step=1, value=20)
        df_word = count_word(df)
        fig2 = px.bar(df_word['words'].value_counts()[1:count],title='Count chat', text_auto='s')
        fig2.update_traces(marker_color='#ff6961')
        fig2.update_layout(template = 'plotly_white')
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


st.header('Number of chat by time')
option = st.selectbox(
     'Group data by: ',
     ('hour','time', 'day', 'dow', 'month', 'year'))
plot_type = st.radio("Average or Cumulative",('Average', 'Cumulative'))
df_plot_weekday = df[df['is_weekday'] == 1]
df_plot_weekend = df[df['is_weekday'] == 0]
grouped_chat_weekday = df_plot_weekday.groupby(option).agg({"name": "count"})
grouped_chat_weekday[option] = grouped_chat_weekday.index
if plot_type  == 'Average':
    grouped_chat_weekday['name'] = grouped_chat_weekday['name']/5
grouped_chat_weekend = df_plot_weekend.groupby(option).agg({"name": "count"})
grouped_chat_weekend[option] = grouped_chat_weekend.index
if plot_type  == 'Average':
    grouped_chat_weekend['name'] = grouped_chat_weekend['name']/2
fig4 = go.Figure()
fig4.add_trace(go.Scatter(y=grouped_chat_weekday['name'].values, x=grouped_chat_weekday[option].values
                              , mode='lines', name='weekday lines'))
fig4.add_trace(go.Scatter(y=grouped_chat_weekend['name'].values, x=grouped_chat_weekend[option].values
                              , mode='lines', name='weekend lines'))
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


st.header('Top chat in a day count')
day_type = st.radio("Weekday or Weekend",('Weekday', 'Weekend'))
if day_type  == 'Weekday':
     chat_day_plot = chat_day[chat_day['is_weekday'] == 1]
else:
     chat_day_plot = chat_day[chat_day['is_weekday'] == 0]
count= st.slider('top chat in a day count:', min_value=0, max_value=100, step=1, value=20)
fig6 = px.bar(chat_day_plot['chat in a day'][:count],title='Chat in a day', text_auto='s')
fig6.update_traces(marker_color='#ff6961')
fig6.update_layout(template = 'plotly_white')
st.plotly_chart(fig6)



pie_chart = df.groupby('dow').agg({"chat":"count"})
fig7 = go.Figure(
    data=[go.Pie(labels=pie_chart.index, values=pie_chart['chat'], hole=0.4, textinfo='label+percent', insidetextorientation='radial')])
st.plotly_chart(fig7)

try:
    st.header('Top count call time')
    count= st.slider('top count call time:', min_value=0, max_value=100, step=1, value=20)
    fig8 = px.bar(call_time['call second'][:count],title='Count call time', text_auto='s')
    fig8.update_traces(marker_color='#ff6961')
    fig8.update_layout(template = 'plotly_white')
    st.plotly_chart(fig8)
except:
    pass

try:
    st.header("Fun fact")
    col1, col2, col3= st.columns(3)

    col1.metric(label="Total chat", value=str(df['chat'].count()))
    col2.metric(label="Top chat by", value=f"{str(df['name'].value_counts().index[0])} : {str(df['name'].value_counts()[0])} ")
    col3.metric(label="Total photos", value = len(df[df['chat']=='[Photo]']))

    col1.metric(label="Total stickers", value = len(df[df['chat']=='[Sticker]']))
    col2.metric(label="Total videos", value=len(df[df['chat'] == '[Video]']))
    col3.metric(label="Total call", value=len(call_time))

    col1.metric(label="Total missed call", value=len(df[df['chat'] == '☎ Missed call']))
    col2.metric(label="Total canceled call", value = len(df[df['chat']=='☎ Canceled call']))
    col3.metric(label="Cumulative call time (second)", value=str(call_time['call second'].sum()))


except:
    pass

mostcommon = wordcloud_value[:100]
wordcloud = WordCloud( font_path='assets/THSarabunNew.ttf', width=1600, height=800, background_color='white',regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",colormap='Set2').generate(str(mostcommon))
wordcloud.to_file('wordcloud.png')
image = Image.open('wordcloud.png')
st.header('Wordcloud of top messages')
st.image(image, caption='Wordcloud of top messages')




#train model
st.header('Train LSTM model to predict word')
if st.button('Train model'):
    from keras.preprocessing.text import Tokenizer
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers import LSTM
    from keras.layers import Embedding
    from keras.layers import SpatialDropout1D
    from keras.models import Sequential
    from keras.layers import Embedding
    train_model_state = st.text('Training model...')
    MAX_WORDS = 2500
    MAX_SEQUENCE_LENGTH = 10
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
    tokenizer.fit_on_texts(df.chat.values)
    word_index = tokenizer.word_index
    X = tokenizer.texts_to_sequences(df.chat.values)

    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    Y = pd.get_dummies(df['name']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 5
    batch_size = 64

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    train_model_state.text('Training model... done!')
    accr = model.evaluate(X_test, Y_test)
    st.write('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    # save the model to disk
    filename = 'model.sav'
    pickle.dump(model, open(filename, 'wb'))
    st.write('model saved!!')
else:
    pass

text = st.text_input('Any words', '...')
try:
    from keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    predicted = get_predict(str(text))
    st.write(f"Predicted writer: {predicted}")
except:
    st.write("Train model first!!")
