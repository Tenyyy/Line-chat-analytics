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
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def who_write(array):
    """this fuction return the name (string) by selecting the max score from prediction"""
    max_index = np.argmax(array)
    name = df['b'].unique()
    return name[max_index]

def get_predict(X_test):
    """this fuction load and predict value of the text classification"""
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

def create_date_time(list):
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
    date_list = list
    day_list = list
    date_list = date_list.str.split(' ', 2, expand=True)
    day_list = day_list.str.split(',', 1, expand=True)
    try:
        df_['date'] = date_list[1]
        df_['day'] = day_list[0]
    except:
        df_['date'] = date_list
        df_['day'] = day_list
        df_['date'] = df_['date'].str[1]
        df_['day'] = df_['day'].str[0]
    df_['is_weekday'] = df_['day'].apply(lambda x: 1 if x in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] else 0)
    return df_

@st.cache
def count_word(df):
    result = []
    df_ = df.copy()
    df_['c'] = df_['c'].astype('str')
    for i in df_['c']:
        result.extend(i)
    words = deepcut.tokenize(result)
    df_word = pd.DataFrame(words,columns=['words'])
    return df_word

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



# personal chat
text = StringIO(deEmojify(string_data))
df = pd.read_csv(text, sep="\t", header=None, names=["a", "b", "c"])
df = df.iloc[2:]
df.reset_index(inplace=True, drop=True)
# st.write(df.head())
df.reset_index(inplace=True)
df['date'] = create_date_time(df['a'].values)
df = extract_day_date(df, df['date'])
null = df[(df.isnull().any(axis=1))&(df.a.str.len() > 10)]
null.drop(columns=['b','c'],inplace=True)
# st.write(null.head())

temp =  null['index'].iloc[1:].values
null = null[:-1]
null['diff_chat'] = temp
# st.write(null.head())

null['chat in a day'] = null['diff_chat'] - null['index']
null = null.set_index('a')
null = null.sort_values(by = 'chat in a day', ascending = False)
null = extract_day_date(null, null.index)

# call time

sd = df[df["a"].str.contains("BE")]
# st.write(sd.head())
# st.write(sd.shape)

try:
    df2 = df.copy()
    df2.dropna(inplace=True)
    dc = df2[df2["c"].str.contains("☎ Call time")]
    dc = dc[dc['c'].str.len() < 30]
    df['c'] = df['c'].astype('str')

    new = dc["c"].str.split("e", n = 1, expand = True)
    dc["call time"]= new[1]

    dc['call second'] = dc['call time'].apply(lambda x: fix_call(x))
    dc = dc.sort_values(by='call second', ascending=False)
    dc.reset_index(inplace=True)
except:
    pass


# count time

value = df['c'].value_counts().index.tolist()


# plot graph

st.header('Top messages')
count= st.slider('top message:', min_value=0, max_value=100, step=1, value=20)
df = df[df['c'] != 'nan']
fig1 = px.bar(df['c'].value_counts()[:count],title='Count chat', text_auto='s')
fig1.update_traces(marker_color='#ff6961')
fig1.update_layout(template = 'plotly_white')
st.plotly_chart(fig1)



st.header('Top words')
if st.button('Top words (take 1-2 minutes)'):
    try:
        count= st.slider('top words:', min_value=0, max_value=100, step=1, value=20)
        df_word = count_word(df)
        fig1 = px.bar(df_word['words'].value_counts()[1:count],title='Count chat', text_auto='s')
        fig1.update_traces(marker_color='#ff6961')
        fig1.update_layout(template = 'plotly_white')
        st.plotly_chart(fig1)
    except:
        pass
else:
    pass

st.header('Top time')
count= st.slider('top time:', min_value=0, max_value=100, step=1, value=20)
fig6 = px.bar(df['a'].value_counts()[:count],title='Count time', text_auto='s')
fig6.update_traces(marker_color='#ff6961')
fig6.update_layout(template = 'plotly_white')
st.plotly_chart(fig6)


st.header('Chat by time')
plot_type = st.radio("Average or Cumulative",('Average', 'Cumulative'))

df_plot_weekday = df[df['is_weekday'] == 1]
df_plot_weekend = df[df['is_weekday'] == 0]
grouped_chat_weekday = df_plot_weekday.groupby('a').agg({"b": "count"})
grouped_chat_weekday['time'] = grouped_chat_weekday.index
if plot_type  == 'Average':
    grouped_chat_weekday['b'] = grouped_chat_weekday['b']/5
grouped_chat_weekend = df_plot_weekend.groupby('a').agg({"b": "count"})
grouped_chat_weekend['time'] = grouped_chat_weekend.index
if plot_type  == 'Average':
    grouped_chat_weekend['b'] = grouped_chat_weekend['b']/2
fig6 = go.Figure()
fig6.add_trace(go.Scatter(y=grouped_chat_weekday['b'].values, x=grouped_chat_weekday['time'].values
                              , mode='lines', name='weekday lines'))
fig6.add_trace(go.Scatter(y=grouped_chat_weekend['b'].values, x=grouped_chat_weekend['time'].values
                              , mode='lines', name='weekend lines'))
fig6.update_xaxes(rangeslider_visible=True)
fig6.update_layout(template='plotly_white')
st.plotly_chart(fig6)

try:
    st.header('Top count call time')
    count= st.slider('top count call time:', min_value=0, max_value=100, step=1, value=20)
    fig7 = px.bar(dc['call second'][:count],title='Count call time', text_auto='s')
    fig7.update_traces(marker_color='#ff6961')
    fig7.update_layout(template = 'plotly_white')
    st.plotly_chart(fig7)
except:
    pass

st.header('Top chat in a day count')
day_type = st.radio("Weekday or Weekend",('Weekday', 'Weekend'))
if day_type  == 'Weekday':
     null_plot = null[null['is_weekday'] == 1]
else:
     null_plot = null[null['is_weekday'] == 0]
count= st.slider('top chat in a day count:', min_value=0, max_value=100, step=1, value=20)
fig12 = px.bar(null_plot['chat in a day'][:count],title='Chat in a day', text_auto='s')
fig12.update_traces(marker_color='#ff6961')
fig12.update_layout(template = 'plotly_white')
st.plotly_chart(fig12)
df = df[df['b'].notna()]

try:
    st.header("Fun fact")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Cumulative call time (second)", value=str(dc['call second'].sum()))
    col2.metric(label="Total chat", value=str(df['b'].count()))
    col3.metric(label="Top chat by", value=f"{str(df['b'].value_counts().index[0])} : {str(df['b'].value_counts()[0])} ")
except:
    pass

mostcommon = value[:100]
wordcloud = WordCloud( font_path='assets/THSarabunNew.ttf', width=1600, height=800, background_color='white',regexp=r"[\u0E00-\u0E7Fa-zA-Z']+",colormap='Set2').generate(str(mostcommon))
wordcloud.to_file('wordcloud.png')
image = Image.open('wordcloud.png')
st.header('Wordcloud of top words')
st.image(image, caption='Wordcloud of top words')




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
    tokenizer.fit_on_texts(df.c.values)
    word_index = tokenizer.word_index
    X = tokenizer.texts_to_sequences(df.c.values)

    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    Y = pd.get_dummies(df['b']).values
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
