import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import re
from io import StringIO
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pickle





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


def get_predict_wandb(X_test,model):
    """this fuction load and predict value of the tect classification"""
    if X_test == '...':
        return '...'
    else:
        X_test = [X_test]
        X_test=tokenizer.texts_to_sequences(X_test)
        X_test=pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
        result = model.predict(X_test)
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

st.header('Upload csv file')
uploaded_file = st.file_uploader("Upload csv file")
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
# st.write(null.head())

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

st.header('Top time')
count= st.slider('top time:', min_value=0, max_value=100, step=1, value=20)
fig6 = px.bar(df['a'].value_counts()[:count],title='Count time', text_auto='s')
fig6.update_traces(marker_color='#ff6961')
fig6.update_layout(template = 'plotly_white')
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
count= st.slider('top chat in a day count:', min_value=0, max_value=100, step=1, value=20)
fig12 = px.bar(null['chat in a day'][:count],title='Chat in a day', text_auto='s')
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
    from keras.preprocessing.sequence import pad_sequences
    predicted = get_predict(str(text))
    st.write(f"Predicted writer: {predicted}")
except:
    st.write("Train model first!!")
