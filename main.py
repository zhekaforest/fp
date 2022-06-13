import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import people_also_ask
import re
import sqlite3
from bs4 import BeautifulSoup
import requests
from sklearn.linear_model import LinearRegression
import numpy as np

@st.cache
def get_airbnb_data():
    df = pd.read_csv("https://github.com/zhekaforest/fp/raw/main/listings.csv")
    return df

def get_wiki(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features="html.parser")
    text =  soup.find("p")
    to_get_rid = text.find_all("sup") + text.find_all('span') + text.find_all("a")
    for element in to_get_rid:
        text = str(text).replace(str(element), '')
    text = text.replace('(', '').replace(')','').replace('-','').replace('<p>','').replace('<b>','').replace('</b>','').replace('</p>','').replace('  ', ' ')
    return text

def better_text(answer):
    if answer['question']=="What do Airbnb mean?":
        answer = answer['raw_text'].replace('Featured snippet from the web', '')
        answer = re.sub("[(A-Z)][(a-z)]{2} ([\d]|[\d]{2}), [\d]{4}(\w|\W)+", '', answer)
        answer = answer.replace("\n,", "Air Bed and Breakfast", 1)
        answer = answer.replace("1.", '')
    else:
        answer = answer['raw_text'].replace('Featured snippet from the web', '')
        answer = re.sub("[(A-Z)][(a-z)]{2} ([\d]|[\d]{2}), [\d]{4}(\w|\W)+", '', answer)
    return answer

def plot_price(df, maxim):
    plot_price = sns.kdeplot(df.price[df.price<maxim], shade=True)
    fig = plot_price.get_figure()
    plt.xlabel("price")
    st.pyplot(fig)
df = get_airbnb_data()

st.header("Какой-то проект про airbnb")
url = "https://ru.wikipedia.org/wiki/Airbnb"
text_def = get_wiki(url)
st.markdown(text_def)
st.code("from bs4 import BeautifulSoup")
questions = people_also_ask.get_related_questions("airbnb")

question = st.selectbox('Peopla also ask', questions)
answer = people_also_ask.get_answer(question)
answer = better_text(answer)
st.markdown(answer)
st.code('import people_also_ask')

st.markdown("Посмотрим на распределение цен на жилье в Амстердаме в Airbnb")
maxim = st.slider('Ценовой диапозон', min_value=100, max_value=8500, value=1000)
plot_price(df, maxim)

st.markdown("Может, вам понравились цены и хотите снять жилье в Амстердаме?")
price_max_inter = st.slider('За сколько будете снимать - максимальная цена', min_value=0, max_value=200, value=100)
room_type_inter = st.selectbox('Какой тип апартаментов?', ['Entire home/apt', 'Private room', 'Shared room'])
df_for_map = df[df['price']<=price_max_inter]
df_for_map = df_for_map[df_for_map['room_type']==room_type_inter]
st.map(df_for_map)

conn = sqlite3.connect('my_data.db')
c = conn.cursor()
df.to_sql('sql_df', conn, if_exists='append', index = False)
sql_room = c.execute("""
    SELECT AVG(price), room_type FROM sql_df
    GROUP BY room_type
""").fetchall()
sql_room = pd.DataFrame(sql_room, columns=['avg_price', 'room_type'])
st.markdown("Посмотрим на среднюю цену для разных видов апартаментов, которую найдем с помощью SQL")
code_sql = '''
sql_room = c.execute("""
    SELECT AVG(price), room_type FROM sql_df
    GROUP BY room_type
""").fetchall()
'''
price = list(sql_room['avg_price'])
types = list(sql_room['room_type'])

fig, ax = plt.subplots()
ax.bar([1,2,3], price, width=0.5, tick_label=types)

st.code(code_sql, language="python")
st.pyplot(fig)

sql_host = c.execute("""
    SELECT COUNT(DISTINCT host_id), COUNT(host_id) FROM sql_df
""").fetchall()
st.write(sql_host)

fig_host, ax = plt.subplots()
ax.scatter(x=df["calculated_host_listings_count"], y=df["price"])
st.pyplot(fig_host)
model = LinearRegression()
model.fit(df[["price"]], df["number_of_reviews"])

x = pd.DataFrame(dict(Reviews=np.linspace(0, max(df['number_of_reviews']))))
fig_model = plt.subplots()
ax.scatter(df["price"], df["number_of_reviews"])
ax.plot(x["Reviews"], model.predict(x), color="C1", lw=2)

st.pyplot(fig_model)
