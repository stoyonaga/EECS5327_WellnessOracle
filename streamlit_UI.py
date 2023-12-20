import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib
import requests
from bs4 import BeautifulSoup
import time
import asyncio
from playwright.sync_api import sync_playwright
import threading
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
import streamlit as st
import re
from selenium.webdriver.common.keys import Keys
from sklearn.linear_model import LogisticRegression

# Load model if there is a saved model, or train a new model
@st.cache_resource
def load_model(model_name):
    messages = pd.read_csv('./Suicide_Detection.csv', engine='python', encoding='utf-8', on_bad_lines='skip')
    messages.dropna(axis=0, inplace=True)
    messages.info()
    print("Dimension of the Dataset: " + str(messages.shape))
    X = messages['text']
    y = messages['class']
    print("Y counts:\n" + str(y.value_counts()))
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=0)

    if(model_name == "Random Forest"):
        model_filename = model_name + ".pkl"
        try:
            model = joblib.load(model_filename)
            print("Loaded pre-trained "+ model_name + " model")
        except FileNotFoundError:
            print("Training model started")
            model = Pipeline(steps=[
                ('text', TfidfVectorizer()),
                ('df', RandomForestClassifier(
                    n_estimators=500,
                    max_depth=125,
                    min_samples_leaf=5,
                    n_jobs=-1
                ))
            ])
            model.fit(train_X, train_y)
            joblib.dump(model, model_filename)
            print(f"Model saved to {model_filename}")
    elif(model_name == "Naive Bayes"):
        model_filename = model_name + ".pkl"
        try:
            model = joblib.load(model_filename)
            print("Loaded pre-trained "+ model_name + " model")
        except FileNotFoundError:
            print("Training model started")
            base_nb = Pipeline(steps=[
                ('text1', TfidfVectorizer()),
                ('rf', MultinomialNB())
            ])
            base_nb.fit(train_X, train_y)
            param_grid = {
                'rf__alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
            }
            grid_search = GridSearchCV(base_nb, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(train_X, train_y)
            model = grid_search.best_estimator_
            joblib.dump(model, model_filename)
            print(f"Model saved to {model_filename}")
    elif(model_name == "Logistic Regression"):
        model_filename = model_name + ".pkl"
        try:
            model = joblib.load(model_filename)
            print("Loaded pre-trained "+ model_name + " model")
        except FileNotFoundError:
            print("Training model started")
            base_lr = Pipeline(steps=[
                ('text5', TfidfVectorizer()),
                ('lr', LogisticRegression(max_iter=1000))  
            ])
            base_lr.fit(train_X, train_y)
            model = base_lr
            joblib.dump(model, model_filename)
            print(f"Model saved to {model_filename}")

    return model

def scrape_reddit(subreddit_id, topic_classifier, number_of_comments, model):
    url = "https://www.reddit.com/r/{}/{}".format(subreddit_id, topic_classifier)

    with st.spinner("Requesting information (json file) from {}...".format(url)):
        headers = {
            'User-Agent': 'shogz-bot'
        }
        response = requests.get(url + ".json", headers=headers)

    if response.ok:
        data = response.json()['data']
        reddit_title = []
        reddit_text = []
        reddit_post_classification = []
        for post in data['children']:
            reddit_title.append(post['data']['title'])
            reddit_text.append(post['data']['selftext'])
            reddit_post_classification.append(model.predict([post['data']['selftext']])[0])

        st.write("Number of scraped posted: {}".format(len(reddit_title)))
        
        for i in range(number_of_comments):
            result = model.predict([reddit_text[i]])[0]
            st.divider()
            if result == "non-suicide":
                st.subheader(result)
            else:
                st.subheader(':red[{}]'.format(result))
            st.text_area("Post Title: {}\n".format(reddit_title[i]), reddit_text[i])
    else:
        st.write('Error {}'.format(response.status_code))

def init_chrome_webdriver(headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox") 
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def quora_scraper_selenium(link, number_of_long_posts, min_post_length_in_chars, loading_time):
    driver = init_chrome_webdriver(headless=True) 
    driver.get(link)
    time.sleep(loading_time) 

    answers = []
    for i in range(number_of_long_posts):
        try:
            continue_reading_buttons = driver.find_elements(By.XPATH, "//button[text()='Continue Reading']")
            if i < len(continue_reading_buttons):
                continue_reading_buttons[i].click()
                time.sleep(1) 
        except NoSuchElementException:
            break

    posts = driver.find_elements(By.CLASS_NAME, "q-box.spacing_log_answer_content")
    for post in posts:
        text = post.text.strip()
        if len(text) > min_post_length_in_chars:
            answers.append(text)

    driver.quit()
    return answers


if 'reddit_open' not in st.session_state:
    st.session_state.reddit_open = False

if 'quora_open' not in st.session_state:
    st.session_state.quora_open = False

st.title('WellnessOracle')

st.text('Mental Health Classifier')
st.text("1. Choose a classifier model.")
model_name = st.radio("Model Type", ["Random Forest", "Naive Bayes", "Logistic Regression"])

st.text("2. Choose a website for scraping.")
col1, col2 = st.columns([1,8])
with col1:
    if st.button('Reddit', type="primary"):
        st.session_state.quora_open = False
        st.session_state.reddit_open = True

with col2:
    if st.button('Quora', type="primary"):
        st.session_state.reddit_open = False
        st.session_state.quora_open = True

if (st.session_state.reddit_open):
    subreddit_id = st.text_input('Subreddit id', 'depression')
    topic_classifier = st.radio("Topic classifier", ["top", "hot", "new"])
    number_of_comments = st.slider('Number of comments', 0, 30, 10)

    if st.button("Scrape"):
        model = load_model(model_name)
        scrape_reddit(subreddit_id, topic_classifier, number_of_comments, model)

if (st.session_state.quora_open):
    link = st.text_input('Quora Link', 'https://www.quora.com/What-is-real-meaning-of-derivative')
    number_of_long_posts = st.slider('Number of Long Posts', 1, 20, 10)
    min_post_length_in_chars = st.slider('Minimum Post Length in Characters', 10, 200, 50)
    loading_time = st.slider('Loading Time (Seconds)', 1, 10, 2)

    if st.button("Scrape"):
        model = load_model(model_name)
        answers = quora_scraper_selenium(link, number_of_long_posts, min_post_length_in_chars, loading_time)
        for answer in answers:
            result = model.predict([answer])[0]
            st.divider()
            if result == "non-suicide":
                st.subheader(result)
            else:
                st.subheader(':red[{}]'.format(result))
            st.text_area("Post Content", value=answer)
