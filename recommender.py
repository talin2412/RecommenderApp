import enum
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

st.header('What Movie Did You Watch?')



movies = movies.merge(credits, on='title')

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)
if st.button('Show Recommendation'):
    movies.dropna(inplace=True)


    def convertCat(tagContent):
        tags = []
        for i in ast.literal_eval(tagContent):
            tags.append(i['name'])
        return tags

    def director(tagContent):
        tags = []
        for i in ast.literal_eval(tagContent):
            if i['job'] == 'Director':
                tags.append(i['name'])
                break
        return tags

    def convertCat2(tagContent):
        tags = []
        counter = 0
        for i in ast.literal_eval(tagContent):
            if counter != 3:
                tags.append(i['name'])
                counter += 1
            else:
                break
        return tags

    movies['genres'] = movies['genres'].apply(convertCat)
    movies['keywords'] = movies['keywords'].apply(convertCat)
    movies['cast'] = movies['cast'].apply(convertCat2)
    movies['crew'] = movies['crew'].apply(director)
    movies['overview'] = movies['overview'].apply(lambda x:x.split())

    movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])


    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    movieFinal = movies[['movie_id', 'title','tags']]
    movieFinal['tags'] = movieFinal['tags'].apply(lambda x:" ".join(x))
    movieFinal['tags'] = movieFinal['tags'].apply(lambda x:x.lower())

    #print(cv.get_feature_names())

    ps = PorterStemmer()

    def stem(text):
        word = []
        for i in text.split():
            word.append(ps.stem(i))

        return " ".join(word)

    movieFinal['tags'] = movieFinal['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movieFinal['tags']).toarray()

    similarMovies = cosine_similarity(vectors)

    def recommend(movie):
        idx = movieFinal[movieFinal['title'] == movie].index[0]
        distances = similarMovies[idx]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

        finalList = []
        for i in movies_list:
            finalList.append(movieFinal.iloc[i[0]].title)

        return finalList

    moviesToRec = recommend(selected_movie)
    col1, col2 = st.columns(2)

    with col1:
        st.text(moviesToRec[0] + '\n' + moviesToRec[1] + '\n' + moviesToRec[2] + '\n' + moviesToRec[3] + '\n' + moviesToRec[4])
    