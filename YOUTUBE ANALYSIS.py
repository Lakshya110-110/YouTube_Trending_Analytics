import pandas as pd

#Loading CSV files
us = pd.read_csv("USvideos.csv")
india = pd.read_csv("INvideos.csv")

#Adding Country Labels
us['country'] = 'US'
india['country'] = 'IN'

# Combining both dataset
df  = pd.concat([us, india], ignore_index=True)

#Cleaning columns
df = df[['video_id', 'title', 'views', 'likes', 'dislikes', 'comment_count','publish_time', 'tags',
         'category_id', 'country']]

import json

#Loading category mapping
with open('US_category_id.json') as f:
    cat_data= json.load(f)

categories = {}
for item in cat_data['items']:
    categories[int(item['id'])] = item['snippet']['title']

#Mapping to main DataFrame
df['category_name'] = df['category_id'].map(categories)


#Sentiment on Titles (VADER)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

df['title_sentiment'] = df['title'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

#Comparing Views by Category and Country

df.groupby(['country', 'category_name'])['views'].mean().unstack().T.plot(kind='bar', figsize=(12,6))

#Top 10 Videos per country

top_us = df[df['country'] == 'US'].nlargest(10, 'views')[['title', 'views']]
top_in = df[df['country'] == 'IN'].nlargest(10, 'views')[['title', 'views']]

#Exporting this clean data

df.to_csv('Youtube_US_IN_Cleaned.csv', index=False)

