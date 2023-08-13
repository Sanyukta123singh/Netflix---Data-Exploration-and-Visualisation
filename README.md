**Please go through the collab link provided for output, business insights and recommendations.
**
# Netflix---Data-Exploration-and-Visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Analysing basic metrics

data = pd.read_csv('netflix_data.csv')
data.shape

data.head()

data.info()

data.describe()

data.columns

# Data Cleaning

data.isnull().sum()

data.nunique()

data.duplicated().sum()

data['country'] = data['country'].fillna(data['country'].mode()[0])
data['country'] = data['country'].astype(str)
data['country'] = data['country'].apply(lambda x : x.split(', ')[0])

data['rating'] = data['rating'].replace({'74 min' : np.nan, '84 min' : np.nan, '66 min': np.nan, 'TV-Y7-FV' : 'TV-Y7','NR': 'Unrated','UR': 'Unrated'})
data['rating'].unique()

data = data.dropna(subset = ['duration'])

data['director'].isnull().sum()

data['director'].fillna('Unknown', inplace= True)
data['cast'].fillna('Unknown', inplace= True)

mode_im = ['date_added','rating','duration']
for i in mode_im:
    data[i] = data[i].fillna(data[i].mode()[0])

data['month'] = data['date_added'].apply(lambda x : x.lstrip().split(' ')[0])
data['year'] = data['date_added'].apply(lambda x : x.split(', ')[-1])

#drop useless columns
data.drop(['show_id','date_added','description'],axis=1, inplace= True)

data.isna().sum()

data.head()

# EDA

data['type'].value_counts()

Year_count = data['release_year'].value_counts()
Year_count

sns.lineplot(data = Year_count, x = Year_count.index, y = Year_count.values)
plt.show()

data['listed_in'].value_counts().head(20)

Rating_count = data['rating'].value_counts().head(11)
Rating_count

plt.figure(figsize=(10, 6))
sns.barplot(x=Rating_count.index,y=Rating_count)
plt.show()

plt.figure(figsize=(15, 6))
sns.countplot(data = data,x='rating',hue = 'type')
plt.show()

data['country'].nunique()

data['country'].value_counts().head(20)

top_10_country = data['country'].value_counts().index[:10]
top_10_country

top_10 = data.loc[(data['country'].isin(top_10_country))]
top_10.shape

plt.figure(figsize=(15, 6))
sns.countplot(data = top_10, x='country',hue='type')

cast_df = pd.DataFrame()
cast_df = data['cast'].str.split(',',expand=True).stack()
cast_df = cast_df.to_frame()
cast_df.columns = ['Actor']
actors = cast_df.groupby(['Actor']).size().reset_index(name = 'Total Count')
actors = actors[actors.Actor != 'Unknown']
actors = actors.sort_values(by=['Total Count'], ascending=False)
top5Actors = actors.head()

barChart2 = sns.barplot(top5Actors, x='Total Count', y='Actor')


cast_df = pd.DataFrame()
cast_df = data['director'].str.split(',',expand=True).stack()
cast_df = cast_df.to_frame()
cast_df.columns = ['Directors']
directors = cast_df.groupby(['Directors']).size().reset_index(name = 'Total Count')
directors = directors[directors.Directors != 'Unknown']
directors = directors.sort_values(by=['Total Count'], ascending=False)
top5directors = directors.head()

barChart2 = sns.barplot(top5directors, x='Total Count', y='Directors')


df1 = data[['type', 'release_year']]
df1 = df1.rename(columns = {"release_year":"Release_Year","type":"Type"})
df2 = df1.groupby(['Release_Year','Type']).size().reset_index(name = 'Total_count')
df2

plt.figure(figsize=(15, 6))
graph = sns.lineplot(df2, x = "Release_Year", y="Total_count", hue = "Type")

data['duration']

data['duration_min'] = data[data['type'] == 'Movie']['duration'].str.extract('(\d+)').astype(float)
data['duration_seasons'] = data[data['type'] == 'TV Show']['duration'].str.extract('(\d+)').astype(float)
**fill NaN values in the new columns with 0**
data[['duration_min', 'duration_seasons']] = data[['duration_min', 'duration_seasons']].fillna(0)
data = data.drop('duration', axis = 1)

data['duration_min']
data['duration_seasons']

sns.histplot(data = data[data['type'] == 'Movie'], x = 'duration_min', bins = 30)

sns.countplot(data = data[data['type'] == 'TV Show'], x = 'duration_seasons',color='skyblue')

genre_counts = data['listed_in'].str.split(', ').explode().value_counts()

**Plot the count of genres using a bar plot**
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Count of Genres')
plt.xticks(rotation=45, ha='right')
plt.show()

**END**

