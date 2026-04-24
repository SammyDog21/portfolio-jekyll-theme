---
layout: post
title: "Jupyter notebook example - my capstone"
---

# What makes a Steam Game successful? Can we predict the positive recommendation percentage from those features? #

When making a game, you want it to be considered a success and make money so what features are the most important when it comes to creating a successful game? What should the price be? What genre should it be? Is it better to have a shorter or longer game? These are all factors that should be considered, but might have different importance and weight behind them. Once these features have been determined, can the positive recommendation percentage be predicted using this data. This would allow developers to have a way to determine whether the game idea has a good chance of being successful or not.

## Importing, Merging, Cleaning and Scaling Data ##

First, we need to import the datasets being used, combine them and clean them up where necessary.


```python
#importing important libraries
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
```


```python
#importing the main dataset and seeing what is inside
steam = pd.read_csv(r"C:\Users\josep\Documents\Data analysis course\Capstone\steam_games_dataset.csv")
steam.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 17 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   appid            10000 non-null  int64  
     1   name             10000 non-null  object 
     2   developer        9967 non-null   object 
     3   publisher        9946 non-null   object 
     4   score_rank       5 non-null      float64
     5   positive         10000 non-null  int64  
     6   negative         10000 non-null  int64  
     7   userscore        10000 non-null  int64  
     8   owners           10000 non-null  object 
     9   average_forever  10000 non-null  int64  
     10  average_2weeks   10000 non-null  int64  
     11  median_forever   10000 non-null  int64  
     12  median_2weeks    10000 non-null  int64  
     13  price            10000 non-null  int64  
     14  initialprice     10000 non-null  int64  
     15  discount         10000 non-null  int64  
     16  ccu              10000 non-null  int64  
    dtypes: float64(1), int64(12), object(4)
    memory usage: 1.3+ MB
    


```python
steam.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>appid</th>
      <th>name</th>
      <th>developer</th>
      <th>publisher</th>
      <th>score_rank</th>
      <th>positive</th>
      <th>negative</th>
      <th>userscore</th>
      <th>owners</th>
      <th>average_forever</th>
      <th>average_2weeks</th>
      <th>median_forever</th>
      <th>median_2weeks</th>
      <th>price</th>
      <th>initialprice</th>
      <th>discount</th>
      <th>ccu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>730</td>
      <td>Counter-Strike: Global Offensive</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>NaN</td>
      <td>7642084</td>
      <td>1173003</td>
      <td>0</td>
      <td>100,000,000 .. 200,000,000</td>
      <td>33852</td>
      <td>708</td>
      <td>6645</td>
      <td>301</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1013936</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1172470</td>
      <td>Apex Legends</td>
      <td>Respawn</td>
      <td>Electronic Arts</td>
      <td>NaN</td>
      <td>668053</td>
      <td>326926</td>
      <td>0</td>
      <td>100,000,000 .. 200,000,000</td>
      <td>10506</td>
      <td>496</td>
      <td>935</td>
      <td>246</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>124262</td>
    </tr>
    <tr>
      <th>2</th>
      <td>578080</td>
      <td>PUBG: BATTLEGROUNDS</td>
      <td>PUBG Corporation</td>
      <td>KRAFTON, Inc.</td>
      <td>NaN</td>
      <td>1520457</td>
      <td>1037487</td>
      <td>0</td>
      <td>100,000,000 .. 200,000,000</td>
      <td>23165</td>
      <td>717</td>
      <td>5622</td>
      <td>261</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>314682</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1623730</td>
      <td>Palworld</td>
      <td>Pocketpair</td>
      <td>Pocketpair</td>
      <td>NaN</td>
      <td>358266</td>
      <td>22443</td>
      <td>0</td>
      <td>50,000,000 .. 100,000,000</td>
      <td>3854</td>
      <td>835</td>
      <td>2213</td>
      <td>257</td>
      <td>2999</td>
      <td>2999</td>
      <td>0</td>
      <td>18028</td>
    </tr>
    <tr>
      <th>4</th>
      <td>440</td>
      <td>Team Fortress 2</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>NaN</td>
      <td>1044264</td>
      <td>117208</td>
      <td>0</td>
      <td>50,000,000 .. 100,000,000</td>
      <td>21244</td>
      <td>736</td>
      <td>4262</td>
      <td>102</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>43819</td>
    </tr>
  </tbody>
</table>
</div>



From these we can see that not all the games in this dataset have a developer or publisher, almost none have a score_rank and although the metacritic user score is filled in for all rows, most seem to just be 0 which is not correct. The score_rank in this case will not be considered for the investigation. These other columns can have their missing data input/replaced by introducing another dataset which also includes platform/OS compatability and two more smaller datasets which include the main genre(s) for the games. Due to all of these datasets being collected using Steam's API, they can all be joined with the appid column.


```python
#importing the 3 new datasets
games = pd.read_csv(r"C:\Users\josep\Documents\Data analysis course\Capstone\applications.csv")


games.info()
games.head()
```

    C:\Users\josep\AppData\Local\Temp\ipykernel_18200\1990862462.py:2: DtypeWarning: Columns (5) have mixed types. Specify dtype option on import or set low_memory=False.
      games = pd.read_csv(r"C:\Users\josep\Documents\Data analysis course\Capstone\applications.csv")
    

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 239664 entries, 0 to 239663
    Data columns (total 30 columns):
     #   Column                 Non-Null Count   Dtype  
    ---  ------                 --------------   -----  
     0   appid                  239664 non-null  int64  
     1   name                   239660 non-null  object 
     2   type                   236997 non-null  object 
     3   is_free                239664 non-null  bool   
     4   release_date           202809 non-null  object 
     5   required_age           239664 non-null  object 
     6   short_description      224171 non-null  object 
     7   supported_languages    222006 non-null  object 
     8   header_image           239664 non-null  object 
     9   background             239664 non-null  object 
     10  metacritic_score       5299 non-null    float64
     11  recommendations_total  22560 non-null   float64
     12  mat_supports_windows   239664 non-null  bool   
     13  mat_supports_mac       239664 non-null  bool   
     14  mat_supports_linux     239664 non-null  bool   
     15  mat_initial_price      144923 non-null  float64
     16  mat_final_price        144923 non-null  float64
     17  mat_discount_percent   144923 non-null  float64
     18  mat_currency           144923 non-null  object 
     19  mat_achievement_count  61401 non-null   float64
     20  mat_pc_os_min          195874 non-null  object 
     21  mat_pc_processor_min   183242 non-null  object 
     22  mat_pc_memory_min      181274 non-null  object 
     23  mat_pc_graphics_min    169242 non-null  object 
     24  mat_pc_os_rec          101890 non-null  object 
     25  mat_pc_processor_rec   97036 non-null   object 
     26  mat_pc_memory_rec      96825 non-null   object 
     27  mat_pc_graphics_rec    92047 non-null   object 
     28  created_at             239664 non-null  object 
     29  updated_at             239664 non-null  object 
    dtypes: bool(4), float64(6), int64(1), object(19)
    memory usage: 48.5+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>appid</th>
      <th>name</th>
      <th>type</th>
      <th>is_free</th>
      <th>release_date</th>
      <th>required_age</th>
      <th>short_description</th>
      <th>supported_languages</th>
      <th>header_image</th>
      <th>background</th>
      <th>...</th>
      <th>mat_pc_os_min</th>
      <th>mat_pc_processor_min</th>
      <th>mat_pc_memory_min</th>
      <th>mat_pc_graphics_min</th>
      <th>mat_pc_os_rec</th>
      <th>mat_pc_processor_rec</th>
      <th>mat_pc_memory_rec</th>
      <th>mat_pc_graphics_rec</th>
      <th>created_at</th>
      <th>updated_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>Counter-Strike</td>
      <td>game</td>
      <td>False</td>
      <td>2000-11-01</td>
      <td>0</td>
      <td>Play the world's number 1 online action game. ...</td>
      <td>English&lt;strong&gt;*&lt;/strong&gt;, French&lt;strong&gt;*&lt;/st...</td>
      <td>https://shared.akamai.steamstatic.com/store_it...</td>
      <td>https://store.akamai.steamstatic.com/images/st...</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2025-09-07 16:27:12.218587+00:00</td>
      <td>2025-09-29 02:01:37.107239+00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>Team Fortress Classic</td>
      <td>game</td>
      <td>False</td>
      <td>1999-04-01</td>
      <td>0</td>
      <td>One of the most popular online action games of...</td>
      <td>English, French, German, Italian, Spanish - Sp...</td>
      <td>https://shared.akamai.steamstatic.com/store_it...</td>
      <td>https://store.akamai.steamstatic.com/images/st...</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2025-09-07 16:27:12.218587+00:00</td>
      <td>2025-09-29 02:01:37.107239+00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>Day of Defeat</td>
      <td>game</td>
      <td>False</td>
      <td>2003-05-01</td>
      <td>0</td>
      <td>Enlist in an intense brand of Axis vs. Allied ...</td>
      <td>English, French, German, Italian, Spanish - Spain</td>
      <td>https://shared.akamai.steamstatic.com/store_it...</td>
      <td>https://store.akamai.steamstatic.com/images/st...</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2025-09-07 16:27:12.218587+00:00</td>
      <td>2025-09-29 02:01:37.107239+00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>Deathmatch Classic</td>
      <td>game</td>
      <td>False</td>
      <td>2001-06-01</td>
      <td>0</td>
      <td>Enjoy fast-paced multiplayer gaming with Death...</td>
      <td>English, French, German, Italian, Spanish - Sp...</td>
      <td>https://shared.akamai.steamstatic.com/store_it...</td>
      <td>https://store.akamai.steamstatic.com/images/st...</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2025-09-07 16:27:12.218587+00:00</td>
      <td>2025-09-29 02:01:37.107239+00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>Half-Life: Opposing Force</td>
      <td>game</td>
      <td>False</td>
      <td>1999-11-01</td>
      <td>0</td>
      <td>Return to the Black Mesa Research Facility as ...</td>
      <td>English, French, German, Korean</td>
      <td>https://shared.akamai.steamstatic.com/store_it...</td>
      <td>https://store.akamai.steamstatic.com/images/st...</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2025-09-07 16:27:12.218587+00:00</td>
      <td>2025-09-29 02:01:37.107239+00:00</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



Here we can see this dataset is much, much larger than the previous one and contains all the games in the original dataset. This is helpful because it contains the metacritic scores for a majority of the games along with new features like OS compatability and achievement count. We can then easily combine these datasets with pandas.


```python
#merging only necessary columns from the second dataset
combined = pd.merge(steam, games[["appid", "metacritic_score", "mat_supports_windows", "mat_supports_mac", 
                                  "mat_supports_linux", "mat_achievement_count"]], on="appid", how="left")
combined.info()
combined.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 22 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   appid                  10000 non-null  int64  
     1   name                   10000 non-null  object 
     2   developer              9967 non-null   object 
     3   publisher              9946 non-null   object 
     4   score_rank             5 non-null      float64
     5   positive               10000 non-null  int64  
     6   negative               10000 non-null  int64  
     7   userscore              10000 non-null  int64  
     8   owners                 10000 non-null  object 
     9   average_forever        10000 non-null  int64  
     10  average_2weeks         10000 non-null  int64  
     11  median_forever         10000 non-null  int64  
     12  median_2weeks          10000 non-null  int64  
     13  price                  10000 non-null  int64  
     14  initialprice           10000 non-null  int64  
     15  discount               10000 non-null  int64  
     16  ccu                    10000 non-null  int64  
     17  metacritic_score       2621 non-null   float64
     18  mat_supports_windows   9418 non-null   object 
     19  mat_supports_mac       9418 non-null   object 
     20  mat_supports_linux     9418 non-null   object 
     21  mat_achievement_count  6891 non-null   float64
    dtypes: float64(3), int64(12), object(7)
    memory usage: 1.7+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>appid</th>
      <th>name</th>
      <th>developer</th>
      <th>publisher</th>
      <th>score_rank</th>
      <th>positive</th>
      <th>negative</th>
      <th>userscore</th>
      <th>owners</th>
      <th>average_forever</th>
      <th>...</th>
      <th>median_2weeks</th>
      <th>price</th>
      <th>initialprice</th>
      <th>discount</th>
      <th>ccu</th>
      <th>metacritic_score</th>
      <th>mat_supports_windows</th>
      <th>mat_supports_mac</th>
      <th>mat_supports_linux</th>
      <th>mat_achievement_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>730</td>
      <td>Counter-Strike: Global Offensive</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>NaN</td>
      <td>7642084</td>
      <td>1173003</td>
      <td>0</td>
      <td>100,000,000 .. 200,000,000</td>
      <td>33852</td>
      <td>...</td>
      <td>301</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1013936</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1172470</td>
      <td>Apex Legends</td>
      <td>Respawn</td>
      <td>Electronic Arts</td>
      <td>NaN</td>
      <td>668053</td>
      <td>326926</td>
      <td>0</td>
      <td>100,000,000 .. 200,000,000</td>
      <td>10506</td>
      <td>...</td>
      <td>246</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>124262</td>
      <td>88.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>578080</td>
      <td>PUBG: BATTLEGROUNDS</td>
      <td>PUBG Corporation</td>
      <td>KRAFTON, Inc.</td>
      <td>NaN</td>
      <td>1520457</td>
      <td>1037487</td>
      <td>0</td>
      <td>100,000,000 .. 200,000,000</td>
      <td>23165</td>
      <td>...</td>
      <td>261</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>314682</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1623730</td>
      <td>Palworld</td>
      <td>Pocketpair</td>
      <td>Pocketpair</td>
      <td>NaN</td>
      <td>358266</td>
      <td>22443</td>
      <td>0</td>
      <td>50,000,000 .. 100,000,000</td>
      <td>3854</td>
      <td>...</td>
      <td>257</td>
      <td>2999</td>
      <td>2999</td>
      <td>0</td>
      <td>18028</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>440</td>
      <td>Team Fortress 2</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>NaN</td>
      <td>1044264</td>
      <td>117208</td>
      <td>0</td>
      <td>50,000,000 .. 100,000,000</td>
      <td>21244</td>
      <td>...</td>
      <td>102</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>43819</td>
      <td>92.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>520.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



Next, we need to obtain the genres. The dataset above doesn't have the genres but they are contained in separate tables from the same dataset as the large set used before.


```python
#Obtaining the genres for all the games.
genres = pd.read_csv(r"C:\Users\josep\Documents\Data analysis course\Capstone\application_genres.csv")
genreKey = games = pd.read_csv(r"C:\Users\josep\Documents\Data analysis course\Capstone\genres.csv")

genres.info()
genres.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 587515 entries, 0 to 587514
    Data columns (total 2 columns):
     #   Column    Non-Null Count   Dtype
    ---  ------    --------------   -----
     0   appid     587515 non-null  int64
     1   genre_id  587515 non-null  int64
    dtypes: int64(2)
    memory usage: 9.0 MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>appid</th>
      <th>genre_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>122</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>122</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>122</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>122</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>122</td>
    </tr>
  </tbody>
</table>
</div>




```python
#need to set the id column as the index here for easier coding later on
genreKey.set_index("id", inplace=True)
genreKey.info()
genreKey.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 154 entries, 1 to 167
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   name    154 non-null    object
    dtypes: object(1)
    memory usage: 2.4+ KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MMO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Indie</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Strategy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Strategy</td>
    </tr>
  </tbody>
</table>
</div>



There are multiple with the same name because the genre was originally in another language but I translated them all into english. Additionally, some games have multiple genres. I think the best way to incorporate this into the dataset is to have each genre as a boolean value with each game being set to true or false if it contains that genre. Only genres with at least 1 game involved will be considered.


```python
#only include the games in the combined dataset
genres = pd.merge(genres, combined["appid"], how="inner")
genres.info()
genres.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 26544 entries, 0 to 26543
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype
    ---  ------    --------------  -----
     0   appid     26544 non-null  int64
     1   genre_id  26544 non-null  int64
    dtypes: int64(2)
    memory usage: 414.9 KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>appid</th>
      <th>genre_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>122</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>122</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>122</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>122</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>122</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check each row in genres
#get the id and the genre
#if new genre create new column and set default to False
#if it contains this genre change to true   

for i in range(len(genres)):
    genre = genreKey.loc[genres.iloc[i,1], "name"]
    id = genres.iloc[i, 0]
    if genre not in combined.columns:
        combined[genre] = False
    combined.loc[combined["appid"] == id, genre] = True

combined.info()
combined.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 50 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   appid                  10000 non-null  int64  
     1   name                   10000 non-null  object 
     2   developer              9967 non-null   object 
     3   publisher              9946 non-null   object 
     4   score_rank             5 non-null      float64
     5   positive               10000 non-null  int64  
     6   negative               10000 non-null  int64  
     7   userscore              10000 non-null  int64  
     8   owners                 10000 non-null  object 
     9   average_forever        10000 non-null  int64  
     10  average_2weeks         10000 non-null  int64  
     11  median_forever         10000 non-null  int64  
     12  median_2weeks          10000 non-null  int64  
     13  price                  10000 non-null  int64  
     14  initialprice           10000 non-null  int64  
     15  discount               10000 non-null  int64  
     16  ccu                    10000 non-null  int64  
     17  metacritic_score       2621 non-null   float64
     18  mat_supports_windows   9418 non-null   object 
     19  mat_supports_mac       9418 non-null   object 
     20  mat_supports_linux     9418 non-null   object 
     21  mat_achievement_count  6891 non-null   float64
     22  Action                 10000 non-null  bool   
     23  Free To Play           10000 non-null  bool   
     24  Adventure              10000 non-null  bool   
     25  Strategy               10000 non-null  bool   
     26  Indie                  10000 non-null  bool   
     27  RPG                    10000 non-null  bool   
     28  Animation & Modeling   10000 non-null  bool   
     29  Video Production       10000 non-null  bool   
     30  Casual                 10000 non-null  bool   
     31  Simulation             10000 non-null  bool   
     32  Racing                 10000 non-null  bool   
     33  MMO                    10000 non-null  bool   
     34  Nudity                 10000 non-null  bool   
     35  Violent                10000 non-null  bool   
     36  Sports                 10000 non-null  bool   
     37  Early Access           10000 non-null  bool   
     38  Gore                   10000 non-null  bool   
     39  Web Publishing         10000 non-null  bool   
     40  Design & Illustration  10000 non-null  bool   
     41  Game Development       10000 non-null  bool   
     42  Education              10000 non-null  bool   
     43  Software Training      10000 non-null  bool   
     44  Utilities              10000 non-null  bool   
     45  Audio Production       10000 non-null  bool   
     46  Photo Editing          10000 non-null  bool   
     47  Movie                  10000 non-null  bool   
     48  Sexual Content         10000 non-null  bool   
     49  Accounting             10000 non-null  bool   
    dtypes: bool(28), float64(3), int64(12), object(7)
    memory usage: 1.9+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>appid</th>
      <th>name</th>
      <th>developer</th>
      <th>publisher</th>
      <th>score_rank</th>
      <th>positive</th>
      <th>negative</th>
      <th>userscore</th>
      <th>owners</th>
      <th>average_forever</th>
      <th>...</th>
      <th>Design &amp; Illustration</th>
      <th>Game Development</th>
      <th>Education</th>
      <th>Software Training</th>
      <th>Utilities</th>
      <th>Audio Production</th>
      <th>Photo Editing</th>
      <th>Movie</th>
      <th>Sexual Content</th>
      <th>Accounting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>730</td>
      <td>Counter-Strike: Global Offensive</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>NaN</td>
      <td>7642084</td>
      <td>1173003</td>
      <td>0</td>
      <td>100,000,000 .. 200,000,000</td>
      <td>33852</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1172470</td>
      <td>Apex Legends</td>
      <td>Respawn</td>
      <td>Electronic Arts</td>
      <td>NaN</td>
      <td>668053</td>
      <td>326926</td>
      <td>0</td>
      <td>100,000,000 .. 200,000,000</td>
      <td>10506</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>578080</td>
      <td>PUBG: BATTLEGROUNDS</td>
      <td>PUBG Corporation</td>
      <td>KRAFTON, Inc.</td>
      <td>NaN</td>
      <td>1520457</td>
      <td>1037487</td>
      <td>0</td>
      <td>100,000,000 .. 200,000,000</td>
      <td>23165</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1623730</td>
      <td>Palworld</td>
      <td>Pocketpair</td>
      <td>Pocketpair</td>
      <td>NaN</td>
      <td>358266</td>
      <td>22443</td>
      <td>0</td>
      <td>50,000,000 .. 100,000,000</td>
      <td>3854</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>440</td>
      <td>Team Fortress 2</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>NaN</td>
      <td>1044264</td>
      <td>117208</td>
      <td>0</td>
      <td>50,000,000 .. 100,000,000</td>
      <td>21244</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



There is likely a more efficient way to write the code above, but it works. Each genre is given its own column and is set to true or false depending on whether the game is considered that genre. This way is better than listing the genres for each game because some games are considered multiple genres.

Next, we need to remove any columns that won't be helpful, and add the review score column which calculates the positive percentage of the game.


```python
#creating a new dataframe without the columns not needed for analysis and calculating the score
final = combined.drop(columns=["appid", "score_rank", "userscore", "owners", "price", "discount", "metacritic_score", "positive", "negative"])
final["score"] = combined["positive"]/(combined["positive"]+combined["negative"])*100
final.info()
final.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 42 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   name                   10000 non-null  object 
     1   developer              9967 non-null   object 
     2   publisher              9946 non-null   object 
     3   average_forever        10000 non-null  int64  
     4   average_2weeks         10000 non-null  int64  
     5   median_forever         10000 non-null  int64  
     6   median_2weeks          10000 non-null  int64  
     7   initialprice           10000 non-null  int64  
     8   ccu                    10000 non-null  int64  
     9   mat_supports_windows   9418 non-null   object 
     10  mat_supports_mac       9418 non-null   object 
     11  mat_supports_linux     9418 non-null   object 
     12  mat_achievement_count  6891 non-null   float64
     13  Action                 10000 non-null  bool   
     14  Free To Play           10000 non-null  bool   
     15  Adventure              10000 non-null  bool   
     16  Strategy               10000 non-null  bool   
     17  Indie                  10000 non-null  bool   
     18  RPG                    10000 non-null  bool   
     19  Animation & Modeling   10000 non-null  bool   
     20  Video Production       10000 non-null  bool   
     21  Casual                 10000 non-null  bool   
     22  Simulation             10000 non-null  bool   
     23  Racing                 10000 non-null  bool   
     24  MMO                    10000 non-null  bool   
     25  Nudity                 10000 non-null  bool   
     26  Violent                10000 non-null  bool   
     27  Sports                 10000 non-null  bool   
     28  Early Access           10000 non-null  bool   
     29  Gore                   10000 non-null  bool   
     30  Web Publishing         10000 non-null  bool   
     31  Design & Illustration  10000 non-null  bool   
     32  Game Development       10000 non-null  bool   
     33  Education              10000 non-null  bool   
     34  Software Training      10000 non-null  bool   
     35  Utilities              10000 non-null  bool   
     36  Audio Production       10000 non-null  bool   
     37  Photo Editing          10000 non-null  bool   
     38  Movie                  10000 non-null  bool   
     39  Sexual Content         10000 non-null  bool   
     40  Accounting             10000 non-null  bool   
     41  score                  9988 non-null   float64
    dtypes: bool(28), float64(2), int64(6), object(6)
    memory usage: 1.3+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>developer</th>
      <th>publisher</th>
      <th>average_forever</th>
      <th>average_2weeks</th>
      <th>median_forever</th>
      <th>median_2weeks</th>
      <th>initialprice</th>
      <th>ccu</th>
      <th>mat_supports_windows</th>
      <th>...</th>
      <th>Game Development</th>
      <th>Education</th>
      <th>Software Training</th>
      <th>Utilities</th>
      <th>Audio Production</th>
      <th>Photo Editing</th>
      <th>Movie</th>
      <th>Sexual Content</th>
      <th>Accounting</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Counter-Strike: Global Offensive</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>33852</td>
      <td>708</td>
      <td>6645</td>
      <td>301</td>
      <td>0</td>
      <td>1013936</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>86.693234</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apex Legends</td>
      <td>Respawn</td>
      <td>Electronic Arts</td>
      <td>10506</td>
      <td>496</td>
      <td>935</td>
      <td>246</td>
      <td>0</td>
      <td>124262</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>67.142422</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PUBG: BATTLEGROUNDS</td>
      <td>PUBG Corporation</td>
      <td>KRAFTON, Inc.</td>
      <td>23165</td>
      <td>717</td>
      <td>5622</td>
      <td>261</td>
      <td>0</td>
      <td>314682</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>59.440590</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Palworld</td>
      <td>Pocketpair</td>
      <td>Pocketpair</td>
      <td>3854</td>
      <td>835</td>
      <td>2213</td>
      <td>257</td>
      <td>2999</td>
      <td>18028</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>94.104946</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Team Fortress 2</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>21244</td>
      <td>736</td>
      <td>4262</td>
      <td>102</td>
      <td>0</td>
      <td>43819</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>89.908668</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>



The id and name aren't helpful for analysis, but name hasn't been removed because it is still helpful to know what the game is in the desciptive analysis section. Additionally, score rank and metacritic score have been remvoed because there is not enough data for analysis. Current price and discount have also been removed because when considering a new game, it will not be discounted immediately so the initial price is the important feature to consider. Number of owners could be considered useful, but with the review score being the target column, and owners being a range, it will not be considerd in this investigation. Finally, the positive and negative columns have been removed because they were used to calculate the score.

There are still a few more tasks of cleaning to do before this dataset is ready.
The NaN values need to be addressed either by removing them or inputting values.
Everything is a numeric value or boolean apart from developer and publisher which need to be given numeric values with meaning or removed too.

First, we need to deal with the NaN values.


```python
#filling na values in the achievements column with 0
final.fillna(value={"mat_achievement_count":0}, inplace=True)

#drop all remaining na values because they can't be filled.
final.dropna(inplace=True)
final.info()
final.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 9338 entries, 0 to 9999
    Data columns (total 42 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   name                   9338 non-null   object 
     1   developer              9338 non-null   object 
     2   publisher              9338 non-null   object 
     3   average_forever        9338 non-null   int64  
     4   average_2weeks         9338 non-null   int64  
     5   median_forever         9338 non-null   int64  
     6   median_2weeks          9338 non-null   int64  
     7   initialprice           9338 non-null   int64  
     8   ccu                    9338 non-null   int64  
     9   mat_supports_windows   9338 non-null   object 
     10  mat_supports_mac       9338 non-null   object 
     11  mat_supports_linux     9338 non-null   object 
     12  mat_achievement_count  9338 non-null   float64
     13  Action                 9338 non-null   bool   
     14  Free To Play           9338 non-null   bool   
     15  Adventure              9338 non-null   bool   
     16  Strategy               9338 non-null   bool   
     17  Indie                  9338 non-null   bool   
     18  RPG                    9338 non-null   bool   
     19  Animation & Modeling   9338 non-null   bool   
     20  Video Production       9338 non-null   bool   
     21  Casual                 9338 non-null   bool   
     22  Simulation             9338 non-null   bool   
     23  Racing                 9338 non-null   bool   
     24  MMO                    9338 non-null   bool   
     25  Nudity                 9338 non-null   bool   
     26  Violent                9338 non-null   bool   
     27  Sports                 9338 non-null   bool   
     28  Early Access           9338 non-null   bool   
     29  Gore                   9338 non-null   bool   
     30  Web Publishing         9338 non-null   bool   
     31  Design & Illustration  9338 non-null   bool   
     32  Game Development       9338 non-null   bool   
     33  Education              9338 non-null   bool   
     34  Software Training      9338 non-null   bool   
     35  Utilities              9338 non-null   bool   
     36  Audio Production       9338 non-null   bool   
     37  Photo Editing          9338 non-null   bool   
     38  Movie                  9338 non-null   bool   
     39  Sexual Content         9338 non-null   bool   
     40  Accounting             9338 non-null   bool   
     41  score                  9338 non-null   float64
    dtypes: bool(28), float64(2), int64(6), object(6)
    memory usage: 1.3+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>developer</th>
      <th>publisher</th>
      <th>average_forever</th>
      <th>average_2weeks</th>
      <th>median_forever</th>
      <th>median_2weeks</th>
      <th>initialprice</th>
      <th>ccu</th>
      <th>mat_supports_windows</th>
      <th>...</th>
      <th>Game Development</th>
      <th>Education</th>
      <th>Software Training</th>
      <th>Utilities</th>
      <th>Audio Production</th>
      <th>Photo Editing</th>
      <th>Movie</th>
      <th>Sexual Content</th>
      <th>Accounting</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Counter-Strike: Global Offensive</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>33852</td>
      <td>708</td>
      <td>6645</td>
      <td>301</td>
      <td>0</td>
      <td>1013936</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>86.693234</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apex Legends</td>
      <td>Respawn</td>
      <td>Electronic Arts</td>
      <td>10506</td>
      <td>496</td>
      <td>935</td>
      <td>246</td>
      <td>0</td>
      <td>124262</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>67.142422</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PUBG: BATTLEGROUNDS</td>
      <td>PUBG Corporation</td>
      <td>KRAFTON, Inc.</td>
      <td>23165</td>
      <td>717</td>
      <td>5622</td>
      <td>261</td>
      <td>0</td>
      <td>314682</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>59.440590</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Palworld</td>
      <td>Pocketpair</td>
      <td>Pocketpair</td>
      <td>3854</td>
      <td>835</td>
      <td>2213</td>
      <td>257</td>
      <td>2999</td>
      <td>18028</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>94.104946</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Team Fortress 2</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>21244</td>
      <td>736</td>
      <td>4262</td>
      <td>102</td>
      <td>0</td>
      <td>43819</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>89.908668</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>



The achievements column can just fill the NaN values with 0 because if a game doesn't have achievements on steam then it just isn't counted in this column and so it can be considered to have 0 achievements.
Any other NaN values can't really be filled and so must be removed. This leaves the dataset with 9338 values which is still a large amount. We still need to deal with the developers and pubilshers.


```python
#find out how many times developers and publishers are mentioned and also how many unique developers and publishers there are.
print(final["developer"].value_counts())
print(final["publisher"].value_counts())
```

    developer
    KOEI TECMO GAMES CO., LTD.            39
    Square Enix                           37
    CAPCOM Co., Ltd.                      37
    Valve                                 31
    Ubisoft Montreal                      21
                                          ..
    Pewter Games Studios, LoPoly Games     1
    Interactive Stone                      1
    Trilobyte Games                        1
    Red Six Development, Male Union        1
    Far Mills, Mysterytag                  1
    Name: count, Length: 6286, dtype: int64
    publisher
    Ubisoft              106
    SEGA                  96
    Square Enix           92
    THQ Nordic            92
    Electronic Arts       92
                        ... 
    Wild Ox Studios        1
    噪点游戏 ZAO GAMES         1
    Jonathan BRASSAUD      1
    DMM GAMES              1
    TVGS                   1
    Name: count, Length: 4718, dtype: int64
    

The best way to include developers and publishers would be to assign boolen values the same way that the genres were done, but unlike genres where theres only around 20, there are 6000+ developers and 4700+ publishers, which is not ideal. Ideally each developer and publisher would have equal weighting, but I could frequency encode these features so the more common developers/publishers have a higher weighting. I think it would be interesting to do this and then compare what the feature importance and predictions look like when including/excluding these features.


```python
#calculating and mapping the frequencies to each row
final["dev_freq"] = final['developer'].map(final['developer'].value_counts())
final["pub_freq"] = final['publisher'].map(final['publisher'].value_counts())

final.info()
final.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 9338 entries, 0 to 9999
    Data columns (total 44 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   name                   9338 non-null   object 
     1   developer              9338 non-null   object 
     2   publisher              9338 non-null   object 
     3   average_forever        9338 non-null   int64  
     4   average_2weeks         9338 non-null   int64  
     5   median_forever         9338 non-null   int64  
     6   median_2weeks          9338 non-null   int64  
     7   initialprice           9338 non-null   int64  
     8   ccu                    9338 non-null   int64  
     9   mat_supports_windows   9338 non-null   object 
     10  mat_supports_mac       9338 non-null   object 
     11  mat_supports_linux     9338 non-null   object 
     12  mat_achievement_count  9338 non-null   float64
     13  Action                 9338 non-null   bool   
     14  Free To Play           9338 non-null   bool   
     15  Adventure              9338 non-null   bool   
     16  Strategy               9338 non-null   bool   
     17  Indie                  9338 non-null   bool   
     18  RPG                    9338 non-null   bool   
     19  Animation & Modeling   9338 non-null   bool   
     20  Video Production       9338 non-null   bool   
     21  Casual                 9338 non-null   bool   
     22  Simulation             9338 non-null   bool   
     23  Racing                 9338 non-null   bool   
     24  MMO                    9338 non-null   bool   
     25  Nudity                 9338 non-null   bool   
     26  Violent                9338 non-null   bool   
     27  Sports                 9338 non-null   bool   
     28  Early Access           9338 non-null   bool   
     29  Gore                   9338 non-null   bool   
     30  Web Publishing         9338 non-null   bool   
     31  Design & Illustration  9338 non-null   bool   
     32  Game Development       9338 non-null   bool   
     33  Education              9338 non-null   bool   
     34  Software Training      9338 non-null   bool   
     35  Utilities              9338 non-null   bool   
     36  Audio Production       9338 non-null   bool   
     37  Photo Editing          9338 non-null   bool   
     38  Movie                  9338 non-null   bool   
     39  Sexual Content         9338 non-null   bool   
     40  Accounting             9338 non-null   bool   
     41  score                  9338 non-null   float64
     42  dev_freq               9338 non-null   int64  
     43  pub_freq               9338 non-null   int64  
    dtypes: bool(28), float64(2), int64(8), object(6)
    memory usage: 1.5+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>developer</th>
      <th>publisher</th>
      <th>average_forever</th>
      <th>average_2weeks</th>
      <th>median_forever</th>
      <th>median_2weeks</th>
      <th>initialprice</th>
      <th>ccu</th>
      <th>mat_supports_windows</th>
      <th>...</th>
      <th>Software Training</th>
      <th>Utilities</th>
      <th>Audio Production</th>
      <th>Photo Editing</th>
      <th>Movie</th>
      <th>Sexual Content</th>
      <th>Accounting</th>
      <th>score</th>
      <th>dev_freq</th>
      <th>pub_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Counter-Strike: Global Offensive</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>33852</td>
      <td>708</td>
      <td>6645</td>
      <td>301</td>
      <td>0</td>
      <td>1013936</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>86.693234</td>
      <td>31</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apex Legends</td>
      <td>Respawn</td>
      <td>Electronic Arts</td>
      <td>10506</td>
      <td>496</td>
      <td>935</td>
      <td>246</td>
      <td>0</td>
      <td>124262</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>67.142422</td>
      <td>2</td>
      <td>92</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PUBG: BATTLEGROUNDS</td>
      <td>PUBG Corporation</td>
      <td>KRAFTON, Inc.</td>
      <td>23165</td>
      <td>717</td>
      <td>5622</td>
      <td>261</td>
      <td>0</td>
      <td>314682</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>59.440590</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Palworld</td>
      <td>Pocketpair</td>
      <td>Pocketpair</td>
      <td>3854</td>
      <td>835</td>
      <td>2213</td>
      <td>257</td>
      <td>2999</td>
      <td>18028</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>94.104946</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Team Fortress 2</td>
      <td>Valve</td>
      <td>Valve</td>
      <td>21244</td>
      <td>736</td>
      <td>4262</td>
      <td>102</td>
      <td>0</td>
      <td>43819</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>89.908668</td>
      <td>31</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 44 columns</p>
</div>



With the frequencies input, the developer and publisher columns can be removed.


```python
#removing the developer and publisher columns
final.drop(columns=["developer", "publisher"], inplace=True)

final.info()
final.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 9338 entries, 0 to 9999
    Data columns (total 42 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   name                   9338 non-null   object 
     1   average_forever        9338 non-null   int64  
     2   average_2weeks         9338 non-null   int64  
     3   median_forever         9338 non-null   int64  
     4   median_2weeks          9338 non-null   int64  
     5   initialprice           9338 non-null   int64  
     6   ccu                    9338 non-null   int64  
     7   mat_supports_windows   9338 non-null   object 
     8   mat_supports_mac       9338 non-null   object 
     9   mat_supports_linux     9338 non-null   object 
     10  mat_achievement_count  9338 non-null   float64
     11  Action                 9338 non-null   bool   
     12  Free To Play           9338 non-null   bool   
     13  Adventure              9338 non-null   bool   
     14  Strategy               9338 non-null   bool   
     15  Indie                  9338 non-null   bool   
     16  RPG                    9338 non-null   bool   
     17  Animation & Modeling   9338 non-null   bool   
     18  Video Production       9338 non-null   bool   
     19  Casual                 9338 non-null   bool   
     20  Simulation             9338 non-null   bool   
     21  Racing                 9338 non-null   bool   
     22  MMO                    9338 non-null   bool   
     23  Nudity                 9338 non-null   bool   
     24  Violent                9338 non-null   bool   
     25  Sports                 9338 non-null   bool   
     26  Early Access           9338 non-null   bool   
     27  Gore                   9338 non-null   bool   
     28  Web Publishing         9338 non-null   bool   
     29  Design & Illustration  9338 non-null   bool   
     30  Game Development       9338 non-null   bool   
     31  Education              9338 non-null   bool   
     32  Software Training      9338 non-null   bool   
     33  Utilities              9338 non-null   bool   
     34  Audio Production       9338 non-null   bool   
     35  Photo Editing          9338 non-null   bool   
     36  Movie                  9338 non-null   bool   
     37  Sexual Content         9338 non-null   bool   
     38  Accounting             9338 non-null   bool   
     39  score                  9338 non-null   float64
     40  dev_freq               9338 non-null   int64  
     41  pub_freq               9338 non-null   int64  
    dtypes: bool(28), float64(2), int64(8), object(4)
    memory usage: 1.3+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>average_forever</th>
      <th>average_2weeks</th>
      <th>median_forever</th>
      <th>median_2weeks</th>
      <th>initialprice</th>
      <th>ccu</th>
      <th>mat_supports_windows</th>
      <th>mat_supports_mac</th>
      <th>mat_supports_linux</th>
      <th>...</th>
      <th>Software Training</th>
      <th>Utilities</th>
      <th>Audio Production</th>
      <th>Photo Editing</th>
      <th>Movie</th>
      <th>Sexual Content</th>
      <th>Accounting</th>
      <th>score</th>
      <th>dev_freq</th>
      <th>pub_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Counter-Strike: Global Offensive</td>
      <td>33852</td>
      <td>708</td>
      <td>6645</td>
      <td>301</td>
      <td>0</td>
      <td>1013936</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>86.693234</td>
      <td>31</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apex Legends</td>
      <td>10506</td>
      <td>496</td>
      <td>935</td>
      <td>246</td>
      <td>0</td>
      <td>124262</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>67.142422</td>
      <td>2</td>
      <td>92</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PUBG: BATTLEGROUNDS</td>
      <td>23165</td>
      <td>717</td>
      <td>5622</td>
      <td>261</td>
      <td>0</td>
      <td>314682</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>59.440590</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Palworld</td>
      <td>3854</td>
      <td>835</td>
      <td>2213</td>
      <td>257</td>
      <td>2999</td>
      <td>18028</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>94.104946</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Team Fortress 2</td>
      <td>21244</td>
      <td>736</td>
      <td>4262</td>
      <td>102</td>
      <td>0</td>
      <td>43819</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>89.908668</td>
      <td>31</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>



Scaling could be an issue, mainly with the playtimes and concurrent users. The playtimes are currently in minutes, so putting them into hours could help here. Concurrent users will be left for now, but could be normalised in the future if there are issues. Additionally, the price is in pennies/cents and so should be converted to dollars.


```python
#converting values into hours and dollars
final["average_forever"] = final["average_forever"] / 60
final["average_2weeks"] = final["average_2weeks"] / 60
final["median_forever"] = final["median_forever"] / 60
final["median_2weeks"] = final["median_2weeks"] / 60
final["initialprice"] = final["initialprice"] / 100

final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>average_forever</th>
      <th>average_2weeks</th>
      <th>median_forever</th>
      <th>median_2weeks</th>
      <th>initialprice</th>
      <th>ccu</th>
      <th>mat_supports_windows</th>
      <th>mat_supports_mac</th>
      <th>mat_supports_linux</th>
      <th>...</th>
      <th>Software Training</th>
      <th>Utilities</th>
      <th>Audio Production</th>
      <th>Photo Editing</th>
      <th>Movie</th>
      <th>Sexual Content</th>
      <th>Accounting</th>
      <th>score</th>
      <th>dev_freq</th>
      <th>pub_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Counter-Strike: Global Offensive</td>
      <td>564.200000</td>
      <td>11.800000</td>
      <td>110.750000</td>
      <td>5.016667</td>
      <td>0.00</td>
      <td>1013936</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>86.693234</td>
      <td>31</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apex Legends</td>
      <td>175.100000</td>
      <td>8.266667</td>
      <td>15.583333</td>
      <td>4.100000</td>
      <td>0.00</td>
      <td>124262</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>67.142422</td>
      <td>2</td>
      <td>92</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PUBG: BATTLEGROUNDS</td>
      <td>386.083333</td>
      <td>11.950000</td>
      <td>93.700000</td>
      <td>4.350000</td>
      <td>0.00</td>
      <td>314682</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>59.440590</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Palworld</td>
      <td>64.233333</td>
      <td>13.916667</td>
      <td>36.883333</td>
      <td>4.283333</td>
      <td>29.99</td>
      <td>18028</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>94.104946</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Team Fortress 2</td>
      <td>354.066667</td>
      <td>12.266667</td>
      <td>71.033333</td>
      <td>1.700000</td>
      <td>0.00</td>
      <td>43819</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>89.908668</td>
      <td>31</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>



This marks the final, cleaned dataset. If the data is too heavily skewed, I will consider scaling/normalising the concurrent users column.

## Descriptive Analytics ##

With the dataset finalised, we can do some general descriptive analysis, find out what is going on in the dataset and how some of the features relate to each other.

### Numeric Features ###


```python
final.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>average_forever</th>
      <th>average_2weeks</th>
      <th>median_forever</th>
      <th>median_2weeks</th>
      <th>initialprice</th>
      <th>ccu</th>
      <th>mat_achievement_count</th>
      <th>score</th>
      <th>dev_freq</th>
      <th>pub_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9338.000000</td>
      <td>9338.000000</td>
      <td>9338.000000</td>
      <td>9338.000000</td>
      <td>9338.000000</td>
      <td>9.338000e+03</td>
      <td>9338.000000</td>
      <td>9338.000000</td>
      <td>9338.000000</td>
      <td>9338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16.901219</td>
      <td>1.940574</td>
      <td>8.395083</td>
      <td>2.047530</td>
      <td>13.559510</td>
      <td>6.023183e+02</td>
      <td>44.040480</td>
      <td>79.434449</td>
      <td>3.304776</td>
      <td>13.730135</td>
    </tr>
    <tr>
      <th>std</th>
      <td>70.389540</td>
      <td>10.732379</td>
      <td>57.421050</td>
      <td>11.930887</td>
      <td>13.681762</td>
      <td>1.168005e+04</td>
      <td>224.955264</td>
      <td>15.534780</td>
      <td>5.251158</td>
      <td>24.200401</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.516667</td>
      <td>0.000000</td>
      <td>1.416667</td>
      <td>0.000000</td>
      <td>2.990000</td>
      <td>1.000000e+00</td>
      <td>0.000000</td>
      <td>71.835044</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.250000</td>
      <td>0.000000</td>
      <td>3.550000</td>
      <td>0.000000</td>
      <td>9.990000</td>
      <td>6.000000e+00</td>
      <td>21.000000</td>
      <td>83.183111</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.562500</td>
      <td>0.200000</td>
      <td>6.866667</td>
      <td>0.212500</td>
      <td>19.990000</td>
      <td>5.000000e+01</td>
      <td>44.000000</td>
      <td>90.915809</td>
      <td>3.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4389.000000</td>
      <td>334.666667</td>
      <td>4486.550000</td>
      <td>334.666667</td>
      <td>149.990000</td>
      <td>1.013936e+06</td>
      <td>9821.000000</td>
      <td>100.000000</td>
      <td>39.000000</td>
      <td>106.000000</td>
    </tr>
  </tbody>
</table>
</div>



Count doesn't matter here. The mean for each of the playtimes column is interersting especially when you compare it with the maximum playtime for each. The mean playtime for average_forever sits at only 16.9 hours which is a lot lower than the maximum of 4389 hours which indicates that the large majority of games have low playtimes. This could mean that there are a lot more short or singleplayer games that you only play once rather than online games that keep players coming back to play more. Alternatively, this could also mean that the casual audience of gamers is the majority with people not putting much playtime into many games with just a few spending thousands of hours in some games. It could also mean a combination of both. 

The stats in the average_2weeks seem to enforce the idea that there is a lagre casual audience with a mean of 2 hours every 2 weeks across all games. Even if we consider short games in this instance, this number wouldn't be that low if the player was more serious about their gaming. For example if a game took 6 hours to complete, a hardcore gamer would be able to complete that in the 2 weeks so it would say 6 hours on the game, not 2 hours. The very low numbers in the 25, 50 and 75 percentiles seem to suggest this too, with a minority of gamers hitting close to the maximum of 335 hours.

The median columns seem to enforce the idea of casual players playing short games taking up the majority of the playerbase. The median isn't as heavily skewed by the large playtimes as the average is, so having the mean at 8 hours, and the 2 weeks mean at 2 hours still, indicates that most people who play any of these games are casual gamers. It is still interesting to note the maximum value in median_forever is 4486 hours which is higher than the average_forever column. This could suggest that for the games where there are very high playtimes, the hardcore gamers take up the majority, but overall the casual playerbase takes up the majority.

The price column is an interesting one to look at. It has a very low mean, an interesting IQR and a very large maximum. The low mean could indicate that the majority of games in this list are made by indie developers and are generally smaller games with lower price tags or there is a large quantity of free games that bring down the average, although these points aren't mutually exclusive. The maximum price at $149.99 is another wildly large number even when compared to the 75th percentile. I will investigate what games are at the maximum/minimum values in the next section to get a clearer explanation of these possible outliers.

The concurrent users should roughly follow the same distribution as the average playtime. The mean sits at 600 with a very large standard deviation due to some games having very little to no concurrent players whereas other, more popular games have 1000000 players concurrently. This makes a lot of sense with the average being fairly low with the casual audience playing these smaller games, but the larger, online and time consuming games having a lot more concurrent players.

Achievement count is an interesting column, because this is not a mandatory column and so a lot of games won't have any achievements, but they are still a nice addition to include, adding some fun challenges for people to complete. The average sitting at 44 with the minimum at 0 enforces this idea. There is a subset of gamers that likes to get all the achievements in a game and so some games will put a large number of easy to obtain achievements in them to try and cater to this audience, potentially boosting the number of sales and positive reviews. The 75th percentile being 44, the same as the average, along with the maximum value being 9821 enforces this idea. Other potential reasons for these large achievement counts could be due to "shovelware" or idle games.

The last major column is the score, our target column. The average sits at 79% which means that the large majority of games are considered good games and are recommended more times than not. The minimum at 0 is interesting because this either means the score is 100% negative (which is unlikely), or it doesn't have any reviews. It is more likely that the game doesn't have many reviews and is a new game from a first time developer just trying to get their game out in the world but hasn't been able to break through the noise. Similarly, the maximum at 100% is also likely a very small/new game with only a couple positive reviews because it is almost impossible for a popular game to have a 100% positive ratio.

Finally, the developer and publisher frequency columns. The means stil at 3 and 13 respectively which makes sense because a lot of developers are indie studios or single people just posting a single or a couple games and so are more saturated, but one publisher can take on many different developers and therefore be displayed on a lot more games.

### Identifying Minimum and Maximums ###


```python
print(final["average_forever"].max())
print(final["average_forever"].idxmax())

print(final.loc[2734])
```

    4389.0
    2734
    name                         懒人修仙传
    average_forever             4389.0
    average_2weeks                 0.0
    median_forever             4486.55
    median_2weeks                  0.0
    initialprice                  3.99
    ccu                             41
    mat_supports_windows          True
    mat_supports_mac              True
    mat_supports_linux            True
    mat_achievement_count          0.0
    Action                       False
    Free To Play                 False
    Adventure                    False
    Strategy                     False
    Indie                         True
    RPG                           True
    Animation & Modeling         False
    Video Production             False
    Casual                        True
    Simulation                    True
    Racing                       False
    MMO                          False
    Nudity                       False
    Violent                      False
    Sports                       False
    Early Access                 False
    Gore                         False
    Web Publishing               False
    Design & Illustration        False
    Game Development             False
    Education                    False
    Software Training            False
    Utilities                    False
    Audio Production             False
    Photo Editing                False
    Movie                        False
    Sexual Content               False
    Accounting                   False
    score                    60.172144
    dev_freq                         2
    pub_freq                         2
    Name: 2734, dtype: object
    

After putting this game into steam, it seems to be an idle/clicker game. This explains the very large average and median playtimes and why it also falls under the "casual" genre because usually you wouldn't consider large playtimes casual. This is one of the games where you mostly just leave it running in the background and there is very little "true" gameplay. The idea behind these types of games is to upgrade buildings and effects to make the number increase. For example, in cookie clicker you want to make as many cookies as possible.


```python
print(final["initialprice"].max())
print(final["initialprice"].idxmax())

print(final.loc[4836])
```

    149.99
    4836
    name                     VEGAS Pro 18 Edit Steam Edition
    average_forever                               100.333333
    average_2weeks                                       0.0
    median_forever                                100.333333
    median_2weeks                                        0.0
    initialprice                                      149.99
    ccu                                                   23
    mat_supports_windows                                True
    mat_supports_mac                                    True
    mat_supports_linux                                  True
    mat_achievement_count                                0.0
    Action                                             False
    Free To Play                                       False
    Adventure                                          False
    Strategy                                           False
    Indie                                              False
    RPG                                                False
    Animation & Modeling                               False
    Video Production                                    True
    Casual                                             False
    Simulation                                         False
    Racing                                             False
    MMO                                                False
    Nudity                                             False
    Violent                                            False
    Sports                                             False
    Early Access                                       False
    Gore                                               False
    Web Publishing                                     False
    Design & Illustration                              False
    Game Development                                   False
    Education                                          False
    Software Training                                  False
    Utilities                                          False
    Audio Production                                    True
    Photo Editing                                      False
    Movie                                              False
    Sexual Content                                     False
    Accounting                                         False
    score                                          20.453573
    dev_freq                                               1
    pub_freq                                               1
    Name: 4836, dtype: object
    

This is very interesting. When you think of steam, you think of a games store and library, but steam has more to offer than that with many different softwares and hardwares available for purchase. This is a video editor pro edition which explains the very large price because games max out at $70, but this isn't a game. It also explains the very low score this has because the reviews are likely people complaining about the price or possibly removed support because a newer version is available.


```python
print(final["ccu"].max())
print(final["ccu"].idxmax())

print(final.loc[0])
```

    1013936
    0
    name                     Counter-Strike: Global Offensive
    average_forever                                     564.2
    average_2weeks                                       11.8
    median_forever                                     110.75
    median_2weeks                                    5.016667
    initialprice                                          0.0
    ccu                                               1013936
    mat_supports_windows                                 True
    mat_supports_mac                                     True
    mat_supports_linux                                   True
    mat_achievement_count                                 1.0
    Action                                               True
    Free To Play                                         True
    Adventure                                           False
    Strategy                                            False
    Indie                                               False
    RPG                                                 False
    Animation & Modeling                                False
    Video Production                                    False
    Casual                                              False
    Simulation                                          False
    Racing                                              False
    MMO                                                 False
    Nudity                                              False
    Violent                                             False
    Sports                                              False
    Early Access                                        False
    Gore                                                False
    Web Publishing                                      False
    Design & Illustration                               False
    Game Development                                    False
    Education                                           False
    Software Training                                   False
    Utilities                                           False
    Audio Production                                    False
    Photo Editing                                       False
    Movie                                               False
    Sexual Content                                      False
    Accounting                                          False
    score                                           86.693234
    dev_freq                                               31
    pub_freq                                               32
    Name: 0, dtype: object
    

The concurrent users being in the millions for this game make sense. This game has always been Steam's most popular game, almost always being at the top of the charts, only occasionally being knocked down to second when a very anticipated game releases, and being a lot of users' reason to download Steam in the first place as it is an original game by Valve and only available on Steam.


```python
print(final["mat_achievement_count"].max())
print(final["mat_achievement_count"].idxmax())

print(final.loc[8207])
```

    9821.0
    8207
    name                     LOGistICAL
    average_forever          162.183333
    average_2weeks                  0.0
    median_forever           162.183333
    median_2weeks                   0.0
    initialprice                   9.99
    ccu                               4
    mat_supports_windows           True
    mat_supports_mac               True
    mat_supports_linux             True
    mat_achievement_count        9821.0
    Action                        False
    Free To Play                  False
    Adventure                     False
    Strategy                       True
    Indie                          True
    RPG                           False
    Animation & Modeling          False
    Video Production              False
    Casual                         True
    Simulation                    False
    Racing                        False
    MMO                           False
    Nudity                        False
    Violent                       False
    Sports                        False
    Early Access                  False
    Gore                          False
    Web Publishing                False
    Design & Illustration         False
    Game Development              False
    Education                     False
    Software Training             False
    Utilities                     False
    Audio Production              False
    Photo Editing                 False
    Movie                         False
    Sexual Content                False
    Accounting                    False
    score                     82.905983
    dev_freq                          1
    pub_freq                          1
    Name: 8207, dtype: object
    

This is a puzzle game with a lot of levels and a lot of DLC with even more levels that all give you achievements every time you do small tasks or complete levels. This game is definitely designed for the people who want to have a large achievement count on their profile or want to customise their profile with achievements with specific icons.


```python
print(final["score"].min())
print(final["score"].idxmin())

print(final.loc[660])
```

    0.0
    660
    name                     Heart of a Warrior
    average_forever                         0.0
    average_2weeks                          0.0
    median_forever                          0.0
    median_2weeks                           0.0
    initialprice                          10.99
    ccu                                       0
    mat_supports_windows                   True
    mat_supports_mac                       True
    mat_supports_linux                     True
    mat_achievement_count                   0.0
    Action                                 True
    Free To Play                          False
    Adventure                              True
    Strategy                              False
    Indie                                  True
    RPG                                    True
    Animation & Modeling                  False
    Video Production                      False
    Casual                                False
    Simulation                            False
    Racing                                False
    MMO                                   False
    Nudity                                False
    Violent                               False
    Sports                                False
    Early Access                          False
    Gore                                  False
    Web Publishing                        False
    Design & Illustration                 False
    Game Development                      False
    Education                             False
    Software Training                     False
    Utilities                             False
    Audio Production                      False
    Photo Editing                         False
    Movie                                 False
    Sexual Content                        False
    Accounting                            False
    score                                   0.0
    dev_freq                                  1
    pub_freq                                  1
    Name: 660, dtype: object
    

As expected, the reason this game has a 0% score is because there are no reviews for the game.

### Summing the boolean features ###


```python
#counting how many True values in each column
for col in final.columns[7:10]:
    print(f"{col}: {final[col].sum()}")
```

    mat_supports_windows: 9338
    mat_supports_mac: 9331
    mat_supports_linux: 9322
    

Every game in this dataset supports Windows which makes sense because it is the most common operating system and if Windows wasn't supported, a large portion of potential customers would be ignored. Mac is almost entirely supported, but a couple games or software doesn't support it. These are likely very small indie games that don't have the development power to support mac. Linux is the least supported with likely the same reason as mac for the games that don't support it, but all 3 operating systems are largely supported. This suggests these 3 features are very important to make sure they are true.


```python
#counting how many True values in each column
for col in final.columns[11:39]:
    print(f"{col}: {final[col].sum()}")
```

    Action: 4477
    Free To Play: 1164
    Adventure: 3876
    Strategy: 2351
    Indie: 5627
    RPG: 2337
    Animation & Modeling: 30
    Video Production: 24
    Casual: 2363
    Simulation: 2093
    Racing: 314
    MMO: 473
    Nudity: 4
    Violent: 18
    Sports: 338
    Early Access: 608
    Gore: 17
    Web Publishing: 21
    Design & Illustration: 37
    Game Development: 19
    Education: 18
    Software Training: 14
    Utilities: 73
    Audio Production: 14
    Photo Editing: 12
    Movie: 1
    Sexual Content: 2
    Accounting: 2
    

The largest genre is Indie which makes sense as although the largest games are created by AAA studios, there are many more indie developers releasing games. The next most common genres are action and adventure, which also make sense as these are the most popular genres across all media and usually what come to mind when you first think of "genre". The rest of the genres in the thousands are the common, classic genres for most games. MMO, racing and sports are the last "proper" genres, but are the least common. Other genres like "Nudity" and "Sexual Content" are very rare because they alienate a large portion of the playerbase being childrena and teenagers. Free to play and early access are similar to the indie genre in that they more descriptors of how a game is developed or released, rather than what type of game it is. The violent and gore genres are largely covered by the more common genres. The rest of the genres are more descriptors of the software and are likely not actual games.

### Displaying Numeric Relationships ###

Before doing feature importance and regression analysis, it is important to find out how features relate to the target column "score" and how some of the features relate to each other.

I will start by comparing score to the numeric features.


```python
#average_forever	average_2weeks	median_forever	median_2weeks	initialprice	ccu	mat_achievement_count	score	dev_freq	pub_freq

#setting y to be the score because this is always going to be the dependent variable.
y = final["score"]

fig, axs = plt.subplots(3, 3, sharey=True, constrained_layout=True)
fig.suptitle("Percent score against all numeric features")
#fig.tight_layout()

#plotting each variable
axs[0,0].scatter(final["average_forever"], y, s=2)
axs[0,0].set_xlabel("average forever [hrs]")

axs[0,1].scatter(final["average_2weeks"], y, s=2)
axs[0,1].set_xlabel("average 2weeks [hrs]")

axs[0,2].scatter(final["median_forever"], y, s=2)
axs[0,2].set_xlabel("median forever [hrs]")

axs[1,0].scatter(final["median_2weeks"], y, s=2)
axs[1,0].set_xlabel("median 2weeks [hrs]")

axs[1,1].scatter(final["initialprice"], y, s=2)
axs[1,1].set_xlabel("price [$]")

axs[1,2].scatter(final["ccu"], y, s=2)
axs[1,2].set_xlabel("concurrent users")

axs[2,0].scatter(final["mat_achievement_count"], y, s=2)
axs[2,0].set_xlabel("achievement count")

axs[2,1].scatter(final["dev_freq"], y, s=2)
axs[2,1].set_xlabel("dev frequency")

axs[2,2].scatter(final["pub_freq"], y, s=2)
axs[2,2].set_xlabel("pub frequency")

plt.show()

```


    
![png](Capstone%20copy_files/Capstone%20copy_51_0.png)
    


The presence of some major outliers in the average forever/median forever, concurrent users and achievement count plots makes it hard to see the distribution of data so lets remove these outliers and rerun the code to see if it improves and analysis can be done. I have also added in a regression line for each of these relationships.


```python
#removing outliers
removed = final.drop([0, 8207, 2734])

#setting y to be the score because this is always going to be the dependent variable.
y = removed["score"]

fig, axs = plt.subplots(3, 3, sharey=True, constrained_layout=True)
fig.suptitle("Percent score against all numeric features - no major outliers")
#fig.tight_layout()

axs[0,0].scatter(removed["average_forever"], y, s=2)
b, a = np.polyfit(removed["average_forever"], y, deg=1)
x = np.linspace(0, removed["average_forever"].max(), num=10000)
axs[0,0].plot(x, a+b*x, color="k", lw=1)
axs[0,0].set_xlabel("average forever [hrs]")

axs[0,1].scatter(removed["average_2weeks"], y, s=2)
b, a = np.polyfit(removed["average_2weeks"], y, deg=1)
x = np.linspace(0, removed["average_2weeks"].max(), num=10000)
axs[0,1].plot(x, a+b*x, color="k", lw=1)
axs[0,1].set_xlabel("average 2weeks [hrs]")

axs[0,2].scatter(removed["median_forever"], y, s=2)
b, a = np.polyfit(removed["median_forever"], y, deg=1)
x = np.linspace(0, removed["median_forever"].max(), num=10000)
axs[0,2].plot(x, a+b*x, color="k", lw=1)
axs[0,2].set_xlabel("median forever [hrs]")

axs[1,0].scatter(removed["median_2weeks"], y, s=2)
b, a = np.polyfit(removed["median_2weeks"], y, deg=1)
x = np.linspace(0, removed["median_2weeks"].max(), num=10000)
axs[1,0].plot(x, a+b*x, color="k", lw=1)
axs[1,0].set_xlabel("median 2weeks [hrs]")

axs[1,1].scatter(removed["initialprice"], y, s=2)
b, a = np.polyfit(removed["initialprice"], y, deg=1)
x = np.linspace(0, removed["initialprice"].max(), num=10000)
axs[1,1].plot(x, a+b*x, color="k", lw=1)
axs[1,1].set_xlabel("price [$]")

axs[1,2].scatter(removed["ccu"], y, s=2)
b, a = np.polyfit(removed["ccu"], y, deg=1)
x = np.linspace(0, removed["ccu"].max(), num=10000)
axs[1,2].plot(x, a+b*x, color="k", lw=1)
axs[1,2].set_xlabel("concurrent users")

axs[2,0].scatter(removed["mat_achievement_count"], y, s=2)
b, a = np.polyfit(removed["mat_achievement_count"], y, deg=1)
x = np.linspace(0, removed["mat_achievement_count"].max(), num=10000)
axs[2,0].plot(x, a+b*x, color="k", lw=1)
axs[2,0].set_xlabel("achievement count")

axs[2,1].scatter(removed["dev_freq"], y, s=2)
b, a = np.polyfit(removed["dev_freq"], y, deg=1)
x = np.linspace(0, removed["dev_freq"].max(), num=10000)
axs[2,1].plot(x, a+b*x, color="k", lw=1)
axs[2,1].set_xlabel("dev frequency")

axs[2,2].scatter(removed["pub_freq"], y, s=2)
b, a = np.polyfit(removed["pub_freq"], y, deg=1)
x = np.linspace(0, removed["pub_freq"].max(), num=10000)
axs[2,2].plot(x, a+b*x, color="k", lw=1)
axs[2,2].set_xlabel("pub frequency")

plt.show()
```


    
![png](Capstone%20copy_files/Capstone%20copy_53_0.png)
    


All of the plots involving playtime have a very similar distribution as would be expected.  All four of these categories seem to show that the more hours people put into a game tend to mostly enjoy the game sitting at around 60% positive or above being fairly consistent, but even here there are some games that drop below this value. In the first quartile, there is a huge variance in score based on hours played and so isn't the greatest indication of score. In both of the 2 weeks playtime graphs, in between 20 and 70 hours there is a clump of points that generally recieve high scores of 75% or above which could indicate that this is the "golden zone" when creating games.

Price is an interesting graph. Firstly, we can see that most games are given specific categories of price with very few games differing from these traditional price points, especially at $30 or above. Secondly, there are a few games at 0 score regardless of price. We know from previous exploration that these games are newly released games by very small developers with no reviews on the game yet. We can see a small positive correlation that as price increases score tends to increase, but this is very loose. Like playtime, there is a large variance when it comes to score regardless of price. 

Concurrent users follows a similar distribution to play time as well with large numbers of concurrent users tending to be more consistent in their scores also tending to be around 60% positive or above. Once again, low values in this feature have a very large variance in score. Similarly to the 2 week graphs, there appears to be a small "golden zone" with consistently high scores.

Achievement count follows the same general pattern as the other features with low numbers having a very large variance and higher achievement counts having more consistency in the score. 

Developer and publisher frequency are interesting graphs to look at. There is a gap in frequency between 20 and 30 developers and 65 and 75 publishers. Additionally, there is no clear difference in distribution before and after the gap in both graphs meaning that the more common developers and publishers don't necessarily make better games. There seems to be a small increase in consistency in developers who appear 20 or less times where the closer to 20, the generally higher scoring their games are. Publishers on the other hand have very little to no correlation with score.

Looking at the regression lines for each of the graphs, it can be seen that there is a small positive relation between each of these features and the score. Publisher frequency is the exception with a slight negative to no correlation at all with score. Developer frequency and achievement count also have very weak correlations with score, but none of these lines are particularly strong in their correlatoins.

### Displaying Developer frequency vs Publisher Frequency ###
It would also be interesting to find what the relation between developer and publisher frequency looks like with the score as a gradient.


```python
plt.scatter(x=final["pub_freq"], y=final["dev_freq"], c=final["score"], s=2)
plt.colorbar(label="Score")
plt.xlabel("Publisher Frequency")
plt.ylabel("Developer Frequency")
plt.title('Dev vs Pub Frequency coloured by Review Score')
plt.show()
```


    
![png](Capstone%20copy_files/Capstone%20copy_56_0.png)
    


I thought there would be more of a correlation between these two variables with the larger publishers taking on larger developers, but there is no clear relationship between these two features. Additionally, there is no clear correlation between score and these values too with the range of scores being spread throughout the entire graph.

### Displaying Binary Features ###


```python
#once again the score is going to be used for all y values
y = final["score"]

fig, axs = plt.subplots(3, 4, sharey=True, constrained_layout=True, figsize=(10,8))
fig.suptitle("Percent score against all binary features with >100 True values")

sns.boxplot(x="Action", y="score", data=final, ax=axs[0,0])
sns.boxplot(x="Free To Play", y="score", data=final, ax=axs[0,1])
sns.boxplot(x="Adventure", y="score", data=final, ax=axs[0,2])
sns.boxplot(x="Strategy", y="score", data=final, ax=axs[0,3])
sns.boxplot(x="Indie", y="score", data=final, ax=axs[1,0])
sns.boxplot(x="RPG", y="score", data=final, ax=axs[1,1])
sns.boxplot(x="Casual", y="score", data=final, ax=axs[1,2])
sns.boxplot(x="Simulation", y="score", data=final, ax=axs[1,3])
sns.boxplot(x="Racing", y="score", data=final, ax=axs[2,0])
sns.boxplot(x="MMO", y="score", data=final, ax=axs[2,1])
sns.boxplot(x="Sports", y="score", data=final, ax=axs[2,2])
sns.boxplot(x="Early Access", y="score", data=final, ax=axs[2,3])
```




    <Axes: xlabel='Early Access', ylabel='score'>




    
![png](Capstone%20copy_files/Capstone%20copy_59_1.png)
    


The reason only features with more than 100 True points is considered is because those with very low values aren't worth doing because the quartiles and IQR become unstable while the outliers dominate the mean and the shape of the graph. Especially considering some of the features have 1-5 True values it would not be worth looking at. The False side of the coin is generally not worth looking at because it is just a combination of every other game and piece of software which doesn't tell us anything useful.

Looking at these graphs, a lot of them seem very similar, but insights can still be made. The MMO genre is by far the worst scored genre with the lowest 3rd quartile, 1st quartile and mean. Additionally, the IQR for MMO games is also small so it is consistent. Therefore, it is likely to be recommended to not make an MMO game. Similarly, free to play and strategy games are slightly lower than the rest of the genres but this is not by much. 

The Early Access genre has the largest IQR and lower quartile of the group which makes a lot of sense. This is the most volatile group as these are games that are released but still in development and not truly finished yet. This means that these games are more likely to be buggy and less likely to be rated well.

On the other hand, the Indie category has one of the smallest IQRs and has the highest mean and quartile values. This means indie games are quite consistently rated well, which means more when you consider that this is the largest genre with the largest number of games belonging to this category. It is likely that the Indie genre will be recommended if we want a game to be rated well.

All other genres fall somewhere between those mentioned in terms of consistency and ratings.
