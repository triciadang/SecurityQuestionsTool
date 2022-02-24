import snscrape.modules.twitter as sntwitter
import pandas as pd
from IPython.display import display

max_rows = None
max_cols = None

pd.set_option("display.max_rows", max_rows, "display.max_columns", max_cols)

# Creating list to append tweet data to
tweets_list1 = []

#twitter_username = 'triciadang7'
twitter_username = 'selenagomez'

# Using TwitterSearchScraper to scrape data and append tweets to list
for i, tweet in enumerate(sntwitter.TwitterSearchScraper('from:' + twitter_username).get_items()):
    if i > 500:
        break
    #tweets_list1.append([tweet.date, tweet.content, tweet.user.username])
    tweets_list1.append([tweet.content, tweet.user.username])

for tweet in tweets_list1:
    if 'food' in tweet[0]:
        print("Food:" + tweet[0])

#print(tweets_list1)
# Creating a dataframe from the tweets list above
#tweets_df1 = pd.DataFrame(tweets_list1, columns=['Datetime', 'Text', 'Username'])
tweets_df1 = pd.DataFrame(tweets_list1, columns=['Text', 'Username'])
display(tweets_df1)