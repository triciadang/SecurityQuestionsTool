import snscrape.modules.twitter as sntwitter
import pandas as pd
from IPython.display import display
import nltk
import gensim
import sys

#nltk.download('stopwords')

# TODO1: first tokenize each tweet and clean up text (remove punctuation)
def get_all_tweets_from_username(twitter_username):
    max_rows = None
    max_cols = None

    pd.set_option("display.max_rows", max_rows, "display.max_columns", max_cols)

    # Creating list to append tweet data to
    tweets_list1 = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('from:' + twitter_username).get_items()):
        if i > 500:
            break
        #tweets_list1.append([tweet.date, tweet.content, tweet.user.username])
        tweets_list1.append([tweet.content, tweet.user.username])

    # Creating a dataframe from the tweets list above
    tweets_df = pd.DataFrame(tweets_list1, columns=['Text'])

    return tweets_df

def get_all_facebook_posts(posts_excel_sheet):
    facebook_df = pd.read_excel(posts_excel_sheet,names=['Text'])

    display(facebook_df)

    #for each in facebook_df.get('Text'):
    #    print("==============")
    #    print(each)

    return facebook_df


# TODO2: Tokenize each tweet and clean up text (remove punctuation)
def preprocess_text(post):

    #tokenize each tweet: converts text to lowercase, split text into words, removes punctuation
    result = gensim.utils.simple_preprocess(str(post), deacc=True)

    #remove stop words
    yield(gensim.parsing.preprocessing.remove_stopword_tokens(result))


#TODO3: Convert cleaned text into numerical representation, create the Document Term Matrix
def get_security_answers(text):
    for each_text in text.get('Text'):
        print(each_text)
        cleaned_text = list(preprocess_text(each_text))
        print(cleaned_text[:1])

#TODO4: Pass vectorized corpus to LDA model

def main():

    #Twitter
    #twitter_username = 'triciadang7'
    #twitter_username = 'selenagomez'
    #tweets_df1 = get_all_tweets_from_username(twitter_username)
    #get_security_answers(tweets_df1)

    #Facebook
    facebook_excel_path = 'C:\\Users\\trici\\OneDrive\\Documents\\GT\\AlexaGilomenFacebook.xlsx'
    facebook_df1 = get_all_facebook_posts(facebook_excel_path)

    get_security_answers(facebook_df1)



if __name__ == "__main__":
    main()

