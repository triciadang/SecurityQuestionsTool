import torch
import nlp
from transformers import LongformerTokenizerFast,LongformerTokenizer, LongformerForQuestionAnswering, EvalPrediction

import string
import re

import requests, lxml
from bs4 import BeautifulSoup



import snscrape.modules.twitter as sntwitter
import pandas as pd
from IPython.display import display

import mysql.connector as mysql

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#Fetching the pretrained model
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
model = LongformerForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

# Basic preprocessing functions
def normalize_text(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)
    text = " ".join(text.split())
    return text

# The actual function that does the job
def longformer(text,question):
    encoding = tokenizer.encode_plus(question, text, return_tensors="pt",max_length=4096,padding='longest', truncation = True)

    input_ids = encoding["input_ids"]

    # default is local attention everywhere
    # the forward method will automatically set global attention on question tokens
    attention_mask = encoding["attention_mask"]

    start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)
    # print(start_scores)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    return answer

def get_all_tweets_from_username(twitter_username):
    max_rows = None
    max_cols = None

    pd.set_option("display.max_rows", max_rows, "display.max_columns", max_cols)

    # Creating list to append tweet data to
    tweets_list1 = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('from:' + twitter_username).get_items()):
        #tweets_list1.append([tweet.date, tweet.content, tweet.user.username])
        tweets_list1.append([tweet.content])

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

def concatenate_all_posts(text):
    all_cleaned_text = ""
    for each_text in text.get('Text'):
        print("Post: " + str(each_text))
        all_cleaned_text += str(each_text)

    return all_cleaned_text


def checkIfUsernameExists(cursor,username):

    cursor.execute("SELECT * FROM users_tweets_posts WHERE user_name= '" + username + "'")
    existingUsername=cursor.fetchall()  # fetch (and discard) remaining rows
    print(existingUsername)

    if len(existingUsername):
        return True
    else:
        return False


def main():
    db = mysql.connect(
        host="localhost",
        user="root",
        passwd="root",
        database = "security_questions",
    )

    cursor = db.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS users_tweets_posts (name VARCHAR(255), user_name VARCHAR(255) PRIMARY KEY,tweets1 LONGTEXT)")

    #cursor.execute("INSERT INTO users_tweets_posts (name,user_name,tweets1) VALUES ('try1', 'trial1', 'here')")


    questions_list = ["What did you call your favorite childhood pet?","What is your favorite food?","Who is your all-time favorite author?","What was your favorite book?","Who is your favorite actor of all time?","Who is your favorite cartoon character?",
"What is your favorite movie?","What is your favorite place to vacation?","What is your favorite sports team?","What was your favorite school teacher’s name?","What was your high school mascot?","Where did you go to high school?"]

    #Twitter
    #twitter_username = 'triciadang7'
    #twitter_username = 'Casey'
    #tweets_df1 = get_all_tweets_from_username(twitter_username)
    #all_posts = concatenate_all_posts(tweets_df1)

    #Facebook
    facebook_excel_path = 'C:\\Users\\trici\\OneDrive\\Documents\\GT\\AlexaGilomenFacebook.xlsx'
    facebook_df1 = get_all_facebook_posts(facebook_excel_path)
    all_posts= concatenate_all_posts(facebook_df1)

    for each_question in questions_list:
        answer = longformer(all_posts, each_question)
        if "Where did you go to high school?" in each_question:
            high_school = answer.split(" ")
            high_school_search = ""
            for each_word in high_school:
                high_school_search += each_word + "%20"

            url = 'https://www.google.com/search?q=' + high_school_search + "mascot"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'}

            try:
                r = requests.get(url, headers=headers)
                soup = BeautifulSoup(r.text, 'lxml')

                result = soup.find('div', class_='Z0LcW')

                print("Question: " + "What was your high school mascot?")
                print("Most Likely Predicted Answer: " + result.text)

            except AttributeError:
                pass

        elif "book" in each_question:
            book = answer.split(" ")
            book_search = ""
            for each_word in book:
                book_search += each_word + "%20"

            url = 'https://www.google.com/search?q=' + book_search + "author"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'}

            try:
                r = requests.get(url, headers=headers)
                soup = BeautifulSoup(r.text, 'lxml')

                result = soup.find('div', class_='Z0LcW')

                print("Question: " + "Who is your all-time favorite author?")
                print("Second Likely Predicted Answer: " + result.text)
            except AttributeError:
                pass


        else:
            print("Question: " + each_question)
            print("Predicted Answer: " + answer)

    #if checkIfUsernameExists(cursor,twitter_username):
    #    pass
        #take data from database
    #else:
    #    cursor.execute("INSERT INTO users_tweets_posts (name,user_name,tweets1) VALUES ('Tricia Dang', 'triciadang7', '" + all_tweets + "')")


    #get_security_answers(tweets_df1,questions_list[1])

    #get_security_answers(tweets_df1)

    #Facebook
    #facebook_excel_path = 'C:\\Users\\trici\\OneDrive\\Documents\\GT\\AlexaGilomenFacebook.xlsx'
    #facebook_df1 = get_all_facebook_posts(facebook_excel_path)

#connect to database - maybe in textfile
# username, all tweets/posts, first question answer, second question,,....
# scrapes twitter and directly put into a format and put directly in database
# extreme: rabbitmq - allow you to have multiple threads that can run in parallel - answer scalability
# amazon sqs - scalability
# collect data of 20 people, build tool that collect their tweets everyday,

if __name__ == "__main__":
    main()


