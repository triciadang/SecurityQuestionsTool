import torch
import nlp
from transformers import LongformerTokenizerFast,LongformerTokenizer, LongformerForQuestionAnswering, EvalPrediction

import json
import os
import re
import string
import numpy as np
import re


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
    encoding = tokenizer.encode_plus(question, text, return_tensors="pt",max_length=750,pad_to_max_length=True, truncation = True)

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
        if i > 500:
            break
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

def get_security_answers(text,question):
    all_cleaned_text = ""
    for each_text in text.get('Text'):
        print("Text: " + each_text)
        all_cleaned_text += normalize_text(each_text) + " "

    print(all_cleaned_text)
    print("Answer: " + longformer(all_cleaned_text,question))

def main():
    db = mysql.connect(
        host="localhost",
        user="root",
        passwd="root",
        database = "security_questions",
    )

    cursor = db.cursor()
    cursor.execute("CREATE TABLE users_tweets_posts (name VARCHAR(255), user_name VARCHAR(255),tweets1 LONGTEXT)")



    print(len("dreaming is believing i still cant believe falcons lost superbowl go falconssss bean sprouts anyone why are bean sprouts so good thunder is best dog ever randy is very good at lost arkand so is chip and ji why is my korean so good bean sprouts is conegamule in korean my favorite food is bean sprouts hi there i like foood "))

    #print(longformer("""I like bean sprouts. Thunder is my dog. Falcons is my favorite football team.""", "What did you call your favorite childhood pet?"))

    questions_list = ["What did you call your favorite childhood pet?","What is your favorite food?","Who is your all-time favorite author?","Who is your favorite actor of all time?","Who is your favorite cartoon character?",
"What is your favorite movie?","What is your favorite place to vacation?","What is your favorite sports team?","What was your favorite school teacher’s name?","What was your high school mascot?"]

    #Twitter
    twitter_username = 'triciadang7'
    #twitter_username = 'selenagomez'
    tweets_df1 = get_all_tweets_from_username(twitter_username)
    get_security_answers(tweets_df1,questions_list[1])

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


