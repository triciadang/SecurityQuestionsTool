import torch
import nlp
from transformers import LongformerTokenizerFast,LongformerTokenizer, LongformerForQuestionAnswering, EvalPrediction
import string
import re
import numpy as np
import requests, lxml
from bs4 import BeautifulSoup
import snscrape.modules.twitter as sntwitter
import pandas as pd
from IPython.display import display
import mysql.connector as mysql
import nltk

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#Fetching the pretrained model
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
model = LongformerForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
high_avg_score = -1000
current_answer = ""

 # Basic preprocessing functions
def normalize_text(text):
     text = text.lower()
     text = "".join(ch for ch in text if ch not in set(string.punctuation))
     regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
     text = re.sub(regex, " ", text)
     text = " ".join(text.split())
     return text

def number_of_tokens(text):
    nltk_tokens = nltk.word_tokenize(str(text))
    return len(nltk_tokens)

# The actual function that does the job
def longformer(text,question):
    global high_avg_score
    global current_answer
    #encoding = tokenizer.encode_plus(question, text, return_tensors="pt",max_length=750,pad_to_max_length=True)
    encoding = tokenizer.encode_plus(question, text, return_tensors="pt", max_length=4096, truncation=True)
    input_ids = encoding["input_ids"]

    # default is local attention everywhere
    # the forward method will automatically set global attention on question tokens
    attention_mask = encoding["attention_mask"]

    start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)
    print(start_scores)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
    highest_start_score = torch.argmax(start_scores)
    highest_end_score = torch.argmax(end_scores)
    highest_start_score = torch.argmax(start_scores)
    print(highest_start_score)
    print(highest_end_score)
    print(float(torch.max(start_scores)))
    print(float(torch.max(end_scores)))
    avg_of_start_end_score = (float(torch.max(start_scores)) + float(torch.max(end_scores)))/2

    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

    if avg_of_start_end_score > high_avg_score:
        high_avg_score = avg_of_start_end_score
        current_answer = answer


    #https://stackoverflow.com/questions/49268359/scrape-google-quick-answer-box-in-python
    if "What high school did you go to?" in question:
        high_school = answer.split(" ")
        high_school_search = ""
        for each_word in high_school:
            high_school_search += each_word + "%20"

        url = 'https://www.google.com/search?q=' + high_school_search + "mascot"
        print(url)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'}

        try:
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, 'lxml')

            result = soup.find('div', class_='Z0LcW')

            answer = result.text

            print("Question: " + "What was your high school mascot?")
            print("Most Likely Predicted Answer: " + result.text)

        except AttributeError:
            pass

    elif "book" in question:
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

            answer = result.text

            print("Question: " + "Who is your all-time favorite author?")
            print("Second Likely Predicted Answer: " + result.text)
        except AttributeError:
            pass



    return

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

    #display(facebook_df)

    #for each in facebook_df.get('Text'):
    #    print("==============")
    #    print(each)

    return facebook_df

def concatenate_all_posts(text):
    all_cleaned_text = ""
    for each_text in text.get('Text'):
        if str(each_text)[:1] in string.punctuation:
            all_cleaned_text += str(each_text)
        else:
            all_cleaned_text += str(each_text) + ". "

    #
    # groups_of_4096_length = []
    # group_index = 0
    # groups_of_4096_length.append("")
    # length = 0

    # for each_text in text.get('Text'):
    #     print(length)
    #     if (length + len(each_text) + len(".")) <= 4096:
    #         if str(each_text)[:1] in string.punctuation:
    #             groups_of_4096_length[group_index] += str(each_text)
    #         else:
    #             groups_of_4096_length[group_index] += str(each_text) + ". "
    #         length += len(each_text)
    #
    #     else:
    #         group_index += 1
    #         groups_of_4096_length.append("")
    #         if str(each_text)[:1] in string.punctuation:
    #             groups_of_4096_length[group_index] += str(each_text)
    #         else:
    #             groups_of_4096_length[group_index] += str(each_text) + ". "
    #         length = len(each_text)

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

    cursor.execute("CREATE TABLE IF NOT EXISTS users_tweets_posts (user_name VARCHAR(255) PRIMARY KEY,tweets1 LONGTEXT)")
    db.commit()

    questions_list = ["What did you call your favorite childhood pet?","What is your favorite food?","Who is your all-time favorite author?","What was your favorite book?","Who is your favorite actor of all time?","Who is your favorite cartoon character?",
"What is your favorite movie?","What is your favorite place to vacation?","What is your favorite sports team?","What was your favorite school teacher’s name?","What was your high school mascot?","What high school did you go to?"]

    #Twitter

    #Working Twitter username Example
    #twitter_username = 'triciadang7'

    #Other tries
    twitter_username = 'EvrydayShortcut'
    #tweets_df1 = get_all_tweets_from_username(twitter_username)
    #tweets_df1.to_excel(twitter_username + ".xlsx")
    excel_path = twitter_username + ".xlsx"

    excel_path = 'triciadang7.xlsx'

    #From Facebook Example
    #Working Facebook excel
    #excel_path = 'Sample2.xlsx'

    facebook_df1 = get_all_facebook_posts(excel_path)

    #Database Infrastructure
    if checkIfUsernameExists(cursor,twitter_username):
        pass
        cursor.execute("SELECT tweets1 FROM users_tweets_posts WHERE user_name= '" + twitter_username + "'")
    # take data from database
    else:
       cursor.execute("INSERT INTO users_tweets_posts (name,user_name,tweets1) VALUES ('Tricia Dang', 'triciadang7', '" + all_tweets + "')")
       db.commit()


    groups_of_token =  ""

    # for each_question in questions_list:
    each_question = questions_list[1]
    for each_text in facebook_df1.get('Text'):
        print(each_text)
        if str(each_text)[-1] in string.punctuation:
            all_cleaned_text = str(each_text)
        else:

            all_cleaned_text = str(each_text) + ". "

        tokens_in_post = number_of_tokens(each_text)

        print(number_of_tokens(groups_of_token))

        #if last one in dataframe, then run longformer function
        if each_text == facebook_df1['Text'].iloc[-1]:
            print(tokens_in_post)
            print(groups_of_token)
            longformer(groups_of_token,each_question)

        #if not last one, keep adding to current group of 4096
        elif tokens_in_post + number_of_tokens(groups_of_token) < 4090:
            groups_of_token += all_cleaned_text

        #once reach length of 4096, run longformer function
        else:
            longformer(groups_of_token, each_question)
            groups_of_token = all_cleaned_text

    print("Question: " + each_question)
    if current_answer is not None:
        print("Answer: " + str(current_answer))
    else:
        print("Answer: Unknown")



if __name__ == "__main__":
    main()


