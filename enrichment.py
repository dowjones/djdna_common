import requests
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from textblob import TextBlob

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def assign_embedding_value(arr_item):
    if(type(arr_item) == list):
        return np.zeros(512)
    return arr_item.numpy()


def add_embedding(articles_df, field) -> pd.DataFrame:
    art_list = articles_df[field].tolist()
    emb_items = []
    i = 0
    for item in art_list:
        i += 1
        if(i % 100 == 0):
            print('[{}]'.format(i), end='')
        else:
            print('.', end='')
        try:
            emb_item = embed([item])[0]
        except Exception as e:
            print("ERR: {0}".format(e))
            emb_item = []
        emb_items.append(emb_item)
    articles_df[field + '_embed'] = [assign_embedding_value(e_item) for e_item in emb_items]
    return articles_df


def get_corenlp_sentiment(text_part, corenlp_host):
    corenlp_url = "{}/?properties={{%22annotators%22:%22sentiment%22,%22outputFormat%22:%22json%22}}".format(corenlp_host)
    payload = text_part
    rsp_text = requests.post(corenlp_url, data=payload)
    obj_rsp = eval(rsp_text.content)
    sum_score = 0
    for stc in obj_rsp['sentences']:
        sum_score += sum(stc['sentimentDistribution'])
    return sum_score / len(obj_rsp['sentences'])


def process_corenlp_sentiment(full_text, corenlp_host):
    doc_sentiments = []
    nl_tokens = full_text.split('\n')
    for nl_token in nl_tokens:
        if len(nl_token.strip()) > 1:
            part_sentiment = get_corenlp_sentiment(nl_token, corenlp_host)
            doc_sentiments.append(part_sentiment)
            # d_tokens = nl_token.strip().split('.')
            # for d_token in d_tokens:
            #     if len(d_token.strip()) > 1:
            #         text_parts.append(d_token.strip())
    return sum(doc_sentiments) / len(nl_tokens)


def add_corenlp_sentiment(articles_df, field, corenlp_host) -> pd.DataFrame:
    stmt_value = articles_df[field].apply(process_corenlp_sentiment, corenlp_host=corenlp_host)
    articles_df['corenlp_sentiment_value'] = stmt_value
    return articles_df


def get_textblob_sentiment(full_text):
    blob = TextBlob(full_text)
    return blob.sentiment


def add_textblob_sentiment(articles_df, field) -> pd.DataFrame:
    sentiment_series = articles_df[field].apply(get_textblob_sentiment)  # Series of type textblob.en.sentiments.Sentiment
    articles_df['textblob_sentiment_polarity'] = sentiment_series.apply(lambda x: x.polarity)
    articles_df['textblob_sentiment_subjectivity'] = sentiment_series.apply(lambda x: x.subjectivity)
    articles_df['textblob_sentiment_score'] = sentiment_series.apply(lambda x: x.polarity * (1 - x.subjectivity))
    return articles_df

