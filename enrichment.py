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
    return articles_df

# my_text2 = "*Celgene: U.S. REVLIMID Patent Litigation With Alvogen Settled29 Mar 2019 12:02 ET *Celgene: Alvogen Licensed to Sell Certain Amounts of Generic Lenalidomide Beginning After March 2022\n\n29 Mar 2019 12:03 ET *Celgene: Alvogen Licensed to Sell Unlimited Volumes of Generic Lenalidomide Beginning Jan. 31, 2026\n\n29 Mar 2019 12:20 ET \nPress Release: Celgene Settles U.S. REVLIMID(R) Patent Litigation with Alvogen\n\nCelgene Settles U.S. REVLIMID(R) Patent Litigation with Alvogen\n\nAlvogen licensed to sell volume-limited amounts of generic lenalidomide in the U.S. beginning on a confidential date after the March 2022 date Celgene previously granted to Natco\n\nAlvogen also licensed to sell generic lenalidomide in the U.S. without volume limitation beginning on January 31, 2026\n\nThe earliest licensed entry of any generic lenalidomide in the U.S. continues to be March 2022, based on settlements reached \n\n\nSUMMIT, N.J.--(BUSINESS WIRE)--March 29, 2019-- \n\nCelgene Corporation (NASDAQ:CELG) and Lotus Pharmaceutical Co., Ltd. and Alvogen Pine Brook, LLC (collectively, Alvogen) today announced the settlement of their litigation relating to patents for REVLIMID(R) (lenalidomide).\n\nAs part of the settlement, the parties will file Consent Judgments with the United States District Court for the District of New Jersey that enjoin Alvogen from marketing generic lenalidomide before the expiration of the patents-in-suit, except as provided for in the settlement, as described below.\n\nIn settlement of all outstanding claims in the litigation, Celgene has agreed to provide Alvogen with a license to Celgene's patents required to manufacture and sell certain volume-limited amounts of generic lenalidomide in the United States beginning on a confidential date that is some time after the March 2022 volume-limited license date that Celgene previously provided to Natco. For each consecutive twelve-month period (or part thereof) following the volume-limited entry date until January 31, 2026, the volume of generic lenalidomide sold by Alvogen cannot exceed certain agreed-upon percentages. Although the agreed-upon percentages are confidential, they increase gradually each period to no more than a single-digit percentage in the final volume-limited period. In addition, Celgene has agreed to provide Alvogen with a license to Celgene's patents required to manufacture and sell an unlimited quantity of generic lenalidomide in the United States beginning no earlier than January 31, 2026.\n\nAlvogen's ability to market lenalidomide in the U.S. will be contingent on its obtaining approval of an Abbreviated New Drug Application.\n\nABOUT CELGENE\n\nCelgene Corporation, headquartered in Summit, New Jersey, is an integrated global pharmaceutical company engaged primarily in the discovery, development and commercialization of innovative therapies for the treatment of cancer and inflammatory diseases through gene and protein regulation. For more information, please visit the Company's website at www.celgene.com. Follow Celgene on Social Media: @Celgene, Pinterest, LinkedIn, Facebook and YouTube.\n\nAbout REVLIMID(R)\n\nIn the U.S., REVLIMID(R) (lenalidomide) in combination with dexamethasone is indicated for the treatment of patients with multiple myeloma. REVLIMID(R) as a single agent is also indicated as a maintenance therapy in patients with multiple myeloma following autologous hematopoietic stem cell transplant. REVLIMID(R) is indicated for patients with transfusion-dependent anemia due to low- or intermediate-1-risk myelodysplastic syndromes (MDS) associated with a deletion 5q cytogenetic abnormality with or without additional cytogenetic abnormalities. REVLIMID(R) is approved in the U.S. for the treatment of patients with mantle cell lymphoma (MCL) whose disease has relapsed or progressed after two prior therapies, one of which included bortezomib. Limitations of Use: REVLIMID(R) is not indicated and is not recommended for the treatment of chronic lymphocytic leukemia (CLL) outside of controlled clinical trials.\n\nForward-Looking Statement\n\nThis press release contains forward-looking statements, which are generally statements that are not historical facts. Forward-looking statements can be identified by the words \"expects,\" \"anticipates,\" \"believes,\" \"intends,\" \"estimates,\" \"plans,\" \"will,\" \"outlook\" and similar expressions. Forward-looking statements are based on management's current plans, estimates, assumptions and projections, and speak only as of the date they are made. We undertake no obligation to update any forward-looking statement in light of new information or future events, except as otherwise required by law. Forward-looking statements involve inherent risks and uncertainties, most of which are difficult to predict and are generally beyond our control. Actual results or outcomes may differ materially from those implied by the forward-looking statements as a result of the impact of a number of factors, many of which are discussed in more detail in our Annual Report on Form 10-K and our other reports filed with the U.S. Securities and Exchange Commission, including factors related to the proposed transaction between Bristol-Myers Squibb and Celgene, such as, but not limited to, the risks that: management's time and attention is diverted on transaction related issues; disruption from the transaction makes it more difficult to maintain business, contractual and operational relationships; pending legal proceedings or any future litigation instituted against Bristol-Myers Squibb, Celgene or the combined company could delay or prevent the proposed transaction; and Bristol-Myers Squibb, Celgene or the combined company is unable to retain key personnel.\n\nHyperlinks are provided as a convenience and for informational purposes only. Celgene bears no responsibility for the security or content of external websites.\n\nView source version on businesswire.com: https://www.businesswire.com/news/home/20190329005384/en/ \n\n\n \n    CONTACT:    Celgene Corporation \n\nInvestors:\n\n908-673-9628\n\nir@celgene.com\n\nMedia:\n\n908-673-2275\n\nmedia@celgene.com\n\n(END) Dow Jones Newswires\n\nMarch 29, 2019 12:20 ET (16:20 GMT)"
# my_text1 = "This product is awesome. I have used it several times with no issues at all"

# all_stcs = process_corenlp_sentiment(my_text2, "http://localhost:9000")
# my_df = pd.DataFrame([[my_text1],[my_text2]], columns=['text'])

# stv = add_corenlp_sentiment(my_df, 'text', "http://localhost:9000")
# print("Done!")
