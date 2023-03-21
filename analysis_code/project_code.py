from mediawiki import MediaWiki
from summa import summarizer, keywords
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Need Function that takes URL as input and generates summary statistics etc
def summarize_wiki(topic):
    '''
    Takes topic as a string and pulls data from MediaWiki 
    1) Generates Wiki Summary using Media Wiki Function
    2) Uses Summa to create summary and find Keywords from Content
    3) Pull Links for later use
    4) Store results in dictionary for later analysis

    '''
    # set site to URL in function in form 'http://.. and pass topic
    site = MediaWiki(url='https://en.wikipedia.org/w/api.php')
    page = MediaWiki().page(topic)

    # generate summary using MediaWiki built in Function
    summary_str = page.summary

    # generate summary and keywords extraction using summa and store keywords in list
    full_content = page.content
    summa_summary = summarizer.summarize(full_content, ratio=0.2, words=None)
    summa_keywords_list = keywords.keywords(full_content, split=True)
    summa_keywords = keywords.keywords(full_content)

    # extracting links
    links = page.links

    # store results
    results = {'Wiki Summary': summary_str,
               "Summa Summary": summa_summary,
               "Summa Keywords as List": summa_keywords_list,
               "Summa Keywords": summa_keywords,
               "Links": links}
    return results


def process_words(words_list):
    '''
    Function used to clean up a list of words by removing stop words (the, of etc and ensure words are alphabetical)
    2) Stem words for further accuracy
    '''

    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for word in words_list:
        if word.isalpha():
            if word.lower() not in stop_words:
                filtered_words.append(word)

    # stem the words
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))

    return stemmed_words


def summary_statistics(dict, result_type):
    '''
    1) Uses nltk to tokenize sentences and words 
    2) Calls remove_stop_words function to clean up words list 
    3) Produce sumamry results from specified dictionary

    '''
    # Tokenize the text summary from previous function into sentences and words
    sentences = sent_tokenize(dict[result_type])
    words = word_tokenize(dict[result_type])

    # Remove stopwords from list calling remove_stop_words() function
    words = process_words(words)

    # create dictionary of summary results
    word_count = len(nltk.word_tokenize(dict[result_type]))
    sentence_count = len(sentences)
    unique_word_count = len(set(words))
    avg_sentence_length = word_count / sentence_count

    # Generate frequency distribution for words
    word_freq_dist = FreqDist(words)
    top_20_words = word_freq_dist.most_common(20)

    result_statistics = {"Word Count": word_count,
                         "Unique Word Count": unique_word_count,
                         "Sentence Count": sentence_count,
                         "Avg. Sentence Length": avg_sentence_length,
                         "Top 20 Most Common Words": top_20_words}

    return result_statistics


def sentiment_analyzer(dict, result_type):
    '''
    generalized function for sentiment analysis built to take results from wiki_summary function


    '''
    # sentiment analysis
    sentiment = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment.polarity_scores(dict[result_type])
    return sentiment_scores


def calculate_cosine_similarity(summary1, summary2):
    '''
    Will take two summary texts - from previous functions -- 
    uses tfid vectorize from sklearn and calculates the cosine similarity using simialrity between vectors

    '''
    # tokenize summaries
    tokenized_text1 = sent_tokenize(summary1)
    tokenized_text2 = sent_tokenize(summary2)

    # process the words
    processed_summary1, processed_summary2 = "".join(process_words(
        tokenized_text1)), " ".join(process_words(tokenized_text2))

    # Vectorize the processed text
    vectorize = TfidfVectorizer()
    vectorized_summary1 = vectorize.fit_transform([processed_summary1])
    vectorized_summary2 = vectorize.fit_transform([processed_summary2])

    # calculate cosine simialrity between vectors
    cosine_similarity = cosine_similarity(
        vectorized_summary1, vectorized_summary2)

    return cosine_similarity


def main():

    # from nltk.stem import PorterStemmer

    # Import required libraries
    # Must install
    # pip install mediawiki.
    # pip install scikit-learn
    # pip install summa
    # pip instal nltk

    topics = {'svb': "Collapse of Silicon Valley Bank",
              'enron': 'Enron Scandal',
              'lehman': 'Bankruptcy of Lehman Brothers',
              'ftx': 'Bankruptcy of FTX',
              'wework': 'WeWork',
              'theranos': 'Theranos'
              }

    # Mock Analysis for SVB
    svb = summarize_wiki(topics['svb'])
    svb_summary_stats = summary_statistics(svb, "Wiki Summary")
    print(svb_summary_stats)
    print(sentiment_analyzer(svb, "Wiki Summary"))
    print(sentiment_analyzer(svb, "Summa Summary"))
    print(sentiment_analyzer(svb, "Summa Keywords"))

    # enron = summarize_wiki('Enron scandal')
    # lehman = summarize_wiki('Bankruptcy of Lehman Brothers')
    # ftx = summarize_wiki('Bankruptcy of FTX')
    # wework = summarize_wiki('WeWork')
    # theranos = summarize_wiki('Theranos')


if __name__ == "__main__":
    main()
