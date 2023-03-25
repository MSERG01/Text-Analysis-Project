import nltk
import pprint
from mediawiki import MediaWiki
from summa import summarizer, keywords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from thefuzz import fuzz
import markovify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re


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
    summary_wiki = page.summary

    # generate summary and keywords extraction using summa and store keywords in list
    full_content = page.content
    summa_summary = summarizer.summarize(full_content, ratio=0.2, words=None)
    summa_keywords_list = keywords.keywords(full_content, split=True)
    summa_keywords = keywords.keywords(full_content)

    # extracting links
    links = page.links

    # store results
    results = {'Wiki Summary': summary_wiki,
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


def summary_statistics(summary):
    '''
    1) Uses nltk to tokenize sentences and words 
    2) Calls remove_stop_words function to clean up words list 
    3) Produce summary results from specified dictionary

    '''
    # Tokenize the text summary from previous function into sentences and words
    sentences = sent_tokenize(summary)
    words = word_tokenize(summary)
    # print(words)
    # Remove stopwords from list calling remove_stop_words() function
    words = process_words(words)

    # create dictionary of summary results
    word_count = len(words)
    sentence_count = len(sentences)
    unique_word_count = len(set(words))
    avg_sentence_length = round((word_count / sentence_count), 0)

    # Generate frequency distribution for words
    word_freq_dist = FreqDist(words)
    top_20_words = word_freq_dist.most_common(20)

    result_statistics = {"Word Count": word_count,
                         "Unique Word Count": unique_word_count,
                         "Sentence Count": sentence_count,
                         "Avg. Sentence Length": avg_sentence_length,
                         "Top 20 Most Common Words": top_20_words}

    return result_statistics


def sentiment_analyzer(text):
    '''
    generalized function for sentiment analysis built 
    '''
    # sentiment analysis
    sentiment = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment.polarity_scores(text)
    return sentiment_scores


def text_similarity(text1, text2):
    '''Uses fuzz library to calc similarity between tow texts with score out of 100'''
    return fuzz.token_sort_ratio(text1, text2)


def markovify_funct(summary):
    ''''
   uses markovify library to PRINT 10 probability bases sentences based on summary provided
   '''
    # 1) Get Raw Text as String
    text = summary  # redundant but helpful if several summaries are inputted

    # 2) Build Model
    text_model = markovify.Text(text)

    # 3) Print Randomly generated text (10 sentences)
    for i in range(10):
        print(text_model.make_sentence())


def gpt_text_generation(sequence):

    # initialize tokenizer and model from pretrained GPT2 Model

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    prompt = " This resulted in, "

    sequence = sequence + prompt

    inputs = tokenizer.encode(sequence, return_tensors='pt')

    # set output parameters
    outputs = model.generate(inputs, max_length=200, do_sample=True)

    # decode array of tokens into words
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text_output


def trim_summary(summary):
    '''
    Used to provide shorter input summary for GPT_text_generation summary
    '''
    sentences = re.split('[.?!]', summary)
    new_text = ". ".join(sentences[:5])

    return new_text


def summarize_all(topics):
    '''
    summarize_all takes dictionary of inputs as input and outputs dictionary with these results and stores them as dictionary pair 
    results = {'Wiki Summary': summary_wiki,
               "Summa Summary": summa_summary,
               "Summa Keywords as List": summa_keywords_list,
               "Summa Keywords": summa_keywords,
               "Links": links}

    for example  --> 'svb: {'Wiki Summary': summary_wiki,
               "Summa Summary": summa_summary,
               "Summa Keywords as List": summa_keywords_list,
               "Summa Keywords": summa_keywords,
               "Links": links} , 'enron': ....

    this calls summarize wiki function and returns dictionary with data for every subsequent analysis

    '''
    results = {}
    for topic in topics:
        results[topic] = summarize_wiki(topics[topic])
    return results


def stats_all(topics, key):
    ''''
    1) Uses nltk to tokenize sentences and words 
    2) Calls remove_stop_words function to clean up words list 
    3) Produce summary results from specified dictionary
    '''
    results = {}
    for topic in topics:
        results[topic] = summary_statistics(topics[topic][key])
    return results


def sentiment_all(topics, key):
    ''''
    1) Uses nltk to tokenize sentences and words 
    2) Calls remove_stop_words function to clean up words list 
    3) Produce summary results from specified dictionary
    '''
    results = {}
    for topic in topics:
        results[topic] = sentiment_analyzer(topics[topic][key])
    return results


def main():

    all_topics = {'svb': "Collapse of Silicon Valley Bank",
                  'enron': 'Enron Scandal',
                  'lehman': 'Bankruptcy of Lehman Brothers',
                  'ftx': 'Bankruptcy of FTX',
                  'wework': 'WeWork',
                  'theranos': 'Theranos',
                  'worldcom': "Worldcom Scandal",
                  'ltcm': "Long-Term Capital Management"
                  }

    ''''
    1. Specify topic ticker (ex. 'svb') 
    all_topics = {'svb': "Collapse of Silicon Valley Bank",
                  'enron': 'Enron Scandal',
                  'lehman': 'Bankruptcy of Lehman Brothers',
                  'ftx': 'Bankruptcy of FTX',
                  'wework': 'WeWork',
                  'theranos': 'Theranos',
                  'worldcom': "Worldcom Scandal",
                  'ltcm': "Long-Term Capital Management"
                  }
    
    2. Specify desired result(s)
    results = {'Wiki Summary': summary_wiki,
               "Summa Summary": summa_summary,
               "Summa Keywords as List": summa_keywords_list,
               "Summa Keywords": summa_keywords,
               "Links": links}
    
    3. For example - to get svb wiki summary the following would be used
    topics_summary_dictionary['svb']['Wiki Summary']
    '''

    # Initially store all data in dictionary to use in later analysis  (see docstring)
    topics_summary_dictionary = summarize_all(all_topics)
    print("-"*50)
    pprint.pprint(topics_summary_dictionary['svb']['Wiki Summary'])

    # use stats_all to return dictionary of each topics and their key stats
    print("-"*50)
    pprint.pprint(stats_all(topics_summary_dictionary, 'Summa Summary'))

    # Sentiment of all summa summaries
    print("-"*50)
    pprint.pprint(sentiment_all(topics_summary_dictionary, 'Summa Summary'))

    # sample of text_similarity function using dictionary
    print("-"*50)
    pprint.pprint(text_similarity(
        topics_summary_dictionary['svb']['Summa Summary'], topics_summary_dictionary['ftx']['Summa Summary']))

    # Uses Markov function to generate 10 sentences based on inputted summary.
    print("-"*50)
    print("The following is randomly generated sentences based off input")
    markovify_funct(topics_summary_dictionary['wework']['Summa Summary'])

    # Lastly use GPT2 preloaded model to generate text based on previous characteristics
    # The Last 200 words after "THIS RESULTS IN" are generated by the code and were not summarized by Wikipedia
    print("-"*50)
    pprint.pprint(gpt_text_generation(trim_summary(
        topics_summary_dictionary['theranos']['Wiki Summary'])))


if __name__ == "__main__":
    main()
