
# Need Function that takes URL as input and generates summary statsitics etc
def wiki_summary(topic):
    # Import required libraries
    # Must install
    # pip install mediawiki.
    # pip install scikit-learn
    # pip install summa
    from mediawiki import MediaWiki
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from summa.summarizer import summarize
    # set site to URL in function in form 'http://
    site = MediaWiki(url='https://en.wikipedia.org/w/api.php')
    page = MediaWiki().page(topic)
    # generate summary in str for topic (given input)
    summary_str = page.summary
    summa_summary = summarize(page, ratio=0.2, words=None)
    # compute summary statistics
    sentences = sent_tokenize(summary_str)
    word_count = len(nltk.word_tokenize(summary_str))
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count
    # sentiment analysis
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(summary_str)\
        # extracting links
    links = page.links
    # keyword extraction
    stop_words = set(stopwords.words('english)'))
    ps = PorterStemmer()
    tokenized_words = [ps.stem(word.lower()) for word in word_tokenize(
        summary_str) if word.lower() not in stop_words and word.isalpha()]
    tfid_matrix = TfidfVectorizer().fit_transform(tokenized_words)
    feature_names =
    # store results
    results = [summary_str, summa_summary, sentiment_score, word_count,
               sentence_count, avg_sentence_length]
    # return statistics
    return results

    # print statistics
    # for i in range(len(results)):
    #     print(f"{results[i]}")
    # print(summary_str)
    # print(f"Sentiment Score: {score}")
    # print(f"Word count: {word_count}")
    # print(f"Sentence Count: {sentence_count}")
    # print(f"Average Sentence Length: {avg_sentence_length}")


def main():
    # results = [summary_str, senitment_score, word_count,sentence_count, avg_sentence_length]
    svb = wiki_summary('Collapse of Silicon Valley Bank')
    enron = wiki_summary('Enron scandal')
    lehman = wiki_summary('Bankruptcy of Lehman Brothers')
    ftx = wiki_summary('Bankruptcy of FTX')
    wework = wiki_summary('WeWork')
    theranos = wiki_summary('Theranos')


if __name__ == "__main__":
    main()
