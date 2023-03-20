
# Need Function that takes URL as input and generates summary statsitics etc
def wiki_summary_url(topic):
    # Import required libraries
    # Must install
    # pip install mediawiki.
    # pip install scikit-learn
    # pip install summa
    from mediawiki import MediaWiki
    import nltk
    from nltk.tokenize import sent_tokenize
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
    # compute summary statistics
    sentences = sent_tokenize(summary_str)
    word_count = len(nltk.word_tokenize(summary_str))
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count
   # sentiment analysis
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(summary_str)

    # store results
    results = [summary_str, score, word_count,
               sentence_count, avg_sentence_length]
    # return statistics
    return results
    # print statistics
    # print(summary_str)
    # print(f"Sentiment Score: {score}")
    # print(f"Word count: {word_count}")
    # print(f"Sentence Count: {sentence_count}")
    # print(f"Average Sentence Length: {avg_sentence_length}")


def main():
    wiki_summary_url('Collapse of Silicon Valley Bank')
    wiki_summary_url('Enron scandal')
    wiki_summary_url('Bankruptcy of Lehman Brothers')
    wiki_summary_url('Bankruptcy of FTX')
    wiki_summary_url('WeWork')
    wiki_summary_url('Theranos')


if __name__ == "__main__":
    main()
