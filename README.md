# Text-Analysis-Project
 
Please read the [instructions](instructions.md).


1. Project Overview (~1 paragraph)

What data source(s) did you use? What technique(s) did you use to process or analyze them? What did you hope to create or learn through this project?

# Project Overview - Data Sources, techniques, and learning goal. 
For this assignment I used Wikipedia data from infamous business failures, frauds, or collapses using the MediaWiki API. These are denotes in def main() under a topics dictionary that pairs the case key (ex. 'svb) to a more specific wikipedia topic such as "Collapse of Silicon Valley Bank".

I wanted to learn how to use API's and other natural language processing libraries to manipulate text. 

I started with a summarize_wiki function --> returns following dictionary 
results = {'Wiki Summary': summary_wiki,
               "Summa Summary": summa_summary,
               "Summa Keywords as List": summa_keywords_list,
               "Summa Keywords": summa_keywords,
               "Links": links}
And from there I experimented with different techniques to process and analyze - these included stop-word removal and word stemming, summary statistics and words frequencies, sentiment analysis, text-similarity, markov analysis, and lastly using GPT-2 model to generate text based on the input summary from previous functions. 

My goal was to learn how to use these tools - and learn how to search for the tools needed (find new libraries and take advantage of open source) - as opposed to preforming one specific analysis. 

2. Implementation (~1-2 paragraphs + screenshots)

Describe your implementation at a system architecture level. You should NOT walk through your code line by line, or explain every function (we can get that from your docstrings). Instead, talk about the major components, algorithms, data structures and how they fit together. You should also discuss at least one design decision where you had to choose between multiple alternatives, and explain why you made the choice. Use screenshots to describe how you used ChatGPT to help you or learn new things.

# Implementation - Approach, Algo 







image.png






3. Results (~2-3 paragraphs + figures/examples)

Present what you accomplished in your project:

If you did some text analysis, what interesting things did you find? Graphs or other visualizations may be very useful here for showing your results.
If you created a program that does something interesting (e.g. a Markov text synthesizer), be sure to provide a few interesting examples of the program's output.





4. Reflection (~1-2 paragraphs)

From a process point of view, what went well? What could you improve? Was your project appropriately scoped? Did you have a good testing plan?

From a learning perspective, mention what you learned through this project, how ChatGPT helped you, and how you'll use what you learned going forward. What do you wish you knew beforehand that would have helped you succeed?