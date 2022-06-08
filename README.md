# NEWS_Article_sentiment_Analysis

The purpose of the document is to explain the details of the peoject:

Code Flow:

Step 1: Get NEWS link from given URL 

a. Pull all the NEWS articles URL from the page url = "https://www.aljazeera.com/where/mozambique/" using python package request and BeautifulSoup.
b. I did not pull only top 10 links becuase few NEWS article were missing description.
c. On close exmination of the NEWS link it was observe that it contains Date in the URL and were arranged from latest to older 
example : https://www.aljazeera.com/news/2022/5/23/floods-hit-south-africas-kwazulu-natal-province-again',
d. So not reordering of NEWS url was required 

step 2: Get NEWS decription and headlines from the extracted link 
a. Pull the Headline and description from the NEWS links 
b. Filter top 10 valid links and break
c. Store the data into Pandas Data frame 

Step 3: Data Cleaning 
a. Apply the below data cleaning for both headline and decription in the dataframe
    convert to lower_case 
    remove punctuation
    lemmatize
    remove stopwords

Step 4: Applied 3 an off-the-shelf library for sentiment analysis (NLTK (VADER), TextBlob)

NLTK (VADER) and TextBlob
Both of these packages rely on a rules-based sentiment analyzer. It, therefore, attaches a positive or negative rating to certain words (ex. horrible has a negative association), pays attention to negation if it exists, and returns values based on these words. This tends to work fine, and has the advantage of being simple and extremely fast, but has some weaknesses.

As sentences get longer, more neutral words exist, and therefore, the overall score tends to normalize more towards neutral as well (or does it)

Flair
Flair is a pre-trained embedding-based model. This means that each word is represented inside a vector space. Words with vector representations most similar to another word are often used in the same context. This allows us, to, therefore, determine the sentiment of any given vector, and therefore, any given sentence. The embeddings are based on this paper if you are curious about the more technical aspects.

Flair tends to be much slower than its rule-based counterparts but comes at the advantage of being a trained NLP model instead of a rule-based model, which, if done well comes with added performance. To put in perspective how much slower, in running 1200 sentences, NLTK took 0.78 seconds, Textblob took an impressive 0.55 seconds, and Flair took 49 seconds (50–100x longer), which begs the question of whether the added accuracy is truly worth the increased runtime.

Step 5: visulization
a. For each Sentiment analysis technique result visulization was done 
b. Viualization picture is stored in visulization folder

Total TIme 
15 Sec