# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:47:36 2018

@author: asengup6
"""
from time import time
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error
from itertools import combinations
from toolz import compose
from sklearn.model_selection import train_test_split
import scipy
from scipy.special import gammaln, psi
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import ast

st = PorterStemmer()
MAX_VOCAB_SIZE = 50000
        
def save_document_image(filename, doc, zoom=2):
    """
    Save document as an image.
    doc must be a square matrix
    """
    height, width = doc.shape
    zoom = np.ones((width*zoom, width*zoom))
    # imsave scales pixels between 0 and 255 automatically
    scipy.misc.imsave(filename, np.kron(doc, zoom))

class SkipGramVectorizer(CountVectorizer):

    def __init__(self, k=1, **kwds):
        super(SkipGramVectorizer, self).__init__(**kwds)
        self.k=k

    def build_sent_analyzer(self, preprocess, stop_words, tokenize):
        return lambda sent : self._word_skip_grams(
                compose(tokenize, preprocess, self.decode)(sent),
                stop_words)

    def build_analyzer(self):    
        preprocess = self.build_preprocessor()
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()
        sent_analyze = self.build_sent_analyzer(preprocess, stop_words, tokenize)

        return lambda doc : self._sent_skip_grams(doc, sent_analyze)

    def _sent_skip_grams(self, doc, sent_analyze):
        skip_grams = []
        for sent in nltk.sent_tokenize(doc):
            skip_grams.extend(sent_analyze(sent))
        return skip_grams

    def _word_skip_grams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        k = self.k
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in np.arange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in np.arange(n_original_tokens - n + 1):
                    # k-skip-n-grams
                    head = [original_tokens[i]]                    
                    for skip_tail in combinations(original_tokens[i+1:i+n+k], n-1):
                        tokens_append(space_join(head + list(skip_tail)))
        return tokens

def sampleFromDirichlet(alpha):
    """
    Sample from a Dirichlet distribution
    alpha: Dirichlet distribution parameter (of length d)
    Returns:
    x: Vector (of length d) sampled from dirichlet distribution
    """
    return np.random.dirichlet(alpha)

def sampleFromCategorical(theta):
    """
    Samples from a categorical/multinoulli distribution
    theta: parameter (of length d)
    Returns:
    x: index ind (0 <= ind < d) based on probabilities in theta
    """
    theta = theta/np.sum(theta)
    return np.random.multinomial(1, theta).argmax()

def word_indices(wordOccuranceVec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in wordOccuranceVec.nonzero()[0]:
        for i in range(int(wordOccuranceVec[idx])):
            yield idx
            
class SentimentLDAGibbsSampler:

    def __init__(self, numTopics, alpha, beta, gamma, numSentiments=100, minlabel = 0, maxlabel = 10, SentimentRange = 10): 
        """
        numTopics: Number of topics in the model
        numSentiments: Number of sentiments (default 2)
        alpha: Hyperparameter for Dirichlet prior on topic distribution
        per document
        beta: Hyperparameter for Dirichlet prior on vocabulary distribution
        per (topic, sentiment) pair
        gamma:Hyperparameter for Dirichlet prior on sentiment distribution
        per (document, topic) pair
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.numTopics = numTopics
        self.numSentiments = numSentiments
        self.minlabel = minlabel
        self.maxlabel = maxlabel
        self.SentimentRange = SentimentRange
        self.probabilities_ts = {}

    def processReviews(self, reviews, window=5):

        self.vectorizer = SkipGramVectorizer(analyzer="word",stop_words="english",max_features=MAX_VOCAB_SIZE,max_df=.75,min_df=10, k = window,ngram_range=(1,2))
        #self.vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words="english",max_features=MAX_VOCAB_SIZE,max_df=.5,min_df=5)
        train_data_features = self.vectorizer.fit_transform(reviews)
        words = self.vectorizer.get_feature_names()
        self.vocabulary = dict(zip(words,np.arange(len(words))))
        self.inv_vocabulary = dict(zip(np.arange(len(words)),words))
        wordOccurenceMatrix = train_data_features.toarray()
        return wordOccurenceMatrix
        
    def create_priorsentiment(self):
        sid = SentimentIntensityAnalyzer()
        l = []
        binsize = self.SentimentRange*1.0/self.numSentiments
        for i in self.vocabulary:
            l.append(sid.polarity_scores(i).get('compound',np.nan))
        clf = MinMaxScaler(feature_range = (self.minlabel,self.maxlabel))
        l = clf.fit_transform(np.array(l))
        l = [min(int(i/binsize)-1,0) for i in l]
        self.priorSentiment = dict(zip(list(self.vocabulary.keys()),l))

    def _initialize_(self, reviews, labels, unlabeled_reviews,skipgramwindow):
        """
        wordOccuranceMatrix: numDocs x vocabSize matrix encoding the
        bag of words representation of each document
        """
        allreviews = reviews + unlabeled_reviews
        self.wordOccuranceMatrix = self.processReviews(allreviews)
        #self.create_priorsentiment()
        numDocs, vocabSize = self.wordOccuranceMatrix.shape
        
        numDocswithlabels = len(labels)
        # Pseudocounts
        self.n_dt = np.zeros((numDocs, self.numTopics))
        self.n_dts = np.zeros((numDocs, self.numTopics, self.numSentiments))
        self.n_d = np.zeros((numDocs))
        self.n_vts = np.zeros((vocabSize, self.numTopics, self.numSentiments))
        self.n_ts = np.zeros((self.numTopics, self.numSentiments))
        self.dt_distribution = np.zeros((numDocs, self.numTopics))
        self.dts_distribution = np.zeros((numDocs, self.numTopics, self.numSentiments))
        self.topics = {}
        self.sentiments = {}
        self.sentimentprior = {}

        self.alphaVec = self.alpha.copy()
        self.gammaVec = self.gamma.copy() #self.gamma * np.ones(self.numSentiments)   

        self.allbigrams = {}
        self.totalbigrams = []
        
        for d in range(numDocs):

            if d < numDocswithlabels:
                Doclabel = labels[d]
                binsize = self.SentimentRange*1.0/self.numSentiments
                DoclabelMatrix = np.eye(self.numSentiments)*.2
                DoclabelMatrix[max(int(Doclabel/binsize)-1,0),max(int(Doclabel/binsize)-1,0)] = 1.2
                gammaVec = np.matmul(DoclabelMatrix,self.gammaVec)
                self.sentimentprior[d] = gammaVec
            else:
                self.sentimentprior[d] = np.array(self.gammaVec)
            
            topicDistribution = sampleFromDirichlet(self.alphaVec)
            sentimentDistribution = np.zeros((self.numTopics, self.numSentiments))
            for t in range(self.numTopics):
                sentimentDistribution[t, :] = sampleFromDirichlet(gammaVec)
            
            temp = SkipGramVectorizer(analyzer="word",stop_words=stopwords.words(),max_features=MAX_VOCAB_SIZE, k = skipgramwindow, ngram_range=(2,2))
            try:
                train_data_features = temp.fit_transform([allreviews[d]])
                bigrams = temp.get_feature_names()
                self.totalbigrams += bigrams
                self.allbigrams[d] = bigrams
                
                for i, w in enumerate(self.allbigrams[d]):
                    t = sampleFromCategorical(topicDistribution)
                    s = sampleFromCategorical(sentimentDistribution[t, :])
                            
                    word1 = w.split()[0]
                    word2 = w.split()[1]
                    if word1 in self.vocabulary and word2 in self.vocabulary:
                        i1 = self.vocabulary[word1]
                        i2 = self.vocabulary[word2]
                        
                        self.topics[(d, i)] = t
                        self.sentiments[(d, i)] = s
                        self.n_dt[d, t] += 1
                        self.n_dts[d, t, s] += 1
                        self.n_d[d] += 1
                        self.n_ts[t, s] += 1   
                        self.n_vts[i1, t, s] += 1
                        self.n_vts[i2, t, s] += 1
            
            except:
                pass
                
            self.dt_distribution[d,:] = (self.n_dt[d] + self.alphaVec) / \
            (self.n_d[d] + np.sum(self.alphaVec))
            for k in range(self.numTopics):
                self.dts_distribution[d,k,:] = (self.n_dts[d, k, :] + self.sentimentprior[d]) / \
                (self.n_dt[d, k] + np.sum(self.sentimentprior[d]))
        
        self.numbigrams = len(set(self.totalbigrams))
        
    def conditionalDistribution(self, d, v):
        """
        Calculates the (topic, sentiment) probability for word v in document d
        Returns:    a matrix (numTopics x numSentiments) storing the probabilities
        """
        probabilities_ts = np.ones((self.numTopics, self.numSentiments))
        firstFactor = (self.n_dt[d] + self.alphaVec) / \
            (self.n_d[d] + np.sum(self.alphaVec))
        
        secondFactor = np.zeros((self.numTopics,self.numSentiments))
        for k in range(self.numTopics):
            secondFactor[k,:] = (self.n_dts[d, k, :] + self.sentimentprior[d]) / \
                (self.n_dt[d, k] + np.sum(self.sentimentprior[d]))
				
        word = self.allbigrams[d][v]
        word1 = word.split()[0]
        word2 = word.split()[1]
        if word1 in self.vocabulary and word2 in self.vocabulary:
            i1 = self.vocabulary[word1]
            i2 = self.vocabulary[word2]
            
            thirdFactor = (self.n_vts[i1, :, :] + self.beta) * (self.n_vts[i2, :, :] + self.beta)/(self.n_ts + self..wordOccuranceMatrix.shape[1] * self.beta)**2
                
            probabilities_ts *= firstFactor[:, np.newaxis]
            probabilities_ts *= secondFactor * thirdFactor
            #probabilities_ts = np.exp(probabilities_ts)
            probabilities_ts /= np.sum(probabilities_ts)

        return probabilities_ts

    def getTopKWords(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(v | t, s) is maximum
        """
        pseudocounts = np.copy(self.n_vts)
        normalizer = np.sum(pseudocounts, (0))
        pseudocounts /= normalizer[np.newaxis, :, :]
        worddict = {}
        for t in range(self.numTopics):
            worddict[t] = {}
            for s in range(self.numSentiments):
                topWordIndices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                vocab = self.vectorizer.get_feature_names()
                worddict[t][s] = [vocab[i] for i in topWordIndices]
        return worddict


    def run(self, reviews, labels, unlabeled_reviews, maxIters=100, skipgramwindow=5):
        """
        Runs Gibbs sampler for sentiment-LDA
        """
        self._initialize_(reviews, labels, unlabeled_reviews, skipgramwindow)
        self.loglikelihoods = np.zeros(maxIters)
        numDocs, vocabSize = self.wordOccuranceMatrix.shape
        for iteration in range(maxIters):
            print ("Starting iteration %d of %d" % (iteration + 1, maxIters))
            loglikelihood = 0
            for d in range(numDocs):
            
                if d in self.allbigrams:
                    
                    for i, w in enumerate(self.allbigrams[d]):
                    
                        word1 = w.split()[0]
                        word2 = w.split()[1]
                        
                        if word1 in self.vocabulary and word2 in self.vocabulary:
                            i1 = self.vocabulary[word1]
                            i2 = self.vocabulary[word2]
                                
                            t = self.topics[(d, i)]
                            s = self.sentiments[(d, i)]
                            self.n_dt[d, t] -= 1
                            self.n_d[d] -= 1
                            self.n_dts[d, t, s] -= 1
                            self.n_vts[i1, t, s] -= 1
                            self.n_vts[i2, t, s] -= 1
                            self.n_ts[t, s] -= 1

                            probabilities_ts = self.conditionalDistribution(d, i)
                            #if v1 in self.priorSentiment and v2 in self.priorSentiment:
                            #    s = int(self.priorSentiment[v1]*.5 + .5*self.priorSentiment[v2])
                            #    t = sampleFromCategorical(probabilities_ts[:, s])
                            #else:
                            ind = sampleFromCategorical(probabilities_ts.flatten())
                            t, s = np.unravel_index(ind, probabilities_ts.shape)
                                    
                            self.probabilities_ts[(d, i)] = probabilities_ts[t,s]
                            #loglikelihood += np.log(self.probabilities_ts[(d, i)])
                            
                            self.topics[(d, i)] = t
                            self.sentiments[(d, i)] = s
                            self.n_dt[d, t] += 1
                            self.n_d[d] += 1
                            self.n_dts[d, t, s] += 1
                            self.n_vts[i1, t, s] += 1
                            self.n_vts[i2, t, s] += 1
                            self.n_ts[t, s] += 1
                                
                if iteration == maxIters - 1:
                    self.dt_distribution[d,:] = (self.n_dt[d,:] + self.alphaVec) / \
                    (self.n_d[d] + np.sum(self.alphaVec))
                    self.dts_distribution[d,:,:] = (self.n_dts[d, :, :] + self.sentimentprior[d]) / \
                    (self.n_dt[d, :] + np.sum(self.sentimentprior[d]))[:,np.newaxis]
            
                    self.dt_distribution = self.dt_distribution/np.sum(self.dt_distribution, axis=1)[:,np.newaxis]
                    self.dts_distribution = self.dts_distribution/np.sum(self.dts_distribution, axis=2)[:,:,np.newaxis]
                
                #loglikelihood += np.sum(gammaln((self.n_dt[d] + self.alphaVec))) - gammaln(np.sum((self.n_dt[d] + self.alphaVec)))
                #loglikelihood -= np.sum(gammaln(self.alphaVec)) - gammaln(np.sum(self.alphaVec))
                
                #for k in range(self.numTopics):
                #    loglikelihood += np.sum(gammaln((self.n_dts[d, k, :] + self.sentimentprior[d]))) - gammaln(np.sum(self.n_dts[d, k, :] + self.sentimentprior[d]))
                #    loglikelihood -= np.sum(gammaln(self.sentimentprior[d])) - gammaln(np.sum(self.sentimentprior[d]))
            
            #for k in range(self.numTopics):
            #    for l in range(self.numSentiments):
            #        loglikelihood += 2*(np.sum(gammaln((self.n_vts[:, k,l] + self.beta))) - gammaln(np.sum((self.n_vts[:, k,l] + self.beta))))
            #        loglikelihood -= 2*(vocabSize * gammaln(self.beta) - gammaln(vocabSize * self.beta))
                
            #self.loglikelihoods[iteration] = loglikelihood
            #print ("Total loglikelihood is {}".format(loglikelihood))
            
            if (iteration+1)%5 == 0:
                # ADJUST ALPHA BY USING MINKA'S FIXED-POINT ITERATION
                numerator = 0
                denominator = 0
                for d in range(numDocs):
                    numerator += psi(self.n_dt[d] + self.alphaVec) - psi(self.alphaVec)
                    denominator += psi(np.sum(self.n_dt[d] + self.alphaVec)) - psi(np.sum(self.alphaVec))
                
                self.alphaVec *= numerator / denominator     
                self.alphaVec = np.maximum(self.alphaVec,self.alpha)
                    

def mape_score(y_true, y_pred):
    l = []
    assert len(y_true) == len(y_pred)
    for i in range(len(y_true)):
        if y_true[i] != 0:
            l.append(np.abs(1-y_pred[i]/y_true[i])*100)
    return np.mean(l)
    
def post_processing(text, index, sampler, worddict):
    #topicindices = np.argwhere(sampler.dt_distribution[index]>0)
    #topicsentiment = np.argwhere(sampler.dts_distribution[index][topicindices] >= 1)
    toptopic = np.argmax(sampler.dt_distribution[index])
    print ("Document id {} has top topic id {}".format(index, toptopic))
    
def coherence_score(sampler,topic_sentiment_df):
    totalcnt = topic_sentiment_df.shape[0]
    total = 0
    for i in range(len(topic_sentiment_df)):
        allwords = topic_sentiment_df.top_words.iloc[i] #ast.literal_eval(topic_sentiment_df.top_words.iloc[i])
        for word1 in allwords:
            for word2 in allwords:
                if word1 != word2:
                    ind1 = sampler.vocabulary[word1]
                    ind2 = sampler.vocabulary[word2]
                    total += np.log((np.matmul(sampler.wordOccuranceMatrix[:,ind1],sampler.wordOccuranceMatrix[:,ind2]) + 1)/np.sum(sampler.wordOccuranceMatrix[:,ind2]))
    return total/(2*totalcnt)

def kl_score(pk,qk):
    return (scipy.stats.entropy(pk,qk)*.5 + scipy.stats.entropy(qk,pk)*.5)
        
def run_experiment(numsentilabel,numtopics,skipgramwindow,alpha,beta,gamma,maxiter,numwordspertopic):
    global sampler, ds_estimated, rmse, coh_score, topic_sentiment_df
    
    binsize = sentirange/numsentilabel
    alpha = alpha/numtopics * np.ones(numtopics)
    gamma = [gamma/(numtopics * numsentilabel)]*numsentilabel
    
    sampler = SentimentLDAGibbsSampler(numtopics, alpha, beta, gamma, numsentilabel, minlabel, maxlabel, sentirange)
    t0 = time()
    sampler.run(list(train_review),list(train_sentiment), list(test_review), maxiter, skipgramwindow)
    worddict = sampler.getTopKWords(numwordspertopic)
    print("done in %0.3fs." % (time() - t0))
    
    ds_estimated = []
    for i in range(len(test_review)):
        sentiment = 0
        index = len(train_review) + i
        temp = np.matmul(sampler.dt_distribution[index,:],sampler.dts_distribution[index,:,:])
        for k, val in enumerate(temp):
            sentiment += (k+1)*binsize*val
        ds_estimated.append(sentiment)
    
    #print ("mean square error and mean absolute percentage error in sentiment estimation are {}, {}%".format(np.sqrt(mean_squared_error(np.array(test_sentiment.values)/10, np.array(ds_estimated)/10)), mape(test_sentiment.values, ds_estimated)))

    temp = []
    for t in range(numtopics):
        for s in range(numsentilabel):
            temp.append([t,s,worddict[t][s]])
    
    topic_sentiment_df = pd.DataFrame(temp,columns = ["topic_id","sentiment_label","top_words"])
    #topic_sentiment_df.to_csv(review_data_file.replace('.csv',"_{}_iter_output_ljstbtm.csv".format(maxiter)), index=False)
    
    #for i in range(0,5):
    #    index = len(train_review) + i
    #    post_processing(test_review.iloc[i],index,sampler,worddict)
    #    save_document_image('../output/{}_review_ljstbtm_{}.png'.format(review_data_file.split('/')[-1].replace('.csv',''),index+1),sampler.dts_distribution[index,:,:])
      
    rmse = np.sqrt(mean_squared_error(np.array(test_sentiment.values)/10, np.array(ds_estimated)/10))
    coh_score = coherence_score(sampler,topic_sentiment_df)
    
    mape = mape_score(test_sentiment.values, ds_estimated)
    
    testlen = test_review.shape[0]
    document_topic = np.zeros((testlen,numtopics))
    for d in range(train_review.shape[0],sampler.dt_distribution.shape[0]):
        document_topic[d-train_review.shape[0],sampler.dt_distribution[d,:].argmax()] = 1
 
    all_kl_scores = np.zeros((testlen,testlen))
    for i in range(testlen-1):
        for j in range(i+1,testlen):
            score = kl_score(sampler.dt_distribution[train_review.shape[0]+i],sampler.dt_distribution[train_review.shape[0]+j])
            all_kl_scores[i,j] = score
            all_kl_scores[j,i] = score

    intradist = 0
    for i in range(numtopics):
       cnt = document_topic[:,i].sum()
       tmp = np.outer(document_topic[:,i],document_topic[:,i])
       tmp = tmp * all_kl_scores
       intradist += tmp.sum()*1.0/(cnt*(cnt-1))
    intradist = intradist/numtopics

    interdist = 0
    for i in range(numtopics):
       for j in range(numtopics):
           if i != j:
             cnt_i = document_topic[:,i].sum()
             cnt_j = document_topic[:,j].sum()
             tmp = np.outer(document_topic[:,i],document_topic[:,j])
             tmp = tmp * all_kl_scores
             interdist += tmp.sum()*1.0/(cnt_i*cnt_j)
    interdist = interdist/(numtopics*(numtopics-1))
    H_score = intradist/interdist
    
    print ("RMSE, MAPE, Coherence ,Hscore values are {},{}%,{},{}".format(rmse,mape,coh_score,H_score))
        
    return rmse, coh_score
  
def f1(params):
    numsentilabel,numtopics = params['numsentilabel'], params['numtopics']
    print (numsentilabel,numtopics,alpha,beta,gamma,maxiter,numwordspertopic)
    return run_experiment(numsentilabel,numtopics,skipgramwindow,alpha,beta,gamma,maxiter,numwordspertopic)[0]
    
def f2(params):
    global topic_sentiment_df
    numwordspertopic = params['numwordspertopic']
    print (numwordspertopic)
    worddict = sampler.getTopKWords(numwordspertopic)
    temp = []
    for t in range(numtopics):
        for s in range(numsentilabel):
            temp.append([t,s,worddict[t][s]])
    
    topic_sentiment_df = pd.DataFrame(temp,columns = ["topic_id","sentiment_label","top_words"])
    coh_score = coherence_score(sampler,topic_sentiment_df)
    print (coh_score)
    return -1*coh_score    

if __name__ == '__main__':

    ''' Fixed parameters '''
    minlabel = 0
    maxlabel = 10
    sentirange = maxlabel - minlabel
    numwordspertopic = 5
    skipgramwindow = 5
    
    ''' Hyperparameters '''
    numsentilabel = 10
    numtopics = 20
    alpha = 10.0
    beta = .01
    gamma = 10.0
    maxiter = 20
    
    testsize = .2
    review_data_file = 'yelp50.csv'
    review_data = pd.read_csv(review_data_file,encoding='cp1250')  
    train_review, test_review, train_sentiment, test_sentiment = train_test_split(review_data.clean_sentence, review_data.sentiment_score, test_size=testsize,random_state=123)
    train_review = train_review.reset_index(drop=True)
    test_review = test_review.reset_index(drop=True)
    
    #run_experiment(numsentilabel,numtopics,alpha,beta,gamma,maxiter)
    
    '''
    document_topic = np.zeros(sampler.dt_distribution.shape)
    for d in range(document_topic.shape[0]):
        document_topic[d,sampler.dt_distribution[d,:].argmax()] = 1
     
    all_kl_scores = np.zeros((sampler.wordOccuranceMatrix.shape[0],sampler.wordOccuranceMatrix.shape[0]))
    for i in range(sampler.wordOccuranceMatrix.shape[0]-1):
        for j in range(i+1,sampler.wordOccuranceMatrix.shape[0]):
            score = kl_score(sampler.dt_distribution[i],sampler.dt_distribution[j])
            all_kl_scores[i,j] = score
            all_kl_scores[j,i] = score
    
    h_score = 0
    for i in range(numtopics):
        cnt = document_topic[:,i].sum()
        tmp = np.outer(document_topic[:,i],document_topic[:,i])
        tmp = tmp * all_kl_scores
        h_score += tmp.sum()*1.0/(cnt*(cnt-1))
    h_score = h_score/numtopics
    '''
    
    spacetosearch = {
    'numsentilabel': hp.choice('numsentilabel', [10,20,25]),
    'numtopics': hp.choice('numtopics', [10,20,25]),
    }
    
    trials = Trials()
    best = fmin(f1, spacetosearch, algo=tpe.suggest, max_evals=9, trials=trials)
    best_params1 = space_eval(spacetosearch, best)
    print ('best {} with params {}'.format(best, best_params1))
    numsentilabel,numtopics = best_params1['numsentilabel'], best_params1['numtopics']
    run_experiment(numsentilabel,numtopics,skipgramwindow,alpha,beta,gamma,maxiter,numwordspertopic)
    
    spacetosearch = {
    'numwordspertopic': hp.choice('numwordspertopic', [5,10,20,25,50])
    }
    
    trials = Trials()
    best = fmin(fn=f2, space=spacetosearch, algo=tpe.suggest, trials=trials, max_evals=5)
    best_params2 = space_eval(spacetosearch, best)
    print ('best {} with params {}'.format(best, best_params2))
    numwordspertopic = best_params2['numwordspertopic']
    
    #run_experiment(numsentilabel,numtopics,alpha,beta,gamma,maxiter,5)
    topic_sentiment_df.to_csv(review_data_file.replace('.csv',"_{}_iter_output_ljstbtm.csv".format(maxiter)), index=False)
    for skipgramwindow in [3,5,7]:
        print (skipgramwindow)
        train_review, test_review, train_sentiment, test_sentiment = train_test_split(review_data.clean_sentence, review_data.sentiment_score, test_size=.2,random_state=123)
        run_experiment(numsentilabel,numtopics,skipgramwindow,alpha,beta,gamma,maxiter,5)
        train_review, test_review, train_sentiment, test_sentiment = train_test_split(review_data.clean_sentence, review_data.sentiment_score, test_size=.1,random_state=123)
        run_experiment(numsentilabel,numtopics,skipgramwindow,alpha,beta,gamma,maxiter,5)
        train_review, test_review, train_sentiment, test_sentiment = train_test_split(review_data.clean_sentence, review_data.sentiment_score, test_size=.25,random_state=123)
        run_experiment(numsentilabel,numtopics,skipgramwindow,alpha,beta,gamma,maxiter,5)
