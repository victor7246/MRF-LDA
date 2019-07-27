"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause
Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in
Finding scientifc topics (Griffiths and Steyvers)
"""

import numpy as np
import datetime
import scipy as sp
from scipy.special import gammaln
from scipy import sparse


def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

class LdaSampler(object):

    def __init__(self, n_topics, n_sentiment, lambda_param, alpha=0.1, beta=0.1, gamma = 0.5, SentimentRange=5):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.n_sentiment = n_sentiment
        self.alpha = alpha
        self.beta = beta
        self.gamma = 10.0/(n_topics * n_sentiment)
        self.gammavec = None
        self.lambda_param = lambda_param
        self.SentimentRange = SentimentRange
        self.probabilities_ts = {}
        self.sentimentprior = {}
        self.count_matrix = None
        self.df_matrix = None
        self.train_sentiment = None
        self.test_sentiment = None
        self.words = None
        self.edge_dict = None
        
    def store_data(self, train_matrix, train_sentiment, test_matrix, test_sentiment, df_matrix, words, edge_dict):
        matrix = np.concatenate([train_matrix, test_matrix], axis=0)
        
        n_docs, vocab_size = matrix.shape
        self.count_matrix = sparse.csr_matrix(matrix)
        self.df_matrix = df_matrix.copy()
        self.train_sentiment = train_sentiment.copy()
        self.test_sentiment = test_sentiment.copy()
        self.words = words
        self.edge_dict = edge_dict
        
    def get_count_matrix(self):
        return np.array(self.count_matrix.todense())
    
    def get_df_matrix(self):
        return np.array(self.df_matrix.todense())

    def _initialize(self, train_matrix, train_sentiment, test_matrix):
        matrix = np.concatenate([train_matrix, test_matrix], axis=0)
        
        n_docs, vocab_size = matrix.shape
        self.count_matrix = sparse.csr_matrix(matrix)
        self.train_sentiment = train_sentiment.copy()

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics))
        self.nmzs = np.zeros((n_docs, self.n_topics, self.n_sentiment))
        self.nm = np.zeros(n_docs)
        self.nzws = np.zeros((self.n_topics, vocab_size, self.n_sentiment))
        self.nzs = np.zeros((self.n_topics, self.n_sentiment))
        self.topics = {}
        self.sentiments = {}
        
        self.gammavec = []
        for i in train_sentiment:
            p = [self.gamma] * self.n_sentiment
            p[int(i)-1] += 1
            self.gammavec.append(p)
        
        for _ in range(len(test_matrix)):
            self.gammavec.append(self.gamma * np.ones(self.n_sentiment))
            
        self.gammavec = np.array(self.gammavec)

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                s = np.random.randint(self.n_sentiment)
                self.nmz[m,z] += 1
                self.nmzs[m,z, s] += 1
                self.nm[m] += 1
                self.nzws[z,w, s] += 1
                self.nzs[z, s] += 1
                self.topics[(m,i)] = z
                self.sentiments[(m,i)] = s

    def _conditional_distribution(self, m, w, edge_dict):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzws.shape[1]
        left = (self.nzws[:, w, :] + self.beta) / (self.nzs + self.beta * vocab_size)
        right = (self.nmz[m,:] + self.alpha) / (self.nm[m] + self.alpha * self.n_topics)

        gammaFactor = np.zeros((self.n_topics, self.n_sentiment))
        for z in range(self.n_topics):
            gammaFactor[z,:] = (self.nmzs[m, z, :] + self.gammavec[m])/(self.nmz[m, z] + np.sum(self.gammavec[m]))

        topic_ass_sent = []
        for z in range(self.n_topics):
            topic_assignment = [0] * self.n_sentiment
            parent = self.nzws[z, w , :]
            try:
                edge_dict[w]
                children = []
                for i in edge_dict[w]:
                    children.append(self.nzws[z, i, :].tolist())
                children = np.array(children)
                children[children>1] = 1
                for idx, i in enumerate(parent):
                    t = 0
                    if i>0:
                        t =  sum(children[z, idx, :])
                    topic_assignment[idx] = t
                if sum(topic_assignment)>0:
                    topic_assignment = topic_assignment / sum(topic_assignment)
            except:
                pass
            topic_assignment = np.exp(np.dot(self.lambda_param, topic_assignment))
            topic_ass_sent.append(topic_assignment)
            
        topic_ass_sent = np.array(topic_ass_sent)
        
        p_zs = left * right[:, np.newaxis] * self.gammavec[m] * topic_ass_sent
        p_zs /= np.sum(p_zs)
        return p_zs

    def loglikelihood(self, docs_edges):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzws.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            for s in xrange(self.n_sentiment):
                lik += log_multi_beta(self.nzws[z, :, s]+self.beta)
                lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            for z in xrange(self.n_topics):
                lik += log_multi_beta(self.nmzs[m, z, :]+self.gammavec[m])
                lik -= log_multi_beta(self.gammavec[m], None)
        
        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)
        
        for i in xrange(n_docs):
            for s in xrange(self.n_sentiment):
                count = 0
                edges_count = 0
#                 print(self.nzws.shape)
                for a, b in (docs_edges[i]):
                    edges_count += 1
                    aa = self.nzws[:, a, s]
                    bb = self.nzws[:, b, s]
                    if aa.argmax() == bb.argmax():
                        count += 1
                if edges_count > 0:
                    lik += np.log(np.exp(self.lambda_param*count/edges_count))

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
#         V = self.nzws.shape[1]
        num = self.nzws + self.beta
        n = np.sum(num, axis=1)
        n = n[:, np.newaxis, :]
        num /= n
        return num
    
    def theta(self):
        V = self.nmz.shape[1]
        num = self.nmz + self.alpha
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num
    
    def pi(self):
        num = self.nmzs + self.gammavec[:, np.newaxis, :]
        n = np.sum(num, axis=2)
        n = n[: ,:, np.newaxis]
        num /= n
        return num
    
    def getTopKWords(self, K, vocab):
        """
        Returns top K discriminative words for topic t v for which p(v | t) is maximum
        """
        pseudocounts = np.copy(self.nzws)
        normalizer = np.sum(pseudocounts, axis = 2)
        normalizer = np.sum(normalizer, axis = 0)
        pseudocounts /= normalizer[np.newaxis, :,  np.newaxis]
        worddict = {}
        for t in range(self.n_topics):
            for s in range(self.n_sentiment):
                worddict[(t, s)] = {}
                topWordIndices = pseudocounts[t, :, s].argsort()[-(K+1):-1]
                worddict[(t, s)] = [vocab[i] for i in topWordIndices]
        return worddict

    def run(self, train_matrix, train_sentiment, test_matrix, edge_dict, maxiter=100):
        """
        Run the Gibbs sampler.
        """
#         print(datetime.datetime.now().time(), "Enter Run")
        n_docs, vocab_size = train_matrix.shape
        self._initialize(train_matrix, train_sentiment, test_matrix)
        self.edge_dict = edge_dict

#         print(datetime.datetime.now().time(), "Start Iterations")
        for it in xrange(maxiter):
#             print(datetime.datetime.now().time(), "IT", it)
            for m in xrange(n_docs):
#                 print(it, datetime.datetime.now().time(), "m", m)
                for i, w in enumerate(word_indices(train_matrix[m, :])):
#                     print(it, datetime.datetime.now().time(), "IT")
                    z = self.topics[(m,i)]
                    s = self.sentiments[(m,i)]
                    self.nmz[m,z] -= 1
                    self.nmzs[m,z, s] -= 1
                    self.nm[m] -= 1
                    self.nzws[z, w, s] -= 1
                    self.nzs[z, s] -= 1

                    p_z = self._conditional_distribution(m, w, edge_dict)
                    ind = sample_index(p_z.flatten())
                    
                    z, s = np.unravel_index(ind, p_z.shape)

                    self.nmz[m,z] += 1
                    self.nmzs[m,z, s] += 1
                    self.nm[m] += 1
                    self.nzws[z,w, s] += 1
                    self.nzs[z, s] += 1
                    
                    self.topics[(m,i)] = z
                    self.sentiments[(m,i)] = s