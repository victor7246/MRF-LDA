"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause
Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in
Finding scientifc topics (Griffiths and Steyvers)
"""

import numpy as np
import scipy as sp
from scipy.special import gammaln

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

    def __init__(self, n_topics, lambda_param, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.lambda_param = lambda_param

    def _initialize(self, matrix):
        n_docs, vocab_size = matrix.shape

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics))
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size))
        self.nm = np.zeros(n_docs)
        self.nz = np.zeros(self.n_topics)
        self.topics = {}

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                self.nmz[m,z] += 1
                self.nm[m] += 1
                self.nzw[z,w] += 1
                self.nz[z] += 1
                self.topics[(m,i)] = z

    def _conditional_distribution(self, m, w, edge_dict):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzw.shape[1]
        left = (self.nzw[:,w] + self.beta) / (self.nz + self.beta * vocab_size)
        right = (self.nmz[m,:] + self.alpha) / (self.nm[m] + self.alpha * self.n_topics)
        topic_assignment = [0] * self.n_topics
        try:
            edge_dict[w]
            parent = self.nzw[:, w]
            children = []
            for i in edge_dict[w]:
                children.append(self.nzw[:, i].tolist())
            children = np.array(children)
            children[children>1] = 1
            for idx, i in enumerate(parent):
                t = 0
                if i>0:
                    t =  sum(children[:, idx])
                topic_assignment[idx] = t
#                 print("a", idx, len(children), sum(children[:, idx]), parent, topic_assignment)
            if sum(topic_assignment)>0:
                topic_assignment = topic_assignment / sum(topic_assignment)
        except:
            pass
        topic_assignment = np.exp(np.dot(self.lambda_param, topic_assignment))
        p_z = left * right * topic_assignment
        p_z /= np.sum(p_z)
        return p_z

    def loglikelihood(self, docs_edges):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzw.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)
            
#             print(self.nzw[z,:])

        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)
        
        for i in xrange(n_docs):
            count = 0
            edges_count = 0
            for a, b in (docs_edges[i]):
                edges_count += 1
                aa = self.nzw[:, a]
                bb = self.nzw[:, b]
                if aa.argmax() == bb.argmax():
                    count += 1
            if edges_count > 0:
                lik += np.log(np.exp(self.lambda_param*count/edges_count))

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.nzw.shape[1]
        num = self.nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num
    
    def getTopKWords(self, K, vocab):
        """
        Returns top K discriminative words for topic t v for which p(v | t) is maximum
        """
        pseudocounts = np.copy(self.nzw.T)
        normalizer = np.sum(pseudocounts, (0))
        pseudocounts /= normalizer[np.newaxis, :]
        worddict = {}
        for t in range(self.n_topics):
            worddict[t] = {}
            topWordIndices = pseudocounts[:, t].argsort()[-(K+1):-1]
            worddict[t] = [vocab[i] for i in topWordIndices]
        return worddict

    def run(self, matrix, edge_dict, maxiter=30):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape
        self._initialize(matrix)

        for it in xrange(maxiter):
            for m in xrange(n_docs):
                for i, w in enumerate(word_indices(matrix[m, :])):
                    z = self.topics[(m,i)]
                    self.nmz[m,z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z,w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(m, w, edge_dict)
                    z = sample_index(p_z)

                    self.nmz[m,z] += 1
                    self.nm[m] += 1
                    self.nzw[z,w] += 1
                    self.nz[z] += 1
                    self.topics[(m,i)] = z

            # FIXME: burn-in and lag!
            yield self.phi()