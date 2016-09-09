# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 14:47:31 2016

@author: Lemon
"""

import theano
import theano.tensor as T
import numpy as np
from optimizer import AdaDeltaOptimizer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__author__ = 'Lemon'

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)
trng = RandomStreams(SEED)
   
class LSTMEncoder(object):

    def __init__(self, word_dim, hidden_dim, embedding, y_dim, Regularization,  dropout=0, verbose=True, use_drop = False):
        if verbose:
            print('Building {}...'.format(self.__class__.__name__))

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.dropout = dropout
        self.embedding = theano.shared(embedding)
        self.y_dim = y_dim
        self.Regularization = Regularization                
        #Used for dropout
        self.use_noise = theano.shared(np.asarray(0., dtype=theano.config.floatX))
        self.use_drop = use_drop

        # for x are word_dim * hidden_dim,  for h/c are hidden_dim * hidden_dim
        self.W_ix = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (word_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_ih = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_ic = theano.shared(np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),
            (hidden_dim, hidden_dim)).astype(theano.config.floatX))

        self.W_fx = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (word_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_fh = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_fc = theano.shared(np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),
            (hidden_dim, hidden_dim)).astype(theano.config.floatX))

        self.W_cx = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (word_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_ch = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,hidden_dim)).astype(theano.config.floatX))

        self.W_ox = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (word_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_oh = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_oc = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim, hidden_dim)).astype(theano.config.floatX))

        # for softmax y
        self.U = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,y_dim)).astype(theano.config.floatX))
        self.b = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (y_dim,)).astype(theano.config.floatX))


        self.params = [self.W_ix,self.W_ih,self.W_ic,
                       self.W_fx,self.W_fh,self.W_fc,
                       self.W_cx,self.W_ch,
                       self.W_ox,self.W_oh,self.W_oc,
                       self.embedding,
                       self.U,self.b,
                      ]

        if verbose:
            print('Architecture of {} built finished'.format(self.__class__.__name__))
            print('Word dimension:  %d' % self.word_dim)
            print('Hidden dimension: %d' % self.hidden_dim)
            print('Dropout Rate:     %f' % self.dropout)

        self.theano = {}
        #self.__build__()
        #self.__build_batch__()

    def _step(self, x_t, h_t_1, c_t_1, W_ix, W_ih, W_ic, W_fx, W_fh, W_fc, W_cx, W_ch, W_ox, W_oh, W_oc, U, b):
        # 1 * hidden_dim = (1 * word_dim * word_dim * hidden_dim) + (1 * hidden_dim * hidden_dim * hidden_dim)
        # = 1 * h + 1 * h = 1 * h    
        # W_oc = theano.printing.Print('W_oc')(W_oc)
        i_t = T.nnet.sigmoid(T.dot(x_t,W_ix) + T.dot(h_t_1,W_ih) + T.dot(c_t_1,W_ic))
        f_t = T.nnet.sigmoid(T.dot(x_t,W_fx) + T.dot(h_t_1,W_fh) + T.dot(c_t_1,W_fc))
        c_t = f_t * c_t_1 + i_t * T.tanh(T.dot(x_t,W_cx) + T.dot(h_t_1,W_ch))

        #1 * word_dim
        o_t = T.nnet.sigmoid(T.dot(x_t,W_ox) + T.dot(h_t_1,W_oh) + T.dot(c_t,W_oc))
        h_t = o_t * T.tanh(c_t)
        #y_t = self.classifier.forward(h_t)
        y_t = T.nnet.softmax(T.dot(h_t,U) + b)

        # h_t = theano.printing.Print('h_t')(h_t)
        return h_t, c_t, y_t
        
    def _step_batch(self, x_t, m_t, h_t_1, c_t_1, W_ix, W_ih, W_ic, W_fx, W_fh, W_fc, W_cx, W_ch, W_ox, W_oh, W_oc, U, b):
        
        #x_t = theano.printing.Print('x')(x_t)
        #batch_size * hidden_dim
        i_t = T.nnet.sigmoid(T.dot(x_t,W_ix) + T.dot(h_t_1,W_ih) + T.dot(c_t_1,W_ic))
        f_t = T.nnet.sigmoid(T.dot(x_t,W_fx) + T.dot(h_t_1,W_fh) + T.dot(c_t_1,W_fc))
        c_t = f_t * c_t_1 + i_t * T.tanh(T.dot(x_t,W_cx) + T.dot(h_t_1,W_ch))

        #batch_size * hidden_dim
        o_t = T.nnet.sigmoid(T.dot(x_t,W_ox) + T.dot(h_t_1,W_oh) + T.dot(c_t,W_oc))
        h_t = o_t * T.tanh(c_t)
        
        #y_t = self.classifier.forward(h_t)
        c_t = m_t[:, None] * c_t + (1. - m_t)[:, None] * c_t_1
        h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_t_1
        
        if self.use_drop:
            self.dropout_layer(h_t, self.use_noise, trng, self.dropout)
            
        #batch * 4
        y_t = T.nnet.softmax(T.dot(h_t,U) + b)
        #y_t = theano.printing.Print('y')(y_t)
               
        return h_t, c_t, y_t
            
        
    def forward_sequence(self, x):
        h0 = theano.shared((np.zeros((1,self.hidden_dim), dtype=theano.config.floatX)), name='h0')
        c0 = theano.shared((np.zeros((1,self.hidden_dim), dtype=theano.config.floatX)), name='c0')

        hs, _ = theano.scan(fn=self._step,
                            sequences=x,
                            outputs_info=[h0, c0, None],
                            non_sequences= [self.W_ix,self.W_ih,self.W_ic,
                                            self.W_fx,self.W_fh,self.W_fc,
                                            self.W_cx,self.W_ch,
                                            self.W_ox,self.W_oh,self.W_oc,
                                            self.U,self.b])

        return hs[2]
        
    def forward_sequence_batch(self,x,mask,batch_size):
        h0 = T.zeros((batch_size,self.hidden_dim), dtype=theano.config.floatX)
        c0 = T.zeros((batch_size,self.hidden_dim), dtype=theano.config.floatX)
        
        hs, _ = theano.scan(fn=self._step_batch,
                            sequences=[x,mask],
                            outputs_info=[h0, c0, None],
                            non_sequences= [self.W_ix,self.W_ih,self.W_ic,
                                            self.W_fx,self.W_fh,self.W_fc,
                                            self.W_cx,self.W_ch,
                                            self.W_ox,self.W_oh,self.W_oc,
                                            self.U,self.b])
        #hs[2] = theano.printing.Print('h2')(hs[2])  
                                  
        return hs[2]
        
    def forward(self, x):
        n_step = x.shape[0]
        pred = self.forward_sequence(x).reshape([n_step,self.y_dim])

        return pred
    
    def forward_batch(self, x, mask, batch_size):
        proj = self.forward_sequence_batch(x, mask, batch_size)
       
        return proj
        
    def __build__(self):
        x = T.imatrix('x')
        y = T.ivector('y')

        n_step = x.shape[0]
        index = x.flatten()
        emb = self.embedding[index].reshape((n_step, self.word_dim))

        regularization = 0.0
        for params in self.params:
            regularization = regularization +(T.sum(params ** 2))**0.5
            
    
        proj = self.forward(emb)

        pred = T.argmax(proj, axis=1)

        acc = (T.eq(y, pred)).sum()

        neg_log_likelihoods = -T.log(proj[(T.arange(n_step),y)]) 
        loss = T.sum(neg_log_likelihoods) + (self.Regularization / 2) * regularization
                
        # SGD
        learning_rate = T.scalar('learning_rate')
        ada_optimizer = AdaDeltaOptimizer(lr=learning_rate, norm_lim=-1)
        except_norm_list = []
        updates = ada_optimizer.get_update(loss, self.params, except_norm_list)

        self.train = theano.function([x, y, learning_rate],[],
                        updates = updates)

        self.ce_loss = theano.function([x, y], loss)

        self.output = theano.function([x], proj)

        self.predict = theano.function([x], pred)

        self.accuracy = theano.function([x, y], acc)
        
    def __build_batch__(self):
        
        
        x = T.itensor3('x')
        y = T.imatrix('y')
        mask = T.matrix('mask')

        n_step = x.shape[0]
        batch_size = x.shape[1]
        
        emb = self.embedding[x.flatten()].reshape((n_step, batch_size, self.word_dim))
        
        proj = self.forward_batch(emb, mask, batch_size)
        #proj = theano.printing.Print('proj')(proj)
        pred = T.argmax(proj, axis=2)
        #pred = theano.printing.Print('pred')(pred)
        proj = proj.reshape([n_step * batch_size, self.y_dim])
        
        regularization = 0.0
        for params in self.params:
            regularization = regularization +(T.sum(params ** 2))**0.5                                    
    
        neg_log_likelihoods = T.log(proj[T.arange(n_step * batch_size), y.flatten()]).reshape([n_step,batch_size]) * mask           
        loss = -T.mean(T.sum(neg_log_likelihoods,axis=0)) + (self.Regularization / 2) * regularization

        # SGD
        learning_rate = T.scalar('learning_rate')
        ada_optimizer = AdaDeltaOptimizer(lr=learning_rate, norm_lim=-1)
        except_norm_list = []
        updates = ada_optimizer.get_update(loss, self.params, except_norm_list)

        self.train_batch = theano.function([x, mask, y, learning_rate],[],
                        updates = updates)

        self.ce_loss = theano.function([x, mask, y], loss)
        self.predict = theano.function([x ,mask], pred)
        return self.use_noise
    
    def save_model(self, filename):
        from six.moves import cPickle
        with open(filename, 'wb') as fout:
            for param in self.params:
                cPickle.dump(param.get_value(), fout, protocol = -1)

    def load_model(self, filename):
        from six.moves import cPickle
        with open(filename, 'rb') as fin:
            for param in self.params:
                param.set_value(cPickle.load(fin))
    
    #dropout_layer
    def dropout_layer(self, state_before, use_noise, trng, dropout_rate):
        proj = T.switch(use_noise,
                        (state_before *
                         trng.binomial(state_before.shape,
                                        p=dropout_rate, n=1,
                                        dtype=state_before.dtype)),
                         state_before * (1-dropout_rate))
        return proj




















