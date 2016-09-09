# -*- coding: utf-8 -*-
# Utils for Gradient

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from abc import abstractmethod, ABCMeta
import logging

__author__ = 'lemon'
logger = logging.getLogger(__name__)

class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, lr, norm_lim=9):
        self.lr = lr
        self.norm_lim = norm_lim

    @abstractmethod
    def get_update(self, loss, params, norm_except_params=None):
        """
        :param loss:
        :param params:
        :param norm_except_params: the name list of the params without norm
        :return:
        """
        pass
    
class SGDOptimizer(Optimizer):
    def __init__(self, lr=0.05, norm_lim=9):
        super(SGDOptimizer, self).__init__(lr, norm_lim)

    def get_update(self, loss, params, norm_exc_params=[]):
        logger.info(params)
        updates = OrderedDict({})
        grad_params = []
        for param in params:
            #gp = theano.printing.Print('gp')(T.grad(loss, param))
            gp = T.grad(loss,param)
            grad_params.append(gp)
        for param, gp in zip(params, grad_params):
            stepped_param = param - gp * self.lr
            param_name = param.name
            if self.norm_lim > 0 and (param.get_value(borrow=True).ndim == 2) and (param_name not in norm_exc_params):
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(self.norm_lim))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param
        return updates

class AdaDeltaOptimizer(Optimizer):
    def __init__(self, lr=0.95, norm_lim=16, epsilon=1e-7):
        super(AdaDeltaOptimizer, self).__init__(lr, norm_lim)
        self.epsilon = epsilon

    def get_update(self, loss, params, norm_exc_params=[]):
        logger.info(params)
        rho = self.lr
        epsilon = self.epsilon
        norm_lim = self.norm_lim
        updates = OrderedDict({})
        exp_sqr_grads = OrderedDict({})
        exp_sqr_ups = OrderedDict({})
        gparams = []
        for param in params:
            exp_sqr_grads[param] = theano.shared(np.zeros(param.get_value().shape,
                dtype = theano.config.floatX),name="exp_grad_%s" % param.name)
            gp = T.grad(loss, param)
            exp_sqr_ups[param] = theano.shared(np.zeros(param.get_value().shape,
                dtype = theano.config.floatX), name="exp_ups_%s" % param.name)
            gparams.append(gp)
        for param, gp in zip(params, gparams):
            exp_sg = exp_sqr_grads[param]
            exp_su = exp_sqr_ups[param]
            up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
            updates[exp_sg] = up_exp_sg
            step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
            updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
            stepped_param = param + step
            param_name = param.name
            if self.norm_lim > 0 and (param.get_value(borrow=True).ndim == 2) and (param_name not in norm_exc_params):
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param
        return updates
