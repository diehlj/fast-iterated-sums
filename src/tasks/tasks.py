from typing import Optional, List, Tuple
import math
import functools
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig
import src.tasks.metrics as M
from src.utils.config import to_list, instantiate

from functools import wraps
import inspect
def wrap_kwargs(f):
    """
    Given a callable f that can consume some named arguments,
    wrap it with a kwargs that passes back any unused args

    EXAMPLES
    --------

    Basic usage:
    def foo(x, y=None):
        return x

    wrap_kwargs(foo)(0, y=1, z=2) == (0, {'z': 2})

    --------

    The wrapped function can return its own argument dictionary,
    which gets merged with the new kwargs.
    def foo(x, y=None):
        return x, {}
    wrap_kwargs(foo)(0, y=1, z=2) == (0, {'z': 2})

    def foo(x, y=None):
        return x, {"y": y, "z": None}
    wrap_kwargs(foo)(0, y=1, z=2) == (0, {'y': 1, 'z': 2})

    --------

    The wrapped function can have its own kwargs parameter:
    def foo(x, y=None, **kw_args):
        return x, {}
    wrap_kwargs(foo)(0, y=1, z=2) == (0, {})

    --------

    Partial functions and modules work automatically:
    class Module:
        def forward(self, x, y=0):
            return x, {"y": y+1}

    m = Module()

    wrap_kwargs(m.forward)(0, y=1, z=2) == (0, {'y': 2, 'z': 2})

    """
    sig = inspect.signature(f)
    # Check if f already has kwargs
    has_kwargs = any([
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    ])
    if has_kwargs:
        @wraps(f)
        def f_kwargs(*args, **kwargs):
            y = f(*args, **kwargs)
            if isinstance(y, tuple) and isinstance(y[-1], dict):
                return y
            else:
                return y, {}
    else:
        param_kwargs = inspect.Parameter("kwargs", kind=inspect.Parameter.VAR_KEYWORD)
        sig_kwargs = inspect.Signature(parameters=list(sig.parameters.values())+[param_kwargs])
        @wraps(f)
        def f_kwargs(*args, **kwargs):
            bound = sig_kwargs.bind(*args, **kwargs)
            if "kwargs" in bound.arguments:
                kwargs = bound.arguments.pop("kwargs")
            else:
                kwargs = {}
            y = f(**bound.arguments)
            if isinstance(y, tuple) and isinstance(y[-1], dict):
                return *y[:-1], {**y[-1], **kwargs}
            else:
                return y, kwargs
    return f_kwargs

def discard_kwargs(f):
    if f is None: return None
    f_kwargs = wrap_kwargs(f)
    @wraps(f)
    def f_(*args, **kwargs):
        return f_kwargs(*args, **kwargs)[0]
    return f_

class BaseTask:
    """ Abstract class that takes care of:
    - loss function
    - arbitrary metrics
    """

    def __init__(self, model=None, loss=None, loss_val=None, metrics=None):
        """ This class is allowed to grab attributes directly off a constructed model object """
        self.model = model
        if metrics is None: metrics = []
        self.metric_names = to_list(metrics)

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.OUTPUT_METRIC_FNS, loss, partial=True)
        # self.loss = discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.OUTPUT_METRIC_FNS, loss_val, partial=True)
            # self.loss_val = discard_kwargs(self.loss_val)


    def metrics(self, x, y, **kwargs):
        """
        Metrics are just functions
        output metrics are a function of output and target
        loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            # name: discard_kwargs(M.OUTPUT_METRIC_FNS[name])(x, y, **kwargs)
            name: M.OUTPUT_METRIC_FNS[name](x, y, **kwargs)
            for name in self.metric_names if name in M.OUTPUT_METRIC_FNS
        }
        loss_metrics = {
            # name: discard_kwargs(M.LOSS_METRIC_FNS[name])(x, y, self.loss, **kwargs)
            name: M.LOSS_METRIC_FNS[name](x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.LOSS_METRIC_FNS
        }
        return {**output_metrics, **loss_metrics}

TASK_REGISTRY = {
    'base': BaseTask,
}