from functools import partial
import io
import logging
import math
import os
import re
import arff
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, roc_auc_score  # just aliasing
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
import numpy as np
from numpy import nan, sort



class ClassificationResult(Result):

    def __init__(self, predictions_df, info=None):
        super().__init__(predictions_df, info)
        self.classes = self.df.columns[:-2].values.astype(str, copy=False)
        self.probabilities = self.df.iloc[:, :-2].values.astype(float, copy=False)
        self.target = Feature(0, 'class', 'categorical', self.classes, is_target=True)
        self.type = 'binomial' if len(self.classes) == 2 else 'multinomial'
        self.truth = self._autoencode(self.truth.astype(str, copy=False))
        self.predictions = self._autoencode(self.predictions.astype(str, copy=False))

    def acc(self):
        return float(accuracy_score(self.truth, self.predictions))

    def balacc(self):
        # return float(balanced_accuracy_score(self.truth, self.predictions))
        pass

    def auc(self):
        if self.type != 'binomial':
            # raise ValueError("AUC metric is only supported for binary classification: {}.".format(self.classes))
            log.warning("AUC metric is only supported for binary classification: %s.", self.classes)
            return nan
        return float(roc_auc_score(self.truth, self.probabilities[:, 1]))

    def cm(self):
        return confusion_matrix(self.truth, self.predictions)

    def f1(self):
        return float(f1_score(self.truth, self.predictions))

    def logloss(self):
        # truth_enc = self.target.one_hot_encoder.transform(self.truth)
        return float(log_loss(self.truth, self.probabilities))

    def _autoencode(self, vec):
        needs_encoding = not _encode_predictions_and_truth_ or (isinstance(vec[0], str) and not vec[0].isdigit())
        return self.target.label_encoder.transform(vec) if needs_encoding else vec

