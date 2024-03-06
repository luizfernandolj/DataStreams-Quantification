import os
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check

from quantifiers.ClassifyCountCorrect.AdjustedClassifyCount import AdjustedClassifyCount
from quantifiers.ClassifyCountCorrect.ClassifyCount import ClassifyCount
from quantifiers.ClassifyCountCorrect.MAX import MAX
from quantifiers.ClassifyCountCorrect.MedianSweep import MedianSweep
from quantifiers.ClassifyCountCorrect.ProbabilisticAdjustedClassifyCount import ProbabilisticAdjustedClassifyCount
from quantifiers.ClassifyCountCorrect.ProbabilisticClassifyCount import ProbabilisticClassifyCount
from quantifiers.ClassifyCountCorrect.T50 import T50
from quantifiers.ClassifyCountCorrect.X import Xqtf
from quantifiers.DistributionMatching.DyS import DyS
from quantifiers.DistributionMatching.HDy import HDy
from quantifiers.DistributionMatching.SORD import SORD

class ApplyQtfs:
  def __init__(self, trainX, trainy, window, model, thr):
    self.trainX = trainX
    self.trainy = trainy
    self.window = window
    self.model = model
    self.thr = thr
    self.quantifiers = ["CC", "ACC", "MS", "DyS"]
    self.quantifiers_initialized = {}
  
  def fit(self, quantifier, clf,  thr, measure, trainX, trainY):
    if quantifier == "CC":
      cc = ClassifyCount(classifier=clf, threshold=thr)
      cc.fit(trainX, trainY)
      self.quantifiers_initialized["CC"] = cc
    if quantifier == "ACC":
      acc = AdjustedClassifyCount(classifier=clf, threshold=thr)
      acc.fit(trainX, trainY)
      self.quantifiers_initialized["ACC"] = acc
    if quantifier == "PCC":
      pcc = ProbabilisticClassifyCount(classifier=clf)
      pcc.fit(trainX, trainY)
      self.quantifiers_initialized["PCC"] = pcc

    if quantifier == "PACC":
      pacc = ProbabilisticAdjustedClassifyCount(classifier=clf, threshold=thr)
      pacc.fit(trainX, trainY)
      self.quantifiers_initialized["PACC"] = pacc

    if quantifier == "X":
      x_qtf = Xqtf(classifier=clf)
      x_qtf.fit(trainX, trainY)
      self.quantifiers_initialized["X"] = x_qtf

    if quantifier == "MAX":
      max_qtf = MAX(classifier=clf)
      max_qtf.fit(trainX, trainY)
      self.quantifiers_initialized["MAX"] = max_qtf

    if quantifier == "T50":
      t50 = T50(classifier=clf)
      t50.fit(trainX, trainY)
      self.quantifiers_initialized["T50"] = t50

    if quantifier == "MS":
      ms = MedianSweep(classifier=clf)
      ms.fit(trainX, trainY)
      self.quantifiers_initialized["MS"] = ms

    if quantifier == "HDy":
      hdy = HDy(classifier=clf)
      hdy.fit(trainX, trainY)
      self.quantifiers_initialized["HDy"] = hdy

    if quantifier == "DyS":
      dys = DyS(classifier=clf, similarity_measure=measure)
      dys.fit(trainX, trainY)
      self.quantifiers_initialized["DyS"] = dys

    if quantifier == "SORD":
      sord = SORD(classifier=clf)
      sord.fit(trainX, trainY)
      self.quantifiers_initialized["SORD"] = sord
   
  def check_train(self, trainX, trainY):
    if (trainX != self.trainX || trainY != self.trainY):
      self.trainX = trainX
      self.trainY = trainY
    for qtf in self.quantifiers:
      # .............Calling of Methods.................
      self.fit(quantifier=qtf, 
               clf=self.model,  
               thr=self.thr,
               measure='topsoe',
               train=trainX, 
               trainY=trainY)
      

  def apply_quantifier(self, quantifier, test):
      if quantifier == "CC":
        return self.quantifiers_initialized["CC"].predict(test)
      if quantifier == "ACC":
        return self.quantifiers_initialized["ACC"].predict(test)
      if quantifier == "PCC":
        return self.quantifiers_initialized["PCC"].predict(test)

      if quantifier == "PACC":
        return self.quantifiers_initialized["PACC"].predict(test)

      if quantifier == "X":
        return self.quantifiers_initialized["X"].predict(test)

      if quantifier == "MAX":
        return self.quantifiers_initialized["MAX"].predict(test)

      if quantifier == "T50":
        return self.quantifiers_initialized["T50"].predict(test)

      if quantifier == "MS":
        return self.quantifiers_initialized["MS"].predict(test)

      if quantifier == "HDy":
        return self.quantifiers_initialized["HDy"].predict(test)

      if quantifier == "DyS":
        return self.quantifiers_initialized["DyS"].predict(test)

      if quantifier == "SORD":
        return self.quantifiers_initialized["SORD"].predict(test)

  def aplly_qtf(self):
    proportions = {}
    for qtf in self.quantifiers:
      # .............Calling of Methods.................
      pred_pos_prop = self.apply_quantifier(quantifier=qtf, test=self.window)
      pred_pos_prop = round(pred_pos_prop[1], 2)# Getting the positive proportion
      proportions[qtf] = pred_pos_prop
    return proportions



    
  def calc_threshold(self, pos_prop, pos_scores):
    # Organiza a lista de probabilidades em ordem crescente
    ordered_pos_scores = sorted(pos_scores, reverse=True)
    
    cut = int(len(ordered_pos_scores) * pos_prop)
        
    if cut == len(ordered_pos_scores):
      threshold = ordered_pos_scores[-1]
    else:
      # Obtém o valor do threshold para a classe atual
      threshold = ordered_pos_scores[cut]
    # Armazena o threshold no dicionário
    return threshold
  
  
  def get_best_threshold(self, pos_prop, pos_scores, thr=0.5, tolerance=0.01):
    min = 0.0
    max = 1.0
    max_iteration = math.ceil(math.log2(len(pos_scores))) * 2 + 10
    for _ in range(max_iteration):
        new_proportion = sum(1 for score in pos_scores if score > thr) / len(pos_scores)
        if abs(new_proportion - pos_prop) < tolerance:
            return thr

        elif new_proportion > pos_prop:
            min = thr
            thr = (thr + max) / 2

        else:
            max = thr
            thr = (thr + min) / 2

    return thr
  





def classifier_accuracy(self, pos_proportion, pos_test_scores, labels):
    sorted_scores = sorted(pos_test_scores)

    threshold = get_best_threshold(pos_proportion, sorted_scores)

    pred_labels = [1 if score >= threshold else 0 for score in pos_test_scores]

    corrects = sum(1 for a, b in zip(pred_labels, labels) if a == b)
    accuracy = corrects / len(pred_labels)

    return accuracy, threshold
      
                           