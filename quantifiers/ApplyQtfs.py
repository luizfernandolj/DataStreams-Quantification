import os
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
from sklearn.metrics import accuracy_score

class ApplyQtfs:
  def __init__(self, trainX, trainy, model, thr):
    self.trainX = trainX
    self.trainy = trainy
    self.model = model
    self.thr = thr
    self.quantifiers = ["CC", "ACC", "MS", "DyS"]
    self.quantifiers_initialized = {}
    self.fit(model, thr, 'topsoe', trainX, trainy)
  
  def fit(self, clf,  thr, measure, trainX, trainY):
      cc = ClassifyCount(classifier=clf, threshold=thr)
      cc.fit(trainX, trainY)
      self.quantifiers_initialized["CC"] = cc
      
      acc = AdjustedClassifyCount(classifier=clf, threshold=thr)
      acc.fit(trainX, trainY)
      self.quantifiers_initialized["ACC"] = acc

      ms = MedianSweep(classifier=clf)
      ms.fit(trainX, trainY)
      self.quantifiers_initialized["MS"] = ms

      dys = DyS(classifier=clf, similarity_measure=measure)
      dys.fit(trainX, trainY)
      self.quantifiers_initialized["DyS"] = dys
   
  def check_train(self, trainX, trainY):
    if ((not trainX.equals(self.trainX)) or (not trainY.equals(self.trainy))):
      self.trainX = trainX
      self.trainy = trainY
        # .............Calling of Methods.................
      self.fit(clf=self.model,
               thr=self.thr,
               measure='topsoe',
               trainX=trainX, 
               trainY=trainY)
      

  def predict_positive_proportion(self, quantifier, test):
    return self.quantifiers_initialized[quantifier].predict(test)
  
  def aplly_qtf(self, window):
    proportions = {}
    for qtf in self.quantifiers:
      # .............Calling of Methods.................
      pred_pos_prop = self.predict_positive_proportion(quantifier=qtf, test=window)
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

  def acc(self, vet_accs):
    real = vet_accs['real']
    for qtf, l in list(vet_accs.items())[1:]:
      print(qtf, accuracy_score(real, l))  
                           