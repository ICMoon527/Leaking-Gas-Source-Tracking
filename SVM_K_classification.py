# -*- coding: utf-8 -*-

'''
#SVM algorithm using sklearn
'''

import os
import numpy as np
import sklearn
import time
from utils.logger import setup_logger
import utils.ReadCSV as ReadCSV
from sklearn.model_selection import train_test_split
import joblib
from sklearn import metrics

logger = setup_logger("SVM", './Results', 0, mode='a+')
logger.info('='*50)

def timeit(func):
    def wrapper(*args, **kwargs):
        name = func
        for attr in ('__qualname__', '__name__'):
            if hasattr(func, attr):
                name = getattr(func, attr)
                break

        logger.info("Start call: {}".format(name))
        now = time.time()
        result = func(*args, **kwargs)
        using = (time.time() - now)
        logger.info("End call {}, using: {:.1f} s".format(name, using))
        
        return result
    return wrapper

class Classify():
    def __init__(self) -> None:
        pass

    @timeit
    def get_datasets(self, path):
        self.data_object = ReadCSV.ReadCSV(path)
        X, Y = self.data_object.readAll()
        # print(X.shape, Y.shape, sep='\n')

        X, Y = sklearn.utils.shuffle(X, Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        return X_train, y_train.ravel(), X_test, y_test.ravel()


    @timeit
    def SVM_K_classification_train(self, X, y):
        from sklearn.svm import SVC
        # from thundersvm import SVC
        model = SVC(kernel='linear', decision_function_shape='ovo')
        model.fit(X, y)
        return model


    @timeit
    def score(self, model, X_train, y_train, X_test, y_test):
        score_train = model.score(X_train, y_train)
        score_test = model.score(X_test, y_test)
        logger.info('Train: {} | Test: {}'.format(score_train, score_test))

        y_test_pred = model.predict(X_test)
        ov_acc = metrics.accuracy_score(y_test_pred,y_test)
        logger.info("overall accuracy: %f"%(ov_acc))
        logger.info("===========================================")
        acc_for_each_class = metrics.precision_score(y_test,y_test_pred,average=None)
        logger.info("acc_for_each_class:\n",acc_for_each_class)
        logger.info("===========================================")
        avg_acc = np.mean(acc_for_each_class)
        logger.info("average accuracy:%f"%(avg_acc))
        logger.info("===========================================")
        classification_rep = metrics.classification_report(y_test,y_test_pred,
                                                target_names=['Source '+str(i+1) for i in range(79)])
        logger.info("classification report: \n",classification_rep)
        logger.info("===========================================")
        confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1),y_test_pred.argmax(axis=1))
        logger.info("confusion metrix:\n",confusion_matrix)
        logger.info("===========================================")
        logger.info("[INFO] Successfully get SVM's classification overall accuracy ! ")

    def process(self, data_path):
        X_train, y_train, X_test, y_test = self.get_datasets(data_path)
        model = self.SVM_K_classification_train(X_train, y_train)
        self.score(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    proj = Classify()
    proj.process('data/First_part.xlsx')
    logger.info('='*50)

