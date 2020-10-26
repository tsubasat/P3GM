from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV, ShuffleSplit


from sklearn.metrics import confusion_matrix

def classify(samples, labels, x_test, y_test):
    
    results = {}
    
    sum_roc = 0
    sum_prc = 0
    rocs = []
    prcs = []
    
    print("LR")
    clf = LogisticRegression()
    score = fit_roc_prc(clf, samples, labels, x_test, y_test)
    
    results["LR"] = score
    
    print("xgb")
    clf = xgb.XGBRegressor(nthread=1)
    score = fit_roc_prc(clf, samples, labels, x_test, y_test)

    results["XgBoost"] = score
    
    print("ADB")
    clf = AdaBoostClassifier()
    score = fit_roc_prc(clf, samples, labels, x_test, y_test)
    
    results["ADB"] = score
    
    print("GBC")
    clf = GradientBoostingClassifier(max_features="sqrt", min_samples_leaf=50, min_samples_split=200, max_depth = 8)
    score = fit_roc_prc(clf, samples, labels, x_test, y_test)
    results["GBC"] = score
    
    print("average")
    results["average"] = list(np.array(list(results.values())).mean(axis=0))
    print(results["average"])
    
    return results

    
def fit_roc_prc(clf, samples, labels, x_test, y_test):
    clf.fit(samples, labels)
    
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x_test)[:,1]
    else:
        proba = clf.predict(x_test)
    
    roc = roc_auc_score(y_test, proba)
    prc = average_precision_score(y_test, proba)
    print("AUROC", roc)
    print("AUPRC", prc)
    
    
    thres = np.linspace(0,1)

    accuracies = []

    for thre in thres:
        pred = [int(val) for val in (proba > thre)]
        accuracies.append(accuracy_score(y_test, pred))

    thre = np.argmax(accuracies)
    accuracy = accuracies[thre]
    pred = [int(val) for val in (proba > thres[thre])]
    
    confusion = confusion_matrix(y_test, pred)
    print("acc", accuracy)

    return [roc, prc, accuracy]