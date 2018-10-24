## Basic packages
import numpy as np
import pandas as pd 
import sys

## xgboost
import xgboost as xgb

## Ploting 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
import seaborn as sns

## sklearn for model selection 
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn import model_selection

## sklearn logistic regression and random forest classifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# default plotting params
def set_mpl_params(style='seaborn-whitegrid', **kw):
    """Set default matplotlib plotting parameters for jupyter."""
    if style not in plt.style.available:
        print(f"style '{style}' not available", file=sys.stderr)
        style = 'seaborn-whitegrid'
    font = kw.pop('font', {'weight': 600, 'size': 16, 'family': 'StixGeneral'})
    line = kw.pop('lines', {'linewidth': 1.5, 'markersize': 10})
    fig = kw.pop('figure', {'figsize': (14, 8), 'dpi': 100,
                            'titlesize': 20, 'titleweight': 700})
    axes = kw.pop('axes', {'linewidth': 0.8,
                           'titlesize': 20, 'titlepad': 4,
                           'labelsize': 18, 'labelweight': 500})
    xtick = kw.pop('xtick', {'labelsize': 14})
    ytick = kw.pop('ytick', xtick)
    mpl.rc('figure', **fig)
    mpl.rc('font', **font)
    mpl.rc('lines', **line)
    mpl.rc('axes', **axes)
    mpl.rc('ytick', **ytick)
    mpl.rc('xtick', **xtick)
    plt.style.use(style)    


## -------------------------------------------------------------------
### Example datasets 
def get_activity_data(data="../data/motion_smartphone.csv"):
    ## Load data 
    df = pd.read_csv("../data/motion_smartphone.csv")
    df = df.drop("subject", axis=1)

    ## Convert the activity labels from 1-6 to 0-5
    df.activity -= 1

    return df.drop(['activity'], axis=1), df.activity


# dataset used in section
def get_carseats_data():
    """
    Get carseats data from ISLR packge in R. Cleans up data for classification
    by binning the Sales variables and converting factors to numerics.

    Returns transformed carseats (DataFrame, Sales)
    """
    url = "https://s3.amazonaws.com/csci-e82-section-data/carseats.csv"
    carseats = pd.read_csv(url,header=0)

    # Bin 'Sales'
    #bins = [-1,5.39,9.32,17]
    #group_names = ['Low', 'Medium', 'High']
    bins = [-1, 7.49, 17]
    group_names = ['Low',  'High']
    carseats['Sales'] = pd.cut(carseats['Sales'], bins, labels=group_names)

    # Freq. count (%) for Sales
    y = carseats['Sales'].map({'Low': 0, 'High': 1})
    carseats.drop('Sales', axis=1, inplace=True)

    # Map labels into categorical variables
    pd.options.mode.chained_assignment = None
    carseats['ShelveLoc'] = carseats['ShelveLoc']\
        .map({'Bad': 0, 'Medium': 1, 'Good':2})
    carseats['Urban'] = carseats['Urban'].map({'No': 0, 'Yes': 1 })
    carseats['US'] = carseats['US'].map({'No': 0, 'Yes': 1 })
    pd.options.mode.chained_assignment = 'warn'

    return carseats, y


## -------------------------------------------------------------------
### General classifaction helpers

def plot_confusion_matrix(cm, target_names):
    """
    Plot confusion matrix (see section 5 notebook).

    Parameters:
      - cm (2darray): confusion matrix
      - target_names (array<str>)
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title('Confusion Matrix')
    plt.set_cmap('Blues')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=60)
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# From scikit-learn examples -- no longer part of module
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

## -------------------------------------------------------------------
### XGBoost 

def plot_xgb_evals(evals, n_plot=1, ax=None, show=True, label=True, **kw):
    """
    Plot validation metrics at iterations of XGBoost.
    
    Parameters:
      - evals(dict): stored results from xgb.train 'evals_result'
      - n_plot(int): if 2, then plot validation scores for test/train on 
                     separate subplots
      - ax: if not None, a pyplot axes to use
      - show(bool): if True show plot
      - label(bool): add label to line if True
    """
    if ax is None:
        fig, ax = plt.subplots(1, n_plot, figsize=kw.pop('figsize', (12, 8)))
    
    if n_plot == 1 and not isinstance(ax, list):
        ax = [ax, ax]

    n = 0
    for i in range(len(ax)):
        ax[i].set_prop_cycle(None)
        
    for dat, metrics in evals.items():
        for metric, vals in metrics.items():
            line, = ax[n].plot(np.arange(0, len(vals)), vals, '-')
            if 'color' in kw:
                line.set_color(kw['color'][n])
            if label:
                line.set_label(f'{dat}-{metric}')
        ax[n].set_xlabel('Iteration')
        if n_plot != 1:
            ax[n].set_title(dat)
        n += 1
    plt.suptitle('Validation Metrics', y=0.92 if n == 1 else 0.95)
    ax[0].set_ylabel('Value')

    if show:
        for i in range(n):
            ax[i].legend(loc='best')
        plt.show()
    else:
        return ax


def xgb_metrics(clf, x_test, y_test):
    """
    Generate confusion matrix/classification report.

    Parameters:
      - clf(model): xgboost model
      - x_test(DMatrix): test data
      - y_test(Series): test labels
    """
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return cm, pd.DataFrame(report)


def fit_xgboost(x_train, y_train, x_test, y_test, bparam, num_round=10,
                verbose=False, **kw):
    """
    Fit XGBoost model on training data and predict test data.

    Parameters:
     - bparam(dict): booster params
     - kw(dict): passed to xdg.train
     - verbose(bool): if True print validation metrics each iteration

    Returns (model, evals, test_predictions).
    """
    ## Load the data into xgb data format
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    
    ## Validations set to watch performance
    evallist = [(dtest, 'test'), (dtrain, 'train')]
    evals = dict()
    
    ## xgboost on a training set 
    model = xgb.train(bparam, dtrain, num_round, evallist, verbose_eval=verbose,
                      evals_result=evals, **kw)
    
    ## Prediction
    y_pred = model.predict(dtest)
    
    return (model, evals, y_pred)


def xgb_accuracy_cv(X, y, bparam, n_splits=10, seed=5, keep_evals=True, **kw):
    """"
    Compute accuracy of XGBoost model on cross-validation sets.

    Parameters:
      - X(DataFrame): predictors
      - y(Series): labels
      - n_splits(int): kfold splits
      - seed(int): kfold seed
      - bparam(dict): booster params
      - keep_evals(bool): if True accumulate loss metrics each iteration
      - kw(dict): passed to xgb.train

    Returns (accuracy, metrics)
    """
    kfold = model_selection.StratifiedKFold(n_splits=n_splits, random_state=seed)
    kfold.get_n_splits()

    evals = dict()              # stores metrics
    accuracy = np.zeros(kfold.n_splits)
    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        x_train, x_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit XGBoost
        result, es, preds =\
            fit_xgboost(x_train, y_train, x_test, y_test, bparam, **kw)

        if keep_evals:
            evals[i] = es

        accuracy[i] = np.sum(preds == y_test)/len(y_test)

    return accuracy, evals
    

def xgb_eval_accuracy(preds, actual):
    """Custom function to be evaluated each boosting iteration."""
    return ("accuracy",
            np.sum(preds == actual.get_label()) / actual.num_row())


def plot_xgb_iter_metrics(evals, metric_name='merror'):
    """Plot validation metrics over iterations (only one metric)."""
    # mean, std merror / iteration
    n_iter = len(evals)
    metrics = np.zeros((n_iter, 4)) # test, train
    for i, d in evals.items():
        for j, (dat, met) in enumerate(d.items()):
            for m, vals in met.items():
                metrics[i][j*2] = np.mean(vals)
                metrics[i][j*2+1] = np.std(vals)
    metrics = pd.DataFrame(metrics, columns=["test-mean", "test-std",
                                             "train-mean", "train-std"])

    plt.figure()
    plt.ylabel(metric_name)
    plt.xlabel("iteration")
    plt.title("Validation Metrics / Iteration")
    plt.grid()
    xs = metrics.index.values
    plt.fill_between(xs, metrics["test-mean"] - metrics["test-std"],
                     metrics["test-mean"] + metrics["test-std"], alpha=0.1,
                     color="r")
    plt.fill_between(xs, metrics["train-mean"] - metrics["train-std"],
                     metrics["train-mean"] + metrics["train-std"], alpha=0.1,
                     color="g")
    plt.plot(xs, metrics["test-mean"], 'o-', color="r", label=f"Mean {metric_name}")
    plt.plot(xs, metrics["train-mean"], 'o-', color="g", label=f"Train {metric_name}")
    plt.legend(loc='best')
    plt.show()
