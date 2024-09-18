import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn import svm
from sklearn.metrics import root_mean_squared_log_error
from copy import deepcopy
from prophet import Prophet

import logging
logging.getLogger("prophet").setLevel(logging.CRITICAL)
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)


def my_GridSearchCV(estimator, X, y, param_grid, cv):   
    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_model_score = float('inf')
    
    for params in permutations_dicts:

        scores = []
        pointer = 0

        for train_index, test_index in cv.split(X, y):  
            model = estimator(**params)                        
            pointer += 1
            x_train, x_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            if (isinstance(estimator(), Prophet)):
                df = pd.concat([x_train, y_train], axis = 1)
                df = df.rename(columns={"visit_date": "ds", "visitors": "y"})
                model.fit(df)

                df = deepcopy(x_test)
                df = df.rename(columns={"visit_date": "ds"})
                pred = model.predict(df)
                pred = pred[["yhat"]]
                pred[pred < 0] = 0

            else:
                model.fit(x_train, y_train)

                pred = model.predict(x_test)
    

            score = root_mean_squared_log_error(y_test, pred)
    
            scores.append(score)

        model_score = np.mean(scores)
        
        if model_score < best_model_score:
            best_model_score = model_score
            best_params = params

    best_model = estimator(**best_params)

    return best_params




def my_nested_cv(estimator, X, y, p_grid, inner_splits=5, outer_splits=5, inner_gap=3, outer_gap=3, test_size=30):
    cv_inner = TimeSeriesSplit(n_splits=inner_splits, gap=inner_gap, test_size=test_size)          
    cv_outer = TimeSeriesSplit(n_splits=outer_splits, gap=outer_gap, test_size=test_size)

    history = []

    pointer = 0
    for train_index, test_index in cv_outer.split(X, y):                          
        pointer += 1
        print('NestedCV: {} of outer fold {}'.format(pointer, cv_outer.get_n_splits()))
        x_train, x_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        
        params = my_GridSearchCV(estimator=estimator, 
                                X=X, 
                                y=y,
                                param_grid=p_grid, 
                                cv=cv_inner) 

        model = estimator(**params)    

         
        if (isinstance(estimator(), Prophet)):
            df = pd.concat([x_train, y_train], axis = 1)
            df = df.rename(columns={"visit_date": "ds", "visitors": "y"})
            model.fit(df)

            df = deepcopy(x_test)
            df = df.rename(columns={"visit_date": "ds"})
            pred = model.predict(df)
            pred = pred[["yhat"]]
            pred[pred < 0] = 0
            
        else:
            model.fit(x_train, y_train)

            pred = model.predict(x_test)

        score = root_mean_squared_log_error(y_test, pred)
    
        print("Score:", score, "\n")
        history.append(score)
        # print(train_index, test_index)

    print('Overall test performance: {}'.format(np.mean(history)))