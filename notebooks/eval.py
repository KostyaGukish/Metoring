import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_log_error
from copy import deepcopy
from prophet import Prophet
from joblib import Parallel, delayed


def split_data(data, k, period = "month"):
    latest_date = data['visit_date'].max()
    
    month = latest_date - pd.DateOffset(months=k)
    next_month = latest_date - pd.DateOffset(months=k-1)

    if period == "week":
        month = latest_date - pd.DateOffset(weeks=k)
        next_month = latest_date - pd.DateOffset(weeks=k-1)
    
    test = np.array(data[(data['visit_date'] >= month) & (data['visit_date'] < next_month)].index)
    train = np.array(data[data['visit_date'] < month].index)

    return train, test


def split(X):
    n_validation_months = X['visit_date'].dt.to_period('M').nunique() - 1
    n_splits = min(n_validation_months, 5)

    if n_validation_months <= 1:
        for i in range(2, 0, -1):
            yield split_data(X, i, "week")
    else:
        for i in range(n_splits, 0, -1):
            train, test = split_data(X, i, "month")
            if train.size > 2 and test.size > 2:
                yield split_data(X, i, "month")


def fit_predict_prophet(model, x_train, y_train, x_test):
    df = pd.concat([x_train, y_train], axis=1)
    df = df.rename(columns={"visit_date": "ds", "visitors": "y"})
    model.fit(df)

    df = deepcopy(x_test)
    df = df.rename(columns={"visit_date": "ds"})
    pred = model.predict(df)
    pred = pred[["yhat"]]
    pred[pred < 0] = 0
    return pred


def fit_model(estimator, X, y, params, train_index, test_index):
    model = estimator(**params)
    x_train, x_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    if isinstance(estimator(), Prophet):
        pred = fit_predict_prophet(model, x_train, y_train, x_test)

    else:
        model.fit(x_train, y_train)

        pred = model.predict(x_test)

    return root_mean_squared_log_error(y_test, pred)


def my_cross_validation(estimator, X, y, params, n_jobs=1):
    scores = []

    for train_index, test_index in split(X):
        scores.append(fit_model(estimator, X, y, params, train_index, test_index))

    model_score = np.mean(scores)

    return model_score


def my_grid_search_cv(
    estimator,
    X,
    y,
    param_grid,
    n_jobs=1,
):
    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    scores = Parallel(n_jobs=n_jobs)(
        delayed(my_cross_validation)(estimator, X, y, params, n_jobs)
        for params in permutations_dicts
    )

    return permutations_dicts[np.argmin(scores)]


def my_nested_cv(
    estimator,
    X,
    y,
    param_grid,
    inner_splits=5,
    outer_splits=5,
    inner_gap=3,
    outer_gap=3,
    test_size=30,
):
    cv_inner = TimeSeriesSplit(
        n_splits=inner_splits, gap=inner_gap, test_size=test_size
    )
    cv_outer = TimeSeriesSplit(
        n_splits=outer_splits, gap=outer_gap, test_size=test_size
    )

    history = []

    pointer = 0
    for train_index, test_index in cv_outer.split(X, y):
        pointer += 1
        print("NestedCV: {} of outer fold {}".format(pointer, cv_outer.get_n_splits()))
        x_train, x_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        params = my_grid_search_cv(
            estimator=estimator, X=X, y=y, param_grid=param_grid, cv=cv_inner, n_jobs=-1
        )

        model = estimator(**params)

        if isinstance(estimator(), Prophet):
            df = pd.concat([x_train, y_train], axis=1)
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

    print("Overall test performance: {}".format(np.mean(history)))
