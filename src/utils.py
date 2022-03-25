import os
import subprocess
import datetime
import joblib
from copy import deepcopy
from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from tqdm.auto import tqdm

import warnings

warnings.filterwarnings("ignore")


def ts_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")


def load_and_preprocess_data(
    path_to_data: Union[str, Path]
) -> Union[pd.DataFrame, None]:

    """
    loads csv or tries to unzip .gz archive first and then loads csv
    then converts datetime features
    """

    remove_extracted = False
    if str(path_to_data).endswith(".gz"):
        return_code = subprocess.call(f"gunzip -k -f '{path_to_data}'", shell=True)
        if return_code:
            print(f"unable to unzip file at {path_to_data}")
        else:
            # return_code == 0 <-> unzipped successful
            path_to_data = str(path_to_data).rstrip(".gz")
            remove_extracted = True
    if os.path.exists(path_to_data) and not str(path_to_data).endswith(".gz"):
        data = pd.read_csv(path_to_data, sep=";")
        data.columns = [f.lower() for f in data.columns]

        data = data.drop_duplicates(keep="first").reset_index(drop=True)

        for f in ["push_time", "push_opened_time", "create_at"]:
            data[f] = pd.to_datetime(data[f])
        if remove_extracted:
            os.remove(path_to_data)
        return data
    return None


def calc_history_daily_features(data: pd.DataFrame) -> pd.DataFrame:

    """
    aggregates pushes history on user on each day in dataframe
    then joins it to the same dataframe as features
    for observations on the next calendar day
    it keeps only unique instances by key ['user_id', 'push_time']
    for the same `user_id` and different `push_time`s
    inside one calendar day features will be the same
    """

    data_hist = data[["user_id", "push_time", "push_opened"]]
    data_hist["push_day"] = data_hist["push_time"].dt.day
    data_hist["push_hour"] = data_hist["push_time"].dt.hour
    data_hist["push_day_start"] = data_hist["push_time"].dt.normalize()

    last_n_hours = [3, 6, 12, 24]

    ### for easier join without mess with suffixes
    data_hist["mean"] = -1
    data_hist["count"] = -1

    for h in tqdm(last_n_hours):
        lb, ub = 24 - h, 24
        mask = (data_hist["push_hour"] >= lb) & (data_hist["push_hour"] < ub)
        gb = (
            data_hist.loc[mask, :]
            .groupby(["user_id", "push_day_start"])["push_opened"]
            .agg(["mean", "count"])
        )
        gb.reset_index(inplace=True)
        gb["push_day_start"] += pd.Timedelta("1d")  ### +1 day for join
        data_hist = data_hist.merge(
            gb,
            how="left",
            on=["user_id", "push_day_start"],
            suffixes=("", f"_ystd_last{h}h"),
        )
    data_hist.drop(
        ["mean", "count", "push_day_start", "push_opened"], axis=1, inplace=True
    )
    data_hist.sort_values(by=["user_id", "push_time"], inplace=True)

    ### в исходных данных могут быть пуши в одно время и ['user_id', 'push_time'] не уникальный ключ
    ### при агрегации это не страшно и учитывается корректно, но дальше при джойнах может быть неприятно
    ### фичи при этом получаются одинаковые для любых наблюдений на одну дату пуша, тк рассчитаны за пред. календарный день

    return data_hist.drop_duplicates(
        ["user_id", "push_time"], keep="first"
    ).reset_index(drop=True)


def predict_by_hours(
    X: pd.DataFrame,
    model: LGBMClassifier,
    return_best_only: bool = False,
    push_hours: List[int] = None,
) -> Union[pd.DataFrame, pd.Series]:

    """
    creates multiple prediction for different hours in a day
    varying 'push_time' feature through values stated in `push_hours`
    then selects optimal one maximizing response probability given other features

    if `return_best_only` is True then returns pd.Series with optimal values
    otherwise pd.DataFrame with probablities for all hour values
    and optimal hour value is returned
    """

    if push_hours is None:
        push_hours = list(range(0, 9)) + list(range(13, 24))
    
    X_h = deepcopy(X)

    predictions = []
    for h in push_hours:
        X_h["push_hour"] = h
        predictions.append(model.predict_proba(X_h)[:, 1])
    P = pd.DataFrame(np.vstack(predictions).T)
    P.columns = push_hours
    P["best_push_hour"] = [push_hours[x] for x in P.values.argmax(axis=1)]
    if return_best_only:
        return P["best_push_hour"]
    return P


def make_next_day_predictions(
    users: List,
    next_day: pd.Timestamp,
    historical_data: pd.DataFrame,
    lgbm_model: LGBMClassifier,
):

    """
    users: targeted user_ids
    next_day: next targeted date for push time selection
    historical_data: all previous push interactions with users up to day before `next_day`
    lgbm_model: fitted model where first feature is `push_hour`

    чтобы рассчитать фичи для следующего дня на инференс
    нужно сконкатенировать данные за последний доступный день
    с датасетом из кортежей псевдоданных ('user_id_i', 'push_time_as_tomorrow_timestamp', -1)

    """
    last_day = next_day - pd.Timedelta("1d")

    data_for_prediction = (
        calc_history_daily_features(
            pd.concat(
                [
                    pd.DataFrame(
                        {"user_id": users, "push_time": next_day, "push_opened": -1}
                    ),
                    historical_data.loc[
                        (historical_data["push_time"].dt.normalize() == last_day)
                        & (historical_data["user_id"].isin(users)),
                        ["user_id", "push_time", "push_opened"],
                    ],
                ],
                axis=0,
                ignore_index=True,
            )
        )
        .query(f'push_time == "{next_day}"')
        .reset_index(drop=True)
    )

    features = lgbm_model.booster_.feature_name()
    data_for_prediction["best_push_hour"] = predict_by_hours(
        data_for_prediction[features], lgbm_model, return_best_only=True
    )

    return data_for_prediction[["user_id", "best_push_hour"]]


def train_model(data, features, target_name):

    MODEL_PARAMS = {
        "boosting_type": "gbdt",
        "num_leaves": 8,
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "class_weight": None,
        "min_split_gain": 0,
        "min_child_weight": 0.001,
        "min_child_samples": 500,
        "subsample": 0.5,
        "subsample_freq": 1,
        "random_state": 8,
        "n_jobs": 4,
        "silent": True,
        "importance_type": "gain",
    }

    model = LGBMClassifier(**MODEL_PARAMS)

    X_train, y_train = data[features], data["push_opened"]
    model.fit(X_train, y_train, categorical_feature=["push_hour"])

    return model


def save_model(model, dir_path):
    path_to_model = dir_path / f"push_model {ts_now()}.pkl"
    joblib.dump(model, path_to_model)
    return path_to_model


def load_model(path_to_model):
    model = joblib.load(path_to_model)
    return model
