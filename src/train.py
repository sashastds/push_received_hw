import os
import pandas as pd
from pathlib import Path

from utils import load_and_preprocess_data, calc_history_daily_features
from utils import train_model, save_model, load_model
from utils import make_next_day_predictions

ENV_PATH_TO_DATA_FILE = "PATH_TO_DATA_FILE"
ENV_PATH_TO_OUTPUT_FILE = "PATH_TO_OUTPUT_FILE"
ENV_MODEL_DIRECTORY = "MODEL_DIRECTORY"

TARGET_NAME = "push_opened"


def main():

    ### we may use argparse but for now using ENV variables for paths
    ### they may be provided with docker run command
    ### or stated by default in Dockerfile

    ### getting mounted directory from env variable
    model_dir_path = Path(os.getenv(ENV_MODEL_DIRECTORY))

    ### loading data
    path_to_data = Path(os.getenv(ENV_PATH_TO_DATA_FILE))
    data = load_and_preprocess_data(path_to_data)

    if data is None:
        with open(model_dir_path / "log.txt", "w") as f_out:
            f_out.write(
                f"was unable to read data from given path: {path_to_data} - exiting\n"
            )
        return
    ### calculating historical features and merging it back to data
    data = data.merge(
        calc_history_daily_features(data), on=["user_id", "push_time"], how="left"
    )

    ### for the 1st day all features will be empty for sure - removing such observations
    min_push_day = data["push_time"].min().normalize() + pd.Timedelta("1d")
    data = data.loc[data["push_time"] >= min_push_day, :]

    ### determining features
    hist_features = [
        f for f in data.columns if f.startswith("mean") or f.startswith("count")
    ]
    features = ["push_hour"] + hist_features

    ### training model
    model = train_model(data, features, TARGET_NAME)

    ### saving model
    path_to_model = save_model(model, model_dir_path)

    ### loading back model
    model = load_model(path_to_model)

    ### creating predictions for the next day
    next_day = data["push_time"].max().normalize() + pd.Timedelta("1d")
    output_predictions = make_next_day_predictions(
        users=sorted(data["user_id"].unique()),
        next_day=next_day,
        historical_data=data,
        lgbm_model=model,
    )

    ### writing predictions to csv file
    path_to_output = Path(os.getenv(ENV_PATH_TO_OUTPUT_FILE))
    output_predictions[["user_id", "best_push_hour"]].to_csv(path_to_output)


if __name__ == "__main__":
    main()
