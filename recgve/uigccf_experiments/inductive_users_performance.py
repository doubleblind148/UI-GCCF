#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import argparse
from datasets.implemented_datasets import *
import wandb
import pandas as pd
import numpy as np
from constants import *
import time
from data.tensorflow_data import TripletsBPRGenerator
from evaluation.topk_evaluator import Evaluator
from uigccf_experiments.best_models import get_wandb_project_dict
from models.tensorflow.uigccf import UIGCCF
import tensorflow as tf
from losses.tensorflow_losses import bpr_loss, l2_reg
from utils import gpu_utils
import os

from utils.early_stopping import EarlyStoppingHandlerTensorFlow
from utils.pandas_utils import remap_column_consecutive, remap_columns_consecutive

PROJECT = "lastfm_dat"
SEEN_USERS_PERCENTAGE = [0.1, 0.3, 0.5, 0.7, 0.9]

if __name__ == "__main__":
    # select free gpu if available
    if gpu_utils.list_available_gpus() is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_gpu_lowest_memory())

    parser = argparse.ArgumentParser()
    ##########################################
    # identifier of WANDB run
    ##########################################
    parser.add_argument("--wandb_project", type=str, default=PROJECT)

    args = vars(parser.parse_args())
    wandb_project_dict = get_wandb_project_dict(args["wandb_project"])

    ##########################################
    # Retrieve run parameters
    ##########################################
    api = wandb.Api()
    run_identifier = "XXX/{}/{}".format(
        args["wandb_project"], wandb_project_dict["igcmf"]
    )
    run_object = api.run(run_identifier)
    run_parameters_dict = run_object.config
    summary = run_object.summary

    ##########################################
    # Load dataset
    ##########################################
    dataset_dict = eval(wandb_project_dict["dataset"])().load_split(
        wandb_project_dict["split_name"]
    )
    train_df = dataset_dict["train"]
    val_df = dataset_dict["val"]
    # use both train and validation data
    train_df = pd.concat([train_df, val_df])
    user_data = {"interactions": train_df}
    test_df = dataset_dict["test"]
    full_data = pd.concat([train_df, val_df, test_df])

    test_evaluator = Evaluator(
        cutoff_list=[5, 20], metrics=["Recall", "NDCG"], test_data=test_df
    )

    ##########################################
    # Split Users
    ##########################################
    # set random seed
    np.random.seed(SEED)
    users = train_df[DEFAULT_USER_COL].unique()
    num_items = len(train_df[DEFAULT_ITEM_COL].unique())
    np.random.shuffle(users)

    for perc in SEEN_USERS_PERCENTAGE:

        idx = round(len(users) * perc)
        users_partitions_list = np.split(users, [idx])

        # Initialize Adam optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=run_parameters_dict["learning_rate"]
        )

        # split train and test based on user id
        train_users = users_partitions_list[0]
        unseen_users = users_partitions_list[1]

        seen_user_train_data = train_df[train_df[DEFAULT_USER_COL].isin(train_users)]
        # I need not remapped users id for predictions
        seen_user_train_data_copy = seen_user_train_data.copy()
        unseen_user_train_data = train_df[train_df[DEFAULT_USER_COL].isin(unseen_users)]

        remap_column_consecutive(seen_user_train_data, DEFAULT_USER_COL, mapping_dict=False)

        data_gen = TripletsBPRGenerator(
            train_data=seen_user_train_data,
            batch_size=run_parameters_dict["batch_size"],
            items_after_users_idxs=False,
            full_data=full_data
        )
        num_batches = data_gen.num_samples // run_parameters_dict["batch_size"]

        @tf.function
        def train_step(idxs):
            with tf.GradientTape() as tape:
                user_emb, item_emb = model(model.urm)
                x_u = tf.gather(user_emb, idxs[0])
                x_i = tf.gather(item_emb, idxs[1])
                x_j = tf.gather(item_emb, idxs[2])
                loss = bpr_loss(x_u, x_i, x_j)
                loss += l2_reg(model, alpha=run_parameters_dict["l2_reg"])
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        run = wandb.init(
            project="{}_inductive_performance".format(args["wandb_project"]), config=run_parameters_dict
        )
        run.name = f"seen_users_percentage_noval_{perc}"
        model = UIGCCF(
            seen_user_train_data,
            embeddings_size=run_parameters_dict["embedding_size"],
            convolution_depth=run_parameters_dict["convolution_depth"],
            edge_dropout=run_parameters_dict["edge_dropout"],
            user_profile_dropout=run_parameters_dict["user_profile_dropout"],
            top_k=run_parameters_dict["top_k"],
            full_data=full_data
        )

        for epoch in range(1, summary["epoch_best_result"]):
            cum_loss = 0
            t1 = time.time()
            for batch in range(num_batches):
                idxs = tf.constant(data_gen.sample())
                loss = train_step(idxs)
                cum_loss += loss

            cum_loss /= num_batches
            log = "Epoch: {:03d}, Loss: {:.4f}, Time: {:.4f}s"
            print(log.format(epoch, cum_loss, time.time() - t1))

        test_evaluator.evaluate_recommender(
            model, user_data={"interactions": seen_user_train_data_copy}
        )
        print("SEEN users performance \n")
        test_evaluator.print_evaluation_results()

        renamed_dict = {}
        for k, v in test_evaluator.result_dict.items():
            renamed_dict[f"{k}_seen_users"] = v
        wandb.log(renamed_dict)

        test_evaluator.evaluate_recommender(
            model, user_data={"interactions": unseen_user_train_data}
        )
        print("UNSEEN users performance \n")
        test_evaluator.print_evaluation_results()

        renamed_dict = {}
        for k, v in test_evaluator.result_dict.items():
            renamed_dict[f"{k}_unseen_users"] = v
        wandb.log(renamed_dict)

        run.finish()
