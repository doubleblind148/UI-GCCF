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
from models.tensorflow.lightgcn import LightGCN
from models.tensorflow.matrix_factorization_bpr import MatrixFactorizationBPR
from models.tensorflow.ngcf import NGCF
from utils import gpu_utils
import os

PROJECT = "gowalla_dat"
ALG = ["ngcf"]

if __name__ == "__main__":
    # select free gpu if available
    if gpu_utils.list_available_gpus() is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_gpu_lowest_memory())

    parser = argparse.ArgumentParser()
    ##########################################
    # identifier of WANDB run
    ##########################################
    parser.add_argument("--wandb_project", type=str, default=PROJECT)
    parser.add_argument("--algorithm", type=str, default=ALG)

    args = vars(parser.parse_args())

    algorithm = args["algorithm"]
    wandb_project_dict = get_wandb_project_dict(args["wandb_project"])

    ##########################################
    # Retrieve run parameters
    ##########################################
    api = wandb.Api()
    run_identifier = "XXX/{}/{}".format(
        args["wandb_project"], wandb_project_dict[algorithm]
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

    train_df = pd.concat([train_df, val_df])

    test_df = dataset_dict["test"]
    user_data = {"interactions": train_df}

    test_evaluator = Evaluator(
        cutoff_list=[5, 20], metrics=["Recall", "NDCG"], test_data=test_df
    )

    run = wandb.init(project=args["wandb_project"], config=run_parameters_dict)
    run.name = algorithm + "_trainval"
    # Initialize Adam optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=run_parameters_dict["learning_rate"]
    )
    if algorithm == "igcmf":
        model = UIGCCF(
            train_df,
            embeddings_size=run_parameters_dict["embedding_size"],
            convolution_depth=run_parameters_dict["convolution_depth"],
            edge_dropout=run_parameters_dict["edge_dropout"],
            user_profile_dropout=run_parameters_dict["user_profile_dropout"],
            top_k=run_parameters_dict["top_k"],
        )

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

        data_gen = TripletsBPRGenerator(
            train_data=train_df,
            batch_size=run_parameters_dict["batch_size"],
            items_after_users_idxs=False,
        )
    else:
        if algorithm == "bprmf":
            model = MatrixFactorizationBPR(
                train_df, embeddings_size=run_parameters_dict["embedding_size"]
            )
        if algorithm == "ngcf":
            model = NGCF(
                train_df,
                embeddings_size=run_parameters_dict["embedding_size"],
                convolution_depth=run_parameters_dict["convolution_depth"],
                mess_dropout=run_parameters_dict["mess_dropout"],
                node_dropout=run_parameters_dict["node_dropout"],
            )
        if algorithm == "lightgcn":
            model = LightGCN(
                train_df,
                embeddings_size=run_parameters_dict["embedding_size"],
                convolution_depth=run_parameters_dict["convolution_depth"],
            )

        @tf.function
        def train_step(idxs):
            with tf.GradientTape() as tape:
                x = model()
                x_u = tf.gather(x, idxs[0])
                x_i = tf.gather(x, idxs[1])
                x_j = tf.gather(x, idxs[2])
                loss = bpr_loss(x_u, x_i, x_j)
                loss += l2_reg(model, alpha=run_parameters_dict["l2_reg"])
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        data_gen = TripletsBPRGenerator(
            train_data=train_df,
            batch_size=run_parameters_dict["batch_size"],
            items_after_users_idxs=True,
        )

    num_batches = data_gen.num_samples // run_parameters_dict["batch_size"]
    start_training = time.time()
    epoch_times = []
    for epoch in range(1, summary["epoch_best_result"]):
        cum_loss = 0
        t1 = time.time()
        for batch in range(num_batches):
            idxs = tf.constant(data_gen.sample())
            loss = train_step(idxs)
            cum_loss += loss

        cum_loss /= num_batches
        log = "Epoch: {:03d}, Loss: {:.4f}, Time: {:.4f}s"
        epoch_times.append(time.time()-t1)
        print(log.format(epoch, cum_loss, time.time() - t1))
        wandb.log({"loss": cum_loss}, step=epoch)

    # log training time
    end_traininig = time.time()
    wandb.log({"training time": "{:.4f}s".format(end_traininig - start_training)})

    std = np.std(np.array(epoch_times))
    mean = np.mean(np.array(epoch_times))
    wandb.log({"time_per_epoch":mean, "time_per_epoch_std":std})

    test_evaluator.evaluate_recommender(model, user_data)
    test_evaluator.print_evaluation_results()

    test_result_dict = {}
    for k, v in test_evaluator.result_dict.items():
        new_key = "test_{}".format(k)
        test_result_dict[new_key] = test_evaluator.result_dict[k]
    wandb.log(test_result_dict)

    save_path = os.path.join(wandb.run.dir, "best_models")
    model_name = "best_model"
    model.save_weights(os.path.join(save_path, model_name))

    run.finish()
