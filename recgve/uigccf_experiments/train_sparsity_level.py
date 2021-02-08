#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import argparse
from datasets.implemented_datasets import *
import wandb
import pandas as pd
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

PROJECT = "ml1m_dat"
CONVOLUTION_DEPTHS = [0,1,2,3]
#DENSITY_SPLIT = [0.01, 0.001, 0.0001]
#DENSITY_SPLIT = [0.005]
DENSITY_SPLIT = [0.02, 0.03]
if __name__ == '__main__':
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

    for convolution_depth in CONVOLUTION_DEPTHS:
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

        N_USERS = len(full_data[DEFAULT_USER_COL].unique())
        N_ITEMS = len(full_data[DEFAULT_ITEM_COL].unique())
        MAX_INTERACTIONS = N_USERS*N_ITEMS

        for density in DENSITY_SPLIT:
            interactions_to_keep = round(MAX_INTERACTIONS*density)
            split_train_df = train_df.sample(n=interactions_to_keep, random_state=SEED, replace=False)

            data_gen = TripletsBPRGenerator(
                train_data=split_train_df,
                batch_size=run_parameters_dict["batch_size"],
                items_after_users_idxs=False,
                full_data=full_data
            )
            num_batches = data_gen.num_samples // run_parameters_dict["batch_size"]

            # Initialize Adam optimizer
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=run_parameters_dict["learning_rate"]
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


            run = wandb.init(
                project="{}_density_performance".format(args["wandb_project"]), config=run_parameters_dict
            )
            run.name = f"density_{density}_conv{convolution_depth}"
            model = UIGCCF(
                split_train_df,
                embeddings_size=run_parameters_dict["embedding_size"],
                convolution_depth=convolution_depth,
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
                model, user_data={"interactions": split_train_df}
            )
            test_evaluator.print_evaluation_results()
            wandb.log(test_evaluator.result_dict)
            run.finish()