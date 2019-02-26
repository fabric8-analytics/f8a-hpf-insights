#!/usr/bin/env python
# coding: utf-8

import os 
import pickle
import numpy as np
import pandas as pd
import hpfrec
import itertools
import json
from rudra.data_store.aws import AmazonS3
import logging


#constants

AWS_S3_ACCESS_KEY_ID = os.environ.get("AWS_S3_ACCESS_KEY_ID", "")
AWS_S3_SECRET_ACCESS_KEY = os.environ.get("AWS_S3_SECRET_ACCESS_KEY", "")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME", "hpf-insights")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "2019-01-03")
DEPLOYMENT_PREFIX = os.environ.get("DEPLOYMENT_PREFIX", "dev")



logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_S3():
    s3_object = AmazonS3(bucket_name=AWS_S3_BUCKET_NAME,
                    aws_access_key_id=AWS_S3_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_S3_SECRET_ACCESS_KEY)

    s3_object.connect()
    logger.info("S3 connection established.")
    return s3_object
 
def load_data(s3_client):
    HPF_output_package_id_dict = os.path.join("maven", DEPLOYMENT_PREFIX, 
                                              MODEL_VERSION, "trained-model/package_id_dict.json")
    HPF_output_manifest_id_dict = os.path.join("maven", DEPLOYMENT_PREFIX, 
                                               MODEL_VERSION, "trained-model/manifest_id_dict.json")
    if ((s3_client.object_exists(HPF_output_package_id_dict)) and
        (s3_client.object_exists(HPF_output_manifest_id_dict))):
        package_id_dict_ = s3_client.read_json_file(HPF_output_package_id_dict)
        manifest_id_dict_ = s3_client.read_json_file(HPF_output_manifest_id_dict)
        return [package_id_dict_, manifest_id_dict_]
    else:
        HPF_output_raw_dict = os.path.join("maven", DEPLOYMENT_PREFIX, 
                                              MODEL_VERSION, "data/manifest.json")
        raw_data_dict_ = s3_client.read_json_file(HPF_output_raw_dict)
        logger.info("Size of Raw Manifest file is: {}".format(len(raw_data_dict_)))
        return [raw_data_dict_]

def generate_package_id_dict(manifest_list):
    package_id_dict = {}
    count = 0
    for manifest in manifest_list:
        for package_name in manifest:
            if package_name in package_id_dict.keys():
                continue
            else:
                self.package_id_dict[package_name] = count
                count += 1
                
    return package_id_dict

def generate_manifest_id_dict(manifest_list, package_id_dict):
    count = 0
    manifest_id_dict = {}
    for manifest in manifest_list:
            package_set = set()
            for each_package in manifest:
                package_set.add(package_id_dict[each_package])
            manifest_id_dict[count] = list(package_set)
            count += 1
            
    return manifest_id_dict
    
def preprocess_raw_data(raw_data_dict):
    all_manifest_list = raw_data_dict.get('package_list', [])
    logger.info("Number of manifests collected = {}".format(
        len(all_manifest_list)))
    trimmed_manifest_list = [
        manifest for manifest in all_manifest_list if 13 < len(manifest) < 15]
    logger.info("Number of trimmed manifest = {}". format(
        len(self.trimmed_manifest_list)))
    del all_manifest_list
    package_id_dict = generate_package_id_dict(trimmed_manifest_list)
    manifest_id_dict = generate_manifest_id_dict(trimmed_manifest_list, package_id_dict)
    return package_id_dict, manifest_id_dict
    
def preprocess_data(data_list):
    if len(data_list) == 2:
        package_dict = (data_list[0])[0].get('package_list')
        manifest_dict = (data_list[1])[0].get('manifest_list')
        logger.info("Size of Package ID dictionary {} and Manifest ID dictionary are: {}".format(
            len(package_dict), len(manifest_dict)))
        return package_dict, manifest_dict
    else:
        raw_data = data_list[0]
        package_dict = preprocess_raw_data(raw_data)[0]
        manifest_dict = preprocess_raw_data(raw_data)[1]
        logger.info("Size of Package ID dictionary {} and Manifest ID dictionary are: {}".format(
            len(package_dict), len(manifest_dict)))
        return package_dict, manifest_dict
    
def make_user_item_df(manifest_dict, package_dict):
    user_item_list = []
    id_package_dict = {v: k for k, v in package_dict.items()} 
    for k, v in manifest_dict.items():
        user_id = int(k)
        for package in v:
            if package in id_package_dict:
                item_id = package
                count = 1
                user_item_list.append(
                {
                        "UserId": user_id,
                        "ItemId": item_id,
                        "Count": 1
                    }
                )
            
    return user_item_list

def train_test_split(data_df):
    data_df = data_df.sample(frac = 1)
    df_user = data_df.drop_duplicates(['UserId'])
    data_df = data_df.sample(frac = 1)
    df_item = data_df.drop_duplicates(['ItemId'])
    train_df = pd.concat([df_user, df_item]).drop_duplicates()
    fraction = round(frac(data_df, train_df), 2)

    if fraction < 0.80:
        df_ = extra_df(fraction, data_df, train_df)
        train_df = pd.concat([train_df, df_])
    test_df = pd.concat([data_df, train_df]).drop_duplicates(keep=False)
    logger.info("Size of Training DF {} and Testing DF are: {}".format(
            len(train_df), len(test_df)))
    return train_df, test_df
    
       
#Finding the unique elements from two lists
def check_unique(list1, list2):
    if set(list2).issubset(set(list1)):
        return True
    return [False, set(list2)&set(list1)]
    
#Calculating the fraction    
def frac(data_df, train_df):
    fraction = (len(train_df.index)/len(data_df.index))
    return fraction

#Calculating DataFrame according to fraction
def extra_df(frac, data_df, train_df):
    remain_frac = float("%.2f" % (0.80-frac))
    len_df = len(data_df.index)
    no_rows = round(remain_frac*len_df)
    df_remain = pd.concat([data_df, train_df]).drop_duplicates(keep = False)
    df_remain_rand = df_remain.sample(frac=1)
    return df_remain_rand[:no_rows]

#Calculating recall according to no of recommendations
def recall_at_m(m, test_df, recommender, user_count):
    recall = []
    for i in range(user_count):
        x = np.array(test_df.loc[test_df.UserId.isin([i])].ItemId)
        rec_l = len(x)
        recommendations = recommender.topN(user=i, n=m, exclude_seen=True)
        intersection_length = len(np.intersect1d(x, recommendations))
        try:
            recall.append({"recall": intersection_length/rec_l, "length": rec_l, "user": i})
        except ZeroDivisionError as e:
            pass
    
    recall_df = pd.DataFrame(recall, index=None)
    return recall_df['recall'].mean()

def precision_at_m(m, test_df, recommender, user_count):
    precision = []
    for i in range(user_count):
        x = np.array(test_df.loc[test_df.UserId.isin([i])].ItemId)
        recommendations = recommender.topN(user=i, n=m, exclude_seen=True)
        l = len(recommendations)
        intersection_length = len(np.intersect1d(x, recommendations))
        try:
            precision.append({"precision": intersection_length/l, "length": l, "user": i})
        except ZeroDivisionError as e:
            pass
    
    precision_df = pd.DataFrame(precision, index=None)
    return precision_df['precision'].mean()

def precision_recall_at_m(m, test_df, recommender, user_item_df):
    user_count = len(user_item_df.groupby("UserId"))
    try:
        precision = precision_at_m(m, test_df, recommender, user_count)
        recall = recall_at_m(m, test_df, recommender, user_count)
    except ValueError as e:
        pass
    logger.info("Precision {} and Recall are: {}".format(
            precision, recall))
    return precision, recall
        
        
def run_recommender(train_df, k):
    #Initialize the recommender(where latent factor is 300)
    recommender = hpfrec.HPF(k=300, random_seed=123,
                      check_every=10, maxiter=400, reindex=False, verbose=True,
                      allow_inconsistent_math=True, ncores=24)
    recommender.step_size = None
    logger.warning("Model is training, Don't interrupt.") 
    recommender.fit(train_df)
    return recommender

def save_model(s3_client, recommender):
    try:
        status = s3_client.write_pickle_file(os.path.join("maven", DEPLOYMENT_PREFIX, 
                                        MODEL_VERSION, "intermediate-model/hpf_model.pkl"), recommender)
        logging.info("Model has been saved {}.".format(status))
    except Exception as exc:
        logging.error(str(exc))

def save_hyperparams(s3_client, content_json):
    try:
        status = s3_client.write_json_file(os.path.join("maven", 
                              DEPLOYMENT_PREFIX, MODEL_VERSION, "intermediate-model/hyperparameters.json"), 
                              content_json)
        logging.info("Precision and Recall has been saved {}.".format(status))
    except Exception as exc:
        logging.error(str(exc)) 

def save_obj(s3_client, trained_recommender, precision, recall):
    logging.info("Trying to save the model.")
    save_model(s3_client, trained_recommender)
    contents = {
        "Precision": precision,
        "Recall": recall
    }
    logging.info("Trying to save the Hyperparameters.")
    save_hyperparams(s3_client, contents)

def train_model():
    s3_obj = load_S3()
    data = load_data(s3_obj)
    package_id_dict, manifest_id_dict = preprocess_data(data)
    user_item_list = make_user_item_df(manifest_id_dict, package_id_dict)
    user_item_df = pd.DataFrame(user_item_list)
    training_df, testing_df = train_test_split(user_item_df)
    trained_recommender = run_recommender(training_df, 300)
    precision, recall = precision_recall_at_m(30, testing_df, trained_recommender, user_item_df)
    try:
        save_obj(s3_obj, trained_recommender, precision, recall)  
    except Exception as error:
        logger.error(error)
        raise
    
if __name__ == "__main__":
    train_model()





