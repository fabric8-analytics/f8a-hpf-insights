#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains training code for maven insights.

Copyright © 2018 Red Hat Inc

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import pandas as pd
import hpfrec
from rudra.data_store.aws import AmazonS3
from rudra.utils.helper import load_hyper_params
import logging
import json
import ruamel.yaml
from github import Github


# constants

AWS_S3_ACCESS_KEY_ID = os.environ.get("AWS_S3_ACCESS_KEY_ID", "")
AWS_S3_SECRET_ACCESS_KEY = os.environ.get("AWS_S3_SECRET_ACCESS_KEY", "")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME", "")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "2019-01-03")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
DEPLOYMENT_PREFIX = os.getenv("DEPLOYMENT_PREFIX", "")

UPSTREAM_REPO_NAME = 'openshiftio'
FORK_REPO_NAME = 'developer-analytics-bot'
PROJECT_NAME = 'saas-analytics'
YAML_FILE_PATH = 'bay-services/f8a-hpf-insights.yaml'

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_s3():
    """Create connection s3."""
    s3_object = AmazonS3(bucket_name=AWS_S3_BUCKET_NAME,
                         aws_access_key_id=AWS_S3_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_S3_SECRET_ACCESS_KEY)
    s3_object.connect()
    if s3_object.is_connected():
        logger.info("S3 connection established for {} bucket".format(AWS_S3_BUCKET_NAME))
        return s3_object
    else:
        raise Exception


def load_data(s3_client):
    """Load data from s3 bucket."""
    HPF_output_raw_dict = os.path.join(MODEL_VERSION, "data/manifest.json")
    logger.info("Reading Manifest file for {} version".format(MODEL_VERSION))
    raw_data_dict_ = s3_client.read_json_file(HPF_output_raw_dict)
    if raw_data_dict_ is None:
        raise Exception("manifest.json not found")
    logger.info("Size of Raw Manifest file is: {}".format(len(raw_data_dict_)))
    return raw_data_dict_


def check_style(data_dict):
    """Check the format of manifest file."""
    if data_dict.get("ecosystem") == "maven":
        package_dict = data_dict.get("package_dict")
        if package_dict:
            try:
                if not all([k in package_dict for k in ("user_input_stack", "bigquery_data")]):
                    logger.info("Keys are missing.")
                    return False
                else:
                    logger.info("Your manifest is in proper format.")
                    return True
            except Exception as e:
                logger.error('%r' % e)
        else:
            logger.error("ERROR - manifest.json is not in proper format.")
            return False
    else:
        logger.info("Skipping because the ecosystem is not maven.")
        return False


def generate_package_id_dict(manifest_list):
    """Generate package id dictionary."""
    package_id_dict = {}
    count = 0
    for manifest in manifest_list:
        for package_name in manifest:
            if package_name in package_id_dict.keys():
                continue
            else:
                package_id_dict[package_name] = count
                count += 1
    return package_id_dict


def generate_manifest_id_dict(manifest_list, package_id_dict):
    """Generate manifest id dictionary."""
    count = 0
    manifest_id_dict = {}
    for manifest in manifest_list:
        package_set = set()
        for each_package in manifest:
            package_set.add(package_id_dict[each_package])
        manifest_id_dict[count] = list(package_set)
        count += 1
    return manifest_id_dict


def format_dict(package_id_dict, manifest_id_dict):
    """Format the dictionaries."""
    format_pkg_id_dict = {'ecosystem': 'maven',
                          'package_list': package_id_dict
                          }
    format_mnf_id_dict = {'ecosystem': 'maven',
                          'manifest_list': manifest_id_dict
                          }
    return format_pkg_id_dict, format_mnf_id_dict


def find_unique_manifest(package_lst):
    """Find uniques manifests from raw data."""
    nested_tuples = [tuple(i) for i in package_lst]
    unique_tuples = list(set(nested_tuples))
    unique_manifests = [list(i) for i in unique_tuples]
    return unique_manifests


def preprocess_raw_data(raw_data_dict, lower_limit, upper_limit):
    """Preprocess raw data."""
    all_manifest_list = \
        [manifest for manifest in raw_data_dict.get('user_input_stack', [])
            if len(manifest) > 1] + \
        [manifest for manifest in raw_data_dict.get('bigquery_data', [])
            if lower_limit < len(manifest) < upper_limit]

    unique_manifests = find_unique_manifest(all_manifest_list)
    logger.info("Number of manifests collected = {}".format(len(unique_manifests)))
    del all_manifest_list

    package_id_dict = generate_package_id_dict(unique_manifests)
    manifest_id_dict = generate_manifest_id_dict(unique_manifests, package_id_dict)
    return package_id_dict, manifest_id_dict


def make_user_item_df(manifest_dict, package_dict, user_input_stacks):
    """Make user item dataframe."""
    user_item_list = []
    set_input_stacks = set()
    for stack in user_input_stacks:
        set_input_stacks.add(frozenset([package_dict.get(package)
                                        for package in stack]))
    id_package_dict = {v: k for k, v in package_dict.items()}
    for k, v in manifest_dict.items():
        user_id = int(k)
        is_user_input_stack = frozenset(v) in set_input_stacks
        for package in v:
            if package in id_package_dict:
                item_id = package
                user_item_list.append(
                    {
                        "UserId": user_id,
                        "ItemId": item_id,
                        "Count": 1,
                        "is_user_input_stack": is_user_input_stack
                    }
                )
    return user_item_list


def train_test_split(data_df):
    """Split for training and testing."""
    user_input_df = data_df.loc[data_df['is_user_input_stack']]
    logger.info("Size of user input df is: {}".format(len(user_input_df)))
    user_input_df = user_input_df.sample(frac=1)
    df_user = user_input_df.drop_duplicates(['UserId'])
    user_input_df = user_input_df.sample(frac=1)
    df_item = user_input_df.drop_duplicates(['ItemId'])
    train_df = pd.concat([df_user, df_item]).drop_duplicates()
    fraction = round(frac(user_input_df, train_df), 2)
    logger.info("Fraction value is: {}".format(fraction))
    if fraction < 0.80:
        df_ = extra_df(fraction, user_input_df, train_df)
        train_df = pd.concat([train_df, df_])
    logger.info("Size of training df is {}".format(len(train_df)))
    test_df = pd.concat([user_input_df, train_df]).drop_duplicates(keep=False)
    test_df = test_df.drop(columns=['is_user_input_stack'])
    data_df = data_df.loc[~data_df['is_user_input_stack']]
    train_df = pd.concat([data_df, train_df])
    train_df = train_df.drop(columns=['is_user_input_stack'])
    logger.info("Size of Training DF {} and Testing DF are: {}".format(
            len(train_df), len(test_df)))
    return train_df, test_df


# Finding the unique elements from two lists
def check_unique(list1, list2):
    """Check unique elements."""
    if set(list2).issubset(set(list1)):
        return True
    return [False, set(list2) & set(list1)]


# Calculating the fraction
def frac(data_df, train_df):
    """Calculate fraction."""
    fraction = (len(train_df.index) / len(data_df.index))
    return fraction


# Calculating DataFrame according to fraction
def extra_df(frac, data_df, train_df):
    """Calculate extra dataframe."""
    remain_frac = float("%.2f" % (0.80 - frac))
    logger.info("Remaining fraction is: {}".format(remain_frac))
    len_df = len(data_df.index)
    no_rows = round(remain_frac * len_df)
    logger.info("Number of rows is : {}".format(no_rows))
    df_remain = pd.concat([data_df, train_df]).drop_duplicates(keep=False)
    df_remain_rand = df_remain.sample(frac=1)
    return df_remain_rand[:no_rows]


# Calculating recall according to no of recommendations
def recall_at_m(m, test_df, recommender, user_count):
    """Calculate recall at `m`."""
    recall = []
    for i in range(user_count):
        x = np.array(test_df.loc[test_df.UserId.isin([i])].ItemId)
        rec_l = len(x)
        recommendations = recommender.topN(user=i, n=m, exclude_seen=True)
        intersection_length = len(np.intersect1d(x, recommendations))
        try:
            recall.append({"recall": intersection_length / rec_l, "length": rec_l, "user": i})
        except ZeroDivisionError:
            pass
    recall_df = pd.DataFrame(recall, index=None)
    return recall_df['recall'].mean()


def precision_at_m(m, test_df, recommender, user_count):
    """Calculate precision at `m`."""
    precision = []
    for i in range(user_count):
        x = np.array(test_df.loc[test_df.UserId.isin([i])].ItemId)
        recommendations = recommender.topN(user=i, n=m, exclude_seen=True)
        _len = len(recommendations)
        intersection_length = len(np.intersect1d(x, recommendations))
        try:
            precision.append({"precision": intersection_length / _len, "length": _len, "user": i})
        except ZeroDivisionError:
            pass
    precision_df = pd.DataFrame(precision, index=None)
    return precision_df['precision'].mean()


def precision_recall_at_m(m, test_df, recommender, user_item_df):
    """Precision and recall at given `m`."""
    user_count = len(user_item_df.groupby("UserId"))
    try:
        precision = precision_at_m(m, test_df, recommender, user_count)
        recall = recall_at_m(m, test_df, recommender, user_count)
    except ValueError:
        pass
    logger.info("Precision {} and Recall are: {}".format(
            precision, recall))
    return precision, recall


def run_recommender(train_df, latent_factor):
    """Start the recommender."""
    recommender = hpfrec.HPF(k=latent_factor, random_seed=123,
                             allow_inconsistent_math=True, ncores=24)
    recommender.step_size = None
    logger.warning("Model is training, Don't interrupt.")
    recommender.fit(train_df)
    return recommender


def save_model(s3_client, recommender):
    """Save model on s3."""
    try:
        status = s3_client.write_pickle_file(
                os.path.join(
                    MODEL_VERSION,
                    "intermediate-model/hpf_model.pkl"),
                recommender)
        logging.info("Model has been saved {}.".format(status))
    except Exception as exc:
        logging.error(str(exc))


def save_hyperparams(s3_client, content_json):
    """Save hyperparameters."""
    try:
        status = s3_client.write_json_file(
                os.path.join(
                    MODEL_VERSION,
                    "intermediate-model/hyperparameters.json"),
                content_json)
        logging.info("Precision and Recall has been saved {}.".format(status))
    except Exception as exc:
        logging.error(str(exc))


def save_dictionaries(s3_client, package_id_dict, manifest_id_dict):
    """Save the ditionaries for scoring."""
    pkg_status = s3_client.write_json_file(
            os.path.join(
                MODEL_VERSION,
                "trained-model/package_id_dict.json"),
            package_id_dict)
    mnf_status = s3_client.write_json_file(
            os.path.join(
                MODEL_VERSION,
                "trained-model/manifest_id_dict.json"),
            manifest_id_dict)

    if not all([pkg_status, mnf_status]):
        raise ValueError("Unable to store data files for scoring")

    logging.info("Saved dictionaries successfully")


def save_obj(s3_client, trained_recommender, hyper_params,
             package_id_dict, manifest_id_dict):
    """Save the objects in s3 bucket."""
    logging.info("Trying to save the model.")
    save_model(s3_client, trained_recommender)

    save_dictionaries(s3_client, package_id_dict, manifest_id_dict)

    logging.info("Trying to save the Hyperparameters.")
    save_hyperparams(s3_client, hyper_params)


def build_hyperparams(lower_limit, upper_limit, latent_factor,
                      precision_30, recall_30, precision_50, recall_50, deployment_type):
    """Build hyper parameter object."""
    return {
        "deployment": deployment_type,
        "model_version": MODEL_VERSION,
        "minimum_length_of_manifest": lower_limit,
        "maximum_length_of_manifest": upper_limit,
        "latent_factor": latent_factor,
        "precision_at_30": precision_30,
        "recall_at_30": recall_30,
        "f1_score_at_30": 2 * ((precision_30 * recall_30) / (precision_30 + recall_30)),
        "precision_at_50": precision_50,
        "recall_at_50": recall_50,
        "f1_score_at_50": 2 * ((precision_50 * recall_50) / (precision_50 + recall_50)),
    }


def get_deployed_model_version(yaml_dict, deployment_type):
    """Read deployment yaml and return the deployed model verison."""
    model_version = None
    environments = yaml_dict.get('services', [{}])[0].get('environments', [])
    for env in environments:
        if env.get('name', '') == deployment_type:
            model_version = env.get('parameters', {}).get('MODEL_VERSION', '')
            break

    if model_version is None:
        raise Exception(f'Model version could not be found for deployment {deployment_type}')

    logger.info('Model version: %s for deployment: %s', model_version, deployment_type)
    return model_version


def update_yaml_data(yaml_dict, deployment_type, model_version, hyper_params):
    """Update the yaml file for given deployment with model data and description as comments."""
    environments = yaml_dict.get('services', [{}])[0].get('environments', [])
    hyper_params = {k: str(v) for k, v in hyper_params.items()}
    for index, env in enumerate(environments):
        if env.get('name', '') == deployment_type:
            yaml_dict['services'][0]['environments'][index]['comments'] = hyper_params
            yaml_dict['services'][0]['environments'][index]['parameters']['MODEL_VERSION'] = \
                model_version
            break

    return ruamel.yaml.dump(yaml_dict, Dumper=ruamel.yaml.RoundTripDumper)


def build_hyper_params_message(hyper_params):
    """Build hyper params data string used for PR description and in yaml comments."""
    return '- Hyper parameters :: {}'.format(json.dumps(hyper_params, indent=4, sort_keys=True))


def format_body(body):
    """Format PR body string to replace decorators."""
    return body.replace('"', '').replace('{', '').replace('}', '').replace(',', '')


def read_deployed_data(upstream_repo, s3_client, deployment_type):
    """Read deployed data like yaml file, hyper params, model version."""
    upstream_latest_commit_hash = upstream_repo.get_commits()[0].sha
    logger.info('Upstream latest commit hash: %s', upstream_latest_commit_hash)

    contents = upstream_repo.get_contents(YAML_FILE_PATH, ref=upstream_latest_commit_hash)
    yaml_dict = ruamel.yaml.load(contents.decoded_content.decode('utf8'),
                                 ruamel.yaml.RoundTripLoader)

    deployed_version = get_deployed_model_version(yaml_dict, deployment_type)
    deployed_file_path = f'{deployed_version}/intermediate-model/hyperparameters.json'
    deployed_hyperparams = s3_client.read_json_file(deployed_file_path)
    if deployed_hyperparams is None: deployed_hyperparams = {}

    deployed_data = {
        'version': deployed_version,
        'hyperparams': deployed_hyperparams
    }
    yaml_data = {
        'content_sha': contents.sha,
        'dict': yaml_dict
    }

    return deployed_data, yaml_data, upstream_latest_commit_hash


def create_branch_and_update_yaml(deployment_type, deployed_data, yaml_data,
                                  hyper_params, latest_commit_hash):
    """Create branch and update yaml content on fork repo."""
    # Update yaml model version for the given deployment
    new_yaml_data = update_yaml_data(yaml_data['dict'], deployment_type,
                                     MODEL_VERSION, hyper_params)
    logger.info('Modified yaml data, new length: %d', len(new_yaml_data))

    # Connect to fabric8 analytic repo & get latest commit hash
    f8a_repo = Github(GITHUB_TOKEN).get_repo(f'{FORK_REPO_NAME}/{PROJECT_NAME}')
    logger.info('f8a fork repo: %s', f8a_repo)

    # Create a new branch on f8a repo
    branch_name = f'bump_f8a-hpf-insights_for_{deployment_type}_to_{MODEL_VERSION}'
    branch = f8a_repo.create_git_ref(f'refs/heads/{branch_name}', latest_commit_hash)
    logger.info('Created new branch [%s] at [%s]', branch, latest_commit_hash)

    # Update the yaml content in branch on f8a repo
    commit_message = f'Bump up f8a-hpf-insights for {deployment_type} from ' \
                     f'{deployed_data["version"]} to {MODEL_VERSION}'
    update = f8a_repo.update_file(YAML_FILE_PATH, commit_message, new_yaml_data,
                                  yaml_data['content_sha'], branch=f'refs/heads/{branch_name}')
    logger.info('New yaml content hash %s', update['commit'].sha)

    return branch_name, commit_message


def create_git_pr(s3_client, hyper_params, deployment_type):  # pragma: no cover
    """Create a git PR automatically if recall_at_30 is higher than previous iteration."""
    upstream_repo = Github(GITHUB_TOKEN).get_repo(f'{UPSTREAM_REPO_NAME}/{PROJECT_NAME}')
    deployed_data, yaml_data, latest_commit_hash = read_deployed_data(upstream_repo, s3_client,
                                                                      deployment_type)

    recall_at_30 = hyper_params['recall_at_30']
    deployed_recall_at_30 = deployed_data['hyperparams'].get('recall_at_30', 0.55)
    logger.info('create_git_pr:: Deployed => Model %s, Recall %f Current => Model %s, Recall %f',
                deployed_data['version'], deployed_recall_at_30, MODEL_VERSION, recall_at_30)
    if recall_at_30 >= deployed_recall_at_30:
        promotion_creteria = 'current_recall_at_30 >= deployed_recall_at_30'

        params = hyper_params.copy()
        params.update({'promotion_criteria': str(promotion_creteria)})
        branch_name, commit_message = create_branch_and_update_yaml(deployment_type, deployed_data,
                                                                    yaml_data, params,
                                                                    latest_commit_hash)

        hyper_params_formated = build_hyper_params_message(hyper_params)
        prev_hyper_params_formated = build_hyper_params_message(deployed_data['hyperparams'])
        body = f'''Current deployed model details:
- Model version :: `{deployed_data['version']}`
{prev_hyper_params_formated}

New model details:
- Model version :: `{MODEL_VERSION}`
{hyper_params_formated}

Criteria for promotion is `{promotion_creteria}`
'''
        pr = upstream_repo.create_pull(title=commit_message, body=format_body(body),
                                       head=f'{FORK_REPO_NAME}:{branch_name}',
                                       base='refs/heads/master')
        logger.info('Raised SAAS %s for review', pr)
    else:
        logger.warn('Ignoring latest model %s as its recall %f is less than '
                    'existing model %s recall %f', MODEL_VERSION, recall_at_30,
                    deployed_data['version'], deployed_recall_at_30)


def train_model():
    """Training model."""
    deployment_prefix_to_type_map = {
        'STAGE': 'staging',
        'PROD': 'production'
    }

    deployment_type = deployment_prefix_to_type_map.get(DEPLOYMENT_PREFIX.upper(), None)
    assert deployment_type is not None, f'Invalid DEPLOYMENT_PREFIX: {DEPLOYMENT_PREFIX}'

    s3_obj = load_s3()
    data = load_data(s3_obj)
    if check_style(data):
        hyper_params = load_hyper_params() or {}
        lower_limit = int(hyper_params.get('lower_limit', 13))
        upper_limit = int(hyper_params.get('upper_limit', 15))
        latent_factor = int(hyper_params.get('latent_factor', 300))
        logger.info("Lower limit {}, Upper limit {} and latent factor {} are used."
                    .format(lower_limit, upper_limit, latent_factor))
        package_id_dict, manifest_id_dict = preprocess_raw_data(
            data.get('package_dict', {}), lower_limit, upper_limit)
        user_input_stacks = data.get('package_dict', {}).\
            get('user_input_stack', [])
        user_item_list = make_user_item_df(manifest_id_dict, package_id_dict, user_input_stacks)
        user_item_df = pd.DataFrame(user_item_list)
        training_df, testing_df = train_test_split(user_item_df)
        format_pkg_id_dict, format_mnf_id_dict = format_dict(package_id_dict, manifest_id_dict)
        del package_id_dict, manifest_id_dict
        trained_recommender = run_recommender(training_df, latent_factor)
        precision_at_30, recall_at_30 = precision_recall_at_m(30, testing_df, trained_recommender,
                                                              user_item_df)
        precision_at_50, recall_at_50 = precision_recall_at_m(50, testing_df, trained_recommender,
                                                              user_item_df)
        try:
            hyper_params = build_hyperparams(lower_limit, upper_limit, latent_factor,
                                             precision_at_30, recall_at_30,
                                             precision_at_50, recall_at_50, deployment_type)
            save_obj(s3_obj, trained_recommender, hyper_params,
                     format_pkg_id_dict, format_mnf_id_dict)
            if GITHUB_TOKEN:
                create_git_pr(s3_obj, hyper_params, deployment_type)
            else:
                logger.info('GITHUB_TOKEN is missing, cannot raise SAAS PR')
        except Exception as error:
            logger.error(error)
            raise
    else:
        logger.error("ERROR: Training will not happen, \
            because of improper format of manifest file.")


if __name__ == "__main__":
    train_model()
