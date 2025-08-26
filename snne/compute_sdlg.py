import os
import logging

import pickle
import yaml
import wandb
from tqdm import tqdm
import torch
import pandas as pd
import evaluate
import numpy as np
from rouge_score import tokenizers

# ----------- 并不确定要不要 -----------
import datasets 
from torch.utils.data import DataLoader, Dataset

# ----------- 并不确定要不要 -----------

from sdlg.args import Args
from sdlg.utils import seed_everything, get_models_and_tokenizers, compute_correctness, compute_semantic_paris
from sdlg.utils import generate_text, prepare_generated_text, compute_likelihood, prepare_likelihood
from sdlg.sdlg import generate_semantically_diverse_output_sequences

from snne.uncertainty.utils.compute_utils import get_parser, setup_wandb, load_precomputed_results, print_best_scores, collect_info
from snne.uncertainty.utils.eval_utils import auroc, auarc, aucpr, is_binary_list
from snne.uncertainty.utils import utils
from snne.uncertainty.utils.entropy_utils import get_sdlg_pair
from snne.uncertainty.utils.metric_utils import get_metric
from snne.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta

# --------- Common Structure --------

# Load pre-computed results

# Load models

# Collect info

# Calculate SDLG

# Output to CSV

# --------- Common Structure --------


# Set up log
utils.setup_logger()

# Parse arguments
args = get_parser()
logging.info("Args: %s", args)
utils.set_all_seeds(args.random_seed)

# Set up wandb
setup_wandb(args, prefix='compute_sdlg')

# Load pre-computed results
precomputed_results = load_precomputed_results(args)
validation_generations = precomputed_results['validation_generations']
save_embedding_path = precomputed_results['save_embedding_path']
save_dict = precomputed_results['save_dict']
sdlg_exist = precomputed_results['sdlg_exist']
list_semantic_ids = precomputed_results['list_semantic_ids']

# Load models
save_list = []
load_list = []

if not sdlg_exist:
    print("Load measurement model")
    entailment_model = EntailmentDeberta()
    save_list = ['sdlg']
else:
    entailment_model = None
    load_list = ['sdlg']
tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize
rouge = evaluate.load('rouge', keep_in_memory=True)

if args.recompute_accuracy:
    # This is usually not enabled.
    logging.warning('Recompute accuracy enabled.')
    metric = get_metric(args.metric)
else:
    metric = None

# Collect info
result_dict = collect_info(
    args, 
    validation_generations, 
    metric, 
    entailment_model, 
    None, 
    rouge,
    tokenizer,
    list_semantic_ids,
    save_dict, 
    save_embedding_path, 
    save_list=save_list,
    load_list=load_list
)

validation_is_true = result_dict['validation_is_true']
list_generation = result_dict['list_generation']
list_sdlg_semantic_pairs = result_dict['llist_sdlg_semantic_pairs']

# Calculate SDLG-pair
list_method_name = []
list_auroc = []
list_auarc = []
list_aucpr = []

validation_is_false = [1.0 - is_t for is_t in validation_is_true]
is_binary = is_binary_list(validation_is_false)
list_responses = [list_generation]

list_similarity_matrix = [list_sdlg_semantic_pairs]
similarity_name_choice = ['sdlg']

for similarity_matrix, similarity_name in zip(list_similarity_matrix, similarity_name_choice):
    list_sdlg = []
    
    for idx in tqdm(range(len(validation_is_true))):
        # Uncertainty      
        list_sdlg.append(get_sdlg_pair(similarity_matrix[idx].numpy())[0]) # get_sdlg_pair should be defined in entropy_utils.py modified from run_experiment.py
    
    if is_binary:
        mat_auroc = auroc(validation_is_false, list_sdlg)
    else:
        mat_auroc = -1
    mat_auarc = auarc(list_sdlg, validation_is_true)
    mat_aucpr = aucpr(list_sdlg, validation_is_true)
    
    list_method_name.append(similarity_name)
    list_auroc.append(mat_auroc)
    list_auarc.append(mat_auarc)
    list_aucpr.append(mat_aucpr)
                
# Output to CSV
data_metrics = {
    'method': list_method_name,
    'auroc': list_auroc,
    'auarc': list_auarc,
    'prr': list_aucpr
}

df_metrics = pd.DataFrame(data_metrics)
logging.info(df_metrics.head())
os.makedirs('sdlg_results', exist_ok=True)
df_metrics.to_csv(f'sdlg_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}.csv', index=False)

wandb.finish()
