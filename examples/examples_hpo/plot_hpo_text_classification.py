r"""
Hyperparameter search for text classification
=============================================

**Author(s)**: Romain Egele, Brett Eiffert.

 
In this tutorial we present how to use hyperparameter optimization on a text classification analysis example from the Pytorch documentation.
 
**Reference**:
This tutorial is based on materials from the Pytorch Documentation: `Text classification with the torchtext library <https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html>`_
"""

# %%
#
# .. code-block:: bash
#
#     %%bash
#     pip install deephyper ray numpy==1.26.4 torch torchtext==0.17.2 torchdata==0.7.1 'portalocker>=2.0.0'

# %%
# Imports
# -------
#
# All imports used in the tutorial are declared at the top of the file.

# .. dropdown:: Imports
import ray
import json
from functools import partial

import torch

from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from torch import nn

# %%
# .. note::
#   The following can be used to detect if **CUDA** devices are available on the current host. Therefore, this notebook will automatically adapt the parallel execution based on the ressources available locally. However, it will not be the case if many compute nodes are requested.
#

# %%
# 
# If GPU is available, this code will enabled the tutorial to use the GPU for pytorch operations.

# .. dropdown:: Code to check if using CPU or GPU
is_gpu_available = torch.cuda.is_available()
n_gpus = torch.cuda.device_count()

# %%
# The dataset
# -----------
#
# The torchtext library provides a few raw dataset iterators, which yield the raw text strings. For example, the :code:`AG_NEWS` dataset iterators yield the raw data as a tuple of label and text. It has four labels (1 : World 2 : Sports 3 : Business 4 : Sci/Tec).
# 

# .. dropdown:: Loading the data
def load_data(train_ratio, fast=False):
    train_iter, test_iter = AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * train_ratio)
    split_train, split_valid = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    
    ## downsample
    if fast:
        split_train, _ = random_split(split_train, [int(len(split_train)*.05), int(len(split_train)*.95)])
        split_valid, _ = random_split(split_valid, [int(len(split_valid)*.05), int(len(split_valid)*.95)])
        test_dataset, _ = random_split(test_dataset, [int(len(test_dataset)*.05), int(len(test_dataset)*.95)])

    return split_train, split_valid, test_dataset

# %%
# Preprocessing pipelines and Batch generation
# --------------------------------------------
#
# Here is an example for typical NLP data processing with tokenizer and vocabulary. The first step is to build a vocabulary with the raw training dataset. Here we use built in
# factory function :code:`build_vocab_from_iterator` which accepts iterator that yield list or iterator of tokens. Users can also pass any special symbols to be added to the
# vocabulary.
# 
# The vocabulary block converts a list of tokens into integers.

# %%
# .. code-block:: python
#
#   vocab(['here', 'is', 'an', 'example'])
#   >>> [475, 21, 30, 5286]

# %%
# The text pipeline converts a text string into a list of integers based on the lookup table defined in the vocabulary. The label pipeline converts the label into integers. For example,

# %%
# .. code-block:: python
#
#   text_pipeline('here is the an example')
#   >>> [475, 21, 2, 30, 5286]
#   label_pipeline('10')
#   >>> 9 

# .. dropdown:: Code to tokenize and build vocabulary for text processing
train_iter = AG_NEWS(split='train')
num_class = 4

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


def collate_batch(batch, device):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

# %%
# .. note:: The :code:`collate_fn` function works on a batch of samples generated from :code:`DataLoader`. The input to :code:`collate_fn` is a batch of data with the batch size in :code:`DataLoader`, and :code:`collate_fn` processes them according to the data processing pipelines declared previously.
#     

# %%
# Define the model
# ----------------
# 
# The model is composed of the `nn.EmbeddingBag <https://pytorch.org/docs/stable/nn.html?highlight=embeddingbag#torch.nn.EmbeddingBag>`_ layer plus a linear layer for the classification purpose.

# .. dropdown:: Defining the Text Classification model
class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# %%
# Define functions to train the model and evaluate results.
# ---------------------------------------------------------

# .. dropdown:: Define the training and evaluation of the Text Classification model
def train(model, criterion, optimizer, dataloader):
    model.train()

    for _, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

def evaluate(model, dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for _, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

# %%
# Define the run-function
# -----------------------
#
# The run-function defines how the objective that we want to maximize is computed. It takes a :code:`config` dictionary as input and often returns a scalar value that we want to maximize. The :code:`config` contains a sample value of hyperparameters that we want to tune. In this example we will search for:
# 
# * :code:`num_epochs` (default value: :code:`10`)
# * :code:`batch_size` (default value: :code:`64`)
# * :code:`learning_rate` (default value: :code:`5`)
# 
# A hyperparameter value can be acessed easily in the dictionary through the corresponding key, for example :code:`config["units"]`.

# .. dropdown:: Run the Text Classification model
def get_run(train_ratio=0.95):
  def run(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = 64
    
    collate_fn = partial(collate_batch, device=device)
    split_train, split_valid, _ = load_data(train_ratio, fast=True) # set fast=false for longer running, more accurate example
    train_dataloader = DataLoader(split_train, batch_size=int(config["batch_size"]),
                                shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(split_valid, batch_size=int(config["batch_size"]),
                                shuffle=True, collate_fn=collate_fn)

    model = TextClassificationModel(vocab_size, int(embed_dim), num_class).to(device)
      
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])

    for _ in range(1, int(config["num_epochs"]) + 1):
        train(model, criterion, optimizer, train_dataloader)
    
    accu_test = evaluate(model, valid_dataloader)
    return accu_test
  return run

# %%
# We create two versions of :code:`run`, one quicker to evaluate for the search, with a small training dataset, and another one, for performance evaluation, which uses a normal training/validation ratio.

# %%
quick_run = get_run(train_ratio=0.3)
perf_run = get_run(train_ratio=0.95)

# %%
# .. note:: The objective maximised by DeepHyper is the scalar value returned by the :code:`run`-function.
# 
# In this tutorial it corresponds to the validation accuracy of the model after training.

# %%
# Define the Hyperparameter optimization problem
# ---------------------------------------------- 
#
# Hyperparameter ranges are defined using the following syntax:
# 
# * Discrete integer ranges are generated from a tuple :code:`(lower: int, upper: int)`
# * Continuous prarameters are generated from a tuple :code:`(lower: float, upper: float)`
# * Categorical or nonordinal hyperparameter ranges can be given as a list of possible values :code:`[val1, val2, ...]`
# 
# We provide the default configuration of hyperparameters as a starting point of the problem.

# %%
from deephyper.hpo import HpProblem

problem = HpProblem()

# Discrete hyperparameter (sampled with uniform prior)
problem.add_hyperparameter((5, 20), "num_epochs", default_value=10)

# Discrete and Real hyperparameters (sampled with log-uniform)
problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=64)
problem.add_hyperparameter((0.1, 10, "log-uniform"), "learning_rate", default_value=5)

problem

# %%
# Evaluate a default configuration
# --------------------------------
#
# We evaluate the performance of the default set of hyperparameters provided in the Pytorch tutorial.

#We launch the Ray run-time and execute the `run` function
#with the default configuration
if is_gpu_available:
    if not(ray.is_initialized()):
        ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)
    
    run_default = ray.remote(num_cpus=1, num_gpus=1)(perf_run)
    objective_default = ray.get(run_default.remote(problem.default_configuration))
else:
    if not(ray.is_initialized()):
        ray.init(num_cpus=1, log_to_driver=False)
    run_default = perf_run
    objective_default = run_default(problem.default_configuration)

print(f"Accuracy Default Configuration:  {objective_default:.3f}")

# %%
# Define the evaluator object
# ---------------------------
#
# The :code:`Evaluator` object allows to change the parallelization backend used by DeepHyper.  
# It is a standalone object which schedules the execution of remote tasks. All evaluators needs a :code:`run_function` to be instantiated.  
# Then a keyword :code:`method` defines the backend (e.g., :code:`"ray"`) and the :code:`method_kwargs` corresponds to keyword arguments of this chosen :code:`method`.

# %%
# .. code-block:: python
#
#   evaluator = Evaluator.create(run_function, method, method_kwargs)
 
# %%
# Once created the :code:`evaluator.num_workers` gives access to the number of available parallel workers.
# 
# Finally, to submit and collect tasks to the evaluator one just needs to use the following interface:

# %%
# .. code-block:: python
#
# 	configs = [...]
# 	evaluator.submit(configs)
#	...
#	tasks_done = evaluator.get("BATCH", size=1) # For asynchronous
#	tasks_done = evaluator.get("ALL") # For batch synchronous

# %%
# .. warning:: Each `Evaluator` saves its own state, therefore it is crucial to create a new evaluator when launching a fresh search.

# %%
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback

def get_evaluator(run_function):
    # Default arguments for Ray: 1 worker and 1 worker per evaluation
    method_kwargs = {
        "num_cpus": 1, 
        "num_cpus_per_task": 1,
        "callbacks": [TqdmCallback()]
    }

    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation
    if is_gpu_available:
        method_kwargs["num_cpus"] = n_gpus
        method_kwargs["num_gpus"] = n_gpus
        method_kwargs["num_cpus_per_task"] = 1
        method_kwargs["num_gpus_per_task"] = 1

    evaluator = Evaluator.create(
        run_function, 
        method="ray", 
        method_kwargs=method_kwargs
    )
    print(f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )
    
    return evaluator

evaluator_1 = get_evaluator(quick_run)

# %%
# Define and run the Centralized Bayesian Optimization search (CBO)
# -----------------------------------------------------------------
#
# We create the CBO using the :code:`problem` and :code:`evaluator` defined above.

# %%
from deephyper.hpo import CBO

# %%
# Instanciate the search with the problem and a specific evaluator
search = CBO(problem, evaluator_1)

# %%  
# .. note:: 
#   All DeepHyper's search algorithm have two stopping criteria:
#       * :code:`max_evals (int)`: Defines the maximum number of evaluations that we want to perform. Default to :code:`-1` for an infinite number.
#       * :code:`timeout (int)`: Defines a time budget (in seconds) before stopping the search. Default to :code:`None` for an infinite time budget.
#

# %%
results = search.search(max_evals=30)

# %%
# The returned :code:`results` is a Pandas Dataframe where columns are hyperparameters and information stored by the evaluator:
# 
# * :code:`job_id` is a unique identifier corresponding to the order of creation of tasks
# * :code:`objective` is the value returned by the run-function
# * :code:`timestamp_submit` is the time (in seconds) when the hyperparameter configuration was submitted by the :code:`Evaluator` relative to the creation of the evaluator.
# * :code:`timestamp_gather` is the time (in seconds) when the hyperparameter configuration was collected by the :code:`Evaluator` relative to the creation of the evaluator.

# %%
results

# %%
# Evaluate the best configuration
# -------------------------------
#
# Now that the search is over, let us print the best configuration found during this run and evaluate it on the full training dataset.

# %%
i_max = results.objective.argmax()
best_config = results.iloc[i_max][:-3].to_dict()
best_config = {k[2:]: v for k, v in best_config.items() if k.startswith("p:")}

print(f"The default configuration has an accuracy of {objective_default:.3f}. \n" 
      f"The best configuration found by DeepHyper has an accuracy {results['objective'].iloc[i_max]:.3f}, \n" 
      f"finished after {results['m:timestamp_gather'].iloc[i_max]:.2f} secondes of search.\n")

print(json.dumps(best_config, indent=4))

# %%
objective_best = perf_run(best_config)
print(f"Accuracy Best Configuration:  {objective_best:.3f}")
