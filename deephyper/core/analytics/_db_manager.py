"""
Database
---------
A tool to interact with a database of Deephyper results.

To view the database run:

.. code-block:: console

    $ deephyper-analytics database --view '' --database db.json

To add an entry to the database run:

.. code-block:: console

    $ deephyper-analytics database --add $log_dir --database db.json

To delete an entry from the database run:

.. code-block:: console

    $ deephyper-analytics database --delete $id --database db.json
"""
import json
import os
import abc
import getpass
import platform
import subprocess

import pandas as pd
import yaml
from datetime import datetime
from typing import Union
from tinydb import TinyDB
from deephyper.core.utils._files import ensure_dh_folder_exists


class DBManager(abc.ABC):
    """Database Manager, for saving DeepHyper experiments and accessing/modifying the resulting databases.

    Example Usage:

        >>> dbm = DBManager(username="Bob", path="path/to/db.json")

    Args:
        username (str, optional): the name of the user accessing the database. Defaults to ``os.getlogin()``.
        path (str, optional): the path to the database. Defaults to ``"~/.deephyper/local_db.json"``.
    """

    def __init__(
        self,
        username: str = None,
        path: str = None,
    ) -> None:
        if path is None:
            path = os.path.join(ensure_dh_folder_exists(), "db.json")
        self._username = username if username else getpass.getuser()
        self._db = TinyDB(path)

    def _get_pip_env(self, pip_versions=True):
        if isinstance(pip_versions, str):
            with open(pip_versions, "r") as file:
                pip_env = json.load(file)
        elif pip_versions:
            pip_list_com = subprocess.run(
                ["pip", "list", "--format", "json"], stdout=subprocess.PIPE
            )
            pip_env = json.loads(pip_list_com.stdout.decode("utf-8"))
        else:
            return None
        return pip_env

    def add(
        self,
        log_dir: str,
        label: str = None,
        description: str = None,
        pip_versions: Union[str, bool] = True,
        metadata: dict = None,
    ) -> int:
        """Adds an experiment to the database.

        Example Usage:

            >>> dbm = DBManager(username="Bob", path="path/to/db.json")
            >>> metadata = {"machine": "ThetaGPU", "n_nodes": 4, "num_gpus_per_node": 8}
            >>> dbm.add("path/to/search/log_dir/", label="exp_101", description="The experiment 101", metadata=metadata)

        Args:
            log_dir (str): the path to the search's logging directory.
            label (str, optional): the label wished for the experiment. Defaults to None.
            description (str, optional): the description wished for the experiment. Defaults to None.
            pip_versions (str or bool, optional): a boolean for which ``False`` means that we don't store any pip version checkpoint, and ``True`` that we store the current pip version checkpoint ; or the path to a ``.json`` file corresponding to the ouptut of ``pip list --format json``. Defaults to True.
            metadata (dict, optional): a dictionary of metadata. When the same key is found in the default `default_metadata` and the passed `metadata` then the values from `default_metadata` are overriden by `metadata` values. Defaults to None.
        """
        context_path = os.path.join(log_dir, "context.yaml")
        with open(context_path, "r") as file:
            context = yaml.load(file, Loader=yaml.SafeLoader)

        results_path = os.path.join(log_dir, "results.csv")
        results = pd.read_csv(results_path, index_col=0).to_dict(orient="list")

        logs = []
        for log_file in context.get("logs", []):
            logging_path = os.path.join(log_dir, log_file)
            logs.append(open(logging_path, "r"))

        metadata = metadata if metadata else {}
        experiment = {
            "metadata": {
                "label": label,
                "description": description,
                "user": self._username,
                "add_date": str(datetime.now()),
                "env": self._get_pip_env(pip_versions=pip_versions),
                "search": context.get("search", None),
                **metadata,
            },
            "data": {
                "search": {
                    "calls": context.get("calls", None),
                    "results": results,
                },
                "logging": logs,  # TODO: how to save files as str ?
            },
        }
        return self._db.insert(experiment)

    def get(self, cond=None, exp_id=None):
        """Retrieve the desired experiment from the database.

        Example Usage:

            >>> dbm = DBManager(username="Bob", path="path/to/db.json")
            >>> dbm.get(23)

        Args:
            cond (tinydb.Query): a search condition.
            exp_id (int): index of the record to delete.

        Returns:
            (list|dict): the retrieved documents in the database.
        """
        if cond is None:
            exp = self._db.get(doc_id=exp_id)
            if exp is not None:
                exp = dict(exp)
                exp["id"] = exp_id
            return exp
        else:
            docs = self._db.search(cond=cond)
            for i, doc in enumerate(docs):
                doc_id = doc.doc_id
                docs[i] = dict(doc)
                docs[i]["id"] = doc_id
            return docs

    def delete(self, ids: list):
        """Deletes an experiment from the database.

        Example Usage:

            >>> dbm = DBManager(username="Bob", path="path/to/db.json")
            >>> dbm.delete([23, 16])

        Args:
            ids (list): indexes of the records to delete.
        """
        self._db.remove(doc_ids=ids)

    def list(self):
        """Returns an iterator over the records stored in the database.

        Example Usage:

            >>> dbm = DBManager(username="Bob", path="path/to/db.json")
            >>> experiments = dbm.list()
            >>> for exp in experiments:
            >>>     ...
        """
        for exp in self._db:
            doc_id = exp.doc_id
            exp = dict(exp)
            exp["id"] = doc_id
            yield (exp)


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "database"
    function_to_call = main

    parser = subparsers.add_parser(subparser_name, help="Interact with a database.")
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        default=getpass.getuser(),
        help=f"Username used to interact with the database. Defaults to '{getpass.getuser()}'.",
    )
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default="~/.deephyper/db.json",
        help="Path to the default database used for the dashboard. Defaults to '~/.deephyper/db.json'.",
    )
    parser.add_argument(
        "-A",
        "--add",
        type=str,
        default=None,
        help="Add an entry to the database. A path to the folder containing the results is expected.",
    )
    parser.add_argument(
        "-D",
        "--delete",
        type=int,
        default=None,
        help="Delete an entry from the database. The 'id' of the entry to delete is expected.",
    )
    parser.add_argument(
        "-v",
        "--view",
        type=str,
        default=None,
        help="Print a view of the database based on filtered labels.",
    )

    return subparser_name, function_to_call


def main(username, database, add, delete, view, **kwargs):
    """
    :meta private:
    """

    if not (os.path.exists(database)):
        print(f"[DBManager] Creating new database: {database}")

    dbm = DBManager(username=username, path=database)

    if add:
        input_logs = os.path.abspath(add)

        print(f"[DBManager] Adding '{input_logs}'")

        metadata = {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
        entry_id = dbm.add(
            input_logs, label=os.path.basename(input_logs), metadata=metadata
        )

        print(f"[DBManager] Entry added with 'id={entry_id}'")
    elif delete:
        entry_id = delete
        print(f"[DBManager] Deleting: {entry_id}")
        dbm.delete([entry_id])
        print("[DBManager] Entry deleted")
    elif view is not None:
        search_value = view
        data_summary = []
        for exp_data in dbm.list():
            label = exp_data["metadata"]["label"]

            if (search_value == "") or (search_value in label):
                data_summary.append(
                    {
                        "id": exp_data["id"],
                        "date created": exp_data["metadata"]["add_date"],
                        "label": exp_data["metadata"]["label"],
                    }
                )

        if len(data_summary) > 0:
            df = pd.DataFrame(data=data_summary)
            print(df)
        else:
            print("[DBManager] No entry found!")
