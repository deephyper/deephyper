import json
import os
import warnings
from datetime import datetime

import numpy as np
import requests
from keras import backend as K
from keras.callbacks import Callback


def compute_trainable_params(model):

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    return {'trainable_params': trainable_count,
            'non_trainable_params': non_trainable_count,
            'total_params': (trainable_count + non_trainable_count)}


class CandleRemoteMonitor(Callback):
    """Capture Run level output and store/send for monitoring
    """

    def __init__(self,
                 params=None):
        super(CandleRemoteMonitor, self).__init__()

        self.global_params = params
        self.has_solr_config = False
        if 'solr_root' in params and params['solr_root'] != '':
            self.has_solr_config = True
            self.root = params['solr_root']
            self.path = '/run/update?commit=true'
            self.headers = {'Content-Type': 'application/json'}

        # init
        self.experiment_id = None
        self.run_id = None
        self.run_timestamp = None
        self.epoch_timestamp = None
        self.log_messages = []

    def on_train_begin(self, logs=None):
        logs = logs or {}
        self.run_timestamp = datetime.now()
        self.experiment_id = self.global_params['experiment_id'] if 'experiment_id' in self.global_params else "EXP_default"
        self.run_id = self.global_params['run_id'] if 'run_id' in self.global_params else "RUN_default"

        run_params = []
        for key, val in self.global_params.items():
            run_params.append("{}: {}".format(key, val))

        send = {'experiment_id': self.experiment_id,
                'run_id': self.run_id,
                'parameters': run_params,
                'start_time': str(self.run_timestamp),
                'status': 'Started'
               }
        # print("on_train_begin", send)
        self.log_messages.append(send)
        if self.has_solr_config:
            self.submit(send)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_timestamp = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        epoch_total = self.global_params['epochs']
        epoch_duration = datetime.now() - self.epoch_timestamp
        epoch_in_sec = epoch_duration.total_seconds()
        epoch_line = "epoch: {}/{}, duration: {}s, loss: {}, val_loss: {}".format(
            (epoch + 1), epoch_total, epoch_in_sec, loss, val_loss)

        send = {'run_id': self.run_id,
                'status': {'set': 'Running'},
                'training_loss': {'set': loss},
                'validation_loss': {'set': val_loss},
                'run_progress': {'add': [epoch_line]}
               }
        # print("on_epoch_end", send)
        self.log_messages.append(send)
        if self.has_solr_config:
            self.submit(send)

    def on_train_end(self, logs=None):
        logs = logs or {}
        run_end = datetime.now()
        run_duration = run_end - self.run_timestamp
        run_in_hour = run_duration.total_seconds() / (60 * 60)

        send = {'run_id': self.run_id,
                'runtime_hours': {'set': run_in_hour},
                'end_time': {'set': str(run_end)},
                'status': {'set': 'Finished'},
                'date_modified': {'set': 'NOW'}
               }
        # print("on_train_end", send)
        self.log_messages.append(send)
        if self.has_solr_config:
            self.submit(send)

        # save to file when finished
        self.save()

    def submit(self, send):
        """Send json to solr

        Arguments:
        send -- json object
        """
        try:
            requests.post(self.root + self.path,
                          json=[send],
                          headers=self.headers)
        except requests.exceptions.RequestException:
            warnings.warn(
                'Warning: could not reach RemoteMonitor root server at ' + str(self.root))

    def save(self):
        """Save log_messages to file
        """
        # path = os.getenv('TURBINE_OUTPUT') if 'TURBINE_OUTPUT' in os.environ else '.'
        path = self.global_params['save'] if 'save' in self.global_params else '.'
        if not os.path.exists(path):
            os.makedirs(path)

        filename = "/run.{}.json".format(self.run_id)
        with open(path + filename, "a") as file_run_json:
            file_run_json.write(json.dumps(self.log_messages, indent=4, separators=(',', ': ')))

class TerminateOnTimeOut(Callback):
    def __init__(self, timeout_in_sec = 10):
        super(TerminateOnTimeOut, self).__init__()
        self.run_timestamp = None
        self.timeout_in_sec = timeout_in_sec
    def on_train_begin(self, logs={}):
        self.run_timestamp = datetime.now()
    def on_epoch_end(self, epoch, logs={}):
        run_end = datetime.now()
        run_duration = run_end - self.run_timestamp
        run_in_sec = run_duration.total_seconds() #/ (60 * 60)
        print('Current time ....%2.3f' % run_in_sec)
        if self.timeout_in_sec != -1:
            if run_in_sec >= self.timeout_in_sec:
                print('Timeout==>Runtime: %2.3fs, Maxtime: %2.3fs' % (run_in_sec, self.timeout_in_sec))
                self.model.stop_training = True



