from datetime import datetime

from tensorflow.keras.callbacks import Callback


class TerminateOnTimeOut(Callback):
    def __init__(self, timeout_in_min=10):
        super(TerminateOnTimeOut, self).__init__()
        self.run_timestamp = None
        self.timeout_in_sec = timeout_in_min * 60
        # self.validation_data = validation_data

    def on_train_begin(self, logs={}):
        self.run_timestamp = datetime.now()

    def on_batch_end(self, epoch, logs={}):
        run_end = datetime.now()
        run_duration = run_end - self.run_timestamp
        run_in_sec = run_duration.total_seconds()  # / (60 * 60)
        # print(' - current training time = %2.3fs/%2.3fs' % (run_in_sec, self.timeout_in_sec))
        if self.timeout_in_sec != -1:
            if run_in_sec >= self.timeout_in_sec:
                print(
                    " - timeout: training time = %2.3fs/%2.3fs"
                    % (run_in_sec, self.timeout_in_sec)
                )
                # print('TimeoutRuntime: %2.3fs, Maxtime: %2.3fs' % (run_in_sec, self.timeout_in_sec))
                self.model.stop_training = True
                # if self.validation_data is not None:
                #    x, y = self.validation_data[0], self.validation_data[1]
                #    loss, acc = self.model.evaluate(x,y)
                #    #print(self.model.history.keys())
