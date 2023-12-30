from keras.callbacks import Callback
from audio_nn import model as model_utils
import os

class SaveModelEachEpoch(Callback):
    def __init__(self, config):
        super(SaveModelEachEpoch, self).__init__()
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        model_suffix = f"_epoch_{epoch + 1}"
        model_utils.save_model(self.config, self.model, model_suffix)
        print("Model saved.")

class SaveBestModel(Callback):
    def __init__(self, config):
        super(SaveBestModel, self).__init__()
        self.config = config
        self.best_loss = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch + 1
            model_suffix =  "_best"
            model_utils.save_model(self.config, self.model, model_suffix)
            print(f"Best model saved.")

class EarlyStoppingAccuracy(Callback):
    def __init__(self, threshold=0.5, patience=3):
        super(EarlyStoppingAccuracy, self).__init__()
        self.threshold = threshold
        self.patience = patience
        self.best_val_accuracy = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get('val_accuracy')
        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_val_accuracy
            self.wait = 0
        elif current_val_accuracy < self.threshold:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nStopping training: val_accuracy under {self.threshold} for {self.patience} epochs")

class EarlyStoppingMixedCriteria(Callback):
    def __init__(self, accuracy_threshold=0.7, min_delta=0.005, loss_patience=3, accuracy_patience=16, early_epoch_threshold=10):
        super(EarlyStoppingMixedCriteria, self).__init__()
        self.accuracy_threshold = accuracy_threshold
        self.min_delta = min_delta
        self.loss_patience = loss_patience
        self.accuracy_patience = accuracy_patience
        self.early_epoch_threshold = early_epoch_threshold
        self.best_val_loss = float('inf')
        self.wait_loss = 0
        self.wait_accuracy = 0
        self.stopped_early = False
        self.val_loss_history = []
        self.val_accuracy_history = []

    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get('val_accuracy')
        current_val_loss = logs.get('val_loss')

        self.val_loss_history.append(current_val_loss)
        self.val_accuracy_history.append(current_val_accuracy)

        if current_val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = current_val_loss
            self.wait_loss = 0
        else:
            self.wait_loss += 1

        if current_val_accuracy < self.accuracy_threshold:
            self.wait_accuracy = 0
        else:
            self.wait_accuracy += 1

        if epoch >= self.early_epoch_threshold:
            if len(self.val_loss_history) > 3 and current_val_loss > self.val_loss_history[-4]:
                print(f"\nStopping training: val_loss increased from 3 epochs ago")
                self.model.stop_training = True
                return

            if len(self.val_accuracy_history) > 3 and current_val_accuracy < self.val_accuracy_history[-4]:
                print(f"\nStopping training: val_accuracy decreased from 3 epochs ago")
                self.model.stop_training = True
                return

        if self.wait_loss >= self.loss_patience and self.wait_accuracy >= self.accuracy_patience:
            self.model.stop_training = True
            self.stopped_early = True
            print(f"\nStopping training: No significant improvement in val_loss for {self.loss_patience} epochs or val_accuracy under {self.accuracy_threshold}")