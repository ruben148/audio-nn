from keras.callbacks import Callback
import os

class SaveModelEachEpoch(Callback):
    def __init__(self, config):
        super(SaveModelEachEpoch, self).__init__()
        self.model_name = os.path.join(config.get("Model", "dir"), config.get("Model", "filename"))

    def on_epoch_end(self, epoch, logs=None):
        model_filename = f"{self.model_name}_epoch_{epoch + 1}.h5"
        self.model.save(model_filename)
        print(f"Model saved as {model_filename}")

class SaveBestModel(Callback):
    def __init__(self, config):
        super(SaveBestModel, self).__init__()
        self.model_name = os.path.join(config.get("Model", "dir"), config.get("Model", "filename"))
        self.best_loss = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch + 1
            best_model_filename = f"{self.model_name}_best.h5"
            self.model.save(best_model_filename)
            print(f"Best model saved as {best_model_filename}")