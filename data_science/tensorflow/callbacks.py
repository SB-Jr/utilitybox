import tensorflow as tf

def tensorboard_callback():
    log_dir = ''
    return tf.keras.callbacks.TensorBoard(logdir)

def checkpoint_callback():
    return tf.keras.callbacks.ModelCheckpoint('saved_model', verbose=1)

def early_stopping_callback():
    return tf.keras.callbacks.EarlyStopping(
            patience=3,
            min_delta=0.05,
            baseline=0.8,
            mode='min',
            monitor='val_loss',
            restore_best_weights=True,
            verbose=1
    )

def csv_logger_callback():
    file_path = ''
    return tf.keras.callbacks.CSVLogger(file_path)

def learning_rate_scheduler_callback():
    def step_decay(epoch):
	initial_lr = 0.01
	drop = 0.5
	epochs_drop = 1
	lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lr
    return tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)

def reduce_lr_on_plateau_callback():
    return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, verbose=1,
            patience=1, min_lr=0.001)
