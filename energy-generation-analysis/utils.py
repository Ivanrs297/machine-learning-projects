import matplotlib.pyplot as plt
import numpy as np

def split_sequence(data, n_steps):
    """Function to convert data into a sequence of sample X and  single step Y """
    X, y = list(), list()

    for i in range(len(data)):

        # get the end of pattern
        end_idx = i + n_steps

        # validate id
        if end_idx > len(data) - 1:
            break

        # input and output sample data
        seq_x, seq_y = data[i:end_idx], data[end_idx]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

def plot_history(history):

    plt.figure(figsize=(10, 10))

    # # summarize history for mean_squared_error
    # plt.subplot(121)
    # plt.plot(history.history['mean_squared_error'])
    # plt.plot(history.history['val_mean_squared_error'])
    # plt.title('model mean_squared_error')
    # plt.ylabel('mean_squared_error')
    # plt.xlabel('epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')

    # summarize history for loss
    # plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('results/Loss Training.png')

    plt.show()