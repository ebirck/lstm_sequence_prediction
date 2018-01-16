import lstm
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    epochs = 3

    X_a, Y_a = lstm.load_data('sensor_a.txt', 1)
    X_b, Y_b = lstm.load_data('sensor_b.txt', 1)
    X_c, Y_c = lstm.load_data('sensor_c.txt', 2)
    X_d, Y_d = lstm.load_data('sensor_d.txt', 2)
    X_b_orig, Y_b_orig = lstm.load_data('sensor_b.txt', 3)
    Sensor_a_x_orig = lstm.load_data('sensor_b.txt', 4)

    X_test = np.array([X_a, X_b, X_c, X_d])
    Y_test = np.array([Y_a, Y_b, Y_c, Y_d])

    X_test = np.swapaxes(X_test, 0, 1)
    X_test = np.swapaxes(X_test, 1, 2)
    Y_test = np.swapaxes(Y_test, 0, 1)

    print(X_test.shape)
    print(Y_test.shape)

    # model_1 = lstm.build_model()

    model_1 = load_model('demo_model.h5')   #pre-compiled model, comment this and uncomment surrounding lines for new model

    # model_1.fit(
    #     X_test,
    #     Y_test,
    #     nb_epoch=epochs)

    # print("Saving model!")
    # model_1.save('demo_model.h5')

    prediction = model_1.predict(X_test)

    temp_predicted_sensor_b = (prediction[:, 0] + 1) * X_b_orig[:, 0]

    sensor_b_y = (Y_test[:, 0] + 1) * X_b_orig[:, 0]

    plot_results(temp_predicted_sensor_b, sensor_b_y)
    plot_results(temp_predicted_sensor_b, X_b_orig[:, 29])
