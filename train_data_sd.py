import numpy as np
np.random.seed(1234)
import tensorflow as tf
#tf.random.set_seed(1234)   
import tensorflow_probability as tfp
import pickle
import matplotlib
import matplotlib.pyplot as plt


# custom gradient allows better numerical precision for functions with diverging or not defined derivative
@tf.custom_gradient
# binfunction selects all outputs from NN and distributes the accompanying events into the certain bin
# left and right edge represent the bin borders
def binfunction(x, right_edge, left_edge):
    # tf.cast casts a tensor to a new type -> here float32, since default is double
    # tf.squeeze removes dimensions of size 1 from the shape of a tensor
    y = tf.squeeze(
        tf.cast(
            tf.cast(x > left_edge, tf.float32) * tf.cast(x <= right_edge, tf.float32), tf.float32
            )
        )
    # for the derivative, the binfunction is approximated by a normal distribution
    def grad(dy):
        width = right_edge - left_edge
        mid = left_edge + 0.5 * width
        sigma = 0.5 * width # careful! necessary?
        gauss = tf.exp(-1.0 * (x - mid)**2 / 2.0 / sigma**2)    # careful!
        g = -1.0 * gauss * (x - mid) / sigma**2
        g = tf.squeeze(g) * tf.squeeze(dy)
        return (g, None, None)
    return y, grad


def main():

    ####
    #### Load data
    ####

    x_train_noshift_signal, x_train_upshift_signal, x_train_downshift_signal, x_train_noshift_background, x_train_upshift_background, x_train_downshift_background, y, w = pickle.load(open("train.pickle", "rb"))
    
    ####
    #### Setup model
    ####

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(2,)),  # input shape required
    tf.keras.layers.Dense(1, activation=tf.sigmoid)
    ])

    batch_len = None

    x_train_noshift = tf.Variable(np.vstack([x_train_noshift_signal, x_train_noshift_background]), tf.float32, shape=[batch_len, 2])
    x_train_upshift = tf.Variable(np.vstack([x_train_upshift_signal, x_train_upshift_background]), tf.float32, shape=[batch_len, 2])
    x_train_downshift = tf.Variable(np.vstack([x_train_downshift_signal, x_train_downshift_background]), tf.float32, shape=[batch_len, 2])

    x_signal_noshift = tf.Variable(x_train_noshift_signal, tf.float32, shape=[batch_len, 2])
    x_background_noshift = tf.Variable(x_train_noshift_background, tf.float32, shape=[batch_len, 2])
    x_background_upshift = tf.Variable(x_train_upshift_background, tf.float32, shape=[batch_len, 2])
    x_background_downshift = tf.Variable(x_train_downshift_background, tf.float32, shape=[batch_len, 2])

    
    ####
    #### Define loss
    ####
    
    bins = np.linspace(0.0, 1.0, 3)
    bin_edges = bins
    right_edges = bin_edges[1:] # all except the first
    left_edges = bin_edges[:-1] # all except the last
    mask_algo = binfunction
    batch_scale = tf.constant(2.0, tf.float32, shape=[])
    
    # assign value to tensor variables
    # default: mu = 1, theta = 0
    mu = tf.Variable(1.0, tf.float32, shape=[])
    theta = tf.Variable(0.0, tf.float32, shape=[])


    # assign value to tensor constants
    epsilon = tf.constant(1e-9, tf.float32)
    null = tf.constant(0.0, tf.float32)
    one = tf.constant(1.0, tf.float32)

    def nll(mu, theta, sig, bkg_noshift):
        value = -1 * tf.math.log(tf.maximum(
            tfp.distributions.Poisson(
                rate=tf.maximum(mu * sig + bkg_noshift, epsilon)
            ).prob(sig + bkg_noshift)\
        , epsilon)) 
        return value


    ####
    #### First define Loss, than define gradient, than minimize via grad()
    ####

    def hist(x, bins):
        counts = []
        # splits histogram in bins regarding their left and right edges
        # zip function puts left and right edge together in one iterable array
        for right_edge, left_edge in zip(bins[1:], bins[:-1]):
            # sums up all 1 entries of each bin 
            counts.append(tf.reduce_sum(binfunction(x, right_edge, left_edge)))
        return tf.squeeze(tf.stack(counts))


    def loss_nll(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_dwonshift, mu, theta):
        nll0 = 0
        f_signal_noshift = tf.squeeze(model(x_signal_noshift))
        f_background_noshift = tf.squeeze(model(x_background_noshift))
        f_background_upshift = tf.squeeze(model(x_background_upshift))
        f_background_downshift = tf.squeeze(model(x_background_downshift))

        #sig = hist((one - f_signal_noshift), bins)
        sig = hist(f_signal_noshift, bins)
        bkg = hist(f_background_noshift, bins)
        bkg_up = hist(f_background_upshift, bins)
        bkg_down = hist(f_background_downshift, bins)

        print("\nSIGNAL:          {:4.2f},     {:4.2f}".format(sig[0].numpy(), sig[1].numpy()))
        print("BACKGROUND:      {:4.2f},     {:4.2f}".format(bkg[0].numpy(), bkg[1].numpy()))
        print("BACKGROUND UP:   {:4.2f},     {:4.2f}".format(bkg_up[0].numpy(), bkg_up[1].numpy()))
        print("BACKGROUND DOWN: {:4.2f},     {:4.2f}\n".format(bkg_down[0].numpy(), bkg_down[1].numpy()))

        nll0 += nll(mu, theta, sig[1], bkg[1])
        nll0 += nll(mu, theta, sig[0], bkg[0])
        
        nll0 += -1 * tf.math.log(tf.maximum(tfp.distributions.Normal(loc=0, scale=1).prob(theta), epsilon))
        
        print("NLL: {}\n".format(nll0.numpy()))
        
        loss_value = nll0
        return loss_value

    def grad_sd(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, parameters):
        mu = parameters[0]
        theta = parameters[1]
        with tf.GradientTape() as backprop:
            with tf.GradientTape() as second_order:
                with tf.GradientTape() as first_order:
                    #loss_value = loss_nll(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, mu, theta)
                    loss_value = loss_nll(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, mu, theta)
                    #print("NLL:\n", loss_value.numpy())

                    gradnll = first_order.gradient(loss_value, mu)
                    #print("GRAD NLL:\n dMU: {},     dTHETA: {}".format(gradnll[0].numpy(), gradnll[1].numpy()))

                    gradgradnll = second_order.gradient(gradnll, mu)

                    poi = gradgradnll
                    standard_deviation = tf.math.sqrt(poi)
                    backpropagation = backprop.gradient(standard_deviation, model.trainable_variables)
        return standard_deviation, backpropagation

    optimizer = tf.keras.optimizers.Adam()


    ####
    #### Training loop
    ####

    steps = []
    loss_train = []
    max_patience = 100
    patience = max_patience
    
    # initial training step:
    loss_value, grads = grad_sd(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta])
    min_loss = loss_value
    print(loss_value.numpy())

    for epoch in range(1, 300):
        steps.append(epoch)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))    # apply grads and vars
        current_loss, _ = grad_sd(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta])
        loss_train.append(current_loss)
        if current_loss >= min_loss:
            patience -= 1
        else:
            min_loss = current_loss
            patience = max_patience
        
        #if epoch % 10 == 0 or patience == 0:
        print("Step: {:02d},         Loss: {:.4f}".format(epoch, current_loss))

        if patience == 0:
            print("Trigger early stopping in epoch {}.".format(epoch))
            break


    ####
    #### Plot NN function
    #### 

    length = 1000
    sensibility = 0.005
    b = np.linspace(-3, 3, length)
    xx, yy = np.meshgrid(b, b, sparse=False)
    x = np.vstack([xx.reshape((length * length)), yy.reshape((length * length))]).T
    c = tf.squeeze(model(x)).numpy()
    c = c.reshape((length, length))
    
    # visualize decision boundary
    boundary = tf.squeeze(model(x)).numpy()
    boundary = boundary.reshape((length, length))
    # iterate through c to highlight decision boundary
    for y in range(0, length):
        for x in range(0, length):
            if c[y][x] > 0.5-sensibility and c[y][x] < 0.5+sensibility:
                boundary[y][x] = 1
            else:
                boundary[y][x] = 0

    limit = [-3, 3]
    plt.figure(figsize=(7, 6))
    cbar = plt.contourf(xx, yy, c + boundary, levels=np.linspace(c.min(), c.max(), 21))
    plt.colorbar(cbar)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Neural network function")
    plt.tight_layout()
    plt.xlim(limit[0], limit[1])
    plt.ylim(limit[0], limit[1])
    #plt.savefig("/home/risto/Masterarbeit/Python/significance_plots/NN_function_{}.png".format(picture_index), bbox_inches = "tight")
    plt.show()
    

if __name__ == "__main__":
    main()
