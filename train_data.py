import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1234)   
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

    x_train_signal_tot, x_train_noshift_background, x_train_upshift_background, x_train_downshift_background = pickle.load(open("train.pickle", "rb"))
    
    ####
    #### Setup model
    ####

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(2,)),  # input shape required
    tf.keras.layers.Dense(1, activation=tf.sigmoid)
    ])

    batch_len = None
    x_signal_noshift = tf.Variable(x_train_signal_tot, tf.float32, shape=[batch_len, 2])
    x_background_noshift = tf.Variable(x_train_noshift_background, tf.float32, shape=[batch_len, 2])
    x_background_upshift = tf.Variable(x_train_upshift_background, tf.float32, shape=[batch_len, 2])
    x_background_downshift = tf.Variable(x_train_downshift_background, tf.float32, shape=[batch_len, 2])


    ####
    #### Define loss
    ####
    
    bins = np.linspace(-0.01, 1.01, 3)
    bin_edges = bins
    right_edges = bin_edges[1:] # all except the first
    left_edges = bin_edges[:-1] # all except the last
    mask_algo = binfunction
    batch_scale = tf.Variable(2.0, tf.float32, shape=[])
    
    # assign value to tensor variables
    # default: mu = 1, theta = 0
    mu = tf.Variable(1.0, tf.float32, shape=[])
    theta = tf.Variable(0.0, tf.float32, shape=[])


    # assign value to tensor constants
    epsilon = tf.Variable(1e-9, tf.float32)
    null = tf.Variable(0.0, tf.float32)
    one = tf.Variable(1.0, tf.float32)

    def nll(mu, theta, sig, bkg_noshift, bkg_upshift, bkg_downshift):
        return -1 * tf.math.log(tf.maximum(
            tfp.distributions.Poisson(
                rate=tf.maximum(
                    mu * sig + bkg_noshift \
                    + tf.maximum(theta, null) * (bkg_upshift - bkg_noshift) \
                    + tf.minimum(theta, null) * (bkg_noshift - bkg_downshift)\
                    , epsilon
                    )).prob(sig + bkg_noshift), epsilon)) # mittlere erwartung, die optimiert werden soll


    ####
    #### First define Loss, than define gradient, than minimize via grad()
    ####

    def loss_nll(model, x_sig, x_bkg, x_bkg_up, x_bkg_down, mu, theta):
        nll0 = 0    # initial NLL value
        for i, right, left in zip(range(len(right_edges)), right_edges, left_edges):
            f_signal_noshift = tf.squeeze(model(x_sig))
            f_background_noshift = tf.squeeze(model(x_bkg))
            f_background_upshift = tf.squeeze(model(x_bkg_up))
            f_background_downshift = tf.squeeze(model(x_bkg_down))

            # declare unshifted models as histograms
            sig = tf.reduce_sum(mask_algo(f_signal_noshift, right, left))
            bkg = tf.reduce_sum(mask_algo(f_background_noshift, right, left))

            # declare shifted models as histograms
            bkg_upshift = tf.reduce_sum(mask_algo(f_background_upshift, right, left))
            bkg_downshift = tf.reduce_sum(mask_algo(f_background_downshift, right, left))

            # NLL
            nll0 += nll(mu, theta, sig, bkg, bkg_upshift, bkg_downshift)

        # Normalized gaussioan constraining the nuisance (see paper) 
        nll0 += -1 * tf.math.log(tf.maximum(tfp.distributions.Normal(loc=0, scale=1).prob(theta), epsilon))
        loss_value = nll0
        return loss_value

    def grad_nll(model, x_sig, x_bkg, x_bkg_up, x_bkg_down, parameters):
        mu = parameters[0]
        theta = parameters[1]
        with tf.GradientTape() as backprop:
            with tf.GradientTape(persistent=True) as second_order:
                with tf.GradientTape() as first_order:
                    loss_value = loss_nll(model, x_sig, x_bkg, x_bkg_up, x_bkg_down, mu, theta)
                    gradnll = first_order.gradient(loss_value, parameters)
                    hessian_rows = [second_order.gradient(g, parameters) for g in gradnll]
                    hesse = tf.stack(hessian_rows, axis=1)
                    variance = tf.linalg.inv(hesse)
                    poi = variance[0][0]
                    standard_deviation = tf.math.sqrt(poi)
                    backpropagation = backprop.gradient(standard_deviation, model.trainable_variables)
        return standard_deviation, backpropagation

    optimizer = tf.keras.optimizers.Adam()


    ####
    #### Training loop
    ####

    steps = []
    loss_train = []
    max_patience = 34
    patience = max_patience
    
    # initial training step:
    loss_value, grads = grad_nll(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta])
    min_loss = loss_value

    for epoch in range(0, 30):
        #current_loss, grads = grad_nll(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta])

        steps.append(epoch)
        print(epoch)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))    # apply grads and vars
        #current_loss = loss_nll(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift).numpy()
        current_loss, _ = grad_nll(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta])

        loss_train.append(current_loss)
        if current_loss >= min_loss:
            patience -= 1
        else:
            min_loss = current_loss
            patience = max_patience
        
        #if epoch % 10 == 0 or patience == 0:
        print("Step: {:02d},         Loss: {:.2f}".format(optimizer.iterations.numpy(), current_loss))
            
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
