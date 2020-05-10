import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1234)   
import tensorflow_probability as tfp
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    #### Loading data and splitting into training and validation with same size
    ####

    x_noshift_signal, x_upshift_signal, x_downshift_signal, x_noshift_background, x_upshift_background, x_downshift_background, y, w = pickle.load(open("train.pickle", "rb"))
    
    x_train_noshift_signal, x_val_noshift_signal, x_train_upshift_signal, x_val_upshift_signal, x_train_downshift_signal, x_val_downshift_signal,\
    x_train_noshift_background, x_val_noshift_background, x_train_upshift_background, x_val_upshift_background, x_train_downshift_background, x_val_downshift_background = train_test_split(
    x_noshift_signal, x_upshift_signal, x_downshift_signal, x_noshift_background, x_upshift_background, x_downshift_background, test_size=0.5, random_state=1234)
    

    ####
    #### Setup model
    ####

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(2,)),  # input shape required
    tf.keras.layers.Dense(1, activation=tf.sigmoid)
    ])

    batch_len = None

    # Training data
    x_signal_noshift = tf.Variable(x_train_noshift_signal, tf.float32, shape=[batch_len, 2])
    x_signal_upshift = tf.Variable(x_train_upshift_signal, tf.float32, shape=[batch_len, 2])
    x_signal_downshift = tf.Variable(x_train_downshift_signal, tf.float32, shape=[batch_len, 2])
    x_background_noshift = tf.Variable(x_train_noshift_background, tf.float32, shape=[batch_len, 2])
    x_background_upshift = tf.Variable(x_train_upshift_background, tf.float32, shape=[batch_len, 2])
    x_background_downshift = tf.Variable(x_train_downshift_background, tf.float32, shape=[batch_len, 2])

    # Validation data
    x_signal_noshift_val = tf.Variable(x_val_noshift_signal, tf.float32, shape=[batch_len, 2])
    x_signal_upshift_val = tf.Variable(x_val_upshift_signal, tf.float32, shape=[batch_len, 2])
    x_signal_downshift_val = tf.Variable(x_val_downshift_signal, tf.float32, shape=[batch_len, 2])
    x_background_noshift_val = tf.Variable(x_val_noshift_background, tf.float32, shape=[batch_len, 2])
    x_background_upshift_val = tf.Variable(x_val_upshift_background, tf.float32, shape=[batch_len, 2])
    x_background_downshift_val = tf.Variable(x_val_downshift_background, tf.float32, shape=[batch_len, 2])

    
    ####
    #### Define loss
    ####
    
    bins = np.linspace(0.0, 1.0, 3)
    bin_edges = bins
    right_edges = bin_edges[1:] # all except the first
    left_edges = bin_edges[:-1] # all except the last
    mask_algo = binfunction
    batch_scale = tf.constant(2.0, tf.float32)
    
    # assign value to tensor variables
    ### default: mu = 1, theta = 0
    mu = tf.Variable(1.0, trainable=True, dtype=tf.float32)
    theta = tf.Variable(0.0, trainable=True, dtype=tf.float32)

    # assign constant value to tensor
    epsilon = tf.constant(1e-9, tf.float32)
    null = tf.constant(0.0, tf.float32)
    one = tf.constant(1.0, tf.float32)


    ####
    #### First define Loss, than define gradient, than minimize via grad(loss, model.variables)
    ####

    def hist(x, bins):
        counts = []
        # splits histogram in bins regarding their left and right edges
        # zip function puts left and right edge together in one iterable array
        for right_edge, left_edge in zip(bins[1:], bins[:-1]):
            # sums up all 1 entries of each bin 
            counts.append(tf.reduce_sum(binfunction(x, right_edge, left_edge)))
        return tf.squeeze(tf.stack(counts))


    def loss_nll(parameters):
        nll0 = null

        mu_nll = parameters[0]
        theta_nll = parameters[1]

        ## NN functions for each origin
        f_signal_noshift = tf.squeeze(model(x_signal_noshift))
        f_background_noshift = tf.squeeze(model(x_background_noshift))
        f_background_upshift = tf.squeeze(model(x_background_upshift))
        f_background_downshift = tf.squeeze(model(x_background_downshift))

        ## Histograms of events separated by decision boundary
        sig = hist(f_signal_noshift, bins)
        bkg = hist(f_background_noshift, bins)
        bkg_up = hist(f_background_upshift, bins)
        bkg_down = hist(f_background_downshift, bins)

        print("\nSIGNAL:          {:4.2f},     {:4.2f}".format(sig[0].numpy(), sig[1].numpy()))
        print("BACKGROUND:      {:4.2f},     {:4.2f}".format(bkg[0].numpy(), bkg[1].numpy()))
        print("BACKGROUND UP:   {:4.2f},     {:4.2f}".format(bkg_up[0].numpy(), bkg_up[1].numpy()))
        print("BACKGROUND DOWN: {:4.2f},     {:4.2f}\n".format(bkg_down[0].numpy(), bkg_down[1].numpy()))

        ## Calculate NLL with nuisance
        for i in range(0, 2):
            exp = mu_nll * sig[i] * bkg[i]
            sys = tf.maximum(theta_nll, null) * (bkg_up[i] - bkg[i]) \
            + tf.minimum(theta_nll, null) * (bkg[i] - bkg_down[i])
            obs = sig[i] + bkg[i]

            nll0 -= tfp.distributions.Poisson(tf.maximum(exp + sys, epsilon)).log_prob(tf.maximum(obs, epsilon))
        nll0 -= tfp.distributions.Normal(loc=0, scale=1).log_prob(theta_nll)
        loss_value = nll0
        return loss_value * 0.01


    def grad_sd(parameters):
        mu_sd = parameters[0]
        theta_sd = parameters[1]
        with tf.GradientTape() as backprop:
            with tf.GradientTape(persistent=True) as second_order:
                with tf.GradientTape() as first_order:
                    loss_value_nll = loss_nll([mu_sd, theta_sd])
                    print("NLL:\n", loss_value_nll.numpy())

                    gradnll = first_order.gradient(loss_value_nll, [mu_sd, theta_sd])
                    print("GRAD NLL:\n dMU: {},     dTHETA: {}".format(gradnll[0].numpy(), gradnll[1].numpy()))

                    hessian_rows = [second_order.gradient(g, [mu_sd, theta_sd]) for g in tf.unstack(gradnll)]
                    #print("HESSIAN ROWS:\n", hessian_rows)
                    
                    hessian_matrix = tf.stack(hessian_rows, axis=1)
                    #print("HESSE MATRIX:\n", hesse.numpy())

                    variance = tf.linalg.inv(hessian_matrix)
                    #print("VARIANZ:\n", variance.numpy())

                    poi = variance[0][0]
                    standard_deviation = tf.math.sqrt(poi)
                    backpropagation = backprop.gradient(loss_value_nll, model.trainable_variables)
        return standard_deviation, backpropagation
    

    def grad_nll(parameters):
        mu_grad_nll = parameters[0]
        theta_grad_nll = parameters[1]
        with tf.GradientTape() as grad:
            loss_value = loss_nll([mu_grad_nll, theta_grad_nll])
            gradnll = grad.gradient(loss_value, model.trainable_variables)
        return loss_value, gradnll

    ## choose optimizer for training
    optimizer = tf.keras.optimizers.Adam()


    ####
    #### Training loop
    ####

    steps = []
    loss_train_list = []
    loss_validation_list = []
    max_patience = 10
    patience = max_patience
    
    ### initial training step:
    #initial_loss, grads = grad_nll(model, x_signal_noshift, x_signal_upshift, x_signal_downshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta])
    initial_loss, grads = grad_sd([mu, theta])
    min_loss = initial_loss
    print(initial_loss.numpy())

    for epoch in range(1, 17):
        steps.append(epoch)

        ## apply gradient step
        optimizer.apply_gradients(zip(grads, model.trainable_variables))    # apply grads and vars
        
        ## save current loss of training and validation
        #current_loss, _ = grad_nll(model, x_signal_noshift, x_signal_upshift, x_signal_downshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta])
        current_loss, _ = grad_sd([mu, theta])
        loss_train_list.append(current_loss)

        #current_loss_val, _ = grad_nll(model, x_signal_noshift_val, x_signal_upshift_val, x_signal_downshift_val, x_background_noshift_val, x_background_upshift_val, x_background_downshift_val, [mu, theta])
        current_loss_val, _ = grad_sd([mu, theta])
        loss_validation_list.append(current_loss_val)

        if current_loss_val >= min_loss:
            patience -= 1
        else:
            min_loss = current_loss_val
            patience = max_patience
        
        if epoch % 10 == 0 or patience == 0:
            print("Step: {:02d},         Loss: {:.4f}".format(epoch, current_loss_val))

        if patience == 0:
            print("Trigger early stopping in epoch {}.".format(epoch))
            break


    ####
    #### Plot histogram displaying significance
    ####

    f_signal_noshift = tf.squeeze(model(x_signal_noshift))
    f_background_noshift = tf.squeeze(model(x_background_noshift))
    f_background_upshift = tf.squeeze(model(x_background_upshift))
    f_background_downshift = tf.squeeze(model(x_background_downshift))

    #sig = hist((one - f_signal_noshift), bins)
    sig = hist(f_signal_noshift, bins)
    bkg = hist(f_background_noshift, bins)
    bkg_up = hist(f_background_upshift, bins)
    bkg_down = hist(f_background_downshift, bins)


    s = sig
    b = bkg
    n = 2500
    bins_for_plots = [0.0, 0.5, 1.0]            # 
    bins_for_plots_middle = []                  # Central Point of Bin 
    for i in range(0, len(bins_for_plots) - 1):
        bins_for_plots_middle.append(bins_for_plots[i] + (bins_for_plots[i + 1] - bins_for_plots[i]) / 2)
    border = bins_for_plots[1]

    opt_sig_significance = s[1] / np.sqrt(s[1] + b[1])
    opt_bkg_significance = b[0] / np.sqrt(s[0] + b[0])

    plt.figure(figsize=(7, 6))
    plt.hist(bins_for_plots_middle, weights= [s[0], s[1]], bins= bins_for_plots, histtype="step", label="Signal", lw=2)
    plt.hist(bins_for_plots_middle, weights= [b[0], b[1]], bins= bins_for_plots, histtype="step", label="Backgorund", lw=2)
    plt.legend(loc= "lower center")
    plt.title("Background Significance: {:.2f},   Signal Significance: {:.2f}".format(opt_bkg_significance, opt_sig_significance))
    plt.xlabel("Projection with decision boundary from NN at {}".format(border))
    plt.ylabel("# Events")
    plt.axvline(x = border, ymin= 0, ymax= max(n, n), color="r", linestyle= "dashed", lw=2)
    #plt.show()

    ####
    #### Plot loss wrt gradient step
    ####

    plt.figure()
    plt.plot(steps, loss_train_list)
    plt.plot(steps, loss_validation_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

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
