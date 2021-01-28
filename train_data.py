import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1234)   
import tensorflow_probability as tfp
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm




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


def main(loss):

    ####
    #### Loading data and splitting into training and validation with same size
    ####

    x_noshift_signal, x_upshift_signal, x_downshift_signal,\
    x_noshift_background, x_upshift_background, x_downshift_background,\
    y, w = pickle.load(open("train.pickle", "rb"))
    
    x_train_noshift_signal, x_val_noshift_signal,\
    x_train_upshift_signal, x_val_upshift_signal,\
    x_train_downshift_signal, x_val_downshift_signal,\
    x_train_noshift_background, x_val_noshift_background,\
    x_train_upshift_background, x_val_upshift_background,\
    x_train_downshift_background, x_val_downshift_background = train_test_split(
        x_noshift_signal, x_upshift_signal, x_downshift_signal,\
        x_noshift_background, x_upshift_background, x_downshift_background,\
        test_size=0.5, random_state=1234
    )
    

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
    x_background_noshift = tf.Variable(x_train_noshift_background, tf.float32, shape=[batch_len, 2])
    x_background_upshift = tf.Variable(x_train_upshift_background, tf.float32, shape=[batch_len, 2])
    x_background_downshift = tf.Variable(x_train_downshift_background, tf.float32, shape=[batch_len, 2])

    # Validation data
    x_signal_noshift_val = tf.Variable(x_val_noshift_signal, tf.float32, shape=[batch_len, 2])
    x_background_noshift_val = tf.Variable(x_val_noshift_background, tf.float32, shape=[batch_len, 2])
    x_background_upshift_val = tf.Variable(x_val_upshift_background, tf.float32, shape=[batch_len, 2])
    x_background_downshift_val = tf.Variable(x_val_downshift_background, tf.float32, shape=[batch_len, 2])

    
    ####
    #### Define losses
    ####
    
    bins = np.linspace(0.0, 1.0, 9)
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
    #### First define Loss, then define gradient, then minimize via grad(loss, model.variables)
    ####

    def hist(x, bins):
        counts = []
        # splits histogram in bins regarding their left and right edges
        # zip function puts left and right edge together in one iterable array
        for right_edge, left_edge in zip(bins[1:], bins[:-1]):
            # sums up all 1 entries of each bin 
            counts.append(tf.reduce_sum(binfunction(x, right_edge, left_edge)))
        return tf.squeeze(tf.stack(counts))


    ## Negative Log Likelihood loss with nuisance
    def loss_nll(model, x_sig, x_bkg, x_bkg_up, x_bkg_down, parameters, nuisance_is_true):
        nll0 = null
        # parameters = [mu, theta]

        ## Histograms of events separated by decision boundary
        sig = hist(tf.squeeze(model(x_sig)), bins) * (50. / 25000.)
        bkg = hist(tf.squeeze(model(x_bkg)), bins) * (1000. / 25000.)
        bkg_up = hist(tf.squeeze(model(x_bkg_up)), bins) * (1000. / 25000.)
        bkg_down = hist(tf.squeeze(model(x_bkg_down)), bins) * (1000. / 25000.)

        ## Calculate NLL with or without nuisance
        for i in range(0, len(sig)):
            if(nuisance_is_true):
                sys = tf.maximum(parameters[1], null) * (bkg_up[i] - bkg[i]) \
                + tf.minimum(parameters[1], null) * (bkg[i] - bkg_down[i])
            else:
                sys = tf.constant(0.0, dtype=tf.float32)
            exp = parameters[0] * sig[i] + bkg[i]
            obs = sig[i] + bkg[i]

            nll0 -= tfp.distributions.Poisson(tf.maximum(exp + sys, epsilon)).log_prob(tf.maximum(obs, epsilon))
        
        if(nuisance_is_true):
            ## Normalized Gaussian constraining the nuisance
            nll0 -= tfp.distributions.Normal(loc=0, scale=1).log_prob(parameters[1])
        return nll0


    ## Standard Deviation loss with and without nuisance
    def loss_sd(model, x_sig, x_bkg, x_bkg_up, x_bkg_down, parameters, nuisance_is_true):
        with tf.GradientTape(persistent=True) as second_order:
            with tf.GradientTape() as first_order:
                if(nuisance_is_true):
                    loss_value_nll = loss_nll(model, x_sig, x_bkg, x_bkg_up, x_bkg_down, parameters, nuisance_is_true)
                    gradnll = first_order.gradient(loss_value_nll, parameters)
                    hessian_rows = [second_order.gradient(g, parameters) for g in tf.unstack(gradnll)]
                    hessian_matrix = tf.stack(hessian_rows, axis=-1)
                    variance = tf.linalg.inv(hessian_matrix)
                    poi = variance[0][0]
                    standard_deviation = tf.math.sqrt(poi)
                else:
                    mu_sd_no_nuisance = parameters[0]
                    loss_value_nll = loss_nll(model, x_sig, x_bkg, x_bkg_up, x_bkg_down, parameters, nuisance_is_true)
                    gradnll = first_order.gradient(loss_value_nll, mu_sd_no_nuisance)
                    gradgradnll = second_order.gradient(gradnll, mu_sd_no_nuisance)
                    covariance = 1 / gradgradnll
                    standard_deviation = tf.math.sqrt(covariance)
        return standard_deviation


    def grad_sd(model, x_sig, x_bkg, x_bkg_up, x_bkg_down, parameters, nuisance_is_true):
        with tf.GradientTape() as backprop:
            loss_value = loss_sd(model, x_sig, x_bkg, x_bkg_up, x_bkg_down, parameters, nuisance_is_true)
            backpropagation = backprop.gradient(loss_value, model.trainable_variables)
        return backpropagation


    ## Cross Entropy Loss
    def loss_ce(model, x_sig, x_bkg):
        f_sig = model(x_sig)
        f_bkg = model(x_bkg)
        ce_loss = -tf.math.reduce_mean(
            tf.math.log(tf.maximum(f_sig, 1e-9)) + tf.math.log(tf.maximum((1 - f_bkg), 1e-9))
        )
        return ce_loss


    def grad_ce(model, x_sig, x_bkg):
        with tf.GradientTape() as backprop:
            loss_value = loss_ce(model, x_sig, x_bkg)
            backpropagation = backprop.gradient(loss_value, model.trainable_variables)
        return backpropagation

    
    ####
    #### Choose optimizer for training
    ####

    optimizer = tf.keras.optimizers.Adam()

    
    ####
    #### Pretraining decisions
    ####
    
    warmup_is_true = True
    nuisance_is_true = True

    if(loss == "Cross Entropy Loss"):
        warmup_is_true = False
    elif(loss == "Standard Deviation Loss"):
        warmup_is_true = False
        nuisance_is_true = False
    else:
        pass


    ## Summery of possible losses
    def model_loss_and_grads(loss):
        if(loss == "Cross Entropy Loss"):
            model_loss      = loss_ce(model, x_signal_noshift, x_background_noshift)
            model_loss_val  = loss_ce(model, x_signal_noshift_val, x_background_noshift_val)
            model_grads     = grad_ce(model, x_signal_noshift, x_background_noshift)

        elif(loss == "Standard Deviation Loss"):
            model_loss      = loss_sd(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta], nuisance_is_true)
            model_loss_val  = loss_sd(model, x_signal_noshift_val, x_background_noshift_val, x_background_upshift_val, x_background_downshift_val, [mu, theta], nuisance_is_true)
            model_grads     = grad_sd(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta], nuisance_is_true)

        elif(loss == "Standard Deviation Loss with nuisance"):
            model_loss      = loss_sd(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta], nuisance_is_true)
            model_loss_val  = loss_sd(model, x_signal_noshift_val, x_background_noshift_val, x_background_upshift_val, x_background_downshift_val, [mu, theta], nuisance_is_true)
            model_grads     = grad_sd(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta], nuisance_is_true)

        return model_loss, model_loss_val, model_grads

    print("\nMethod: {}\n".format(loss))

    ####
    #### WARMUP 
    ####
    
    if(warmup_is_true):
        print("\n\
    ######################\n\
    # Warmup Initialized #\n\
    ######################\n")
        for warmup_step in tqdm(range(0, 70)):
            ## Warmup trains model without nuisance to increase stability
            grads = grad_sd(model, x_signal_noshift, x_background_noshift, x_background_upshift, x_background_downshift, [mu, theta], False)    # nuisance has to be FALSE here
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    

    ####
    #### Training loop
    ####

    ## prerequisites for training
    steps = []
    loss_train_list = []
    loss_validation_list = []
    max_patience = 15
    patience = max_patience

    ## initial loss:
    min_loss, _, _ = model_loss_and_grads(loss)

    ## Training loop
    for epoch in range(1, 80):
        current_loss, current_loss_val, grads = model_loss_and_grads(loss)

        ## apply grads and vars
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        print("\n\nTHETA: {}\nMU: {}".format(theta.numpy(), mu.numpy()))

        ## monitor loss
        steps.append(epoch)
        loss_train_list.append(current_loss)
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
    #### Save Model and histogram
    ####

    model.save('./mymodel')

    s = hist(tf.squeeze(model(x_signal_noshift)), bins) * (50. / 25000.)
    b = hist(tf.squeeze(model(x_background_noshift)), bins) * (1000. / 25000.)
    b_up = hist(tf.squeeze(model(x_background_upshift)), bins) * (1000. / 25000.)
    b_down = hist(tf.squeeze(model(x_background_downshift)), bins) * (1000. / 25000.)

    print("SIGNAL:           {}, SUM:   {}".format(s, np.sum(s)))
    print("BACKGROUND:       {}, SUM:   {}".format(b, np.sum(b)))
    print("BACKGROUND UP:    {}, SUM:   {}".format(b_up, np.sum(b_up)))
    print("BACKGROUND DOWN:  {}, SUM:   {}".format(b_down, np.sum(b_down)))

    pickle.dump([s, b, b_up, b_down, bins], open("plot_histogram.pickle", "wb"))


    ####
    #### Plot loss wrt gradient step
    ####

    plt.figure()
    plt.plot(steps, loss_train_list)
    plt.plot(steps, loss_validation_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("./plots/loss_opt_steps_{}".format(plot_label), bbox_inches = "tight")
    

if __name__ == "__main__":
    ## load plot_label to choose loss method according to label
    plot_label = pickle.load(open("plot_label.pickle", "rb"))

    if("CE" in plot_label):
        loss_choice = "Cross Entropy Loss"
    elif("SD_no" in plot_label):
        loss_choice = "Standard Deviation Loss"
    elif("SD_with" in plot_label):
        loss_choice = "Standard Deviation Loss with nuisance"
    
    main(loss_choice)
