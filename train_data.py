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
    mask = tf.squeeze(
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
    return mask, grad


def main(loss):

    ####
    #### Loading data and splitting into training and validation with same size
    ####

    x, x_up, x_down, y, w = pickle.load(open("train.pickle", "rb"))

    # Compute class weights for CE loss
    w_sumsig = np.sum(w[y == 1])
    w_sumbkg = np.sum(w[y == 0])
    w_class = np.zeros(w.shape)
    w_class[y == 1] = (w_sumsig + w_sumbkg) / w_sumsig
    w_class[y == 0] = (w_sumsig + w_sumbkg) / w_sumbkg
    

    ## Make training and validation datasets
    x_train, x_val, x_up_train, x_up_val, x_down_train, x_down_val, y_train, y_val, w_train, w_val, w_class_train, w_class_val = train_test_split(
        x, x_up, x_down, y, w, w_class, test_size=0.5, random_state=1234)
    

    ####
    #### Setup model
    ####

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(2,)),  # input shape required
    tf.keras.layers.Dense(1, activation=tf.sigmoid)
    ])

    batch_size = x_train.shape[0]
    batch_scale = tf.constant(2.0, tf.float32)
    batch_len = None

    # Training data
    x = tf.Variable(x_train, tf.float32, shape=[batch_len, 2])
    x_up = tf.Variable(x_up_train, tf.float32, shape=[batch_len, 2])
    x_down = tf.Variable(x_down_train, tf.float32, shape=[batch_len, 2])

    # Validation data
    x_val = tf.Variable(x_val, tf.float32, shape=[batch_len, 2])
    x_up_val = tf.Variable(x_up_val, tf.float32, shape=[batch_len, 2])
    x_down_val = tf.Variable(x_down_val, tf.float32, shape=[batch_len, 2])

    ## Class weights
    w = tf.placeholder(tf.float32, shape=[batch_len])
    w_class = tf.placeholder(tf.float32, shape=[batch_len])

    
    ####
    #### Define losses
    ####
    
    bins = np.linspace(0, 1, 9)
    bin_edges = bins
    right_edges = bin_edges[1:] # all except the first
    left_edges = bin_edges[:-1] # all except the last
    mask_algo = binfunction
    
    # assign value to tensor variables
    ### default: mu = 1, theta = 0
    mu = tf.Variable(1.0, trainable=True, dtype=tf.float32)
    theta = tf.Variable(0.0, trainable=True, dtype=tf.float32)

    # assign constant value to tensor
    epsilon = tf.constant(1e-9, tf.float32)
    null = tf.constant(0.0, tf.float32)
    one = tf.constant(1.0, tf.float32)


    ## Calculationg NLL
    def loss_nll(model, x, y, w, right_edges, left_edges, mu, theta, nuisance_is_true):
        nll0 = null

        for i, right_edge, left_edge in zip(range(len(left_edges)), right_edges, left_edges):
            print("Bin (Right edge, left edge, mid): {:g} / {:g} / {:g}".format(
                right_edge, left_edge, left_edge + 0.5 * (right_edge - left_edge)))
            right_edge_ = tf.constant(right_edge, tf.float32)
            left_edge_ = tf.constant(left_edge, tf.float32)

            
            ## Nominal
            mask = mask_algo(model(x), right_edge_, left_edge_)
            sig = tf.reduce_sum(mask * y * w * batch_scale)
            bkg = tf.reduce_sum(mask * (one - y) * w * batch_scale)


            ## Shifts
            mask_up = mask_algo(model(x_up), right_edge_, left_edge_)
            bkg_up = tf.reduce_sum(mask_up * (one - y) * w * batch_scale)

            mask_down = mask_algo(model(x_down), right_edge_, left_edge_)
            bkg_down = tf.reduce_sum(mask_down * (one - y) * w * batch_scale)


            if(nuisance_is_true):
                sys = tf.maximum(theta, null) * (bkg_up - bkg) + tf.minimum(theta, null) * (bkg - bkg_down)
                exp = mu * sig + bkg
                obs = sig + bkg
                nll0 -= tfp.distributions.Poisson(tf.maximum(exp + sys, epsilon)).log_prob(tf.maximum(obs, epsilon))
            else:
                sys = tf.constant(0.0, dtype=tf.float32)
                exp = mu * sig + bkg
                obs = sig + bkg
                nll0 -= tfp.distributions.Poisson(tf.maximum(exp + sys, epsilon)).log_prob(tf.maximum(obs, epsilon))
            
        if(nuisance_is_true):
            ## Normalized Gaussian constraining the nuisance
            nll0 -= tfp.distributions.Normal(loc=0, scale=1).log_prob(theta)
        return nll0


    ## Standard Deviation loss with and without nuisance
    def loss_sd(nll0, parameters, nuisance_is_true):
        with tf.GradientTape(persistent=True) as second_order:
            with tf.GradientTape() as first_order:
                if(nuisance_is_true):
                    loss_value_nll = nll0
                    gradnll = first_order.gradient(loss_value_nll, parameters)
                    hessian_rows = [second_order.gradient(g, parameters) for g in tf.unstack(gradnll)]
                    hessian_matrix = tf.stack(hessian_rows, axis=-1)
                    variance = tf.linalg.inv(hessian_matrix)
                    poi = variance[0][0]
                    standard_deviation = tf.math.sqrt(poi)
                else:
                    mu_sd_no_nuisance = parameters[0]
                    loss_value_nll = nll0
                    gradnll = first_order.gradient(loss_value_nll, mu_sd_no_nuisance)
                    gradgradnll = second_order.gradient(gradnll, mu_sd_no_nuisance)
                    covariance = 1 / gradgradnll
                    standard_deviation = tf.math.sqrt(covariance)
        return standard_deviation


    def grad_sd(nll0, parameters, nuisance_is_true):
        with tf.GradientTape() as backprop:
            loss_value = loss_sd(nll0, parameters, nuisance_is_true)
            backpropagation = backprop.gradient(loss_value, model.trainable_variables)
        return backpropagation


    ## Cross Entropy Loss
    def loss_ce(model, x, y):
        f = model(x)
        ce_loss = -tf.math.reduce_mean(
            tf.math.log(y * tf.maximum(f, epsilon)) + (one - y) * tf.math.log(tf.maximum((1 - f), epsilon))
        )
        return ce_loss


    def grad_ce(model, x, y):
        with tf.GradientTape() as backprop:
            loss_value = loss_ce(model, x, y)
            backpropagation = backprop.gradient(loss_value, model.trainable_variables)
        return backpropagation
    
    
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

    nll0 = loss_nll(model, x, y, w, right_edges, left_edges, mu, theta, nuisance_is_true)
    nll0_val = loss_nll(model, x_val, y_val, w_val, right_edges, left_edges, mu, theta, nuisance_is_true)

    
    ####
    #### Choose optimizer for training
    ####

    optimizer = tf.keras.optimizers.Adam()


    ## Summery of possible losses
    def model_loss_and_grads(loss):
        if(loss == "Cross Entropy Loss"):
            model_loss      = loss_ce(model, x, y)
            model_loss_val  = loss_ce(model, x_val, y_val)
            model_grads     = grad_ce(model, x, y)

        elif(loss == "Standard Deviation Loss"):
            model_loss      = loss_sd(nll0, [mu, theta], nuisance_is_true)
            model_loss_val  = loss_sd(nll0_val, [mu, theta], nuisance_is_true)
            model_grads     = grad_sd(nll0, [mu, theta], nuisance_is_true)

        elif(loss == "Standard Deviation Loss with nuisance"):
            model_loss      = loss_sd(nll0, [mu, theta], nuisance_is_true)
            model_loss_val  = loss_sd(nll0_val, [mu, theta], nuisance_is_true)
            model_grads     = grad_sd(nll0, [mu, theta], nuisance_is_true)

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
        for warmup_step in tqdm(range(0, 30)):
            ## Warmup trains model without nuisance to increase stability
            grads = grad_sd(nll0, [mu, theta], False)    # nuisance has to be FALSE here
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    

    ####
    #### Training loop
    ####

    ## prerequisites for training
    steps = []
    max_steps = 1000
    loss_train_list = []
    loss_validation_list = []
    max_patience = 30
    patience = max_patience

    ## initial loss:
    min_loss, _, _ = model_loss_and_grads(loss)

    ## Training loop
    for epoch in range(1, max_steps):
        current_loss, current_loss_val, grads = model_loss_and_grads(loss)

        ## apply grads and vars
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 

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


    s =[]
    b = []
    b_up = []
    b_down = []

    for i, right_edge, left_edge in zip(range(len(left_edges)), right_edges, left_edges):
        right_edge_ = tf.constant(right_edge, tf.float32)
        left_edge_ = tf.constant(left_edge, tf.float32)

        
        ## Nominal
        mask = mask_algo(model(x), right_edge_, left_edge_)
        s.append(tf.reduce_sum(mask * y * w * batch_scale))
        b.append(tf.reduce_sum(mask * (one - y) * w * batch_scale))


        ## Shifts
        mask_up = mask_algo(model(x_up), right_edge_, left_edge_)
        b_up.append(tf.reduce_sum(mask_up * (one - y) * w * batch_scale))

        mask_down = mask_algo(model(x_down), right_edge_, left_edge_)
        b_down.append(tf.reduce_sum(mask_down * (one - y) * w * batch_scale))

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
