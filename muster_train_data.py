import sys
import numpy as np
np.random.seed(1230)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(1230)
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
from data import variables

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


"""
variables = [
    "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
    "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt",
    "PRI_tau_eta", "PRI_tau_phi", "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi",
    "PRI_met", "PRI_met_phi", "PRI_met_sumet", "PRI_jet_num",
    "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt", "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi", "PRI_jet_all_pt"
]
"""


def load(filename):
    x = pandas.read_csv(filename, usecols=variables).values
    y = pandas.read_csv(filename, usecols=["Label"]).values.squeeze()
    y[y == 's'] = 1.0
    y[y == 'b'] = 0.0
    y = np.array(y, dtype="float")
    w = pandas.read_csv(filename, usecols=["Weight"]).values.squeeze()
    w_up = pandas.read_csv(filename, usecols=["WeightUp"]).values.squeeze()
    w_down = pandas.read_csv(filename, usecols=["WeightDown"]).values.squeeze()
    return x, y, w, w_up, w_down


@tf.custom_gradient
def count_fwd_gauss_bkw_mask(x, up, down):
    mask = tf.cast(
            tf.cast(x > down, tf.float32) * tf.cast(x <= up, tf.float32),
            tf.float32)
    mask = tf.squeeze(mask)

    def grad(dy):
        width = up - down
        mid = down + 0.5 * width
        sigma = 0.5 * width
        gauss = tf.exp(-1.0 * (x - mid)**2 / 2.0 / sigma**2)
        g = -1.0 * gauss * (x - mid) / sigma**2
        g = tf.squeeze(g) * tf.squeeze(dy)
        return (g, None, None)

    return mask, grad


def model(x, reuse=False):
    with tf.variable_scope("model", reuse=reuse) as scope:
        hidden_nodes = 100
        w1 = tf.get_variable('w1', shape=(len(variables), hidden_nodes), dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b1 = tf.get_variable('b1', shape=(hidden_nodes), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))
        w2 = tf.get_variable('w2', shape=(hidden_nodes, 1), dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b2 = tf.get_variable('b2', shape=(1), dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))

    l1 = tf.nn.relu(tf.add(b1, tf.matmul(x, w1)))
    logits = tf.add(b2, tf.matmul(l1, w2))
    f = tf.sigmoid(logits)
    f = tf.squeeze(f)
    return f, (w1, b1, w2, b2)


def main(use_adversary):
    # Load data
    x, y, w, w_up, w_down = load("train.csv")

    from sklearn.preprocessing import StandardScaler
    preprocessing_input = StandardScaler()
    x = preprocessing_input.fit_transform(x)

    pickle.dump(preprocessing_input, open('higgs_preprocessing.pickle', 'wb'))

    w_sumsig = np.sum(w[y == 1])
    w_sumbkg = np.sum(w[y == 0])
    w_class = np.zeros(w.shape)
    w_class[y == 1] = (w_sumsig + w_sumbkg) / w_sumsig
    w_class[y == 0] = (w_sumsig + w_sumbkg) / w_sumbkg

    # Make training and validation datasets
    x_train, x_val, y_train, y_val, w_nom_train, w_nom_val, w_up_train, w_up_val, w_down_train, w_down_val, w_class_train, w_class_val = train_test_split(
        x, y, w, w_up, w_down, w_class, test_size=0.5, random_state=1234)

    # Setup model
    # NOTE: We reuse the names x_nom, x_up, ... and so on
    batch_scale = tf.placeholder(tf.float32, shape=[])
    batch_len = None
    x = tf.placeholder(tf.float32, shape=[batch_len, len(variables)])
    w_up = tf.placeholder(tf.float32, shape=[batch_len])
    w_down = tf.placeholder(tf.float32, shape=[batch_len])
    w_nom = tf.placeholder(tf.float32, shape=[batch_len])
    w_class = tf.placeholder(tf.float32, shape=[batch_len])
    f, w_vars = model(x)
    y = tf.placeholder(tf.float32, shape=[batch_len])

    # Define loss
    bins = np.linspace(0, 1, 9)
    pickle.dump(bins, open("bins.pickle", "wb"))
    upper_edges = bins[1:]
    lower_edges = bins[:-1]
    mask_algo = count_fwd_gauss_bkw_mask

    theta0 = tf.constant(0.0, tf.float32)
    mu0 = tf.constant(1.0, tf.float32)

    one = tf.constant(1, tf.float32)
    zero = tf.constant(0, tf.float32)
    epsilon = tf.constant(1e-9, tf.float32)

    nll0 = zero
    nll0_statsonly = zero
    for i, up, down in zip(range(len(upper_edges)), upper_edges, lower_edges):
        # Bin edges
        print("Bin (up, down, mid): {:g} / {:g} / {:g}".format(
            up, down, down + 0.5 * (up - down)))
        up_ = tf.constant(up, tf.float32)
        down_ = tf.constant(down, tf.float32)

        # Nominal
        mask = mask_algo(f, up_, down_)
        sig_nom = tf.reduce_sum(mask * y * w_nom * batch_scale)
        bkg_nom = tf.reduce_sum(mask * (one - y) * w_nom * batch_scale)

        # Shifts
        sig_up = tf.reduce_sum(mask * y * w_nom * w_up * batch_scale)
        sig_down = tf.reduce_sum(mask * y * w_nom * w_down * batch_scale)
        bkg_up = tf.reduce_sum(mask * (one - y) * w_nom * w_up * batch_scale)
        bkg_down = tf.reduce_sum(mask * (one - y) * w_nom * w_down * batch_scale)

        # Likelihood
        exp = mu0 * sig_nom + bkg_nom
        sys = tf.maximum(theta0, zero) * (sig_up - sig_nom + bkg_up - bkg_nom) + tf.minimum(theta0, zero) * (sig_nom - sig_down + bkg_nom - bkg_down)
        obs = sig_nom + bkg_nom
        nll0 -= tfp.distributions.Poisson(tf.maximum(exp + sys, epsilon)).log_prob(tf.maximum(obs, epsilon))
        nll0_statsonly -= tfp.distributions.Poisson(tf.maximum(exp, epsilon)).log_prob(tf.maximum(obs, epsilon))

    # Nuisance constraint
    nll0 -= tfp.distributions.Normal(loc=0, scale=1).log_prob(theta0)

    # POI constraint (full NLL)
    def get_constraint(nll, params):
        hessian = [tf.gradients(g, params) for g in tf.unstack(tf.gradients(nll, params))]
        inverse = tf.matrix_inverse(hessian)
        covariance_poi = inverse[0][0]
        constraint = tf.sqrt(covariance_poi)
        return constraint

    constraint = get_constraint(nll0, [mu0, theta0])

    # POI constraint (stats only)
    constraint_statsonly = get_constraint(nll0_statsonly, [mu0])

    # CE loss
    ce = -tf.reduce_mean((y * tf.log(tf.maximum(f, epsilon)) + (one - y) * tf.log(tf.maximum(one - f, epsilon))) * w_nom * w_class)

    # Train
    def minimize_op(loss, variables):
        optimizer = tf.train.AdamOptimizer()
        #optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9)
        return optimizer.minimize(loss, var_list=w_vars)
        #gvs = optimizer.compute_gradients(loss, var_list=variables)
        #capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
        #return optimizer.apply_gradients(capped_gvs)

    minimize_constraint = minimize_op(constraint, w_vars)
    minimize_constraint_statsonly = minimize_op(constraint_statsonly, w_vars)
    minimize_ce = minimize_op(ce, w_vars)

    loss = constraint
    minimize_loss = minimize_constraint
    if "statsonly" in mode:
        loss = constraint_statsonly
        minimize_loss = minimize_constraint_statsonly
    elif "ceonly" in mode:
        loss = ce
        minimize_loss = minimize_ce

    steps = []
    loss_train = []
    loss_val = []
    constraint_train = []
    constraint_val = []
    save_steps = 10
    if mode == "nll":
        warmup_steps = 0
    else:
        warmup_steps = 0
    max_steps = 100000
    patience = 100
    min_better = 0.999
    best_loss_val = 99999
    patience_counter = patience
    save_always = False
    saver = tf.train.Saver(max_to_keep=1)

    sess.run(
        tf.global_variables_initializer()
        )

    feed_dict_train = {x: x_train,
                       y: y_train,
                       w_nom: w_nom_train,
                       w_up: w_up_train,
                       w_down: w_down_train,
                       w_class: w_class_train,
                       batch_scale: 2.0,
                       }

    feed_dict_val   = {x: x_val,
                       y: y_val,
                       w_nom: w_nom_val,
                       w_up: w_up_val,
                       w_down: w_down_val,
                       w_class: w_class_val,
                       batch_scale: 2.0,
                       }

    for i_step in range(max_steps + 1):
        is_verbose = np.mod(i_step, save_steps) == 0
        is_warmup = i_step < warmup_steps
        if is_verbose:
            print(">>> Step / patience: {} / {}".format(i_step, patience_counter))

        # Optimize NN
        try:
            if is_warmup:
                loss_train_, _ = sess.run([constraint_statsonly, minimize_constraint_statsonly], feed_dict_train)
            else:
                constraint_train_, loss_train_, _ = sess.run([constraint, loss, minimize_loss], feed_dict_train)
        except:
            print(">>> [ERROR] Failed to optimize parameters.")
            break

        if is_verbose:
            if is_warmup:
                print("+++ Warm-up loss: {:g}".format(loss_train_))

        # Validate
        if is_verbose and not is_warmup:
            # Compute validation loss
            try:
                constraint_val_, loss_val_ = sess.run([constraint, loss], feed_dict_val)
            except:
                print(">>> [ERROR] Failed to calculate validation metrics.")
                break

            loss_val.append(loss_val_)
            constraint_val.append(constraint_val_)
            steps.append(i_step)

            # Save and print
            tag = ""
            if loss_val[-1] < best_loss_val * min_better or save_always:
                best_loss_val = loss_val[-1]
                path = saver.save(sess, "model.ckpt", global_step=i_step)
                tag = " -> Save to " + path
                patience_counter = patience
            else:
                patience_counter += -1

            loss_train.append(loss_train_)
            constraint_train.append(constraint_train_)
            print("+++ Loss (train / val): {:g} / {:g} {:}".format(loss_train[-1], loss_val[-1], tag))
            print("+++ Constraint (train / val): {:g} / {:g}".format(constraint_train[-1], constraint_val[-1]))

            if patience_counter == 0:
                print("+++ Early stopping triggered.")
                break


    # Plot loss
    plt.figure(figsize=(6, 6))
    plt.plot(steps, loss_train, lw=3, color="C0", label="Training")
    plt.plot(steps, loss_val, lw=3, color="C1", label="Validation")
    plt.xlabel("Gradient step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png", bbox_inches="tight")

    plt.figure(figsize=(6, 6))
    plt.plot(steps, constraint_train, lw=3, color="C0", label="Training")
    plt.plot(steps, constraint_val, lw=3, color="C1", label="Validation")
    plt.xlabel("Gradient step")
    plt.ylabel("constraint")
    plt.legend()
    plt.savefig("constraint.png", bbox_inches="tight")


if __name__ == "__main__":
    mode = "nll"
    if len(sys.argv) == 2:
        if "ceonly" in sys.argv[1]:
            mode = "ceonly"
        elif "statsonly" in sys.argv[1]:
            mode = "statsonly"
    main(mode)