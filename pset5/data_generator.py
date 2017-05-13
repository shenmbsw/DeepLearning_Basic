import numpy as np
import numpy.random as npr


def _build_rnn_testdata_matrix(n_hidden_dim=10, n_input_dim=10, n_steps=100):
    """
    returns: (X, y)
    
        x is (n_steps, n_input_dim)
        h is (n_steps, n_hidden_dim)
    """
    x = npr.randn(n_steps, n_input_dim)
    h = np.zeros((n_steps, n_hidden_dim))
    
    h_0 = npr.randn(n_hidden_dim)
    w = npr.randn(n_input_dim, n_hidden_dim)
    
    h_prev = h_0
    for i in range(n_steps):
        h[i] = np.maximum(0, 1 - (h_prev + x[i].dot(w)))
        h_prev = h[i]
    return (h_0, w), x, h


def _build_lstm_testdata_matrix(n_hidden_dim=10, n_input_dim=10, n_steps=100):
    """
    returns: (X, y)
    
        x is (n_steps, n_input_dim)
        h is (n_steps, n_hidden_dim)
    """
    
    def _sigmoid(_x):
        return 1/(1+np.exp(-_x))
    
    x = npr.randn(n_steps, n_input_dim)
    h = np.zeros((n_steps, n_hidden_dim))
    
    h_0 = npr.randn(n_hidden_dim)
    c_0 = npr.randn(n_hidden_dim)
    w_i = npr.randn(n_input_dim, n_hidden_dim)
    w_c = npr.randn(n_input_dim, n_hidden_dim)
    w_f = npr.randn(n_input_dim, n_hidden_dim)
    w_o = npr.randn(n_input_dim, n_hidden_dim)
    u_i = npr.randn(n_hidden_dim, n_hidden_dim)
    u_c = npr.randn(n_hidden_dim, n_hidden_dim)
    u_f = npr.randn(n_hidden_dim, n_hidden_dim)
    u_o = npr.randn(n_hidden_dim, n_hidden_dim)
    v_o = npr.randn(n_hidden_dim, n_hidden_dim)
    weights = (h_0, c_0, w_i, w_c, w_f, w_o, u_i, u_c, u_f, u_o, v_o)
    
    h_prev = h_0
    c_prev = c_0
    for t in range(n_steps):
        i_t = _sigmoid(np.dot(x[t], w_i) + np.dot(u_i, h_prev))
        c_bar_t = np.tanh(np.dot(x[t], w_c) + np.dot(u_c, h_prev))
        f_t = _sigmoid(np.dot(x[t], w_f) + np.dot(u_f, h_prev))
        c_t = i_t*c_bar_t + f_t*c_prev
        o_t = _sigmoid(np.dot(x[t], w_o) + np.dot(u_o, h_prev) + np.dot(v_o, c_t))
        h_t = o_t * np.tanh(c_t)
    
        h[t] = h_t
        h_prev = h_t
        c_prev = c_t
    return weights, x, h


def build_dataset(dataset_name, n_hidden_dim=10, n_input_dim=15, n_batch_size=20,
                  n_steps_per_batch=100, n_batches=30, noise=0.0):
    """
    [.. ,(X_i, y_i), ..] - n_batches
    
        X_i : (n_batch_size, n_steps, n_input_dim)
        y_i : (n_batch_size, n_hidden_dim)
    """
    dataset_functions = {
        'lstm': _build_lstm_testdata_matrix,
        'rnn': _build_rnn_testdata_matrix
    }
    function = dataset_functions[dataset_name]
    weights, x, h = function(n_hidden_dim, n_input_dim,
                             n_batch_size*n_steps_per_batch*n_batches)
    # x of shape (n_batches*n_batch_size*n_steps_per_batch, n_input_dim)
    # h of shape (n_batches*n_batch_size*n_steps_per_batch, n_hidden_dim)
    x += npr.random(x.shape) * noise
    
    data = []
    for batch_n in range(n_batches-1):
        batch_tensor_X_rows = []
        batch_tensor_y_rows = []
        for line_n in range(n_batch_size):
            from_idx = batch_n*n_batch_size*n_steps_per_batch + line_n*n_steps_per_batch
            to_idx = batch_n*n_batch_size*n_steps_per_batch + (line_n+1)*n_steps_per_batch
            batch_x = x[from_idx:to_idx]
            batch_y = h[to_idx-1]
            batch_tensor_X_rows.append(batch_x)
            batch_tensor_y_rows.append(batch_y)
        
        data.append([np.array(batch_tensor_X_rows), np.array(batch_tensor_y_rows)])
        
    return weights, data


def _dataset_specs():
    import matplotlib.pyplot as plt
    (h_0, w), x, h = _build_rnn_testdata_matrix()
    for v, name in zip([h_0, w, x], ['h0', 'w', 'x']):
        print(name, v.shape, np.min(v), np.max(v))
    norm_x_t = np.sum(x*2, axis=1)
    plt.plot(np.arange(norm_x_t.shape[0]), norm_x_t)
    plt.show()
    
    weights, x, h = _build_lstm_testdata_matrix()
    for v in (x,) + weights:
        print(v.shape, np.min(v), np.max(v))
    norm_x_t = np.sum(x*2, axis=1)
    plt.plot(np.arange(norm_x_t.shape[0]), norm_x_t)
    plt.show()
    
if __name__ == '__main__':
    _dataset_specs()
