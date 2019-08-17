import numpy as np
import tensorflow as tf


class lstm_layer1:
    # Create weights and initialize
    def __init__(self, inp_size, hidden_size, n_id):
        input_size = inp_size + hidden_size
        stdini = np.sqrt(1.0/input_size)
        self.n_id = n_id
        self.wf = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wf"+str(n_id))
        self.bf = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bf"+str(n_id))
        self.wi = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wi"+str(n_id))
        self.bi = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bi"+str(n_id))
        self.wc = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wc"+str(n_id))
        self.bc = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bc"+str(n_id))
        self.wo = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wo"+str(n_id))
        self.bo = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bo"+str(n_id))
        ws = [self.wf, self.wi, self.wc, self.wo]
        for w in ws:
            tf.add_to_collection('reg', w)
    # Call function
    def __call__(self, inp, h, c, dropout={'active': False}):
        inp2 = tf.concat([h, inp], axis=1, name="concat_"+str(self.n_id))
        if dropout['active']:
            inp2 = tf.nn.dropout(inp2, rate=dropout['rate'])
        ft = tf.nn.sigmoid(tf.matmul(inp2,self.wf) + self.bf, name="ft_"+str(self.n_id))
        it = tf.nn.sigmoid(tf.matmul(inp2,self.wi) + self.bi, name="it_"+str(self.n_id))
        Cbt = tf.nn.tanh(tf.matmul(inp2,self.wc) + self.bc, name="Cbt_"+str(self.n_id))
        ot = tf.nn.sigmoid(tf.matmul(inp2,self.wo) + self.bo, name="ot_"+str(self.n_id))
        c = ft * c + it * Cbt
        h = ot * tf.nn.tanh(c)
        return h, h, c


class lstm_layer2:
    # Create weights and initialize
    def __init__(self, inp_size, hidden_size, n_id):
        input_size = inp_size + hidden_size
        stdini = np.sqrt(1.0/input_size)
        stdini2 = np.sqrt(1.0/(input_size+hidden_size))
        self.n_id = n_id
        self.wf = tf.Variable(tf.random_normal([input_size+hidden_size, hidden_size], stddev=stdini2, dtype=tf.float32), name="wf"+str(n_id))
        self.bf = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bf"+str(n_id))
        self.wi = tf.Variable(tf.random_normal([input_size+hidden_size, hidden_size], stddev=stdini2, dtype=tf.float32), name="wi"+str(n_id))
        self.bi = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bi"+str(n_id))
        self.wc = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wc"+str(n_id))
        self.bc = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bc"+str(n_id))
        self.wo = tf.Variable(tf.random_normal([input_size+hidden_size, hidden_size], stddev=stdini2, dtype=tf.float32), name="wo"+str(n_id))
        self.bo = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bo"+str(n_id))
        ws = [self.wf, self.wi, self.wc, self.wo]
        for w in ws:
            tf.add_to_collection('reg', w)
    # Call function
    def __call__(self, inp, h, c, dropout={'active': False}):
        inp2 = tf.concat([h, inp], axis=1, name="concat_"+str(self.n_id))
        if dropout['active']:
            inp2 = tf.nn.dropout(inp2, rate=dropout['rate'])
        ft = tf.nn.sigmoid(tf.matmul(tf.concat([c,inp2],axis=1),self.wf) + self.bf, name="ft_"+str(self.n_id))
        it = tf.nn.sigmoid(tf.matmul(tf.concat([c,inp2],axis=1),self.wi) + self.bi, name="it_"+str(self.n_id))
        Cbt = tf.nn.tanh(tf.matmul(inp2,self.wc) + self.bc, name="Cbt_"+str(self.n_id))
        ot = tf.nn.sigmoid(tf.matmul(tf.concat([c,inp2],axis=1),self.wo) + self.bo, name="ot_"+str(self.n_id))
        c = ft * c + it * Cbt
        h = ot * tf.nn.tanh(c)
        return h, h, c


class lstm_layer3:
    # Create weights and initialize
    def __init__(self, inp_size, hidden_size, n_id):
        input_size = inp_size + hidden_size
        stdini = np.sqrt(1.0/input_size)
        self.n_id = n_id
        self.wf = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wf"+str(n_id))
        self.bf = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bf"+str(n_id))
        self.wc = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wc"+str(n_id))
        self.bc = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bc"+str(n_id))
        self.wo = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wo"+str(n_id))
        self.bo = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bo"+str(n_id))
        ws = [self.wf, self.wc, self.wo]
        for w in ws:
            tf.add_to_collection('reg', w)
    # Call function
    def __call__(self, inp, h, c, dropout={'active': False}):
        inp2 = tf.concat([h, inp], axis=1, name="concat_"+str(self.n_id))
        if dropout['active']:
            inp2 = tf.nn.dropout(inp2, rate=dropout['rate'])
        ft = tf.nn.sigmoid(tf.matmul(inp2,self.wf) + self.bf, name="ft_"+str(self.n_id))
        Cbt = tf.nn.tanh(tf.matmul(inp2,self.wc) + self.bc, name="Cbt_"+str(self.n_id))
        ot = tf.nn.sigmoid(tf.matmul(inp2,self.wo) + self.bo, name="ot_"+str(self.n_id))
        c = ft * c - ft * Cbt
        h = ot * tf.nn.tanh(c)
        return h, h, c


class lstm_layer4:
    # Create weights and initialize
    def __init__(self, inp_size, hidden_size, n_id):
        input_size = inp_size + hidden_size
        stdini = np.sqrt(1.0/input_size)
        self.n_id = n_id
        self.wr = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wr"+str(n_id))
        self.br = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="br"+str(n_id))
        self.wz = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wz"+str(n_id))
        self.bz = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bz"+str(n_id))
        self.wt = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=stdini, dtype=tf.float32), name="wt"+str(n_id))
        self.bt = tf.Variable(tf.zeros([hidden_size], dtype=tf.float32), name="bt"+str(n_id))
        ws = [self.wr, self.wz, self.wt]
        for w in ws:
            tf.add_to_collection('reg', w)
    # Call function
    def __call__(self, inp, h, c, dropout={'active': False}):
        inp2 = tf.concat([h, inp], axis=1, name="concat_"+str(self.n_id))
        if dropout['active']:
            inp2 = tf.nn.dropout(inp2, rate=dropout['rate'])
        rt = tf.nn.sigmoid(tf.matmul(inp2,self.wr) + self.br, name="rt_"+str(self.n_id))
        zt = tf.nn.sigmoid(tf.matmul(inp2,self.wz) + self.bz, name="zt_"+str(self.n_id))
        ht = tf.nn.tanh(tf.matmul(tf.concat([rt*h, inp], axis=1), self.wt) + self.bt, name="ht_"+str(self.n_id))
        h = -zt * h + zt * ht
        return h, h, c


class conv_layer:
    # Create filters and initialize
    def __init__(self, in_channels, out_channels, filter_height, filter_width, n_id):
        input_size = in_channels*filter_height*filter_width
        stdini = np.sqrt(1.0/input_size)
        self.n_id = n_id
        self.w = tf.Variable(tf.random_normal([filter_height, filter_width, in_channels, out_channels], stddev=stdini, dtype=tf.float32), name="wcnn"+str(n_id))
        self.b = tf.Variable(tf.zeros([1,1,out_channels], dtype=tf.float32), name="bcnn"+str(n_id))
        tf.add_to_collection('reg2', self.w)
    # Call function
    def __call__(self, inp, activation, std=[1,1,1,1], pad="SAME", dropout={'active': False}):
        if dropout['active']:
            inp = tf.nn.dropout(inp, rate=dropout['rate'])
        out = tf.nn.conv2d(inp, filter=self.w, strides=std, padding=pad, name="conv_"+str(self.n_id))
        if activation is not None:
            out = activation(out)
        out += self.b
        return out


class fc_layer:
    # Create weights and initialize
    def __init__(self, input_size, out_size, n_id):
        stdini = np.sqrt(1.0/input_size)
        self.n_id = n_id
        self.wfc = tf.Variable(tf.random_normal([input_size, out_size], stddev=stdini, dtype=tf.float32), name="wfc"+str(n_id))
        self.bfc = tf.Variable(tf.zeros([out_size], dtype=tf.float32), name="bfc"+str(n_id))
        tf.add_to_collection('reg', self.wfc)
    # Call function
    def __call__(self, inp, activation, dropout={'active': False}):
        if dropout['active']:
            inp = tf.nn.dropout(inp, rate=dropout['rate'])
        out = None
        if activation is not None:
            out = activation(tf.matmul(inp,self.wfc) + self.bfc, name="fc_"+str(self.n_id))
        else:
            out = tf.matmul(inp,self.wfc) + self.bfc
        return out

