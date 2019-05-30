import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow.python.ops import control_flow_ops

raw_df = pd.read_csv("C:/Users/a0988/Desktop/Meeting/stocks_data/2409.TW.csv")
p = raw_df['Close'].values  # np.ndarray
n = len(p)
z = np.array([p[i+1] - p[i] for i in range(n-1)])  # (4499,)

m = 50  # no of price ticks[inputs]  50筆算一段
ts = n - m  # time steps
c = 0  # transaction cost 交易成本
batch_size = 5
numClusters = 3


def create_batch_generator(X, batch_size=20):
    X_copy = np.array(X)
    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i:i+batch_size, :], X_copy[i+1:i+1+batch_size, -1])  # 生成一段batch和下一個時間點價差


tf.reset_default_graph()

g1 = tf.Graph()
with g1.as_default():
    tf.set_random_seed(123)
    features = tf.placeholder(tf.float32, shape=[None], name='features')  # 原始close data的價差(4499,)
    l = tf.placeholder(tf.float32, shape=[None], name='l')  # 存slice好的data (4450,50)
    with tf.variable_scope("data_pre"):
        def slice_data(features, l):  # 資料切片
            temp_list = []
            for i in range(1, ts + 1):
                f = tf.slice(features, [i - 1], [m])  # 把features切片,從[i-1,0]開始, 切出[m,1]
                temp_list.append(f)
            l = tf.stack(temp_list)
            return l
        slice_l = slice_data(features, l)  # 轉換好(4450,50)的差值時間差資料

        init = tf.global_variables_initializer()
with tf.Session(graph=g1) as sess:  # 執行session
    sess.run(init)
    slice_z = sess.run(slice_l, feed_dict={features: z, l: []})

g = tf.Graph()
with g.as_default():  # 設計graph
    tf.set_random_seed(123)
    # placeholder
    batch__X = tf.placeholder(tf.float32, shape=[None, 50], name='batch__X')
    next__z = tf.placeholder(tf.float32, shape=[None], name='next__z')

    with tf.variable_scope("k-means"):
        total_data = np.asarray(a=slice_z, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS  # 起始中心選擇方式
        compactness, label_kmeans, centers = cv2.kmeans(
            data=total_data, K=3, bestLabels=None, criteria=criteria, attempts=10, flags=flags)
        List0 = []
        List1 = []
        List2 = []
        for ii in range(n-m):  # 把資料分到三類去
            if 0 == label_kmeans[ii][0]:
                List0.append(total_data[ii])
            if 1 == label_kmeans[ii][0]:
                List1.append(total_data[ii])
            if 2 == label_kmeans[ii][0]:
                List2.append(total_data[ii])
        mean0, variance0 = np.mean(List0, axis=0), np.var(List0, axis=0)
        mean1, variance1 = np.mean(List1, axis=0), np.var(List1, axis=0)
        mean2, variance2 = np.mean(List2, axis=0), np.var(List2, axis=0)

    with tf.variable_scope("fuzzy-layer"):
        # 取exp(取負值(標準化)) paper (7)式
        fuzzy0 = tf.exp(tf.negative(tf.nn.batch_normalization(x=batch__X, mean=mean0,
                                                              variance=variance0, offset=None, scale=None,
                                                              variance_epsilon=0.001)))
        fuzzy1 = tf.exp(tf.negative(tf.nn.batch_normalization(x=batch__X, mean=mean1,
                                                              variance=variance1, offset=None, scale=None,
                                                              variance_epsilon=0.001)))
        fuzzy2 = tf.exp(tf.negative(tf.nn.batch_normalization(x=batch__X, mean=mean2,
                                                              variance=variance2, offset=None, scale=None,
                                                              variance_epsilon=0.001)))
        fuzzyOut = tf.concat(values=[fuzzy0, fuzzy1, fuzzy2], axis=0, name="FuzzyOut")  # 連接[fuzzy0,fuzzy1,fuzzy2]
        fuzzyOut = tf.reshape(tensor=fuzzyOut, shape=[batch_size, 150])


    with tf.variable_scope("AutoEncoder"):
        h1 = tf.layers.dense(inputs=fuzzyOut, units=100, activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(inputs=h1, units=60, activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(inputs=h2, units=40, activation=tf.nn.leaky_relu)
        h4 = tf.layers.dense(inputs=h3, units=30, activation=tf.nn.leaky_relu)
        AE_out = tf.layers.dense(inputs=h4, units=10)

    with tf.variable_scope("DRL"):  # 深度循環類神經 計算delta (動作) paper (3)式
        rnn_In = tf.reshape(tensor=AE_out, shape=[1, batch_size, 10], name="reshape1")
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=1, activation=tf.tanh)  # 創建最小單位(output)
        initial_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)  # 初始化狀態
        delta, final_state = tf.nn.dynamic_rnn(rnn_cell, rnn_In, initial_state=initial_state,
                                               dtype=tf.float32, time_major=False)

        delta = tf.reshape(tensor=delta, shape=[batch_size])
        d = delta

        delta = tf.map_fn(
            lambda x: tf.case(
                pred_fn_pairs=[
                    (tf.greater(x, 0.33), lambda: tf.math.tanh(x*100)),
                    (tf.less(x, -0.33), lambda: tf.math.tanh(x*100))],
                default=lambda: tf.math.tanh(x/100)), delta)

        delta = tf.reshape(tensor=delta, shape=[batch_size, 1])

    with tf.variable_scope("UT_loss"):  # 計算總利潤 paper(1)式

        def cal_UT(delta, next__z):
            r = []
            for i in range(1, delta.shape[0]):
                Rt = delta[i - 1] * next__z[i - 1] - c * tf.abs(delta[i] - delta[i - 1])
                r.append(Rt)
            UT = sum(r)
            return UT

        loss = (-1) * cal_UT(delta, next__z)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss=loss)
    init = tf.global_variables_initializer()
with tf.Session(graph=g) as sess:  # 執行session
    sess.run(init)
    each_epoch_total_reward = []
    epoch_times = []
    for i in range(20):  # epochs
        epoch_total_reward = []
        batch_idx = []
        count = 0
        current_total_loss = 0
        batch_generator = create_batch_generator(slice_z, batch_size=batch_size)
        print("Epochs : ", i + 1)
        for batch_X, next_z in batch_generator:
            if batch_X.shape == (batch_size, 50):
                count += 1
                _, total_loss, d_out, delta_out, next_one = sess.run([train_op, loss, d, delta, next__z], feed_dict={batch__X: batch_X, next__z: next_z})
                print(d_out)
                print(delta_out)
                print(next_one)
                print(total_loss)

                current_total_loss += total_loss[0]
                print("Batch Times : ", count, "Current_total_reward :", (-1) * current_total_loss)
                epoch_total_reward.append((-1)*current_total_loss)
                batch_idx.append(count)

        each_epoch_total_reward.append((-1) * current_total_loss)
        epoch_times.append(i)
        print("\n==================================================\n")

        plt.title("2409's trading total_reward from FRDNN ")
        plt.xlabel('batch_idx')
        plt.ylabel('total_reward')
        plt.plot(batch_idx, epoch_total_reward)
    plt.grid()
    plt.show()

    plt.title("2409's epoch times with final reward ")
    plt.xlabel('epoch_idx')
    plt.ylabel('total_reward')
    plt.plot(epoch_times, each_epoch_total_reward)
    plt.grid()
    plt.show()
