TextCNN-Multilabel-Classification
===============
# 语言
Python3.6
# 依赖库
tensorflow=1.8.0

numpy=1.14.3

pandas=0.23.0

scikit-learn=0.19.1
# 项目介绍
## 目的
利用深度神经网络对四万多条的具有多标签（共有60个不同标签）的留言进行训练，实现对新留言的多标签分类
## 基于卷积神经网络的深度学习
本项目采用了TextCNN作为深度学习的cnn模型，包括word_embedding layer，convolutional layer,
max_pooling layer,全连接层。textcnn的优点在于把局部词序信息也关联了起来，即局部相关性
## 实现思想
首先先把原始数据分词，去掉停用词，每句话用一个list存储，处理好的词作为字符串作为这个list的元素，然后
所有的句子再组成一个list。每个句子对应的多标签转化为二值化标签，例如[0,1,0,1,0],列数代表标签总量，
'1'代表命中的标签，然后再打乱、分离数据集，最后使用batch生成器，把固定大小的batch输入到TextCNN中
训练，设定到一定的step评估、保存一次模型
# 用法介绍
## 文本预处理和batch生成器函数：data_helper.py
- data_solve():对原始数据进行处理
- batch_iter():batch生成器
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer#用来把多标签二值化
def data_solve(data_file):
    data = pd.read_csv(data_file)#读数据，读取文件类型是csv
    data_keyword = []#用来存每个句子的list
    data_label_temp = []#用来存label的list
    for (num,x) in enumerate(data['keyword']):#一行一行地读取数据
        temp = x
        data_keyword.append(temp)
    for (num,y) in enumerate(data['label']):
        temp = y.strip().split(' ')
        data_label_temp.append(temp)
    #把label标签化
    label = MultiLabelBinarizer()#每个多标签二值化后会返回一个list
    data_label = label.fit_transform(data_label_temp)
    return data_keyword,data_label
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    "num_epoch是指所有样本一共要训练的次数，例如：100个样本，batch_size是10，则10个batch才需练完，这就是一个num_epoch"
    data = np.array(data)#注意，要把输入的list转化为ndarray的格式
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1#每一个epoch所包含的batch个数
    for epoch in range(num_epochs):#一共要走的epoch
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):#每个epoch中生成每个batch去训练
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)#担心最后一个batch不够batch_size的大小
            yield shuffled_data[start_index:end_index]#返回batche，生成器
```
## TextCNN模型函数：textCNN.py
- 多类分类和二分类：损失函数用tf.nn.softmax_cross_entropy_with_logits()，因为softmax()适用于每个类别相互独立且排斥的情况。
评估函数使用textcnn中提供的cnn.accuracy

- 多标签分类：损失函数用tf.nn.sigmoid_cross_entropy_with_logits()，sigmoid()适用于每个类别相互独立但互不排斥的情况,评估函数
使用function_.py来评估
```python
import tensorflow as tf
class TextCNN(object):#定义一个textcnn的类
    '''embedding层，卷积层，池化层，softmax层'''
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        '''      sequence_size = 句子的长度
                 num_classes = 输出层的类别
                 vocab_size = 词汇大小
                 embedding_size = 嵌入层的维度
                 filter_sizes = 卷积滤波器覆盖的字数
                 num_filter = 每个滤波器尺寸对应的滤波器数量
        '''
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        # 定义一个operation，名称input_x,利用参数sequence_length，None表示样本数不定，
        # 这是一个placeholder
        # 数据类型int32，（样本数*句子长度）的tensor，每个元素为一个单词
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        # 这个placeholder的数据输入类型为float，（样本数*类别）的tensor
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # placeholder表示图的一个操作或者节点，用来喂数据，进行name命名方便可视化

        l2_loss = tf.constant(0.0)
        # l2正则的初始化

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 封装了一个叫做“embedding'的模块，使用设备cpu，模块里3个operation
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # operation1，一个（词典长度*embedding_size）tensor，作为W，也就是最后的词向量，建立一个查找表
            #每个字、词都转化成了词ID，每个词都会对应一个词向量
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # operation2，input_x的tensor维度为[none，seq_len],那么这个操作的输出为none*seq_len*em_size
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # 增加一个维度，变成，batch_size*seq_len*em_size*channel(=1)的4维tensor，符合图像的习惯

        pooled_outputs = []
        for (i, filter_size) in enumerate(filter_sizes):#比如（0，3），（1，4），（2，5）
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 循环第一次，建立一个名称为如”conv-ma-3“的模块
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # operation1，没名称，卷积核参数，高*宽*通道*卷积个数
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # operation2，名称”W“，变量维度filter_shape的tensor
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # operation3，名称"b",变量维度卷积核个数的tensor
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],#样本，height，width，channel移动距离
                    padding="VALID",
                    name="conv")
                # operation4，卷积操作，名称”conv“，与w系数相乘得到一个矩阵
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # operation5，加上偏置，进行relu，名称"relu"
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                # 每个卷积核和pool处理一个样本后得到一个值
                # 三种卷积核，append3次

        num_filters_total = num_filters * len(filter_sizes)
        # operation，每种卷积核个数与卷积核种类的积
        self.h_pool = tf.concat(pooled_outputs, 3)
        # operation，将outputs在第4个维度上拼接，如本来是128*1*1*64的结果3个，拼接后为128*1*1*192的tensor
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # operation，结果reshape为128*192的tensor

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)#一定概率是神经元失活，防止过拟合
        # 添加一个"dropout"的模块，里面一个操作，输出为dropout过后的128*192的tensor

        with tf.name_scope("output"):#添加一个”output“的模块，多个operation
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            # operation1，系数tensor，如192*2，192个features分2类，名称为"W"，注意这里用的是get_variables
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # operation2,偏置tensor，如2，名称"b"
            l2_loss += tf.nn.l2_loss(W)
            # operation3，loss上加入w的l2正则
            l2_loss += tf.nn.l2_loss(b)
            # operation4，loss上加入b的l2正则
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # operation5，scores计算全连接后的输出，如[0.2,0.7]名称”scores“
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            '''
            以下三个为计算多标签预测值，predictions为一个矩阵，矩阵每行的元素为self.scores每行最大的三个值的索引，即有3列
            
            scores_ = function_.turn_tensor_to_array(self.scores)
            max_index = function_.return_matrix_max_indice(scores_,3)
            predictions = function_.to_tensor(max_index)
'''
            # operations，计算预测值，输出最大值的索引，0或者1，名称”predictions“1

        with tf.name_scope("loss"):#定义一个”loss“的模块
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # operation1，定义losses，交叉熵，如果是一个batch，那么是一个长度为batchsize1的tensor？
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            # operation2，计算一个batch的平均交叉熵，加上全连接层参数的正则
            
        with tf.name_scope("accuracy"):#定义一个名称”accuracy“的模块
            '''
            以下三个函数即把input_y也返回为元素为1的索引值，共有三个索引          
            temp1 = function_.turn_tensor_to_array(self.input_y)
            temp2 = function_.return_matrix_max_indice(temp1,3)
            input_y_change = function_.to_tensor(temp2)
            correct_predictions = tf.equal(predictions,input_y_change)'''
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # operation1，根据input_y和predictions是否相同，得到一个矩阵batchsize大小的tensor
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"),name="accuracy")
            # operation2，计算均值即为准确率，名称”accuracy“
```
## 训练模型函数：train.py
- 导入库
```python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from textCNN import TextCNN
from tensorflow.contrib import learn
import function_
```
- 参数定义
```python
#处理好的数据文件路径和分离数据集的百分比
tf.flags.DEFINE_float("dev_sample_percentage", 0.3, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "./data/data_del_symbol_stopword.csv", "Data source for the positive data.")
# textCNN模型的参数
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '2,3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob",0.7, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularization lambda (default: 0.0)")
# 训练batch的参数设置
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# tensorflow会话的配置设置
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
```
- 对导入数据的一些处理
```python
def preprocess():
    #把data_helper.py中处理好的数据导入
    x_text, y = data_helper.data_solve(FLAGS.data_file)
    #构建字典。把词转化成ID
    max_document_length = max([len(x.split(" ")) for x in x_text])#数据中文本最大的长度
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))#转化成ndarray的数组格式
    #打乱数据集
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    #分离数据集
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x, y, x_shuffled, y_shuffled#清理内存
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev
```
- 开始训练模型
```python
def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    with tf.Graph().as_default():#使用这个配置的图
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)#会话中的配置
        sess = tf.Session(config=session_conf)
        with sess.as_default():#使用这个配置的会话
            cnn = TextCNN(
                sequence_length=x_train.shape[1],#取训练集这个np数组的列数
                num_classes=y_train.shape[1],#取二值化label的列数，即多少个label
                vocab_size=len(vocab_processor.vocabulary_),#字典的词ID个数，模型的embedding层会初始化对应个词向量
                embedding_size=FLAGS.embedding_dim,#词的维度，自己定义，一般256或128
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),#卷积核的大小，一次卷积包含多少个词的个数
                num_filters=FLAGS.num_filters,#卷积核的个数
                l2_reg_lambda=FLAGS.l2_reg_lambda)#L2正则
            #定义训练的程序
            global_step = tf.Variable(0, name="global_step", trainable=False)#步数
            optimizer = tf.train.AdamOptimizer(1e-3)#优化器
            grads_and_vars = optimizer.compute_gradients(cnn.loss)#梯度和变量？
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)#保存模型，max_to_keep是指保留最新的多少个模型

            vocab_processor.save(os.path.join(out_dir, "vocab"))#保存字典
            sess.run(tf.global_variables_initializer())  #初始化全部变量
            def train_step(x_batch, y_batch):
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }#把数据喂进图中
                _, step, summaries, loss, accuracy,scores= sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,cnn.scores],
                    feed_dict)#开始运行，并得到sess.run()里面参数的结果
                #用来测试准确率
                predictions_label = function_.return_matrix_max_indice(np.array(scores), 3)#得到scores中每一行最大的三个值的索引并作为list返回
                mark_label = function_.return_matrix_max_indice(np.array(y_batch), 3)#把label每一行最大的三个值的索引并作为list返回，
                                                                                     # 虽然label本来就是1或0，并只有三个1，走个形式
                predictions_label_length = np.array(predictions_label).shape[0]#一个多少句话
                acc_sum =[]#因为每个batch有很多句话，所以用来存每句话的准确值，然后再求平均
                for i in range(predictions_label_length):
                "测准确率思想是：求出索引值相同的个数，然后用相同个数除以原本label命中的个数，最后一个batch所有的准确率加起来，求平局得出该batch的准确率"
                    set1 = set(predictions_label[i])
                    set2 = set(mark_label[i])
                    the_same = set1 & set2#求并集
                    the_same_num = len(the_same)#集和的个数
                    the_same_acc = the_same_num / 3
                    acc_sum.append(the_same_acc)
                gentle = sum(acc_sum) / predictions_label_length#平均准确率
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                print("真正的准确率为：",gentle)
                
                # 定义了一个函数，用于验证集，输入为一个batch，方法和train_step()的思想一样
            def dev_step(x_batch, y_batch,writer = None):
                # 验证集太大，会爆内存，采用batch的思想进行计算，下面生成多个子验证集
                num = 20
                x_batch = x_batch.tolist()
                y_batch = y_batch.tolist()
                l = len(y_batch)
                l_20 = int(l / num)
                x_set = []
                y_set = []
                for i in range(num - 1):
                    x_temp = x_batch[i * l_20:(i + 1) * l_20]
                    x_set.append(x_temp)
                    y_temp = y_batch[i * l_20:(i + 1) * l_20]
                    y_set.append(y_temp)
                x_temp = x_batch[(num - 1) * l_20:]
                x_set.append(x_temp)
                y_temp = y_batch[(num - 1) * l_20:]
                y_set.append(y_temp)
                # 每个batch验证集计算一下准确率，num个batch再平均
                lis_loss = []
                lis_accu = []#假的验证准确率，留着只是懒得改TextCNN模型和summary的函数了
                gentle_accu = []#真正的验证准确率
                for i in range(num):
                    step, summaries,loss, accuracy,scores = sess.run(
                        [global_step, dev_summary_op,cnn.loss, cnn.accuracy,cnn.scores],
                        feed_dict={
                            cnn.input_x: np.array(x_set[i]),
                            cnn.input_y: np.array(y_set[i]),
                            cnn.dropout_keep_prob: 0.7}
                    )
                    lis_loss.append(loss)
                    lis_accu.append(accuracy)
                   #用来测试准确率
                    predictions_dev_label = function_.return_matrix_max_indice(np.array(scores), 3)
                    mark_dev_label = function_.return_matrix_max_indice(np.array(y_batch), 3)
                    predictions_dev_label_length = np.array(predictions_dev_label).shape[0]
                    acc_dev_sum = []
                    for i in range(predictions_dev_label_length):
                        set1 = set(predictions_dev_label[i])
                        set2 = set(mark_dev_label[i])
                        the_same = set1 & set2
                        the_same_num = len(the_same)
                        the_same_acc = the_same_num / 3
                        acc_dev_sum.append(the_same_acc)
                    gentle = sum(acc_dev_sum) / predictions_dev_label_length
                    gentle_accu.append(gentle)#准确率
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    print("该个验证机准确率为：",gentle)
                print("test_loss and test_acc" + "\t\t" + str(sum(lis_loss) / num) + "\t\t" + str(sum(lis_accu) / num))
                print("验证集准确率为：",sum(gentle_accu) / num)

            #batch生成
            batches = data_helper.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # 训练batch的循环
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)#当前步数
                if current_step % FLAGS.evaluate_every == 0:#每隔evaluate_every步就评估
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:#每隔checkpoint_every步就保存模型
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
```
## 测准确率函数：function_.py
- 主要用在了train.py的train_step()和dev_step()中，返回索引
- return_max_list()和return_matrix_max_number()配合使用
- return_max_index()和return_matrix_max_indice()配合使用
- turn_tensor_to_array()转成ndarray格式，用于好处理tensor里的数值
- to_tensor()转成tensor类型
```python
import tensorflow as tf
import numpy as np

def return_max_list(list_word,number):
    '''
    输入一个list，该list的最大前number个值的位置置1，其他位置置0，返回该list
     example:>>l = [1,4,6,7,875,2,-3,5]
            >>y = return_max_list(l,4)
            >>print(y)
    result:>>[0, 0, 1, 1, 1, 0, 0, 1]
    '''
    length = len(list_word)
    prediction = [[0] * length][0]
    max_ = [[0] * number][0]        # 用于存储前number个最大值
    for x in range(number):
        max_[x] = max(list_word)
        index_ = list_word.index(max_[x])
        prediction[index_] = 1
        list_word[index_] = -10000
    return prediction

def to_tensor(list_tensor):
    #把list转化为tensor
    return tf.convert_to_tensor(list_tensor,dtype=tf.float32)

def return_max_index(list_,number):
    '''
    返回最大前number个值的索引
    example:>>l = [1,4,6,7,875,2,-3,5]
            >>y = return_max_index(l,3)
            >>print(y)
    result:>>[4 3 2]
            如果 l = [1,0,1,0,1,0,1,0]
                 g = return_max_index(l,3)
                 print(g)
            >>[6 4 2]
            即元素值相等，则原来索引大的则在返回list中排前面
    '''
    list_ = np.array(list_)
    #list_index = list_.argsort()[-1:(number+1):-1]
    list_index = list_.argsort()[-(number):]
    return list_index

def turn_tensor_to_array(d_tensor_):
    '''
    tensor: 输入一个tensor，最好是不超过三维,注意要在tf.Session()中使用这个函数
    return: 返回一个内容和形状与该tensor一样的np.array
    '''
    temp = d_tensor_.eval()
    return np.array(temp)

def return_matrix_max_indice(matrix_,number):
    '''
    matrix_:输入np.array矩阵
    number:每一行需要最大的前number个值
    return: 返回一个list，每一行为matrix_每行最大前number个值的索引，行数与matrix_一样，列数为number
    '''
    list_ = list(matrix_)
    max_indice = []
    for a in list_:
        max_indice.append(return_max_index(a,number))
    return np.array(max_indice)

def return_matrix_max_number(matrix_,number):
    '''
    matrix_: 输入一个np.array矩阵
    number: 每一行需要最大的前number个值
    return: 返回一个list，大小与matrix_一样，只是只有最大number个值所在位置置1，其他全为0
    '''
    list_ = list(matrix_)
    max_indice = []
    for a in list_:
        max_indice.append(return_max_list(a, number))
    return np.array(max_indice)
```
## 评估新样本函数：eval.py
- 加载已训练的模型需要的文件有：checkpoint,model-400.data-00000-of-00001,model.index,model.meta,vocab
- “.meta”文件：包含图形结构
- “.data”文件：包含变量的值
- “.index”文件：标识检查点
- “checkpoint”文件：具有最近检查点列表的协议缓冲区
- “vocab”文件：字典文件
- **待更新**
```python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from textCNN import TextCNN
from tensorflow.contrib import learn
import csv

# Data Parameters
tf.flags.DEFINE_string("pos_file", "normal_label.csv", "Data source for the positive data.")
tf.flags.DEFINE_string("neg_file", "unnormal_label.csv", "Data source for the negative data.")
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir","", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helper.data_solve(FLAGS.pos_file, FLAGS.neg_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir,r"E:\forworking\Iflycompany\深度学习-骚扰留言-多标签\fighting_textcnn_OVR\model","vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
print("\nEvaluating...\n")
# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(r"E:\forworking\Iflycompany\深度学习-骚扰留言-多标签\fighting_textcnn_OVR\model\\")
print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    print('begin to eval')
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        print('finish meta')
        saver.restore(sess,checkpoint_file)#不能直接输入模型文件名，会报错，输入其前缀，前缀可有tf.train.latest_checkpoint()获得
        print('finish model')
        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        # Generate batches for one epoch
        batches = data_helper.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
```
## tensorborad
- 找到summary文件夹，在终端打开该文件夹，命令：tensorboard --logdir=./放可视化文件的文件夹 --host 0.0.0.0
- 用chorme，打开http：//localhost:6006,即看到可视化的图表 ![image](E:\forworking\Iflycompany\深度学习-骚扰留言\Success_textcnn_mutillabel\runs\1532499610\summaries\tensorgrogh.jpg)
![image](E:\forworking\Iflycompany\深度学习-骚扰留言\Success_textcnn_mutillabel\runs\1532499610\summaries\tensorboard-loss.jpg)
```python
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
```
