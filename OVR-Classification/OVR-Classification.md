# Text-Classification-OVR
## **语言**
Python3.6
## **依赖库**
numpy=1.14.3

pandas=0.23.0

scikit-learn=0.19.1

jieba=0.39

Mysql.connector-python=8.0.11
## **项目介绍**
### 监督学习
通过已有单个标签的留言内容数据进行训练，实现新留言内容的多类分类
### 实现思想
由于数据量大，这次训练采用了八百万骚扰号码（包含了多个骚扰类别）
和五百万非骚扰号码的原始数据。首先对原始数据分词，去掉停用词，再把
处理好的数据存入数据库里，然后从数据库里提取所需要的量的数据，导出为csv
文件，再读取数据，打乱数据，分好测试集和训练集用scikit-learn的机器学习
算法训练数据，保存模型文件，预测准确率
## **用法介绍**
### 文本预处理函数：preprocess.py
- 从文中提取关键词，去掉停用词，返回一个list
```python
import jieba
stop=[line.strip() for line in open('stop.txt','r',encoding='gb18030').readlines()]
def get_keyword(string):
    '''
    从文本中获取关键词
    string:待提取文本
    stop:停用词列表
    '''
    words=jieba.cut(string,cut_all=False)
    keywords=list(set(words)-set(stop))
    return keywords
```
- 把两个list合并成一个元组
```python
import random
def combine_xy(x,y):
    '''
    将x,y结合为一个元组，并返回乱序元组列表
    要求:x,y必须为一维列表并等长度，需import random
    x:列表x
    y:列表y
    '''
    data_xy=[]
    lists=list(range(len(x)))
    random.shuffle(lists)
    for i in lists:
        data_xy.extend([[x[i],y[i]]])
    return data_xy
```
- 把数据存为磁盘文件，以便以后加载使用
```python
import pickle
import sys
def save_data(path,filename,data):
    '''
    将获得的数据存储为pkl文件
    path:存放目标文件路径
    filename:文件名
    data:待存储数据
    '''
    try:
        f=open(path+filename+'.pkl','wb')
        pickle.dump(data,f)
        f.close()
    except IOError as e:
        print('Unable to open file. %s'%e)
    except:
        print("Unexcepted error:",sys.exc_info())
```
### 传输数据：now_linux_import_sql（1/2）.py
- 先建立pycharm与mysql的连接
```python
import mysql.connector
#连接数据库
cnn = mysql.connector.connect(
                              user = 'root',
                              password = 'Iflytek@1234',
                              host = 'localhost',
                              database = 'TESTDB'
                            )
                            
cursor = cnn.cursor()#获取操作游标
```
- 按行读取原始数据（骚扰和非骚扰留言内容），每行内容分为主叫号码，被叫号码
，留言内容，命中关键字，标签，时间，并用“#”隔开，所以利用正则表达式提取字符串
```python
import re
import preprocess
#读数据文件，然后逐行的进行分词
fd = open('data.txt')
for lines in fd.readlines():
    all_comment = re.split(r'#',lines)#正则表达提取出了六个字符串信息

    sentance_temp = preprocess.get_keyword(all_comment[2])
    sentance_comment = ' '.join(sentance_temp)

    importance_temp = re.split(r'\+',all_comment[3])
    importance_comment = ' '.join(importance_temp)
```
- 在for循环内，每处理一行，把数据内容传入到数据库里

数据库表格格式如下：
|MO|SI|Sentance|Importance|Number_label|Time|
```python
    # sql插入语句
    cursor = cnn.cursor()#获取操作游标
    sql = """insert into talks(MO,SI,Sentance,Importance,Number_label,Time)
                 values(%s,%s,%s,%s,%s,%s)"""
    cursor.execute(sql,(all_comment[0],all_comment[1],sentance_comment,importance_comment,
                            all_comment[4],all_comment[5]))#内容对号入座
```
### 从数据库把数据导出为csv格式文件
```mysql
select * from talks   #数据表的名字   
into outfile './now_linux_talks_data.csv'  #导出文件位置和文件名   
fields terminated by ','  #字段间以,号分隔
optionally enclosed by '"'  #字段用"号括起
escaped by '"'  #字段中使用的转义符为"
lines terminated by '\r\n';  #行以\r\n结束
```
### 模型训练和保存：gentle_train_talks_data.py
- 需要导入库列表
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import preprocess
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
```
- 训练数据的整合
```python
fd = pd.read_csv('now_train_talks_data.csv')#读取文件
different = fd.drop_duplicates('Sentance',keep='first')#把Sentance中内容相同的只保留第一个，其他删除
```
- 标签数值化
```python
le = LabelEncoder()#定义标签化转化器
y_Encoder = le.fit_transform(list(data['Number_label']))#标签转化
preprocess.save_data('trained_data/','labelencoder',le)#保存标签转化器
```
- 分离训练集和测试集，并打乱
```python
X_train,X_test,y_train_Encoder,y_test_Encoder=train_test_split(list(data['Sentance'].values.astype('U')),y_Encoder,test_size=0.3,random_state=0,shuffle=True)
```
- 用词袋模型进行训练
```python
tfidfv = TfidfVectorizer(max_features=1000000,ngram_range=(1,3),norm='l2')#定义字典属性
word_dictionary = tfidfv.fit_transform(X_train)#生成字典
```
- 用一对多分类器进行训练
```python
classifier=OneVsRestClassifier(SGDClassifier())#定义分类器
classifier.fit(word_dictionary,y_train_Encoder)#训练
```
- 模型和字典保存
```python
joblib.dump(classifier,'trained_data/classifier.clf')#模型
preprocess.save_data('trained_data/','tfidfv',tfidfv)#字典
```
- 测试准确率
```python
predicted=classifier.predict(tfidfv.transform(X_test))
```
## **模型调用和新留言的类别预测举例**
```python
from sklearn.externals import joblib
import jieba
import pickle
classifier = joblib.load('classifier/classifier.clf')#加载分类器
labelencoder = pickle.load(open('classifier/labelencoder.pkl','rb'))#加载标签转化器
tfidfv = pickle.load(open('classifier/tfidfv.pkl','rb'))#加载字典
temp = input('Please input the sentence you want to judge(input nothing and enter will teminal): ')
while(temp!=''):
    print('you had input: %s'%temp)
    key_words = jieba.cut(temp,cut_all=False)
    key_words = ' '.join(key_words)
    key_words = tfidfv.transform([key_words])
    predict = classifier.predict(key_words)#预测数值
    predict = labelencoder.inverse_transform(predict)#根据预测数值反向输出预测标签
    print('result: %s'%predict)
    del temp,key_words
    temp = input('Please input the sentence you want to judge(input nothing and enter will teminal): ')
```
## **数据样本生成器**
- 根据你想要的样本数量，从原始样本中提取相应数量样本，用于轻量级模型训练
```python
disturb = open('data.txt','r',encoding='utf-8')#原始数据
count = 100
data = open('data3.txt','w',encoding='utf-8')#提取数据样本
while(count > 0):
    data.write(disturb.readline())
    count -= 1
disturb.close()
data.close()
```
