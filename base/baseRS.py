# -*- coding:utf-8 -*-
# 使用surprise构建推荐系统的demo

import io
import cPickle as pickle
from datetime import datetime
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import Dataset, Reader
from surprise import evaluate, print_perf

def read_item_name(file_name, sep):
    """
        获取 电影名到电影ID的映射 和 电影ID到电影名的映射
        item_rid_to_name: 电影ID到电影名的映射
        item_name_to_rid: 电影名到电影ID的映射
    """
    item_rid_to_name = {}
    item_name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            content = line.split(sep)
            item_rid_to_name[content[0]] = content[1]
            item_name_to_rid[content[1]] = content[0]
    return item_rid_to_name, item_name_to_rid


def read_rating_data(file_name, sep):
    """
        按照指定的格式读取数据
    """
    # 告诉阅读器文本的格式,必须是user item rating timestamp这样的关键字（rating写成ratings也会报错）
    reader = Reader(line_format='user item rating timestamp', sep=sep)
    # 加载数据
    data = Dataset.load_from_file(file_name, reader)
    return data


def save(file, algo, type):
    """
        保存模型和训练集
    """
    print 'Save {0} ...'.format(type)
    pickle.dump(algo, open(file, 'wb'))


def load(file, type):
    """
        加载模型和训练集
    """
    print 'Load {0} ...'.format(type)
    return pickle.load(open(file, 'rb'))


def train_model(data):
    """
        构建训练数据并训练模型
    """
    # 构建训练数据
    trainset = data.build_full_trainset()
    # 选用pearson相似度，采用基于物品的协同过滤
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    algo = KNNBaseline(sim_options = sim_options)
    algo.train(trainset)
    return algo, trainset


def get_item_neighbors(algo, item_name, item_rid_to_name, item_name_to_rid, k):
    """
        寻找相似的item
        item_raw_id: 是我们自定义的item_id
        item_inner_id: 是效用矩阵内部的item_id
        to_inner_iid可以将item_raw_id转化为item_inner_id
        to_raw_iid可以将item_inner_id转化为item_raw_id
        下面score函数中：
        user_inner_id: 是效用矩阵内部的user_id
        to_inner_uid可以将user_raw_id转化为user_inner_id
        to_raw_uid可以将user_inner_id转化为user_raw_id
    """
    item_raw_id = item_name_to_rid[item_name]
    item_inner_id = algo.trainset.to_inner_iid(item_raw_id)
    neighbors = algo.get_neighbors(item_inner_id, k=k)
    neighbors = [algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors]
    neighbors = [item_rid_to_name[rid] for rid in neighbors]
    print 'The 10 nearest neighbors of {0} are:'.format(item_name)
    for item in neighbors:
        print item


def score(algo, trainset, user_id, item_rid_to_name, item_name_to_rid):
    """
        用模型给用户打分过的商品打分，用于对比
    """
    user_inner_id = algo.trainset.to_inner_uid(user_id)
    # 获取用户所有打分过的商品((item,score),(item,score),...)
    user_rating = trainset.ur[user_inner_id] 
    for item_rating in user_rating:
        item = item_rating[0]
        rating = item_rating[1]
        print algo.predict(user_inner_id, item, r_ui=rating)


def test(algo, trainset, item_rid_to_name, item_name_to_rid):
    """
        1.寻找item的k个相似相似item
        2.预测user对item的打分
        3.给user做推荐
        ### 数据中没有user_name，只有user_id
    """
    user_id = '4'
    item_name = 'Toy Story (1995)'
    k = 10 # k个相似item
    print '1.寻找item的k个相似相似item'
    get_item_neighbors(algo, item_name, item_rid_to_name, item_name_to_rid, k)
    print '2.预测user对item的打分'
    score(algo, trainset, user_id, item_rid_to_name, item_name_to_rid)


def algo_comp(data, algo):
	"""
		比较算法的效果和运行时间
	"""
    print '---------------{0}---------------'.format(algo.__name__)
    start_time = datetime.now()
    algo = algo()
    pref = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(pref)
    end_time = datetime.now()
    print '耗时：{}s'.format((end_time - start_time).seconds)
    print '-----------------------------------------------'


def cv(data):
    """
        交叉验证比较不同模型
    """
    # 指定将数据分为3折（方便经销交叉验证）
    data.split(n_folds = 3)
    algos = [
        NormalPredictor, 
        BaselineOnly,
        KNNBasic,     # 基本的协同过滤算法
        KNNBaseline,  
        KNNWithMeans, # 去掉均值偏差的协同过滤算法
        SVD,          # SVD
        SVDpp,        # SVD++
        NMF
    ]
    for algo in algos:
        algo_comp(data, algo)

if __name__ == '__main__':
    dir_data = '../data/ml-100k/'
    dir_model = '../model/'
    # 用户数据
    file_user = dir_data + 'u.user'
    sep_user = '|'
    # 电影（item）数据
    file_movie = dir_data + 'u.item'
    sep_movie = '|'
    # 评分数据
    file_rating = dir_data + 'u.data'
    sep_rating = '\t'
    file_model = dir_model + 'base.model'
    file_trainset = dir_model + 'base.trainset'
    # 使用：1.训练模型 2.测试模型 3.模型对比
    selected = 3
    # 读取训练数据
    data = read_rating_data(file_rating, sep_rating)
    if selected == 1:
        # 训练模型并获得模型和训练数据集
        algo, trainset = train_model(data)
        save(file_model, algo, 'model')
        save(file_trainset, trainset, 'trainset')
    elif selected == 2:
        # 测试模型
        algo = load(file_model, 'model')
        trainset = load(file_trainset, 'trainset')
        item_rid_to_name, item_name_to_rid = read_item_name(file_movie, sep_movie)
        test(algo, trainset, item_rid_to_name, item_name_to_rid)
    elif selected == 3:
        # 交叉验证对比各种模型
        cv(data)
    else:
        print 'ERROR'
    



