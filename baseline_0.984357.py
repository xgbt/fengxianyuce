import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import csv

pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


def main():
    # 读入数据
    base_info_pd = pd.read_csv(r'./datasets/train/base_info.csv', index_col=0)
    label_pd = pd.read_csv(r'./datasets/train/entprise_info.csv')
    news_info_pd = pd.read_csv(r'./datasets/train/news_info.csv')
    other_info_pd = pd.read_csv(r'./datasets/train/other_info.csv')
    change_info_pd = pd.read_csv(r'./datasets/train/change_info.csv')
    # 企业变更信息的次数
    base_info_pd['change_count'] = 0
    for item in change_info_pd.groupby('id')['id'].count().reset_index(name='count').to_numpy():
        base_info_pd.loc[item[0], 'change_count'] = item[1]
    # 积极舆论数量
    base_info_pd['positive'] = 0
    # 消极舆论数量
    base_info_pd['negative'] = 0
    # 中立舆论数量
    base_info_pd['neutral'] = 0
    ids_in_news_info = np.unique(news_info_pd['id'])
    for id in ids_in_news_info:
        count = news_info_pd[news_info_pd['id'] == id].drop(['public_date'], axis=1).groupby('positive_negtive')[
            'id'].count().reset_index(name='count')
        count = count.to_numpy()
        for item in count:
            if item[0] == '积极':
                base_info_pd.loc[id, 'positive'] = item[1]
            elif item[0] == '中立':
                base_info_pd.loc[id, 'neutral'] = item[1]
            elif item[0] == '消极':
                base_info_pd.loc[id, 'negative'] = item[1]
    # 基于id对数据进行连接，这里
    merged_pd = pd.merge(base_info_pd, label_pd, on=['id'], how='left')
    merged_pd = pd.merge(merged_pd, other_info_pd, on=['id'], how='left')
    # 缺失值太多的列
    missing_value_feature_label = ['enttypeitem', 'opto', 'empnum', 'compform', 'parnum',
                                   'exenum', 'opform', 'ptbusscope', 'venind', 'enttypeminu',
                                   'midpreindcode', 'protype', 'reccap', 'forreccap',
                                   'forregcap', 'congro', 'legal_judgment_num', 'brand_num', 'patent_num']
    # 单一值太多的列
    same_value_feature_label = ['dom', 'opscope', 'oploc']
    # 移除缺失值和单一值太多的列
    # for label in missing_value_feature_label:
    #     del merged_pd[label]
    for label in ['opto', 'opform']:
        del merged_pd[label]
    for label in same_value_feature_label:
        del merged_pd[label]
    # 拆分年和月
    merged_pd['year'] = merged_pd['opfrom'].apply(lambda x: int(x.split('-')[0]))
    merged_pd['month'] = merged_pd['opfrom'].apply(lambda x: int(x.split('-')[1]))
    del merged_pd['opfrom']
    dataset = merged_pd.copy()
    # 不需要的训练特征
    drop_feature_label = ['id', 'label']
    # 类别特征
    category_feature_label = ['industryphy']
    features = []
    for feature_name in list(dataset.columns):
        if feature_name in drop_feature_label:
            continue
        if feature_name in category_feature_label:
            dataset[feature_name] = dataset[feature_name].astype('category')
        features.append(feature_name)
    # 部分模型需要指定样本的权重，这里全部填充为1
    if 'sample_weight' not in dataset.keys():
        dataset['sample_weight'] = 1
    # 取出label为空的数据索引
    test_set_index = (dataset['label'].isnull()) | (dataset['label'] == -1)
    # 根据label是否为空划分训练集和测试集
    train_data = dataset[~test_set_index].reset_index(drop=True)
    test_data = dataset[test_set_index]
    # # 使用LabelEncoder对字段类型为string的列进行编码（也可以使用OneHotEncoder）
    # label_encoder = LabelEncoder()
    # label_encoder.fit(dataset['industryphy'])
    # train_data['industryphy'] = label_encoder.transform(train_data['industryphy'])
    # test_data['industryphy'] = label_encoder.transform(test_data['industryphy'])
    # # 处理部分缺失值，直接填充0
    # train_data['industryco'] = train_data['industryco'].fillna(0)
    # train_data['regcap'] = train_data['regcap'].fillna(0)
    # test_data['industryco'] = test_data['industryco'].fillna(0)
    # test_data['regcap'] = test_data['regcap'].fillna(0)
    for label in missing_value_feature_label:
        if label == 'opto' or label == 'opform':
            continue
        train_data[label] = train_data[label].fillna(0)
        test_data[label] = test_data[label].fillna(0)
    # 实例化模型
    model = lgb.LGBMClassifier(boosting_type='rf', num_leaves=128, reg_alpha=0., reg_lambda=0.01, metric='rmse',
                               max_depth=-1, learning_rate=0.01, min_child_samples=10, seed=1018, n_estimators=3000,
                               subsample=0.7, colsample_bytree=0.7, subsample_freq=1)
    # model = cb.CatBoostClassifier(iterations=5000, learning_rate=0.04, random_seed=1018, verbose=100)
    # model = xgb.XGBClassifier(n_estimators=5000, max_depth=0, learning_rate=0.05)
    # 训练模型
    model.fit(train_data[features], train_data['label'])
    # # 直接写入标签而不是得分
    # predict_result = model.predict(test_data[features])
    # result = {}
    # for id, predict_label in zip(test_data['id'].to_numpy(), predict_result):
    #     result[id] = predict_label
    # submission_pd = pd.read_csv(r'./datasets/entprise_submit.csv')
    # with open('result.csv ', 'w', encoding='utf-8', newline='') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(submission_pd.columns.values)
    #     for submission in submission_pd.values:
    #         # csv_writer.writerow([submission[0], result[submission[0]]])
    #         csv_writer.writerow([submission[0], result[submission[0]]])

    # 预测得分
    proba = model.predict_proba(test_data[features])
    # 读取结果并写入到csv文件
    result = {}
    for id, predict_score in zip(test_data['id'].to_numpy(), proba):
        result[id] = predict_score[1]
    submission_pd = pd.read_csv(r'./datasets/entprise_submit.csv')
    with open('result.csv ', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(submission_pd.columns.values)
        for submission in submission_pd.values:
            # csv_writer.writerow([submission[0], result[submission[0]]])
            csv_writer.writerow([submission[0], '{:.16f}'.format(result[submission[0]])])

    # kfold = KFold(n_splits=10, shuffle=True)
    # for train_index, test_index in kfold.split(train_data):
    #     train_x = train_data.loc[train_index][features]
    #     train_y = train_data.loc[train_index]['label']
    #     test_x = train_data.loc[test_index][features]
    #     test_y = train_data.loc[test_index]['label']
    #     train_data.loc[test_index, 'predict_label'] = model.predict(test_x)
    #     if len(test_data) != 0:
    #         test_data['predict_label'] = test_data['predict_label'] + model.predict(test_data[features])
    # test_data['predict_label'] = test_data['predict_label'] / 10


if __name__ == '__main__':
    main()
