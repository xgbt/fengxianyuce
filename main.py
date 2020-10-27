import time
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


# 模型预测模块化
def get_predict_w(model, data, label='label', feature=[], cate_feature=[], random_state=2018, n_splits=5,
                  model_type='lgb'):
    if 'sample_weight' not in data.keys():
        data['sample_weight'] = 1
    model.random_state = random_state
    predict_label = 'predict_' + label
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    data[predict_label] = 0
    test_index = (data[label].isnull()) | (data[label] == -1)  # 找到要预测的数据集
    train_data = data[~test_index].reset_index(drop=True)  # 分割出预测集训练集
    test_data = data[test_index]

    for train_idx, val_idx in kfold.split(train_data):
        model.random_state = model.random_state + 1

        train_x = train_data.loc[train_idx][feature]
        train_y = train_data.loc[train_idx][label]

        test_x = train_data.loc[val_idx][feature]
        test_y = train_data.loc[val_idx][label]
        if model_type == 'lgb':
            try:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=400,
                          eval_metric='mae',
                          callbacks=[lgb.reset_parameter(learning_rate=lambda iter: max(0.005, 0.5 * (0.99 ** iter)))],
                          categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=100)
            except:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=200,
                          eval_metric='mae',
                          callbacks=[lgb.reset_parameter(learning_rate=lambda iter: max(0.005, 0.5 * (0.99 ** iter)))],
                          categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=100)
        elif model_type == 'ctb':
            model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=200,
                      # eval_metric='mae',
                      # callbacks=[lgb.reset_parameter(learning_rate=lambda iter: max(0.005, 0.5 * (0.99 ** iter)))],
                      cat_features=cate_feature,
                      sample_weight=train_data.loc[train_idx]['sample_weight'],
                      verbose=100)
        train_data.loc[val_idx, predict_label] = model.predict(test_x)
        if len(test_data) != 0:  # 预测集的预测
            test_data[predict_label] = test_data[predict_label] + model.predict(test_data[feature])
    test_data[predict_label] = test_data[predict_label] / n_splits
    print((train_data[label], train_data[predict_label]) * 5, train_data[predict_label].mean(),
          test_data[predict_label].mean())

    return pd.concat([train_data, test_data], sort=True, ignore_index=True), predict_label


def main():
    # 读文件
    base = pd.read_csv('./datasets/train/base_info.csv', index_col=0)
    label = pd.read_csv('./datasets/train/entprise_info.csv')

    news_info_pd = pd.read_csv(r'./datasets/train/news_info.csv')
    other_info_pd = pd.read_csv(r'./datasets/train/other_info.csv')
    change_info_pd = pd.read_csv(r'./datasets/train/change_info.csv')
    # 企业变更信息的次数
    base['change_count'] = 0
    for item in change_info_pd.groupby('id')['id'].count().reset_index(name='count').to_numpy():
        base.loc[item[0], 'change_count'] = item[1]
    # 积极舆论数量
    base['positive'] = 0
    # 消极舆论数量
    base['negative'] = 0
    # 中立舆论数量
    base['neutral'] = 0
    ids_in_news_info = np.unique(news_info_pd['id'])
    for id in ids_in_news_info:
        count = news_info_pd[news_info_pd['id'] == id].drop(['public_date'], axis=1).groupby('positive_negtive')[
            'id'].count().reset_index(name='count')
        count = count.to_numpy()
        for item in count:
            if item[0] == '积极':
                base.loc[id, 'positive'] = item[1]
            elif item[0] == '中立':
                base.loc[id, 'neutral'] = item[1]
            elif item[0] == '消极':
                base.loc[id, 'negative'] = item[1]
    # 合并文件
    base = pd.merge(base, label, on=['id'], how='left')
    base = pd.merge(base, other_info_pd, on=['id'], how='left')
    # 删除有缺失的、单一值
    drop = ['enttypeitem', 'opto', 'empnum', 'compform', 'parnum',
            'exenum', 'opform', 'ptbusscope', 'venind', 'enttypeminu',
            'midpreindcode', 'protype', 'reccap', 'forreccap',
            'forregcap', 'congro', 'legal_judgment_num', 'brand_num', 'patent_num']
    for label in drop:
        if label == 'opto' or label == 'opform':
            continue
        base[label] = base[label].fillna(0)
    print(base)

    # for f in drop:
    #     del base[f]
    for label in ['opto', 'opform']:
        del base[label]

    del base['dom'], base['opscope']  # 单一值太多
    del base['oploc']

    # 拆分年月特征
    base['year'] = base['opfrom'].apply(lambda x: int(x.split('-')[0]))
    base['month'] = base['opfrom'].apply(lambda x: int(x.split('-')[1]))
    del base['opfrom']

    # 复制
    data = base.copy()

    num_feat = []
    cate_feat = []

    # 不需要的特征
    drop = ['id', 'label']
    # 类别特征
    # cat = ['industryphy']
    cat = ['protype', 'midpreindcode', 'venind', 'enttypeminu', 'enttypeitem', 'industryphy', 'compform', 'opform',
           'ptbusscope']
    for j in list(data.columns):
        if j in drop:
            continue
        if j in cat:
            cate_feat.append(j)
        else:
            num_feat.append(j)

    # 将类别特征转换为category类型
    for i in cate_feat:
        data[i] = data[i].astype('category')
    features = num_feat + cate_feat

    # 训练
    lgb_model = lgb.LGBMRegressor(
        num_leaves=64, reg_alpha=0., reg_lambda=0.01, metric='rmse',
        max_depth=-1, learning_rate=0.03, min_child_samples=10, seed=1018,
        n_estimators=3000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    )
    xgb_model = xgb.sklearn.XGBRegressor()

    data, predict_label = get_predict_w(lgb_model, data, label='label',
                                        feature=features, cate_feature=cate_feat,
                                        random_state=1018, n_splits=10, model_type='lgb')

    data['score'] = data[predict_label]
    result = {}
    for id, predict_score in zip(data['id'].to_numpy(), data['score'].to_numpy()):
        result[id] = predict_score
    submission_pd = pd.read_csv(r'./datasets/entprise_submit.csv')
    with open('result.csv ', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(submission_pd.columns.values)
        for submission in submission_pd.values:
            # csv_writer.writerow([submission[0], result[submission[0]]])
            # csv_writer.writerow([submission[0], result[submission[0]]])
            csv_writer.writerow([submission[0], '{:.16f}'.format(result[submission[0]])])
    # # data['forecastVolum'] = data['lgb'].apply(lambda x: -x if x < 0 else x)
    # df = data[data.label.isnull()][['id', 'score']]
    # # 对结果进行修正
    # df['score'] = df['score'].apply(lambda x: 0 if x < 0 else x)
    # df['score'] = df['score'].apply(lambda x: 1 if x > 1 else x)
    # # 输出结果
    # df.to_csv('results/result.csv', index=False)


if __name__ == '__main__':
    main()
