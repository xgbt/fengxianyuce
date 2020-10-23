import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import random


def main():
    base_info = pd.read_csv(r'./datasets/train/base_info.csv')
    enterprise_info = pd.read_csv(r'./datasets/train/entprise_info.csv')
    merged_data = pd.merge(base_info, enterprise_info, how='left', on='id')
    industryphy = merged_data['industryphy'].to_numpy()
    industryphy = industryphy.reshape([industryphy.shape[0], 1])
    # one_hot = OneHotEncoder()
    # one_hot.fit(industryphy)
    labeled_data = merged_data[~merged_data['label'].isnull()]
    x = labeled_data[['industryphy', 'townsign']].to_numpy()
    y = labeled_data['label'].to_numpy()
    for i in range(len(x)):
        x[i][0] = ord(x[i][0])
    for index, item in enumerate(x):
        if np.isnan(item[1]):
            item[1] = 0
    # 简单根据已有的数据集进行一下测试
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    random_forest = RandomForestClassifier()
    mlp = MLPClassifier()
    xgb = XGBClassifier()
    svc = SVC()
    gnb = GaussianNB()
    random_forest.fit(x_train, y_train)
    y_pred = random_forest.predict(x_test)
    print('accuracy_score:', accuracy_score(y_test, y_pred))

    random_forest = RandomForestClassifier()
    mlp = MLPClassifier()
    xgb = XGBClassifier()
    svc = SVC(probability=True)

    random_forest.fit(x, y)
    unlabeled_data = merged_data[merged_data['label'].isnull()]
    x_test = unlabeled_data[['industryphy', 'townsign']].to_numpy()
    for i in range(len(x_test)):
        x_test[i][0] = ord(x_test[i][0])
    for index, item in enumerate(x_test):
        if np.isnan(item[1]):
            item[1] = 0
    # x_test = x_test.reshape([x_test.shape[0], 1])
    # x_test = one_hot.transform(x_test)
    result = {}
    predict_result = random_forest.predict_proba(x_test)
    for id, predict_score in zip(unlabeled_data['id'].to_numpy(), predict_result):
        result[id] = predict_score[1]

    submission_pd = pd.read_csv(r'./datasets/entprise_submit.csv')
    with open('result.csv ', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(submission_pd.columns.values)
        for submission in submission_pd.values:
            csv_writer.writerow([submission[0], round(result[submission[0]], 4)])


if __name__ == '__main__':
    main()
