import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, classification_report
import warnings
warnings.filterwarnings('ignore')
# 导入数据
train = pd.read_csv('./data/1. 主数据集_25k/df_train_b1.csv')
test = pd.read_csv('./data/1. 主数据集_25k/df_test_b1.csv')
# 训练测试 合并
data = pd.concat([train, test]).reset_index(drop=True)


#
def all_dataset_merge():
    train = pd.read_csv('./data/1. 主数据集_25k/df_train_b1.csv')
    test = pd.read_csv('./data/1. 主数据集_25k/df_test_b1.csv')
    data = pd.concat([train, test]).reset_index(drop=True)

    basic_b1 = pd.read_csv('./data/2. 细分数据集_25k/df_basic_b1.csv')
    basic_b1.columns = ['cust_id', 'basic_1_other', 'basic_2_other', 'basic_3_other']

    corp = pd.read_csv('./data/2. 细分数据集_25k/df_corp_b1.csv')

    query = pd.read_csv('./data/2. 细分数据集_25k/df_query_b1.csv')
    query.columns = ['cust_id', 'date_1_query', 'query_1_other', 'query_2_other', 'query_3_other',
                     'query_4_other', 'query_5_other', 'query_6_other']

    loan2 = pd.read_csv('./data/2. 细分数据集_25k/df_loan2_b1.csv')
    loan2.columns = ['cust_id', 'loan2_1_other', 'loan2_2_other', 'loan2_3_other', 'loan2_4_other', 'loan2_5_other']

    data = pd.merge(data, basic_b1, on='cust_id', how='left')
    data = pd.merge(data, corp, on='cust_id', how='left')
    data = pd.merge(data, query, on='cust_id', how='left')
    data = pd.merge(data, loan2, on='cust_id', how='left')
    return data


data = all_dataset_merge()
# 数据预处理

# 特征工程

# 训练测试 分离
train = data[~data['label'].isna()].reset_index(drop=True)
test = data[data['label'].isna()].reset_index(drop=True)
# train.replace(-99, np.nan, inplace=True)
# test.replace(-99, np.nan, inplace=True)

no_train_fea = ['label', 'cust_id', 'date_1_query', 'scope',]
#'basic_8', 'basic_9', 'basic_11', 'basic_12', 'basic_15', 'loan1_8', 'loan1_9', 'overdue_1', 'overdue_5', 'overdue_6', 'overdue_10', 'overdue_11', 'overdue_12', 'overdue_13', 'overdue_14', 'overdue_15', 'overdue_16', 'overdue_17', 'overdue_18', 'overdue_24', 'overdue_25', 'query_9']

# 特征
features = [i for i in train.columns if i not in no_train_fea]
# 存储 label
y = train['label']
# 分层 K折
KF = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
# 特征重要性 df 初始化
feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
# 参数
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'n_jobs': 30,
    'learning_rate': 0.01,
    'num_leaves': 16,
    'max_depth': 4,
    'tree_learner': 'serial',
    'colsample_bytree': 0.85,
    'subsample_freq': 1,
    'subsample': 0.8,
    'num_boost_round': 5000,
    'max_bin': 255,
    'verbose': -1,
    'seed': 2021,
    'bagging_seed': 2021,
    'feature_fraction_seed': 2021,
    'early_stopping_rounds': 100,
}
# 训练集标签预测 df 初始化
oof_lgb = np.zeros(len(train))
# 测试集标签预测 df 初始化
predictions_lgb = np.zeros((len(test)))

# 模型训练
for fold_, (trn_idx, val_idx) in enumerate(KF.split(train.values, y.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=y.iloc[val_idx])
    num_round = 3000
    clf = lgb.train(
        params,
        trn_data,
        num_round,
        valid_sets=[trn_data, val_data],
        verbose_eval=100,
        early_stopping_rounds=20,
    )

    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb[:] += clf.predict(test[features], num_iteration=clf.best_iteration) / 5
    feat_imp_df['imp'] += clf.feature_importance() / 5
# 计算 metrics
print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

# 提交结果
test['label'] = predictions_lgb
test[['cust_id', 'label']].to_csv('baseline.csv', index=False)