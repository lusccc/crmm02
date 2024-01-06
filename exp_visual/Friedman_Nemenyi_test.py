import pandas as pd

from nemenyi_plot_tool import nemenyi_plot_multiple

"""
FOR ROLLING SPLIT DATASET EXP RESULTS
两个数据集rolling的结果，每一个数据集rolling划分可得四个数据集，那么共有8个数据集。
8个数据集的结果我都放在all_results里了
"""

all_results = [
    """
    CNCT	0.7658 	0.8474 	0.8031 	0.7479 
    CNCT-no-ensemble	0.7660 	0.8496 	0.8008 	0.7469 
    CNCT-no-NLL	0.5931 	0.7482 	0.4822 	0.5228 
    CNCT-no-MLP	0.7885 	0.8689 	0.8136 	0.7786 
    LogR	0.6923 	0.7500 	0.7659 	0.7707 
    SVM	0.7596 	0.8244 	0.8011 	0.8015 
    KNN	0.6827 	0.6936 	0.7568 	0.7610 
    DT	0.7019 	0.6904 	0.7482 	0.7483 
    MLP	0.7237 	0.7393 	0.7843 	0.7879 
    Adaboost	0.7564 	0.7972 	0.8081 	0.8107 
    XGBoost	0.7372 	0.8082 	0.7929 	0.7955 
    GBDT	0.7824 	0.8397 	0.8280 	0.8305 
    RF	0.7891 	0.8558 	0.8308 	0.8326 
    GATE	0.6962 	0.7834 	0.7230 	0.7326 
    TabTransformer	0.6631 	0.7150 	0.7036 	0.7068 
    AutoInt	0.6846 	0.8241 	0.7813 	0.7956 
    """,
    """
    CNCT	0.8457 	0.9109 	0.8734 	0.8290 
    CNCT-no-ensemble	0.8401 	0.9140 	0.8632 	0.8366 
    CNCT-no-NLL	0.8198 	0.8980 	0.8474 	0.8080 
    CNCT-no-MLP	0.8326 	0.9116 	0.8557 	0.8288 
    LogR	0.7702 	0.8336 	0.8254 	0.8292 
    SVM	0.7441 	0.8492 	0.8130 	0.8199 
    KNN	0.7076 	0.7772 	0.7871 	0.7941 
    DT	0.7587 	0.7508 	0.7963 	0.7964 
    MLP	0.7321 	0.7903 	0.7833 	0.7860 
    Adaboost	0.7990 	0.8533 	0.8386 	0.8394 
    XGBoost	0.8277 	0.8845 	0.8559 	0.8559 
    GBDT	0.8355 	0.8911 	0.8665 	0.8670 
    RF	0.8535 	0.9133 	0.8804 	0.8808 
    GATE	0.7457 	0.8355 	0.7875 	0.7927 
    TabTransformer	0.7663 	0.8351 	0.7964 	0.7978 
    AutoInt	0.7616 	0.8539 	0.8182 	0.8233 
    """,
    """
    CNCT	0.8589 	0.9287 	0.8868 	0.8477 
    CNCT-no-ensemble	0.8557 	0.9291 	0.8841 	0.8427 
    CNCT-no-NLL	0.8276 	0.9023 	0.8636 	0.8076 
    CNCT-no-MLP	0.8546 	0.9242 	0.8793 	0.8556 
    LogR	0.7333 	0.8002 	0.7981 	0.7994 
    SVM	0.7688 	0.8325 	0.8279 	0.8301 
    KNN	0.7542 	0.8059 	0.8179 	0.8203 
    DT	0.7496 	0.7388 	0.7956 	0.7958 
    MLP	0.7373 	0.7954 	0.7858 	0.7873 
    Adaboost	0.7750 	0.8460 	0.8218 	0.8219 
    XGBoost	0.8188 	0.8871 	0.8595 	0.8599 
    GBDT	0.8004 	0.8712 	0.8435 	0.8437 
    RF	0.8021 	0.8893 	0.8440 	0.8442 
    GATE	0.8029 	0.8425 	0.8381 	0.8396 
    TabTransformer	0.7973 	0.8611 	0.8329 	0.8339 
    AutoInt	0.8185 	0.8811 	0.8542 	0.8546 
    """,
    """
    CNCT	0.8456 	0.9067 	0.8314 	0.8437 
    CNCT-no-ensemble	0.8367 	0.9040 	0.8249 	0.8357 
    CNCT-no-NLL	0.8294 	0.9029 	0.8115 	0.8251 
    CNCT-no-MLP	0.8304 	0.8990 	0.8114 	0.8258 
    LogR	0.6799 	0.7568 	0.6097 	0.6143 
    SVM	0.7780 	0.8388 	0.7439 	0.7454 
    KNN	0.7126 	0.7998 	0.7146 	0.7173 
    DT	0.6850 	0.6824 	0.6547 	0.6547 
    MLP	0.6722 	0.7538 	0.6327 	0.6392 
    Adaboost	0.7664 	0.8550 	0.7475 	0.7475 
    XGBoost	0.7967 	0.8665 	0.7873 	0.7878 
    GBDT	0.7995 	0.8753 	0.7856 	0.7857 
    RF	0.7678 	0.8487 	0.7499 	0.7500 
    GATE	0.8023 	0.8698 	0.7823 	0.7847 
    TabTransformer	0.8147 	0.8782 	0.7956 	0.7978 
    AutoInt	0.8124 	0.8814 	0.7913 	0.7922 
    """,
    """
    CNCT	0.8833 	0.9411 	0.9129 	0.8583 
    CNCT-no-ensemble	0.8862 	0.9451 	0.9151 	0.8618 
    CNCT-no-NLL	0.8792 	0.9366 	0.9076 	0.8645 
    CNCT-no-MLP	0.8844 	0.9410 	0.9118 	0.8709 
    LogR	0.8010 	0.8496 	0.8585 	0.8603 
    SVM	0.8068 	0.8792 	0.8656 	0.8688 
    KNN	0.7888 	0.8015 	0.8516 	0.8542 
    DT	0.7421 	0.7186 	0.8018 	0.8019 
    MLP	0.6901 	0.6835 	0.7561 	0.7581 
    Adaboost	0.7851 	0.7545 	0.8396 	0.8398 
    XGBoost	0.8753 	0.9228 	0.9076 	0.9080 
    GBDT	0.8683 	0.8943 	0.9001 	0.9001 
    RF	0.8723 	0.9317 	0.9036 	0.9036 
    GATE	0.8464 	0.9000 	0.8824 	0.8835 
    TabTransformer	0.8650 	0.9285 	0.8959 	0.8964 
    AutoInt	0.8797 	0.9377 	0.9085 	0.9088 
    """,
    """
    CNCT	0.9300 	0.9664 	0.9495 	0.9186 
    CNCT-no-ensemble	0.9260 	0.9648 	0.9469 	0.9098 
    CNCT-no-NLL	0.9265 	0.9675 	0.9470 	0.9138 
    CNCT-no-MLP	0.9295 	0.9598 	0.9489 	0.9204 
    LogR	0.8153 	0.8369 	0.8733 	0.8742 
    SVM	0.8410 	0.8981 	0.8902 	0.8909 
    KNN	0.8323 	0.8797 	0.8842 	0.8849 
    DT	0.7944 	0.7701 	0.8486 	0.8489 
    MLP	0.7752 	0.7913 	0.8368 	0.8397 
    Adaboost	0.8184 	0.7989 	0.8721 	0.8724 
    XGBoost	0.8807 	0.9278 	0.9162 	0.9165 
    GBDT	0.8548 	0.9145 	0.8986 	0.8990 
    RF	0.8705 	0.9338 	0.9088 	0.9090 
    GATE	0.9253 	0.9597 	0.9462 	0.9463 
    TabTransformer	0.9289 	0.9656 	0.9485 	0.9487 
    AutoInt	0.9269 	0.9655 	0.9476 	0.9476 
    """,
    """
    CNCT	0.9199 	0.9540 	0.9388 	0.9060 
    CNCT-no-ensemble	0.9204 	0.9536 	0.9420 	0.9059 
    CNCT-no-NLL	0.9197 	0.9632 	0.9412 	0.9090 
    CNCT-no-MLP	0.9186 	0.9461 	0.9401 	0.9098 
    LogR	0.7769 	0.8124 	0.8482 	0.8501 
    SVM	0.8089 	0.8645 	0.8642 	0.8645 
    KNN	0.8351 	0.8799 	0.8822 	0.8824 
    DT	0.7781 	0.7454 	0.8380 	0.8380 
    MLP	0.7842 	0.8114 	0.8491 	0.8511 
    Adaboost	0.8021 	0.7750 	0.8620 	0.8628 
    XGBoost	0.8758 	0.9275 	0.9120 	0.9124 
    GBDT	0.8318 	0.8875 	0.8803 	0.8805 
    RF	0.8328 	0.9052 	0.8792 	0.8792 
    GATE	0.9105 	0.9486 	0.9346 	0.9346 
    TabTransformer	0.9180 	0.9598 	0.9399 	0.9400 
    AutoInt	0.9149 	0.9586 	0.9381 	0.9381 
    """,
    """
    CNCT	0.9118 	0.9581 	0.9215 	0.9093 
    CNCT-no-ensemble	0.9064 	0.9617 	0.9168 	0.9036 
    CNCT-no-NLL	0.9013 	0.9660 	0.9116 	0.8991 
    CNCT-no-MLP	0.9006 	0.9533 	0.9096 	0.9010 
    LogR	0.7487 	0.7941 	0.7791 	0.7792 
    SVM	0.8062 	0.8585 	0.8293 	0.8294 
    KNN	0.8215 	0.8810 	0.8435 	0.8438 
    DT	0.7190 	0.7215 	0.7489 	0.7490 
    MLP	0.7378 	0.7845 	0.7683 	0.7700 
    Adaboost	0.7487 	0.8341 	0.7660 	0.7665 
    XGBoost	0.8133 	0.8963 	0.8293 	0.8294 
    GBDT	0.7781 	0.8644 	0.7969 	0.7971 
    RF	0.7878 	0.8838 	0.8097 	0.8098 
    GATE	0.8645 	0.9264 	0.8783 	0.8789 
    TabTransformer	0.8887 	0.9573 	0.9018 	0.9023 
    AutoInt	0.8791 	0.9387 	0.8919 	0.8921 
    """
]


def parse_results(results):
    data_lines = results.strip().split("\n")
    parsed_data = [line.split() for line in data_lines]
    models = [line[0] for line in parsed_data]
    acc = [float(line[1]) for line in parsed_data]
    auc = [float(line[2]) for line in parsed_data]
    f1 = [float(line[3]) for line in parsed_data]
    g_mean = [float(line[4]) for line in parsed_data]
    return pd.DataFrame(
        {
            "Model": models,
            "ACC": acc,
            "AUC": auc,
            "F1": f1,
            "G-mean": g_mean,
        }
    )


def parse_df(all_results, metric, rank_data=False):
    n_dataset = len(all_results)
    all_res_df = []
    for res in all_results:
        res_df = parse_results(res)
        all_res_df.append(res_df)
    merged_res_df = pd.concat(all_res_df, axis=1, keys=[f'{i}' for i in range(n_dataset)])
    merged_res_df.columns = ['_'.join(col).strip() for col in merged_res_df.columns.values]
    metric_res_all_mdl_dt = merged_res_df[['0_Model'] + [f'{i}_{metric}' for i in range(n_dataset)]]
    if rank_data:
        # 对每一列进行排名
        rank_df = metric_res_all_mdl_dt.iloc[:, 1:].rank(method='min', ascending=False)

        # 将排名结果替换原 dataframe 中的准确率结果
        metric_res_all_mdl_dt.iloc[:, 1:] = rank_df
        # 对每一行求平均值
        metric_res_all_mdl_dt['Avg_Rank'] = metric_res_all_mdl_dt.iloc[:, 1:].mean(axis=1)
        print(metric_res_all_mdl_dt.iloc[:, 1:].values.tolist())
    return metric_res_all_mdl_dt

res_dicts = []
metrics = ['ACC', 'AUC', 'F1', 'G-mean']
for metric in metrics:
    res_df = parse_df(all_results, metric)
    res_dict = res_df.set_index('0_Model').T.to_dict('list')
    res_dicts.append(res_dict)

# nemenyi_plot_multiple(res_dicts, [f"({chr(ord('e') + i)}) {m}" for i, m in enumerate(metrics)], row=4, col=1)
nemenyi_plot_multiple(res_dicts, [f"({chr(ord('a') + i)}) {m}" for i, m in enumerate(metrics)], row=2, col=2)
