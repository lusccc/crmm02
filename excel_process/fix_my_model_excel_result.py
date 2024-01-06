import glob
import os

import pandas as pd

excel_files = glob.glob('./my_file/*.xlsx')

for excel_file in excel_files:
    nrows = 50 if 'cr2' in excel_file else 40
    fname = os.path.basename(excel_file)
    # 加载Excel文件
    xls = pd.ExcelFile(excel_file, engine='openpyxl')

    val_eval_avg_acc_roc_auc_f1_gmean_best_record = []
    # 创建一个新的Excel writer对象
    with pd.ExcelWriter(f'processed_{fname}') as writer:
        # for sheet_name in xls.sheet_names[:-1]:
        for sheet_name in xls.sheet_names[1:-1]:  # for cr2
            # 读取当前sheet
            df = pd.read_excel(xls, sheet_name=sheet_name, nrows=nrows)
            loc = df.columns.get_loc("val_eval_type2_acc") + 1
            df.rename(columns={df.columns[17]: 'val_eval_sqrt_roc_auc_gmean'}, inplace=True)
            df.rename(columns={df.columns[18]: 'val_eval_avg_roc_auc_gmean'}, inplace=True)
            df.rename(columns={df.columns[19]: 'val_eval_avg_acc_roc_auc_gmean'}, inplace=True)
            df.rename(columns={df.columns[20]: 'val_eval_avg_acc_roc_auc_f1_gmean'}, inplace=True)

            # （1）将列test_eval_acc、test_eval_roc_auc移到最后
            columns_to_move = ['test_eval_acc', 'test_eval_roc_auc']
            df = df[[c for c in df if c not in columns_to_move] + columns_to_move]

            # （2）根据列计算recall、precision、f1
            # 计算recall
            df['test_eval_recall_.5threshold'] = df['test_eval_tp'] / (df['test_eval_tp'] + df['test_eval_fn'])
            # 计算precision
            df['test_eval_precision_.5threshold'] = df['test_eval_tp'] / (df['test_eval_tp'] + df['test_eval_fp'])
            # 计算f1
            df['test_eval_f1_.5threshold'] = 2 * (
                    (df['test_eval_precision_.5threshold'] * df['test_eval_recall_.5threshold']) /
                    (df['test_eval_precision_.5threshold'] + df['test_eval_recall_.5threshold'])
            )

            # 将新列添加到test_eval_roc_auc之后
            # columns_order = df.columns.tolist()
            # position = columns_order.index('test_eval_roc_auc') + 1
            # columns_order[position:position] = ['test_eval_recall_correct', 'test_eval_precision_correct',
            #                                     'test_eval_f1_correct']
            # df = df[columns_order]

            # （3）将列test_eval_ks、test_eval_gmean、test_eval_type1_acc、test_eval_type2_acc移到最后
            columns_to_move = ['test_eval_ks', 'test_eval_gmean', 'test_eval_type1_acc', 'test_eval_type2_acc']
            df = df[[c for c in df if c not in columns_to_move] + columns_to_move]

            # （6）找出eval_final_metric列中最大的值所在行
            best_record = df.loc[df["val_eval_avg_acc_roc_auc_f1_gmean"].idxmax()]
            val_eval_avg_acc_roc_auc_f1_gmean_best_record.append(best_record)

            # 将处理过的sheet写入新的Excel文件
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 将val_eval_avg_acc_roc_auc_f1_gmean_best_record写入到一个新的sheet中
        df_val_eval_avg_acc_roc_auc_f1_gmean_best_record = pd.DataFrame(val_eval_avg_acc_roc_auc_f1_gmean_best_record)
        df_val_eval_avg_acc_roc_auc_f1_gmean_best_record.to_excel(writer, sheet_name='Best Records', index=False)
