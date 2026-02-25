import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.colors as mcolors

matplotlib.rcParams['path.simplify_threshold'] = 0.5
#plt.rc('font', family='Times New Roman')
# Set global font sizes
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 'Microsoft YaHei' is another good option
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 14  # Main font size for all text
plt.rcParams['axes.labelsize'] = 14  # Font size for axes labels
plt.rcParams['axes.titlesize'] = 16  # Font size for title
plt.rcParams['xtick.labelsize'] = 14  # Font size for X tick labels
plt.rcParams['ytick.labelsize'] = 14  # Font size for Y tick labels
plt.rcParams['legend.fontsize'] = 15  # Font size for legend

#sns.set_style("ticks", {"font.family": "Times New Roman"})
DPI = 100

def load_data(file_paths, model_names):
    """Load data from multiple CSV files into a single DataFrame."""
    dataframes = []
    for file_path, model_name in zip(file_paths, model_names):
        df = pd.read_csv(file_path)
        df['Model'] = model_name
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def plot_mean_compare():
    # Define the fields for each metric
    fields = {'err_t': ['dx', 'dy', 'dz'],
              'err_r': ['ax', 'ay', 'az'],
              'iou': ['iou'],
              'pixacc': ['pixacc'],}
              #'recall': ['recall']}

    # Create an empty list to store the data for each model and metric
    data_list = []
    rt_data_list = []
    plot_data_list = []
    # Loop through the models and metrics, and collect data
    for model_name in model_names:
        for metric, metric_fields in fields.items():
            # Load the data from the CSV file for the current model and metric
            file_path = in_dirs[model_name] + metric + '.csv'
            df = pd.read_csv(file_path)

            if len(metric_fields) == 1:
                # For single-field metrics (e.g., 'iou', 'pixacc', 'recall')
                abs_values = df[metric_fields[0]].abs()
                mean_v = abs_values.mean()

                # Append the data to the list
                data_list.append({
                    'Model': model_name,
                     metric: f"{mean_v * 100:.2f}%"
                })
                plot_data_list.append({
                    'Model': model_name,
                    metric: mean_v* 100
                })
            else:
                # For multi-field metrics (e.g., 'err_r' -> ['ax', 'ay', 'az'])
                # Combine the fields by calculating the norm of each row
                abs_values = df[metric_fields].abs()
                #norm_values = (abs_values ** 2).sum(axis=1) ** 0.5
                norm_values = abs_values.mean(axis=1)
                mean_v = norm_values.mean()
                std_v = norm_values.std()

                for fld in metric_fields:
                    v = df[fld].abs()
                    mv = v.mean()
                    std = v.std()

                    rt_data_list.append({
                        'Model': model_name,
                            fld: f"{mv:.2f} ± {std:.2f}"
                    })

                # Append the data to the list
                data_list.append({
                    'Model': model_name,
                     metric: f"{mean_v:.2f} ± {std_v:.2f}"
                })

    # Convert the list into a unified DataFrame
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.float_format', '{:.2f}'.format)

    combined_df = pd.DataFrame(data_list)
    pivot_df = combined_df.pivot_table(index='Model', aggfunc='first')
    pivot_df = pivot_df.reset_index()
    print(pivot_df)

    combined_rt_df = pd.DataFrame(rt_data_list)
    pivot_rt_df = combined_rt_df.pivot_table(index='Model', aggfunc='first')
    pivot_rt_df = pivot_rt_df.reset_index()
    print(pivot_rt_df)

    plot_df = pd.DataFrame(plot_data_list)
    pivot_plot_df = plot_df.pivot_table(index='Model', aggfunc='first')
    pivot_plot_df = pivot_plot_df.reset_index()

    colors = ['yellow', 'blue', 'green']
    plt.figure(figsize=(8, 6))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    #pivot_plot_df[['pixacc', 'iou']].plot(kind='bar', colormap=custom_cmap, legend=True)
    df_melted = pivot_plot_df.melt(id_vars='Model', value_vars=['pixacc', 'iou'],
                        var_name='Metric', value_name='Value')
    sns.barplot(data=df_melted, x='Model', y='Value', hue='Metric')


    # Set plot title and labels
    plt.title('像素准确率和IoU')
    plt.ylabel('百分比(%)')
    plt.xlabel('')

    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adjust subplot layout
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()  # Adjust the plot to fit within the figure area
    plt.savefig(out_dir + 'comparison_of_metrics_across_models.png',dpi=DPI)


def plot_rt_and_recall():
    fields = {'err_r': ['ax', 'ay', 'az'],
              'err_t': ['dx', 'dy', 'dz'], }

    t_thresholds = np.linspace(0, 0.5, 50)
    r_thresholds = np.linspace(0, 5, 50)

    all_data = {}
    for model_name in model_names:
        metric_name = 'err_t'
        file_path = in_dirs[model_name] + metric_name + '.csv'
        df = pd.read_csv(file_path)
        metric_fields = fields[metric_name]
        err_t = 0
        #for metric in metric_fields:
        #    err_t += df[metric] ** 2
        #err_t = (err_t ** 0.5).rename('err_t')

        err_t = df[metric_fields].abs().mean(axis=1).rename('err_t')

        metric_name = 'err_r'
        file_path = in_dirs[model_name] + metric_name + '.csv'
        df = pd.read_csv(file_path)
        metric_fields = fields[metric_name]
        err_r = 0
        #for metric in metric_fields:
        #    err_r += df[metric] ** 2
        #err_r = (err_r ** 0.5).rename('err_r')
        err_r = df[metric_fields].abs().mean(axis=1).rename('err_r')

        recall_values = []
        recall_values_t = []
        recall_values_r = []
        for i in range(0, len(t_thresholds)):
            #recall = ((err_t < t_thresholds[i]) & (err_r < r_thresholds[i])).any().mean()  # Proportion of instances with RTE < threshold
            #recall_values.append(recall)
            recall_t = (err_t < t_thresholds[i]) .mean()
            recall_values_t.append(recall_t* 100)

            recall_r = (err_r < r_thresholds[i]) .mean()
            recall_values_r.append(recall_r* 100)

        #all_data[model_name] = {'err_t': t_thresholds, 'err_r': r_thresholds, 'recall_values': recall_values}
        all_data[model_name] = {'err_t': t_thresholds, 'err_r': r_thresholds,
                                'recall_t': recall_values_t ,
                                'recall_r': recall_values_r }

    plt.figure(figsize=(8, 6))
    for model_name, model_data in all_data.items():
        plt.plot(model_data['err_t'], model_data['recall_t'], label=model_name)
    plt.xlabel('平均平移误差阈值(m)')
    plt.ylabel('召回率(%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(out_dir + 'RTE_recall.png', dpi=DPI)
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_name, model_data in all_data.items():
        plt.plot(model_data['err_r'], model_data['recall_r'], label=model_name)
    plt.xlabel('平均旋转误差阈值(度)')
    plt.ylabel('召回率(%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(out_dir + 'RRE_recall.png', dpi=DPI)

def print_recall():
    fields = {'err_r': ['ax', 'ay', 'az'],
              'err_t': ['dx', 'dy', 'dz'], }

    t_threshold = 0.3
    r_threshold = 1

    all_data = pd.DataFrame()
    for model_name in model_names:
        metric_name = 'err_t'
        file_path = in_dirs[model_name] + metric_name + '.csv'
        df = pd.read_csv(file_path)
        metric_fields = fields[metric_name]
        err_t = 0
        '''for metric in metric_fields:
            err_t += df[metric] ** 2
        err_t = (err_t ** 0.5).rename('err_t')'''
        err_t = df[metric_fields].abs().mean(axis=1).rename('err_t')

        metric_name = 'err_r'
        file_path = in_dirs[model_name] + metric_name + '.csv'
        df = pd.read_csv(file_path)
        metric_fields = fields[metric_name]
        err_r = 0
        '''for metric in metric_fields:
            err_r += df[metric] ** 2
        err_r = (err_r ** 0.5).rename('err_r')'''
        err_r = df[metric_fields].abs().mean(axis=1).rename('err_r')

        recall = ((err_t < t_threshold) & (err_r < r_threshold)).mean()
        all_data[model_name] = {'recall(%)': f"{recall * 100:.2f}%"}
    print('t_threshold:%.2f'% t_threshold)
    print('r_threshold:%.2f' % r_threshold)
    print(all_data.T)


def main():
    plot_mean_compare()
    plot_rt_and_recall()
    print_recall()


if __name__ == '__main__':
    in_dir = '/mnt/e/data/hn/temp/'
    in_dirs = {'VisionHDAlignNet':in_dir + 'a2_reg_deeplab/eval_metrics-pose//',
               'RegNet': in_dir + 'regnet_results/val_process/eval_metrics//',
               'CalibNet': in_dir + 'calibnet_results/val_process/eval_metrics//'
    }
    out_dir = '/mnt/e/data/hn/temp/compare/'
    os.makedirs(out_dir,exist_ok=True)

    model_names = ['VisionHDAlignNet', 'RegNet', 'CalibNet']
    main()
