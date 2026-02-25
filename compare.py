from deeplab.utils import plot_metrics_from_csv

def compare_loss():
    csv_files = [out_dir + out_csv_name]
    plot_metrics_from_csv(csv_files, out_dir)


if __name__ == '__main__':
    root_dir = '/mnt/e/data/hn/temp/'
    deeplab_str = 'a0_deeplabv3'
    deeplab_attention_str = 'a1_attention_deeplab'
    deeplab_reg_str = 'a2_reg_deeplab'
