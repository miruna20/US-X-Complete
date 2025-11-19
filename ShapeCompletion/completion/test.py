import logging
import os
import sys
import importlib
import argparse
import numpy as np
import h5py
import subprocess
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import munch
import yaml
from train_utils import *
from dataset import verse2020_lumbar
import time
try:
    import wandb
except ImportError:
    wandb = None

import warnings

warnings.filterwarnings("ignore")

device = 'cuda'
device_ids = [0]

wandb_enabled = False
wandb_run = None

def plot_category_boxplots(category_metrics_results, metrics, cat_name, log_dir):
    """
    Plot boxplots for each category and save them to the specified directory.

    Args:
    - category_metrics_results (dict): Dictionary with category-wise metric results.
    - metrics (list): List of metric names.
    - cat_name (list): List of category names.
    - log_dir (str): Directory to save the plots.
    """
    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Prepare data for boxplot
        data = [category_metrics_results[cat][metric] if 'f1' in metric else [arr *10000 for arr in category_metrics_results[cat][metric]]for cat in cat_name]
        labels = [f' L{i + 1}' for i in range(len(data))]
        labels.append('All Categories')

        combined_data = np.concatenate(data)
        data.append(combined_data)

        # Check if data is empty
        if any(len(d) == 0 for d in data):
            logging.warning(f"No data for metric {metric}. Skipping boxplot.")
            continue

        # Create a boxplot
        plt.boxplot(data, labels = labels)
        plt.xticks(ticks=np.arange(1, len(data) + 1), labels=labels)
        plt.xlabel('Category')
        plt.ylabel(metric)
        plt.title(f'Boxplot of {metric} for each category')

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{metric}_boxplot.png'))
        plt.close()

def test():
    logging.info(str(args))

    prefix = args.data_to_test

    use_labelmaps = (args.use_labelmaps_in_PMNET or args.use_labelmaps_in_RENet)

    dataset_test = verse2020_lumbar(train_path=args.path_to_train_dataset,
                                    val_path=args.path_to_val_dataset,
                                    test_path=args.path_to_test_dataset,
                                    apply_trafo=args.apply_trafo,
                                    sigma=args.sigma,
                                    Xray_labelmap=use_labelmaps,
                                    prefix=prefix,
                                    num_partial_scans_per_mesh=args.num_partial_scans_per_mesh)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # Load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')

    modelPath = args.load_model

    net = torch.nn.DataParallel(model_module.Model(args), device_ids=device_ids)
    net.to(device)
    net.module.load_state_dict(torch.load(modelPath)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    # Metrics we would like to compute
    if args.eval_emd:
        metrics = ['cd_p', 'cd_p_arch', 'cd_p_body', 'cd_t', 'cd_t_arch', 'cd_t_body', 'emd', 'emd_arch', 'emd_body', 'f1', 'f1_arch', 'f1_body']
    else:
        metrics = ['cd_p', 'cd_p_arch', 'cd_p_body', 'cd_t', 'cd_t_arch', 'cd_t_body', 'f1', 'f1_arch', 'f1_body']

    # Dictionary with all of the metrics
    test_loss_meters = {m: AverageValueMeter() for m in metrics}

    # Number of samples per class
    number_samples_per_class = dataset_test.number_per_classes
    num_partial_scans_per_category = dataset_test.num_partial_scans_per_mesh

    # Number of categories
    num_categories = len(number_samples_per_class)

    # Metrics for each category present
    test_loss_cat = torch.zeros([num_categories, len(metrics)], dtype=torch.float32).cuda()

    # Number of samples from each category
    cat_num = torch.ones([num_categories, 1], dtype=torch.float32).cuda()

    cat_name = ['L1', 'L2', 'L3', 'L4', 'L5']

    for class_ in range(0, len(number_samples_per_class)):
        cat_num[class_] = number_samples_per_class[class_] * num_partial_scans_per_category

    logging.info('Testing on ' + prefix + ' dataset')

    with torch.no_grad():
        results_list = []
        coarse_results_list = []

        # Added dictionary to collect all metric results
        all_metrics_results = {m: [] for m in metrics}

        # Added dictionary to collect category-wise results
        category_metrics_results = {cat: {m: [] for m in metrics} for cat in cat_name}

        labels_seen = []
        for i, data in enumerate(dataloader_test):


            label, partial_pcd, labelmap, gt = data

            partial_pcd = partial_pcd.float().to(device)
            partial_pcd = partial_pcd.transpose(2, 1).contiguous()

            if use_labelmaps:
                labelmap = labelmap.float().to(device)
                labelmap = labelmap.transpose(2, 1).contiguous()

            gt = gt.float().to(device)

            result_dict = net(x_pcd= partial_pcd,  x_labelmap= labelmap,  gt= gt, prefix="test")

            # Update average metrics and collect results
            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())
                all_metrics_results[k].append(result_dict[k].cpu().numpy())

            # Sum up results for each metric and collect category-wise results
            for j, l in enumerate(label):
                for ind, m in enumerate(metrics):
                    test_loss_cat[int(l), ind] += result_dict[m][int(j)]
                    category_metrics_results[cat_name[int(l)]][m].append(result_dict[m][int(j)].cpu().numpy())

            # Append shape completion results
            results_list.append(result_dict['result'].cpu().numpy())
            coarse_results_list.append(result_dict['out1'].cpu().numpy())

            if i % args.step_interval_to_print == 0:
                logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

        logging.info('Loss per category:')
        category_log = ''
        table_data = []

        for i in range(num_categories):
            category_log += '\ncategory name: %s' % (cat_name[i])
            row_data = {
                "Category Name": cat_name[i]
            }
            for ind, m in enumerate(metrics):
                scale_factor = 1 if m in ['f1', 'f1_arch', 'f1_body'] else 10000
                mean_value = test_loss_cat[i, ind] / cat_num[i] * scale_factor
                std_dev = np.std(category_metrics_results[cat_name[i]][m]) * scale_factor
                median_value = np.median(category_metrics_results[cat_name[i]][m]) * scale_factor
                mean_std_median_str = f"{mean_value.item():.4f} ± {std_dev:.4f} (Median: {median_value:.4f})"
                category_log += ' %s: %s' % (m, mean_std_median_str)
                row_data[m] = mean_std_median_str

            table_data.append(row_data)

        logging.info(category_log)
        if wandb_enabled and wandb_run is not None:
            wandb.log({"Per category results": category_log})

        logging.info('Overview results:')
        overview_log = ''
        overview_results = []

        for metric, meter in test_loss_meters.items():
            scale_factor = 1 if metric in ['f1', 'f1_arch', 'f1_body'] else 10000
            mean_value = meter.avg * scale_factor
            metric_values = np.concatenate(all_metrics_results[metric], axis=0) * scale_factor
            std_dev = np.std(metric_values)
            median_value = np.median(metric_values)
            mean_std_median_str = f"{mean_value:.4f} ± {std_dev:.4f} (Median: {median_value:.4f})"
            overview_log += '%s: %s ' % (metric, mean_std_median_str)
            overview_results.append((metric, mean_std_median_str))

        logging.info(overview_log)
        if wandb_enabled and wandb_run is not None:
            wandb.log({"Overview Results": overview_log})

        # Concatenate all results
        all_results = np.concatenate(results_list, axis=0)
        all_coarse_results = np.concatenate(coarse_results_list, axis=0)

        # Write results to HDF5 file
        with h5py.File(os.path.join(log_dir, 'results.h5'), 'w') as f:
            f.create_dataset('results', data=all_results)
            f.create_dataset('coarse_results', data=all_coarse_results)
            for metric in metrics:
                f.create_dataset(metric, data=np.concatenate(all_metrics_results[metric], axis=0))

        # Create submission zip file
        cur_dir = os.getcwd()
        cmd = "cd %s; zip -r submission.zip results.h5 ; cd %s" % (log_dir, cur_dir)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, _ = process.communicate()
        print("Submission file has been saved to %s/submission.zip" % (log_dir))

    # Plot and save the boxplots
    plot_category_boxplots(category_metrics_results, metrics, cat_name, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)

    torch.cuda.empty_cache()
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    #wandb.login(key="845cb3b94791a8d541b28fd3a9b2887374fe8b2c")
    #run = wandb.init(project="Multimodal Shape Completion", tags=['inference'])
    #wandb.config.update(args)

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    output_directory = "results"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(output_directory, args.name_folder_to_save_results)

    # In case the log_dir does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])


    wandb_enabled, wandb_run = setup_wandb(args)

    test()

    if wandb_enabled and wandb_run is not None:
        wandb.finish()
