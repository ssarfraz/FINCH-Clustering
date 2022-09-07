import glob
import os
import time
import numpy as np
import pandas as pd
from python.read_utils import get_mapping, read_gt_label, decode_hung_labels, estimate_cost_matrix, avg_gt_activity_datasets
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from python.twfinch import FINCH
import argparse

def run_twfinch(
        dataset_name='50Salads',
        datasets_path='./Action_Segmentation_Datasets',
        tw_finch=True,
        verbose=False
):
    # setup paths
    ds_name = dataset_name
    path_ds = os.path.join(datasets_path, ds_name)
    path_gt = os.path.join(path_ds, 'groundTruth/')
    path_mapping = os.path.join(path_ds, 'mapping', 'mapping.txt')
    path_features = os.path.join(path_ds, 'features/')

    # %% Load all needed files: Descriptor, GT & Mapping
    if os.path.exists(path_mapping):
        # Create the Mapping dict
        mapping_dict = get_mapping(path_mapping)
    else:
        mapping_dict = None

    # Load all filenames from ds path        
    if ds_name == "MPII_Cooking":  # Load MPI test set files
        mpi_df = pd.read_csv(os.path.join(path_ds, 'mapping', 'sequencesTest.txt'), names=['filename'], header=None)
        mpi_df['filename'] = path_features + mpi_df['filename'] + '-cam-002.txt'
        filenames = mpi_df.filename.tolist()
    else:
        filenames = glob.glob(os.path.join(path_features, '**/*.txt'), recursive=True)

    # Create the result matrix to hold all values in the end
    results_matrix_ds = np.zeros(shape=(len(filenames), 7), dtype=object)

    # %% Loop over each file
    for file_num, cur_filename in enumerate(filenames):

        # Load the Descriptor        
        cur_desc = np.loadtxt(cur_filename, dtype='float32')

        # Load the GT_labels, map them to the corresponding ID    
        video_name = os.path.basename(cur_filename)[:-4]
        if ds_name == "MPII_Cooking":
            activity_name = video_name.split("-")[1]
        else:
            activity_name = os.path.basename(os.path.split(cur_filename)[0])

        # n_clusters for TWFINCH
        n_clusters = int(avg_gt_activity_datasets[ds_name][activity_name])
        gt_label_path = os.path.join(path_gt, video_name)
        if not os.path.exists(gt_label_path):
            gt_label_path = os.path.join(path_gt, video_name + '.txt')

        gt_labels, n_labels = read_gt_label(gt_label_path, mapping_dict=mapping_dict)

        # cluster data with Finch        
        start = time.time()
        c, num_clust, req_c = FINCH(cur_desc, req_clust=n_clusters, verbose=verbose, tw_finch=tw_finch)

        # Find best assignment through Hungarian Method
        cost_matrix = estimate_cost_matrix(gt_labels, req_c)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # decode the predicted labels
        y_pred = decode_hung_labels(req_c, col_ind)

        # Calculate the metrics (External libraries)
        mof = -cost_matrix[row_ind, col_ind].sum() / len(
            cur_desc)  # MoF accuracy can also be computed with sklearn metrics.accuracy_score
        cur_acc = metrics.accuracy_score(gt_labels, y_pred)

        f1_macro = metrics.f1_score(gt_labels, y_pred, average='macro')  # F1-Score
        # iou_macro = metrics.jaccard_score(gt_labels, y_pred, average='macro')  # IOU
        # penalize equally over/under clustering
        iou = np.sum(metrics.jaccard_score(gt_labels, y_pred, average=None)) / n_clusters
        end = time.time()
        if verbose:
            print(f'Evaluation on Video {activity_name}/{video_name} finshed in {np.round(end - start)} seconds: accuracy = {cur_acc} IoU = {iou} and f1 ={f1_macro}')

        # Transfer all the calculated metrics into the result matrix 
        results_matrix_ds[file_num][0] = activity_name + '/' + video_name
        results_matrix_ds[file_num][1] = cur_acc  # Mean over frame accuracy
        results_matrix_ds[file_num][2] = iou  # IOU
        results_matrix_ds[file_num][3] = f1_macro  # F1 from Sklearn
        results_matrix_ds[file_num][4] = n_clusters  # Number of desired clusters for TW-Finch
        results_matrix_ds[file_num][5] = n_labels  # Number of ground truth clusters
        results_matrix_ds[file_num][5] = gt_labels  # Encoded GT Data
        results_matrix_ds[file_num][6] = y_pred  # Encoded cluster labels n_clusters     

    avg_score = np.mean(results_matrix_ds[:, 1:4], axis=0)
    print(f'Overal Results on {ds_name} Dataset: MOF:{avg_score[0]}, IOU: {avg_score[1]}, F-Score: {avg_score[2]}')
    return results_matrix_ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True, help='Specify the path to your data csv file.')
    parser.add_argument('--datasets-path', required=True, help='Specify the root folder of all datsets')
    parser.add_argument('--tw-finch', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()
    results = run_twfinch(dataset_name=args.dataset_name,
                          datasets_path=args.datasets_path,
                          tw_finch=args.tw_finch,
                          verbose=args.verbose)