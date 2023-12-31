import argparse
import os
from collections import defaultdict

import time
import torch

import dataset.data_loader as data_loader
import model.net as net
from common import utils
from common.manager import Manager
from loss.loss import compute_loss, compute_metrics

import matplotlib.pyplot as plt

from common import quaternion, se3

import numpy as np

import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./experiments/experiment_omnet",
                    help="Directory containing params.json")
parser.add_argument("--restore_file", type=str, help="name of the file in --model_dir containing weights to load")


def plot_3d_points(set1, set2, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    color_set1 = 'r'
    color_set2 = 'b'
    # color_set3 = 'g'

    ax.scatter(set1[:, 0], set1[:, 1], set1[:, 2], c=color_set1, marker='o')
    ax.scatter(set2[:, 0], set2[:, 1], set2[:, 2], c=color_set2, marker='^')
    # ax.scatter(set3[:, 0], set3[:, 1], set3[:, 2], c=color_set3, marker='x')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(title)

    plt.show()


def plot_3_3d_points(set1_input, set2_input, set1_output, set2_output, set1_gt, set2_gt):
    fig = plt.figure(figsize=(30, 20))

    # Plot input
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(set1_input[:, 0], set1_input[:, 1], set1_input[:, 2], c='r', marker='o')
    ax1.scatter(set2_input[:, 0], set2_input[:, 1], set2_input[:, 2], c='b', marker='^')
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    ax1.set_title('Input')

    # Plot output
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(set1_output[:, 0], set1_output[:, 1], set1_output[:, 2], c='r', marker='o')
    ax2.scatter(set2_output[:, 0], set2_output[:, 1], set2_output[:, 2], c='b', marker='^')
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('Z Label')
    ax2.set_title('Output')

    # Plot ground truth
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(set1_gt[:, 0], set1_gt[:, 1], set1_gt[:, 2], c='r', marker='o')
    ax3.scatter(set2_gt[:, 0], set2_gt[:, 1], set2_gt[:, 2], c='b', marker='^')
    ax3.set_xlabel('X Label')
    ax3.set_ylabel('Y Label')
    ax3.set_zlabel('Z Label')
    ax3.set_title('Ground Truth')

    plt.show()


def test(model, manager):
    # set model to evaluation mode
    torch.cuda.empty_cache()
    model.eval()

    with torch.no_grad():
        # compute metrics over the dataset
        # if manager.dataloaders["val"] is not None:
        #
        #     # inference time
        #     total_time = 0.
        #     all_endpoints = defaultdict(list)
        #     # loss status and val status initial
        #     manager.reset_loss_status()
        #     manager.reset_metric_status("val")
        #
        #     # Visualization
        #     pcd = o3d.io.read_point_cloud("dataset/lander.ply")
        #     points_src = np.asarray(pcd.points).astype(np.float32)
        #     points_src = points_src - np.mean(points_src, axis=0)
        #
        #     pcd = o3d.io.read_point_cloud("dataset/seg_target_object1.ply")
        #     points_ref = np.asarray(pcd.points)
        #     points_ref = points_ref - np.mean(points_ref, axis=0)
        #
        #     theta = np.radians(78)
        #
        #     R = np.array([
        #         [np.cos(theta), -np.sin(theta), 0],
        #         [np.sin(theta), np.cos(theta), 0],
        #         [0, 0, 1]
        #     ])
        #
        #     points_ref = np.dot(points_ref, R.T).astype(np.float32)
        #
        #     # find the global max and min values
        #     max_value_global = np.max([np.abs(points_src.min()), np.abs(points_src.max()),
        #                                np.abs(points_ref.min()), np.abs(points_ref.max())])
        #
        #     # Scale the points using the global maximum value
        #     points_src = points_src / max_value_global
        #     points_ref = points_ref / max_value_global
        #
        #     # Ensure that the points are within the range of -1 and 1
        #     points_src = 2 * (points_src - points_src.min()) / (points_src.max() - points_src.min()) - 1
        #     points_ref = 2 * (points_ref - points_ref.min()) / (points_ref.max() - points_ref.min()) - 1
        #
        #     N = points_ref.shape[0]
        #     indices = np.random.choice(N, size=points_src.shape[0], replace=False)
        #
        #     points_ref = points_ref[indices, :]
        #
        #     plot_3d_points(points_ref, points_src, "Input")
        #
        #     print("hi")
        #     start_time = time.time()
        #     points_src, points_ref, src_pred_mask, ref_pred_mask, _ = model.module.my_eval(points_src, points_ref)
        #     print(time.time() - start_time)
        #     print("hi")
        #
        #
        #     plot_3d_points(points_ref, points_src, "Output")
        #
        #     # for data in manager.dataloaders["train"]:
        #     #     for i in range(5):
        #     #         points_src = data["points_src"].cpu().numpy()[i]
        #     #         points_ref = data["points_ref"].cpu().numpy()[i]
        #     #         plot_3d_points(points_ref, points_src)
        #     #
        #     #         points_src, points_ref, src_pred_mask, ref_pred_mask, _ = model.module.my_eval(points_src,
        #     #                                                                                        points_ref)
        #     #
        #     #         plot_3d_points(points_ref, points_src)
        #
        #     for batch_idx, data_batch in enumerate(manager.dataloaders["val"]):
        #         # move to GPU if available
        #         data_batch = utils.tensor_gpu(data_batch)
        #         # compute model output
        #         output_batch = model(data_batch)
        #
        #         # real batch size
        #         batch_size = data_batch["points_src"].size()[0]
        #         # compute all loss on this batch
        #         loss = compute_loss(output_batch, manager.params)
        #
        #         manager.update_loss_status(loss, batch_size)
        #         # compute all metrics on this batch
        #         metrics = compute_metrics(output_batch, manager.params)
        #         manager.update_metric_status(metrics, "val", batch_size)
        #
        #     # compute RMSE metrics
        #     manager.summarize_metric_status(metrics, "val")
        #     # For each epoch, update and print the metric
        #     manager.print_metrics("val", title="Val", color="green")

        if manager.dataloaders["test"] is not None:
            # inference time
            total_time = {"total": 0.}
            total_time_outside = 0.
            all_endpoints = defaultdict(list)
            # loss status and test status initial
            manager.reset_loss_status()
            manager.reset_metric_status("test")
            for batch_idx, data_batch in enumerate(manager.dataloaders["test"]):
                # move to GPU if available
                data_batch = utils.tensor_gpu(data_batch)
                print(data_batch['label'].cpu().numpy()[0])
                # if data_batch['label'].cpu().numpy()[0] != 40:
                #     continue
                # compute model output
                start_time = time.time()
                output_batch = model(data_batch)
                print(time.time() - start_time)

                # real batch size
                batch_size = data_batch["points_src"].size()[0]
                # compute all loss on this batch
                loss = compute_loss(output_batch, manager.params)

                for i in range(batch_size):
                    points_src_input = data_batch["points_src"][i].cpu().numpy()
                    points_ref = data_batch["points_ref"][i].cpu().numpy()
                    # plot_3d_points(points_ref, points_src, "Input")

                    for j in range(params.titer):
                        # cls loss

                        print("cls: " + str(loss['cls_{}'.format(j)].item()))
                        print("quat: " + str(loss["quat_{}".format(j)].item()))
                        print("translate: " + str(loss["translate_{}".format(j)].item()))


                    print("total: " + str(loss['total'].item()))
                    print("======================")

                    points_src_output = quaternion.torch_quat_transform(output_batch['pose_pair'][1],
                                                                        data_batch["points_src"])[i].cpu().numpy()
                    # plot_3d_points(points_ref, points_src, "Output")

                    points_src_gt = quaternion.torch_quat_transform(output_batch['pose_pair'][0],
                                                                    data_batch["points_src"])[i].cpu().numpy()
                    # plot_3d_points(points_ref, points_src, "Ground Truth")

                    plot_3_3d_points(points_ref, points_src_input, points_ref, points_src_output, points_ref,
                                   points_src_gt)

                manager.update_loss_status(loss, batch_size)
                # compute all metrics on this batch
                metrics = compute_metrics(output_batch, manager.params)
                manager.update_metric_status(metrics, "test", batch_size)

            # compute RMSE metrics
            manager.summarize_metric_status(metrics, "test")
            # For each epoch, print the metric
            manager.print_metrics("test", title="Test", color="red")


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Use GPU if available
    params.cuda = torch.cuda.is_available()
    if params.cuda:
        num_gpu = torch.cuda.device_count()
        if num_gpu > 0:
            torch.cuda.set_device(0)
        gpu_ids = ", ".join(str(i) for i in [j for j in range(num_gpu)])
        logger.info("Using GPU ids: [{}]".format(gpu_ids))
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.fetch_net(params)

    # Initial status for checkpoint manager
    manager = Manager(model=model, optimizer=None, scheduler=None, params=params, dataloaders=dataloaders,
                      logger=logger)

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    test(model, manager)
