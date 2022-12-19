#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2, Image
from open3d_ros_helper import open3d_ros_helper as orh
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from perception.laptop_perception_helpers import RealsenseHelpers
import cv2
from ros_numpy import numpify


class PCLProcessor:
  def __init__(self):
    rospy.init_node('segment_pcl', anonymous=True)
    # rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pcl_callback, queue_size=1)
    self.rs_helpers = RealsenseHelpers()
  
  def segment_pcl(self, visualize=False):
    rospy.loginfo("Received PointCloud2 message")
    msg = rospy.wait_for_message("/camera/depth/color/points", PointCloud2)
    pcd = orh.rospc_to_o3dpc(msg) 
    np_points = np.asarray(pcd.points)
    x_cond = np.logical_and(np_points[:, 0] >= 1.58-2, np_points[:, 0] <= 2.212-2)
    z_cond = np_points[:, 2] <= 1.36
    filtered_np_points = np.where(np.logical_and(x_cond, z_cond))
    pcd.points = o3d.utility.Vector3dVector(np_points[filtered_np_points])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.0047, ransac_n=3, num_iterations=1000)
    # print(plane_model)
    plane_cloud = pcd.select_by_index(inliers)
    # plane_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
    objects_cloud = pcd.select_by_index(inliers, invert=True)
    plane_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    if visualize:
      o3d.visualization.draw_geometries([plane_cloud, objects_cloud])
    labels = np.array(objects_cloud.cluster_dbscan(eps=0.008, min_points=5))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label 
    if max_label > 0 else 1))
    colors[labels < 0] = 0
    objects_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    if visualize:
      o3d.visualization.draw_geometries([objects_cloud])
    objects_boundaries = []
    for i in range(max_label+1):
      label_indices = np.where(labels == i)[0]
      if len(label_indices) < 60:
        continue
      cluster = objects_cloud.select_by_index(label_indices)
      if visualize:
        o3d.visualization.draw_geometries([cluster])
      points = np.asarray(cluster.points).T
      intrinsics = self.rs_helpers.get_intrinsics(self.rs_helpers.color_intrin_topic)
      extrinsics = self.rs_helpers.get_depth_to_color_extrinsics()
      pixels = self.rs_helpers.calculate_pixels_from_points(points, intrinsics, cam_to_cam_extrinsics=extrinsics)
      objects_boundaries.append([(min(pixels[1]), min(pixels[0])), (max(pixels[1]), max(pixels[0]))])
      if visualize:
        msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        image_np = numpify(msg)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        cv2.imshow("image", image_np[min(pixels[1]):max(pixels[1]), min(pixels[0]):max(pixels[0])])
        cv2.waitKey(0)
    return objects_boundaries


if __name__ == '__main__':
  try:
    pcl_processor = PCLProcessor()
    pcl_processor.segment_pcl(visualize=True)
  except rospy.ROSInterruptException:
    pass