#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from open3d_ros_helper import open3d_ros_helper as orh
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# from perception.laptop_perception_helpers import RealsenseHelpers, constrain_environment


class SegmentPCL:
  def __init__(self):
    rospy.init_node('segment_pcl', anonymous=True)
    rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pcl_callback, queue_size=1)
  
  def pcl_callback(self, msg):
    rospy.loginfo("Received PointCloud2 message")
    pcd = orh.rospc_to_o3dpc(msg) 
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    print(plane_model)
    plane_cloud = pcd.select_by_index(inliers)
    plane_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
    objects_cloud = pcd.select_by_index(inliers, invert=True)
    plane_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([plane_cloud, objects_cloud])
    labels = np.array(objects_cloud.cluster_dbscan(eps=0.005, min_points=5))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label 
    if max_label > 0 else 1))
    colors[labels < 0] = 0
    objects_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([objects_cloud])
    print(objects_cloud)


if __name__ == '__main__':
  try:
    SegmentPCL()
    rospy.spin()
  except rospy.ROSInterruptException:
    pass