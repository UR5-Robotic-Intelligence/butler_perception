#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2, Image
from open3d_ros_helper import open3d_ros_helper as orh
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image as PILImage
from perception.laptop_perception_helpers import RealsenseHelpers
import cv2
from ros_numpy import numpify
import clip
import torch


class PCLProcessor:
  def __init__(self):
    rospy.init_node('segment_pcl', anonymous=True)
    # rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pcl_callback, queue_size=1)
    self.rs_helpers = RealsenseHelpers()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and image preprocessing
    self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
  
  def segment_pcl(self, visualize=False):
    msg = rospy.wait_for_message("/camera/depth/color/points", PointCloud2)
    rospy.loginfo("Received PointCloud2 message")
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
      msg = rospy.wait_for_message("/camera/color/image_raw", Image)
      image_np = numpify(msg)
      image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
      if visualize:
        cv2.imshow("image", image_np[min(pixels[1]):max(pixels[1]), min(pixels[0]):max(pixels[0])])
        cv2.waitKey(0)
    return objects_boundaries, image_np
  
  def find_object(self, object_names):
      objects_on_table_roi, image_np = self.segment_pcl(visualize=False)
      # image = PILImage.fromarray(np.uint8(image_np)*255)
      objects_images = []
      for object_roi in objects_on_table_roi:
          obj_image = PILImage.fromarray(image_np[object_roi[0][0]-10:object_roi[1][0]+10, object_roi[0][1]-10:object_roi[1][1]+10])
          cv2.imshow("image", np.array(obj_image))
          cv2.waitKey(0)
          objects_images.append(obj_image)

      text_snippets = ["a photo of a {}".format(name) for name in object_names]
      text_snippets.append("a photo of something else")
      # pre-process text
      text = clip.tokenize(text_snippets).to(self.device)
      
      # with torch.no_grad():
      #     text_features = model.encode_text(text)
      detected_objects = []
      for i, object_image in enumerate(objects_images):
          # pre-process image
          prepro_image = self.preprocess(object_image).unsqueeze(0).to(self.device)
          
          # with torch.no_grad():
          #     image_features = model.encode_image(prepro_image)
          
          with torch.no_grad():
              logits_per_image, logits_per_text = self.model(prepro_image, text)
              probs = logits_per_image.softmax(dim=-1).cpu().numpy()
          # print("Label probs:", ["{0:.10f}".format(i) for i in probs[0]])
          obj_idx = np.argmax(probs[0])
          print(obj_idx)
          if (obj_idx < len(object_names)) and (obj_idx != len(object_names) - 1):
              print("Object {} is {}".format(i, object_names[obj_idx]))
              detected_objects.append((obj_idx, objects_on_table_roi[i]))
      
      for i, detected_object in detected_objects:
          # print("Object {} is {}".format(i, object_names[i]))
          if detected_object is not None:
              cv2.rectangle(image_np, (detected_object[0][1], detected_object[0][0]), (detected_object[1][1], detected_object[1][0]), (0, 255, 0), 2)
              cv2.putText(image_np, object_names[i], (detected_object[0][1], detected_object[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
      cv2.imshow("image", image_np)
      cv2.waitKey(0)
      return detected_objects


if __name__ == '__main__':
  try:
    pcl_processor = PCLProcessor()
    pcl_processor.find_object(object_names=["cup", "container cover", "tea packet", "headset cover", "computer mouse"])
  except rospy.ROSInterruptException:
    pass