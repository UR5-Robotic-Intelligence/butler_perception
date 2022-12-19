#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from open3d_ros_helper import open3d_ros_helper as orh
import open3d as o3d


class SegmentPCL:
  def __init__(self):
    rospy.init_node('segment_pcl', anonymous=True)
    rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pcl_callback, queue_size=1)
  
  def pcl_callback(self, msg):
    rospy.loginfo("Received PointCloud2 message")
    o3dpc = orh.rospc_to_o3dpc(msg) 
    o3d.visualization.draw_geometries([o3dpc])


if __name__ == '__main__':
  try:
    SegmentPCL()
    rospy.spin()
  except rospy.ROSInterruptException:
    pass