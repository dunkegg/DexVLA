from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image

def ros_image_to_cv2(msg: Image) -> np.ndarray:
    """
    将 ROS Image 消息转换为 OpenCV 图像 (numpy array)
    只支持常见编码：'rgb8', 'bgr8', 'mono8'
    """
    if msg.encoding == 'rgb8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        # OpenCV 默认是 BGR，如果需要 BGR 可交换通道
        img = img[:, :, ::-1]  # RGB -> BGR
    elif msg.encoding == 'bgr8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    elif msg.encoding == 'mono8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    return img
class ImageSubscriber:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/realsense_head/color/image_raw", Image, self.image_callback)
 
    def image_callback(self, data):

        cv_image = ros_image_to_cv2(data)

 
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)
 
if __name__ == '__main__':
    rospy.init_node('image_subscriber')
    image_subscriber = ImageSubscriber()
    rospy.spin()