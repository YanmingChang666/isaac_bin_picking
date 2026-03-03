"""
run_ros_test.py
功能：基于 FoundationPose 的 ROS 2 位姿估计节点（测试/简化版）。

与 run_ros.py 的主要区别：
- 无触发信号（/trigger）控制，图像非空即持续运行；
- 无 TF 坐标变换，位姿保留在相机坐标系中；
- 无 PoseStamped / Float32MultiArray 位姿发布，仅发布可视化图像；
- 适合在没有完整机器人 TF 树的环境中快速验证检测效果。
"""

from estimater import *
from datareader import *
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import threading
from cv_bridge import CvBridge
from ultralytics import YOLO


# ─────────────────────────────────────────────
# 全局共享变量（由订阅节点写入，由 Estimator 读取）
# ─────────────────────────────────────────────
bridge = CvBridge()                                    # ROS Image ↔ OpenCV 互转工具

rgb_image   = np.zeros((480, 640, 3), np.uint8)        # 彩色图像缓冲区（默认黑图）
depth_image = np.zeros((480, 640), np.uint8)           # 深度图像缓冲区（默认全零）
trigger_signal = False                                 # 触发信号（本文件中定义但未使用）


# ─────────────────────────────────────────────
# 节点 1：深度图像订阅器
# ─────────────────────────────────────────────
class Depth_CameraSubscriber(Node):
    """订阅 /depth 话题，将 ROS Image 消息转换为 NumPy 深度图并存入全局变量。"""

    def __init__(self):
        super().__init__('depth_subscriber')
        # 订阅深度图话题，队列深度为 10
        self.subscription = self.create_subscription(
            Image,
            '/depth',
            self.camera_callback,
            10)
        self.subscription  # 防止订阅对象被垃圾回收

    def camera_callback(self, data):
        """深度图回调：将 ROS Image 消息转为 NumPy 数组（保留原始编码）。"""
        global depth_image
        # 'passthrough' 保留原始像素格式（通常为 float32 或 uint16，单位：米或毫米）
        depth_image = bridge.imgmsg_to_cv2(data, 'passthrough')


# ─────────────────────────────────────────────
# 节点 2：彩色图像订阅器
# ─────────────────────────────────────────────
class RGB_CameraSubscriber(Node):
    """订阅 /rgb 话题，将 ROS Image 消息转换为 BGR NumPy 图像并存入全局变量。"""

    def __init__(self):
        super().__init__('rgb_subscriber')
        # 订阅彩色图话题，队列深度为 10
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.camera_callback,
            10)
        self.subscription  # 防止订阅对象被垃圾回收

    def camera_callback(self, data):
        """彩色图回调：将 ROS Image 消息转为 BGR 格式的 NumPy 数组。"""
        global rgb_image
        # "bgr8" 对应 OpenCV 默认的 BGR 通道顺序
        rgb_image = bridge.imgmsg_to_cv2(data, "bgr8")


# ─────────────────────────────────────────────
# 节点 3：位姿估计主节点（简化版，无 TF / 触发控制）
# ─────────────────────────────────────────────
class Estimator(Node):
    """
    核心位姿估计节点（测试版），周期性执行以下逻辑：
    1. 图像非全黑时立即开始处理（无需等待触发信号）；
    2. 首帧：YOLO 分割目标 → FoundationPose.register 初始化位姿；
    3. 后续帧：FoundationPose.track_one 跟踪位姿；
    4. 在彩色图上绘制 3D 包围盒和坐标轴后发布可视化图像。

    注意：位姿结果保留在相机坐标系中，不做世界坐标系变换，
         也不发布 PoseStamped 或 Float32MultiArray。
    """

    def __init__(self, mesh, debug, debug_dir, video_dir):
        """
        初始化估计器节点。

        参数：
            mesh:      trimesh 网格对象，表示待估计目标的 3D 模型
            debug:     调试级别（0=关闭，3=保存中间文件）
            debug_dir: 调试文件输出目录
            video_dir: 场景数据目录，包含 cam_K.txt 相机内参文件
        """
        super().__init__('recorder')

        # ── 网格预处理 ──
        # 计算网格的有向包围盒变换矩阵及尺寸，用于后续绘制 3D 框
        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        # 将包围盒尺寸整理为 [[-dx/2,-dy/2,-dz/2],[dx/2,dy/2,dz/2]] 格式
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        # ── 相机内参 ──
        # 从文本文件读取 3x3 相机内参矩阵 K
        self.K = np.loadtxt(f'{video_dir}/cam_K.txt').reshape(3,3)

        # ── FoundationPose 模型初始化 ──
        scorer  = ScorePredictor()           # 位姿评分预测器
        refiner = PoseRefinePredictor()      # 位姿精细化预测器
        glctx   = dr.RasterizeCudaContext()  # CUDA 光栅化上下文（用于渲染）
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=debug_dir,
            debug=debug,
            glctx=glctx
        )
        logging.info("estimator initialization done")

        # ── 状态标志 ──
        self.first_step = True  # True 表示需要执行初始化注册（首帧）

        # ── YOLO 实例分割模型 ──
        # 加载自定义训练的分割模型权重（路径硬编码，测试版使用相对路径热身）
        self.model = YOLO(os.path.join(os.environ["HOME"], "isaac_bin_picking/yolo/best.pt"))

        # 热身推理：消除首次推理的延迟，此处使用本地测试图片，置信度阈值 0.8
        # 注意：与 run_ros.py 不同，这里使用相对路径而非环境变量拼接的绝对路径
        _ = self.model("./demo_data/bolt/rgb/0000001.png", save=False, conf=0.8)

        # ── 定时器（主处理循环）──
        timer_period = 0.1  # 定时周期：0.1 秒（约 10 Hz）
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # ── 发布器 ──
        # 发布带 3D 包围盒可视化的彩色图像（仅此一个，无位姿话题发布）
        self.pub_img = self.create_publisher(Image, '/foundation_pose_result_img', 10)

    def normalize(self, v):
        """对向量 v 进行 L2 归一化，返回单位向量。"""
        return v / np.linalg.norm(v)

    def timer_callback(self):
        """
        定时器回调（约 10 Hz），核心处理流程：
        1. 图像非全黑时立即执行（无触发信号判断）；
        2. 首帧执行 YOLO 分割 + FoundationPose 注册；
        3. 后续帧执行 FoundationPose 跟踪；
        4. 绘制可视化结果并发布；
        5. 图像全黑时打印等待提示。

        注意：此版本位姿保留在相机坐标系，不做世界坐标系变换。
        """
        global rgb_image, depth_image, trigger_signal

        # 获取当前帧的彩色图和深度图（浅拷贝引用）
        color = rgb_image
        depth = depth_image

        # ── 主处理条件：图像非全黑（无触发信号判断，与 run_ros.py 不同）──
        if np.all(rgb_image == 0) == False:
            if self.first_step:
                # ── 首帧 —— YOLO 分割获取目标掩码 ──
                # 置信度阈值 0.8（run_ros.py 使用 0.9）
                results = self.model(rgb_image, save=False, conf=0.8)

                for r in results:
                    masks = r.masks  # 实例分割掩码（可能为 None）
                    if masks is not None:
                        self.first_step = False  # 检测到目标，准备注册
                        for mask in masks:
                            # 将 GPU Tensor 转为 CPU NumPy，并整形为 (480,640,1)
                            x = mask.data.to('cpu').detach().numpy().copy()
                            bolt_mask = x.reshape(480, 640, 1)
                            # 将掩码值从 [0,1] 缩放到 [0,255]（uint8 格式）
                            mask = bolt_mask[:,:,0]*255

                if self.first_step == False:
                    # ── FoundationPose 注册（首帧位姿初始化）──
                    # register 需要相机内参、彩色图、深度图、目标掩码
                    # 返回物体在相机坐标系中的 4x4 变换矩阵
                    pose = self.est.register(K=self.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

                    # 调试级别 ≥ 3 时保存中间结果到磁盘
                    if debug>=3:
                        m = mesh.copy()
                        m.apply_transform(pose)               # 将模型变换到估计位姿
                        m.export(f'{debug_dir}/model_tf.obj') # 保存变换后的网格
                        xyz_map = depth2xyzmap(depth, self.K) # 深度图转点云 xyz
                        valid = depth>=0.001                  # 过滤无效深度点
                        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
            else:
                # ── 后续帧 —— FoundationPose 跟踪 ──
                # track_one 基于上一帧位姿进行快速轻量跟踪
                pose = self.est.track_one(rgb=color, depth=depth, K=self.K, iteration=args.track_refine_iter)

            if self.first_step == False:
                # ── 可视化：在彩色图上绘制 3D 包围盒和坐标轴 ──
                # 将位姿从物体局部包围盒坐标系变换到相机坐标系（用于正确绘制）
                center_pose = pose@np.linalg.inv(self.to_origin)
                # 绘制 3D 包围盒（黄色框）
                vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
                # 绘制 XYZ 坐标轴（scale=0.1 米，transparency=0 表示不透明）
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)

                # 将 RGB → BGR（OpenCV 默认通道）后转为 ROS Image 并发布
                img_msg = bridge.cv2_to_imgmsg(vis[...,::-1])
                self.pub_img.publish(img_msg)
        else:
            # ── 等待图像：图像全黑时打印提示（run_ros.py 中无此分支）──
            print(f"Waiting for an image...")

        # 保存上一帧触发状态（此处 trigger_signal 未参与逻辑，属于遗留代码）
        self.prev_trigger = trigger_signal


# ─────────────────────────────────────────────
# 主程序入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    rclpy.init(args=None)  # 初始化 ROS 2 通信

    # ── 命令行参数解析 ──
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))  # 脚本所在目录

    # 注释掉的选项为备用的 mustard0 目标，当前默认使用 bolt 目标
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/bolt/mesh/bolt.obj',
                        help='目标物体 3D 模型文件路径（.obj 格式）')
    #parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/bolt',
                        help='场景数据目录，需包含 cam_K.txt 相机内参文件')
    parser.add_argument('--est_refine_iter',   type=int, default=5,
                        help='首帧注册阶段的精细化迭代次数')
    parser.add_argument('--track_refine_iter', type=int, default=2,
                        help='跟踪阶段的精细化迭代次数（越小越快）')
    parser.add_argument('--debug',     type=int, default=1,
                        help='调试级别：0=关闭，3=保存中间文件')
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug',
                        help='调试文件输出目录')
    args = parser.parse_args()

    # ── 日志与随机种子 ──
    set_logging_format()  # 设置统一的日志格式
    set_seed(0)           # 固定随机种子，保证结果可复现

    # ── 加载 3D 网格模型 ──
    mesh      = trimesh.load(args.mesh_file)
    debug     = args.debug
    debug_dir = args.debug_dir
    video_dir = args.test_scene_dir

    # 清空并重建调试目录（track_vis 存跟踪可视化，ob_in_cam 存相机坐标系中的物体位姿）
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    # ── 创建 ROS 2 节点（无 Trigger 节点，与 run_ros.py 不同）──
    estimator               = Estimator(mesh, debug, debug_dir, video_dir)  # 位姿估计主节点
    rgb_camera_subscriber   = RGB_CameraSubscriber()                         # 彩色图订阅节点
    depth_camera_subscriber = Depth_CameraSubscriber()                       # 深度图订阅节点

    # ── 多线程执行器：允许多节点并行回调 ──
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(estimator)
    executor.add_node(rgb_camera_subscriber)
    executor.add_node(depth_camera_subscriber)
    # 注意：run_ros.py 还会 add_node(trigger)，此处无该节点

    # 在独立守护线程中启动执行器，避免阻塞主线程
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # ── 主线程：保持运行，等待 Ctrl+C 或 ROS 关闭信号 ──
    try:
        rate = estimator.create_rate(2)  # 主线程以 2 Hz 空转（避免 CPU 满载）
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        print("Ctrl+C pressed.")
    except rclpy.exceptions.ROSInterruptException:
        print("ROS shutdown triggered — stopping loop safely.")
    finally:
        # ── 优雅关闭：依次停止执行器、ROS、等待线程退出 ──
        executor.shutdown()
        rclpy.shutdown()
        executor_thread.join()
        print("ROS nodes stopped.")
