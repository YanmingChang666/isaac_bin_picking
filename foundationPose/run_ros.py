"""
run_ros.py
功能：基于 FoundationPose 的 ROS 2 六自由度位姿估计节点。

主要流程：
1. 通过 /rgb 和 /depth 话题订阅相机图像；
2. 通过 /trigger 话题接收触发信号；
3. 触发后使用 YOLO 实例分割获取目标掩码；
4. 首帧用 FoundationPose.register 初始化位姿，后续帧用 track_one 跟踪；
5. 将位姿从相机坐标系变换到世界坐标系，并发布到 /target_bolt_pose 和 /pos_angle。
"""

from estimater import *
from datareader import *
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
# from tf2_msgs.msg import TFMessage  # 备用，当前通过 tf2_ros 直接获取 TF
from tf2_ros import Buffer, TransformListener
import threading
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from math import pi


# ─────────────────────────────────────────────
# 全局共享变量（由各订阅节点写入，由 Estimator 读取）
# ─────────────────────────────────────────────
bridge = CvBridge()                                    # ROS Image ↔ OpenCV 互转工具

rgb_image   = np.zeros((480, 640, 3), np.uint8)        # 彩色图像缓冲区（默认黑图）
depth_image = np.zeros((480, 640), np.uint8)           # 深度图像缓冲区（默认全零）
trigger_signal = False                                 # 位姿估计触发信号（True 时开始处理）


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
# 节点 3：触发信号订阅器
# ─────────────────────────────────────────────
class Trigger(Node):
    """订阅 /trigger 话题，将布尔触发信号存入全局变量以控制位姿估计的启停。"""

    def __init__(self):
        super().__init__('trigger')
        # 订阅触发话题，队列深度为 10
        self.subscription = self.create_subscription(
            Bool,
            '/trigger',
            self.trigger_callback,
            10)
        self.subscription  # 防止订阅对象被垃圾回收

    def trigger_callback(self, data):
        """触发信号回调：True 表示开始位姿估计，False 表示停止。"""
        global trigger_signal
        trigger_signal = data.data


# ─────────────────────────────────────────────
# 节点 4：位姿估计主节点
# ─────────────────────────────────────────────
class Estimator(Node):
    """
    核心位姿估计节点，周期性执行以下逻辑：
    1. 收到触发信号后，首次获取相机 → 世界坐标系的静态 TF；
    2. 首帧：YOLO 分割目标 → FoundationPose.register 初始化位姿；
    3. 后续帧：FoundationPose.track_one 跟踪位姿；
    4. 将位姿变换到世界坐标系并发布。
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
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        # ── 相机内参 ──
        # 从文本文件读取 3x3 相机内参矩阵 K
        self.K = np.loadtxt(f'{video_dir}/cam_K.txt').reshape(3, 3)

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
        self.first_step    = True   # True 表示需要执行初始化注册（首帧）
        self.get_camera_tf = False  # True 表示已成功获取相机 TF

        # ── YOLO 实例分割模型 ──
        yolo_dir = os.path.join(os.environ["HOME"], "isaac_bin_picking/yolo")
        # 加载自定义训练的分割模型权重
        self.model = YOLO(os.path.join(yolo_dir, "best.pt"))
        # 热身推理：消除首次推理的延迟，置信度阈值 0.9
        _ = self.model(os.path.join(yolo_dir, "data_for_test/img14.jpg"), save=False, conf=0.9)

        # ── 定时器（主处理循环）──
        timer_period = 0.1  # 定时周期：0.1 秒（约 10 Hz）
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # ── 发布器 ──
        # 发布带 3D 包围盒可视化的彩色图像
        self.pub_img = self.create_publisher(Image, '/foundation_pose_result_img', 10)
        # 发布目标螺栓在世界坐标系中的 PoseStamped
        self.pub_target_bolt_pose = self.create_publisher(PoseStamped, '/target_bolt_pose', 10)
        # 发布位置与校正角度：[x, y, z, corrected_angle]
        self.pos_angle = self.create_publisher(Float32MultiArray, "/pos_angle", 10)

        # ── TF 监听器 ──
        self.tf_buffer   = Buffer()                              # TF 数据缓冲
        self.tf_listener = TransformListener(self.tf_buffer, self)  # 自动订阅 /tf

        # 相机到世界坐标系的齐次变换矩阵（World ← Camera），初始为单位阵
        self.T_wc = np.eye(4)

        # ── 目标位姿初始化（世界坐标系）──
        self.target_bolt_pose = PoseStamped()
        self.target_bolt_pose.header.frame_id = 'World'
        # 初始位置设为原点
        self.target_bolt_pose.pose.position.x    = 0.0
        self.target_bolt_pose.pose.position.y    = 0.0
        self.target_bolt_pose.pose.position.z    = 0.0
        # 初始姿态设为单位四元数（无旋转）
        self.target_bolt_pose.pose.orientation.x = 0.0
        self.target_bolt_pose.pose.orientation.y = 0.0
        self.target_bolt_pose.pose.orientation.z = 0.0
        self.target_bolt_pose.pose.orientation.w = 1.0

        # 上一帧的触发信号状态（用于检测上升沿，即重新触发时重置首帧标志）
        self.prev_trigger = False

    def normalize(self, v):
        """对向量 v 进行 L2 归一化，返回单位向量。"""
        return v / np.linalg.norm(v)

    def timer_callback(self):
        """
        定时器回调（约 10 Hz），核心处理流程：
        1. 检查图像非空且触发信号为 True；
        2. 首次触发时通过 TF 树获取相机外参；
        3. 首帧执行 YOLO 分割 + FoundationPose 注册；
        4. 后续帧执行 FoundationPose 跟踪；
        5. 将结果变换到世界坐标系并发布；
        6. 检测触发信号上升沿，重置首帧标志。
        """
        global rgb_image, depth_image, trigger_signal

        # 获取当前帧的彩色图和深度图（浅拷贝引用，保证线程安全的读取）
        color = rgb_image
        depth = depth_image

        # ── 主处理条件：图像非全黑 且 触发信号为 True ──
        if np.all(rgb_image == 0) == False and trigger_signal == True:

            # ── 阶段 A：首次获取相机 TF（静态，只需查询一次）──
            if self.get_camera_tf == False:
                try:
                    # 查询 World → Camera 的坐标变换（使用最新可用时刻）
                    tf = self.tf_buffer.lookup_transform(
                        'World',   # 目标坐标系（父系）
                        'Camera',  # 源坐标系（子系）
                        rclpy.time.Time()  # 时间戳为 0 表示使用最新变换
                    )

                    pos = tf.transform.translation  # 平移分量
                    rot = tf.transform.rotation     # 旋转分量（四元数）

                    # 将四元数转换为旋转矩阵
                    t    = np.array([pos.x, pos.y, pos.z])
                    q    = np.array([rot.x, rot.y, rot.z, rot.w])
                    R_wc = R.from_quat(q).as_matrix()

                    # 组装 4x4 齐次变换矩阵 T_wc（World ← Camera）
                    self.T_wc[:3, :3] = R_wc
                    self.T_wc[:3, 3]  = t

                    self.get_logger().info('\033[32m' + "Got static TF once" + '\033[0m')
                    self.get_camera_tf = True  # 标记已获取，后续不再查询

                except Exception:
                    # TF 尚未准备好，打印警告并等待下一帧重试
                    self.get_logger().info('\033[31m' + "Could not get TF." + '\033[0m')

            else:
                # ── 阶段 B：已有 TF，执行位姿估计 ──

                if self.first_step:
                    # ── B-1：首帧 —— YOLO 分割获取目标掩码 ──
                    # 对当前彩色图运行 YOLO，置信度阈值 0.9，不保存结果图
                    results = self.model(rgb_image, save=False, conf=0.9)

                    for r in results:
                        masks = r.masks  # 实例分割掩码（可能为 None）
                        if masks is not None:
                            self.first_step = False  # 检测到目标，准备注册
                            for mask in masks:
                                # 将 GPU Tensor 转为 CPU NumPy，并整形为 (480,640,1)
                                x = mask.data.to('cpu').detach().numpy().copy()
                                bolt_mask = x.reshape(480, 640, 1)
                                # 将掩码值从 [0,1] 缩放到 [0,255]（uint8 格式）
                                mask = bolt_mask[:, :, 0] * 255

                    if self.first_step == False:
                        # ── B-2：FoundationPose 注册（首帧位姿初始化）──
                        # register 需要相机内参、彩色图、深度图、目标掩码
                        # 返回物体在相机坐标系中的 4x4 变换矩阵
                        pose = self.est.register(
                            K=self.K,
                            rgb=color,
                            depth=depth,
                            ob_mask=mask,
                            iteration=args.est_refine_iter  # 精细化迭代次数
                        )

                        # 调试级别 ≥ 3 时保存中间结果到磁盘
                        if debug >= 3:
                            m = mesh.copy()
                            m.apply_transform(pose)            # 将模型变换到估计位姿
                            m.export(f'{debug_dir}/model_tf.obj')  # 保存变换后的网格
                            xyz_map = depth2xyzmap(depth, self.K)  # 深度图转点云 xyz
                            valid   = depth >= 0.001               # 过滤无效深度点
                            pcd     = toOpen3dCloud(xyz_map[valid], color[valid])
                            o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
                else:
                    # ── B-3：后续帧 —— FoundationPose 跟踪 ──
                    # track_one 基于上一帧位姿进行快速轻量跟踪
                    pose = self.est.track_one(
                        rgb=color,
                        depth=depth,
                        K=self.K,
                        iteration=args.track_refine_iter  # 跟踪精细化迭代次数
                    )

                # ── 阶段 C：坐标系变换 + 结果发布 ──
                if self.first_step == False:

                    # 将物体位姿从相机坐标系变换到世界坐标系
                    # T_WO = T_wc（World←Camera）× pose（Camera←Object）
                    T_WO = self.T_wc @ pose

                    # 从世界坐标系旋转矩阵提取四元数（scipy 返回 [x,y,z,w]）
                    bolt_quat = R.from_matrix(T_WO[:3, :3]).as_quat()

                    # ── 发布 PoseStamped（目标在世界坐标系中的位姿）──
                    self.target_bolt_pose.header.stamp     = self.get_clock().now().to_msg()
                    self.target_bolt_pose.pose.position.x  = T_WO[:3, 3][0]
                    self.target_bolt_pose.pose.position.y  = T_WO[:3, 3][1]
                    self.target_bolt_pose.pose.position.z  = T_WO[:3, 3][2]
                    self.target_bolt_pose.pose.orientation.x = bolt_quat[0]
                    self.target_bolt_pose.pose.orientation.y = bolt_quat[1]
                    self.target_bolt_pose.pose.orientation.z = bolt_quat[2]
                    self.target_bolt_pose.pose.orientation.w = bolt_quat[3]
                    self.pub_target_bolt_pose.publish(self.target_bolt_pose)

                    # ── 计算螺栓在地面投影方向角 ──
                    rot_mat = T_WO[:3, :3]
                    # 取物体 Z 轴在世界坐标系中的方向（即螺栓轴方向）
                    bolt_axis_world = rot_mat[:, 2]
                    world_up = np.array([0, 0, 1])          # 世界 Z 轴（竖直向上）
                    # 将螺栓轴投影到水平面（去除竖直分量）
                    bolt_xy = bolt_axis_world - np.dot(bolt_axis_world, world_up) * world_up
                    bolt_xy = self.normalize(bolt_xy)        # 单位化水平投影向量
                    # 计算水平投影向量相对于世界 X 轴的方位角（弧度，范围 [-π, π]）
                    bolt_angle = np.arctan2(bolt_xy[1], bolt_xy[0])

                    # ── 角度区间映射（将任意方位角映射到抓取所需的等效角度范围）──
                    # 注：注释掉的旧方法已替换为以下分段线性映射
                    '''
                    if bolt_angle <= 0:
                        b = bolt_angle + pi
                    else:
                        b = bolt_angle - pi
                    if -pi <= b < -pi/4:
                        corrected_bolt_angle = b + 5*pi/4
                    else:
                        corrected_bolt_angle = b - 3*pi/4
                    '''
                    # 当前映射：将 [-2π, 2π) 的方位角映射到 [-3π/4, 5π/4) 区间
                    if -2*pi <= bolt_angle < -5*pi/4:
                        corrected_bolt_angle = bolt_angle + 9*pi/4
                    elif -5*pi/4 <= bolt_angle < 3*pi/4:
                        corrected_bolt_angle = bolt_angle + pi/4
                    else:
                        corrected_bolt_angle = bolt_angle - 7*pi/4

                    # ── 发布位置与校正角度：[x, y, z, corrected_angle] ──
                    array_forPublish = Float32MultiArray(
                        data=[T_WO[:3, 3][0], T_WO[:3, 3][1], T_WO[:3, 3][2], corrected_bolt_angle]
                    )
                    self.pos_angle.publish(array_forPublish)

                    # ── 可视化：在彩色图上绘制 3D 包围盒和坐标轴 ──
                    # 将位姿从物体局部包围盒坐标系变换到相机坐标系（用于正确绘制）
                    center_pose = pose @ np.linalg.inv(self.to_origin)
                    # 绘制 3D 包围盒（黄色框）
                    vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
                    # 绘制 XYZ 坐标轴（scale=0.1 米，透明度 0 表示不透明）
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1,
                                        K=self.K, thickness=3, transparency=0, is_input_rgb=True)

                    # 将 RGB → BGR（OpenCV 默认通道）后转为 ROS Image 并发布
                    img_msg = bridge.cv2_to_imgmsg(vis[..., ::-1])
                    self.pub_img.publish(img_msg)

        # ── 检测触发信号上升沿（False → True）：重置为首帧模式 ──
        # 当新的一次触发到来时，需要重新运行 YOLO 分割和 register 初始化
        if self.prev_trigger == False and trigger_signal == True:
            self.first_step = True

        # 更新上一帧触发状态
        self.prev_trigger = trigger_signal


# ─────────────────────────────────────────────
# 主程序入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    rclpy.init(args=None)  # 初始化 ROS 2 通信

    # ── 命令行参数解析 ──
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))  # 脚本所在目录

    parser.add_argument('--mesh_file',        type=str, default=f'{code_dir}/demo_data/bolt/mesh/bolt.obj',
                        help='目标物体 3D 模型文件路径（.obj 格式）')
    parser.add_argument('--test_scene_dir',   type=str, default=f'{code_dir}/demo_data/bolt',
                        help='场景数据目录，需包含 cam_K.txt 相机内参文件')
    parser.add_argument('--est_refine_iter',  type=int, default=5,
                        help='首帧注册阶段的精细化迭代次数')
    parser.add_argument('--track_refine_iter',type=int, default=2,
                        help='跟踪阶段的精细化迭代次数（越小越快）')
    parser.add_argument('--debug',            type=int, default=1,
                        help='调试级别：0=关闭，3=保存中间文件')
    parser.add_argument('--debug_dir',        type=str, default=f'{code_dir}/debug',
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

    # ── 创建各 ROS 2 节点 ──
    estimator               = Estimator(mesh, debug, debug_dir, video_dir)  # 位姿估计主节点
    rgb_camera_subscriber   = RGB_CameraSubscriber()                         # 彩色图订阅节点
    depth_camera_subscriber = Depth_CameraSubscriber()                       # 深度图订阅节点
    trigger                 = Trigger()                                       # 触发信号订阅节点

    # ── 多线程执行器：允许多节点并行回调 ──
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(estimator)
    executor.add_node(rgb_camera_subscriber)
    executor.add_node(depth_camera_subscriber)
    executor.add_node(trigger)

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
