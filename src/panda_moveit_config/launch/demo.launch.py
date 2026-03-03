import os
# launch 核心模块：用于构建 ROS 2 启动描述
from launch import LaunchDescription
# 声明启动参数的动作类
from launch.actions import DeclareLaunchArgument
# 启动配置读取器 和 路径拼接替换器
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
# 条件判断：当条件为真时才启动对应节点
from launch.conditions import IfCondition
# ROS 2 节点启动动作
from launch_ros.actions import Node
# 查找 ROS 2 功能包共享目录的替换器
from launch_ros.substitutions import FindPackageShare
# Python 接口：获取功能包的共享目录路径
from ament_index_python.packages import get_package_share_directory
# MoveIt 配置构建工具
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """
    生成 Panda 机械臂 MoveIt 演示的启动描述。

    该启动文件会同时启动以下组件：
      - MoveIt move_group 规划服务器
      - RViz 可视化界面
      - 静态 TF 变换发布器（world -> panda_link0）
      - robot_state_publisher（发布机器人 TF 树）
      - ros2_control 控制器管理器
      - joint_state_broadcaster（关节状态广播器）
      - panda_arm_controller（机械臂关节轨迹控制器）
      - panda_hand_controller（夹爪控制器）
      - MongoDB 仓库服务（可选，受 db 参数控制）
    """

    # ──────────────────────────────────────────────────────────────
    # 启动参数声明
    # ──────────────────────────────────────────────────────────────

    # RViz 配置文件名参数，默认使用 moveit.rviz
    # 启动时可通过 rviz_config:=<文件名> 覆盖
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="moveit.rviz",
        description="RViz configuration file",
    )

    # 获取 panda_moveit_config 功能包的共享目录绝对路径
    # 用于后续拼接配置文件路径
    arm_robot_sim_path = os.path.join(
        get_package_share_directory('panda_moveit_config'))

    # 是否启动 MongoDB 仓库数据库的开关参数，默认关闭
    # 启用后可持久化存储运动规划结果（运动学数据库、机器人状态等）
    db_arg = DeclareLaunchArgument(
        "db", default_value="False", description="Database flag"
    )

    # ros2_control 硬件接口类型参数
    # mock_components：使用虚假硬件（仿真/离线测试）
    # isaac：对接 NVIDIA Isaac Sim 仿真器
    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="mock_components",
        description="ROS 2 control hardware interface type to use for the launch file -- possible values: [mock_components, isaac]",
    )

    # ──────────────────────────────────────────────────────────────
    # MoveIt 配置构建
    # 使用 MoveItConfigsBuilder 链式 API 加载所有 MoveIt 相关配置
    # ──────────────────────────────────────────────────────────────
    moveit_config = (
        MoveItConfigsBuilder("panda")
        # 加载 URDF/Xacro 机器人描述文件，并传入硬件类型参数
        # xacro 文件内部会根据该参数切换 ros2_control 硬件插件
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={
                "ros2_control_hardware_type": LaunchConfiguration(
                    "ros2_control_hardware_type"
                )
            },
        )
        # 加载 SRDF 语义机器人描述（规划组、末端执行器、碰撞矩阵等）
        .robot_description_semantic(file_path="config/panda.srdf")
        # 配置规划场景监视器，向外发布机器人描述话题供其他节点订阅
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        # 加载轨迹执行配置（绑定 MoveIt 控制器与 ros2_control 控制器的映射关系）
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        # 加载 MoveItCpp 额外设置（如规划器参数、执行超时等）
        .moveit_cpp(arm_robot_sim_path + "/config/controller_setting.yaml")
        # 构建并返回最终的 MoveIt 配置对象
        .to_moveit_configs()
    )

    # ──────────────────────────────────────────────────────────────
    # move_group 节点：MoveIt 规划与执行的核心服务
    # 提供 MoveGroup Action、规划场景管理、运动规划等功能
    # ──────────────────────────────────────────────────────────────
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",                          # 日志输出到终端
        parameters=[moveit_config.to_dict()],     # 传入完整 MoveIt 配置字典
        arguments=["--ros-args", "--log-level", "info"],  # 设置日志级别为 info
    )

    # ──────────────────────────────────────────────────────────────
    # RViz 可视化节点
    # 加载指定配置文件并传入机器人描述、规划管线、运动学参数等
    # ──────────────────────────────────────────────────────────────

    # 从启动参数中读取 RViz 配置文件名
    rviz_base = LaunchConfiguration("rviz_config")
    # 拼接完整路径：<panda_moveit_config 包>/launch/<rviz_config>
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("panda_moveit_config"), "launch", rviz_base]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",                             # 日志输出到文件（不打印到终端）
        arguments=["-d", rviz_config],            # 指定 RViz 配置文件
        parameters=[
            moveit_config.robot_description,          # URDF 机器人描述
            moveit_config.robot_description_semantic, # SRDF 语义描述
            moveit_config.planning_pipelines,         # 规划管线（OMPL 等）配置
            moveit_config.robot_description_kinematics,  # 运动学插件配置（KDL/IKFast）
            moveit_config.joint_limits,               # 关节限位配置
        ],
    )

    # ──────────────────────────────────────────────────────────────
    # 静态 TF 变换发布器
    # 将 world 坐标系与机器人基座 panda_link0 绑定（平移/旋转均为零）
    # 告知系统机器人固定于世界原点
    # ──────────────────────────────────────────────────────────────
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        # 参数顺序：x y z roll pitch yaw 父坐标系 子坐标系
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "panda_link0"],
    )

    # ──────────────────────────────────────────────────────────────
    # robot_state_publisher 节点
    # 订阅 /joint_states 话题，结合 URDF 计算并发布完整的 TF 树
    # RViz、MoveIt 等组件依赖此节点获取实时关节位置
    # ──────────────────────────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",                            # 同时输出到终端和日志文件
        parameters=[moveit_config.robot_description],  # 传入 URDF 机器人描述
    )

    # ──────────────────────────────────────────────────────────────
    # ros2_control 控制器管理器节点
    # 根据硬件类型加载相应的硬件接口插件，统一管理所有控制器的生命周期
    # ──────────────────────────────────────────────────────────────

    # 获取 ros2_controllers.yaml 配置文件的绝对路径
    # 该文件定义了控制器类型、关节名称、PID 参数等
    ros2_controllers_path = os.path.join(
        get_package_share_directory("panda_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[
            # 将控制器管理器的机器人描述话题重映射到全局 /robot_description
            # 保证控制器管理器与 robot_state_publisher 使用同一份 URDF
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="screen",
    )

    # ──────────────────────────────────────────────────────────────
    # 控制器 Spawner 节点（由 controller_manager 动态加载）
    # ──────────────────────────────────────────────────────────────

    # 关节状态广播器：读取 ros2_control 硬件接口的关节状态
    # 并发布到 /joint_states 话题，供 robot_state_publisher 消费
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    # Panda 机械臂关节轨迹控制器
    # 接收 MoveIt 发送的 JointTrajectory 目标并驱动 7 个关节运动
    panda_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"],
    )

    # Panda 夹爪控制器
    # 控制夹爪的开合动作（panda_finger_joint1、panda_finger_joint2）
    panda_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_hand_controller", "-c", "/controller_manager"],
    )

    # ──────────────────────────────────────────────────────────────
    # MongoDB 仓库服务（可选）
    # 当 db:=True 时启动，用于持久化存储机器人状态、规划场景等数据
    # 端口 33829，连接本地 MongoDB 实例
    # ──────────────────────────────────────────────────────────────
    db_config = LaunchConfiguration("db")
    mongodb_server_node = Node(
        package="warehouse_ros_mongo",
        executable="mongo_wrapper_ros.py",
        parameters=[
            {"warehouse_port": 33829},            # MongoDB 服务端口
            {"warehouse_host": "localhost"},       # MongoDB 主机地址
            {"warehouse_plugin": "warehouse_ros_mongo::MongoDatabaseConnection"},
        ],
        output="screen",
        condition=IfCondition(db_config),         # 仅当 db:=True 时才启动此节点
    )

    # ──────────────────────────────────────────────────────────────
    # 组装并返回完整的启动描述
    # 节点启动顺序说明：
    #   1. 参数声明（rviz_config_arg、db_arg、ros2_control_hardware_type）
    #   2. RViz 可视化界面
    #   3. 静态 TF 变换（world -> panda_link0）
    #   4. robot_state_publisher（发布动态 TF 树）
    #   5. move_group（MoveIt 规划服务器）
    #   6. ros2_control_node（硬件接口 + 控制器管理器）
    #   7. joint_state_broadcaster（关节状态广播）
    #   8. panda_arm_controller（机械臂控制器）
    #   9. panda_hand_controller（夹爪控制器）
    #  10. mongodb_server_node（数据库，可选）
    # ──────────────────────────────────────────────────────────────
    return LaunchDescription(
        [
            rviz_config_arg,
            db_arg,
            ros2_control_hardware_type,
            rviz_node,
            static_tf_node,
            robot_state_publisher,
            move_group_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            panda_arm_controller_spawner,
            panda_hand_controller_spawner,
            mongodb_server_node,
        ]
    )
