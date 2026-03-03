/******************************************************************************
 * 程序名称: bin_picking.cpp (螺栓抓取与放置系统)
 * 功能描述: 基于ROS2和MoveIt2的Panda机械臂自动化抓取系统
 * 主要功能:
 *   1. 接收外部视觉系统提供的螺栓位姿和角度信息
 *   2. 控制Panda机械臂和夹爪执行抓取动作
 *   3. 将螺栓按序放置到指定位置(螺栓支架上)
 *   4. 添加碰撞检测网格(螺栓支架和料盘)
 *   5. 实现完整的10步抓取-放置循环流程
 ******************************************************************************/
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <std_msgs/msg/bool.hpp>
#include "std_msgs/msg/float32_multi_array.hpp"
#include <shape_msgs/msg/mesh.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shapes.h>
#include <array>

#define PI 3.14159265359

// ANSI escape codes
// ANSI颜色代码，用于终端输出带颜色的日志信息
#define COLOR_RED     "\033[31m"      // 红色
#define COLOR_GREEN   "\033[32m"      // 绿色
#define COLOR_YELLOW  "\033[33m"      // 黄色
#define COLOR_BLUE    "\033[34m"      // 蓝色
#define COLOR_MAGENTA "\033[35m"      // 品红色
#define COLOR_CYAN    "\033[36m"      // 青色
#define COLOR_RESET   "\033[0m"       // 重置颜色

/******************************************************************************
 * 类名: ControlRobot
 * 描述: 机器人控制类，继承自rclcpp::Node
 * 功能: 
 *   - 订阅视觉系统发布的螺栓位姿信息
 *   - 控制机械臂执行完整的抓取-放置序列
 *   - 管理碰撞检测场景
 *   - 控制夹爪开闭
 ******************************************************************************/
class ControlRobot : public rclcpp::Node {
public:
    // ========== 成员变量：位姿和状态信息 ==========
    geometry_msgs::msg::PoseStamped current_pose;           // 当前末端执行器位姿
    geometry_msgs::msg::Pose target_pose;                   // 目标位姿
    geometry_msgs::msg::Quaternion q_current;               // 当前四元数姿态
    sensor_msgs::msg::JointState current_joint_state_;      // 当前关节状态
    std_msgs::msg::Bool msg;                                // 布尔消息(用于触发信号)

    // ========== 状态控制变量 ==========
    int sequence = 0;                                       // 当前执行序列号(0-10步)
    bool planning = false;                                  // 规划状态标志
    bool bolt_pose_received_ = false;                       // 螺栓位姿接收标志
    std::array<float, 4> bolt_pose_angle_;                  // 螺栓位姿和角度 [x, y, z, angle]
    bool bolt_pose_angle_received_ = false;                 // 位姿角度数据接收标志
    double roll_current, pitch_current, yaw_current;        // 当前欧拉角
    size_t i = 0;                                           // 当前放置位置索引

    // ========== 螺栓放置位置数组 ==========
    // 定义3个螺栓的目标放置坐标 (x, y, z)，位于螺栓支架上
    std::array<std::array<double, 3>, 3> bolt_places = {{
        {0.3, -0.24, 0.05},     // 第1个螺栓位置
        {0.3, -0.30, 0.05},     // 第2个螺栓位置
        {0.3, -0.36, 0.05}      // 第3个螺栓位置
    }};

    /******************************************************************************
     * 构造函数: ControlRobot
     * 功能: 初始化节点、创建订阅器、发布器、定时器和回调组
     ******************************************************************************/
    ControlRobot()
        : Node("robot_controller")
    {
        // ========== 创建回调组 ==========
        // 使用可重入回调组允许多个回调函数并发执行
        pose_angle_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::Reentrant
        );

        timer_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::Reentrant
        );

        // ========== 位姿角度订阅器 ==========
        // 订阅话题: /pos_angle
        // 消息类型: Float32MultiArray [x, y, z, angle]
        // 来源: 外部视觉系统(如FoundationPose)提供的螺栓6D位姿
        rclcpp::SubscriptionOptions pose_angle_options;
        pose_angle_options.callback_group = pose_angle_group_;

        pos_angle_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/pos_angle", 10,
            std::bind(&ControlRobot::pose_angleCallback, this, std::placeholders::_1),
            pose_angle_options
        );    

        // ========== 触发信号发布器 ==========
        // 发布话题: /trigger
        // 用途: 通知视觉系统可以继续进行下一个物体的检测
        bool_pub = this->create_publisher<std_msgs::msg::Bool>(
            "/trigger",
            10
        );

        // ========== 主控制定时器 ==========
        // 周期: 100ms
        // 功能: 定期执行timerCallback，驱动状态机
        rclcpp::TimerBase::SharedPtr timer =
            this->create_wall_timer(
                std::chrono::milliseconds(100),
                std::bind(&ControlRobot::timerCallback, this),
                timer_group_
            );
        timer_ = timer;

        // ========== 规划场景发布器 ==========
        // 用于向MoveIt发布碰撞物体信息
        planning_scene_publisher_ =
            this->create_publisher<moveit_msgs::msg::PlanningScene>(
                "/planning_scene",
                rclcpp::QoS(1).transient_local()
            );

        // ========== 碰撞网格添加定时器 ==========
        // 延迟500ms执行，确保MoveIt已准备好接收碰撞物体
        timer_collision_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&ControlRobot::addCollisionMesh, this)
        );

        RCLCPP_INFO(this->get_logger(), "ControlRobot created. Waiting for MoveGroup init...");
    }

    /******************************************************************************
     * 函数: waitForRobotState
     * 参数: timeout_sec - 超时时间(秒)
     * 返回: bool - 是否成功获取机器人状态
     * 功能: 等待机器人状态就绪，避免在初始化时出错
     ******************************************************************************/
    bool waitForRobotState(double timeout_sec = 2.0)
    {
        auto start = this->now();
        rclcpp::Duration timeout = rclcpp::Duration::from_seconds(timeout_sec);

        while ((this->now() - start) < timeout)
        {
            auto current_state = arm_move_group->getCurrentState(0.1);
            if (current_state)
            {
                return true;    // 成功获取状态
            }
            rclcpp::sleep_for(std::chrono::milliseconds(50));
        }
        return false;   // 超时
    }

    /******************************************************************************
     * 函数: initializeMoveGroup
     * 功能: 初始化MoveIt运动规划组
     *   - arm_move_group: 控制Panda机械臂(规划组名: "panda_arm")
     *   - gripper_move_group: 控制夹爪(规划组名: "hand")
     ******************************************************************************/
    void initializeMoveGroup()
    {   
        // 创建机械臂运动组接口
        arm_move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "panda_arm");
        
        // 创建夹爪运动组接口
        gripper_move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "hand");

        RCLCPP_INFO(this->get_logger(), "Waiting for robot state...");
        
        // 等待机器人状态就绪
        if (!waitForRobotState(2.0))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to receive robot state");
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Robot state ready");
        }
    }

private:
    /******************************************************************************
     * 工具函数: apply_sleep
     * 参数: s - 睡眠时间(秒)
     * 功能: 线程睡眠，用于在动作之间添加延迟
     ******************************************************************************/
    void apply_sleep(double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); }
    
    /******************************************************************************
     * 函数: addCollisionMesh
     * 功能: 向MoveIt规划场景添加碰撞网格模型
     *   1. 螺栓支架 (bolt_stand.stl)
     *   2. 料盘 (dish.stl)
     * 作用: 避免机械臂在运动规划时与这些物体碰撞
     ******************************************************************************/
    void addCollisionMesh()
        {
            moveit::planning_interface::PlanningSceneInterface psi;

            // ========== 添加螺栓支架碰撞物体 ==========
            moveit_msgs::msg::CollisionObject collision_object;
            collision_object.header.frame_id = "panda_link0";   // 基坐标系
            collision_object.id = "bolt_stand";                 // 物体ID

            // 从文件加载网格模型
            shapes::Mesh* mesh = shapes::createMeshFromResource(
                "package://bin_packing_manipulation/meshes/bolt_stand.stl"
            );
            
            // 转换网格格式: shapes::Mesh -> shape_msgs::msg::Mesh
            shape_msgs::msg::Mesh mesh_msg;
            shapes::ShapeMsg mesh_msg_tmp;
            shapes::constructMsgFromShape(mesh, mesh_msg_tmp);
            mesh_msg = boost::get<shape_msgs::msg::Mesh>(mesh_msg_tmp);

            collision_object.meshes.push_back(mesh_msg);

            // 设置网格在世界坐标系中的位姿
            geometry_msgs::msg::Pose mesh_pose;
            mesh_pose.position.x = 0.3;
            mesh_pose.position.y = -0.3;
            mesh_pose.position.z = 0.0;
            //mesh_pose.orientation.w = 1.0;
            
            // 绕Y轴旋转-90度(模型需要调整方向)
            // Rotate -90° about Y axis
            tf2::Quaternion q;
            q.setRPY(0.0, -M_PI_2, 0.0);   // roll, pitch, yaw
            mesh_pose.orientation = tf2::toMsg(q);

            collision_object.mesh_poses.push_back(mesh_pose);
            collision_object.operation = collision_object.ADD;  // 添加操作

            psi.applyCollisionObject(collision_object);

            // ========== 添加料盘碰撞物体 ==========
            moveit_msgs::msg::CollisionObject dish_collision_object;
            dish_collision_object.header.frame_id = "panda_link0";
            dish_collision_object.id = "dish";
            
            // 加载料盘网格
            shapes::Mesh* dish_mesh = shapes::createMeshFromResource(
                "package://bin_packing_manipulation/meshes/dish.stl"
            );

            shape_msgs::msg::Mesh dish_mesh_msg;
            shapes::ShapeMsg dish_mesh_msg_tmp;
            shapes::constructMsgFromShape(dish_mesh, dish_mesh_msg_tmp);
            dish_mesh_msg = boost::get<shape_msgs::msg::Mesh>(dish_mesh_msg_tmp);

            dish_collision_object.meshes.push_back(dish_mesh_msg);

            geometry_msgs::msg::Pose dish_mesh_pose;
            dish_mesh_pose.position.x = 0.31;
            dish_mesh_pose.position.y = 0.32;
            dish_mesh_pose.position.z = 0.0;
            dish_mesh_pose.orientation.w = 1.0;

            dish_collision_object.mesh_poses.push_back(dish_mesh_pose);
            dish_collision_object.operation = dish_collision_object.ADD;

            psi.applyCollisionObject(dish_collision_object);            

            // 只执行一次，然后取消定时器
            timer_collision_->cancel();
        }

    // ==================== 回调函数 ====================

    /******************************************************************************
     * 回调函数: pose_angleCallback
     * 参数: msg - 包含螺栓位姿和角度的数组 [x, y, z, angle]
     * 功能: 接收视觉系统发布的螺栓位置和旋转角度
     * 触发: 当/pos_angle话题有新消息时
     ******************************************************************************/
    void pose_angleCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(pose_angle_mutex);
        std::copy_n(msg->data.begin(), 4, bolt_pose_angle_.begin());
        /*
        float x = bolt_pose_angle[0];
        float y = bolt_pose_angle[1];
        float z = bolt_pose_angle[2];
        float angle = bolt_pose_angle[3];
        */
        bolt_pose_angle_received_ = true;
    }


    /******************************************************************************
     * 回调函数: timerCallback
     * 功能: 主控制循环 - 实现完整的11步状态机(序列0-10)
     * 执行频率: 100ms (由定时器触发)
     * 
     * 状态机完整序列:
     *   序列0: 移动到螺栓XY位置(保持当前Z高度)
     *   序列1: 旋转夹爪对齐螺栓角度
     *   序列2: 张开夹爪准备抓取
     *   序列3: 垂直下降到螺栓抓取高度
     *   序列4: 闭合夹爪抓取螺栓
     *   序列5: 垂直上升提起螺栓
     *   序列6: 移动到放置位置并旋转夹爪到-45度
     *   序列7: 垂直下降到放置高度
     *   序列8: 张开夹爪释放螺栓
     *   序列9: 返回准备位置(ready)
     *   序列10: 发送触发信号,继续下一个螺栓
     ******************************************************************************/
    void timerCallback()
    {   
        // ========== 前置检查: MoveIt接口是否已初始化 ==========
        if (!arm_move_group)
        {
            RCLCPP_WARN(this->get_logger(), "MoveGroupInterface not initialized yet.");
            return;
        }

        // ========== 前置检查: 是否需要请求新的螺栓位姿 ==========
        // 如果尚未接收到位姿数据 且 还有螺栓需要放置
        if (!bolt_pose_angle_received_ && i < bolt_places.size())
        {   
            // 发送触发信号，请求视觉系统提供新的螺栓位姿
            msg.data = true;
            bool_pub->publish(msg);
            RCLCPP_WARN(this->get_logger(), COLOR_YELLOW "Bolt pose is not received yet." COLOR_RESET);
            return; // 等待位姿数据，本次循环结束
        }

            // ========================================================================
            // 序列0: 移动到螺栓XY平面位置(保持当前Z高度不变)
            // 目的: 先在安全高度移动到螺栓正上方，避免碰撞
            // ========================================================================
            if (sequence == 0 && planning == false) //Moving to the bolt position
            {
                planning = true;

                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 0: Moving to the bolt position" COLOR_RESET);
                
                // 启动状态监控器
                arm_move_group->startStateMonitor();
                
                // 等待3秒，确保FoundationPose视觉系统更新完成
                // (有时视觉推理较慢，需要等待最新数据)
                rclcpp::sleep_for(std::chrono::milliseconds(3000));

                // ========== 线程安全地复制螺栓位姿数据 ==========
                std::array<float, 4> local_bolt_pose_angle;
                {
                    std::lock_guard<std::mutex> lock(pose_angle_mutex);
                    if (!bolt_pose_angle_received_) return;     // 双重检查
                    local_bolt_pose_angle = bolt_pose_angle_;
                }

                // 停止发送触发信号(已获取到位姿数据)
                msg.data = false;
                bool_pub->publish(msg);
                
                // 设置起始状态为当前状态
                arm_move_group->setStartStateToCurrentState();
                // 获取末端执行器(panda_link8)的当前位姿
                current_pose = arm_move_group->getCurrentPose("panda_link8");

                // ========== 构造目标位姿: 只改变XY，保持Z不变 ==========
                target_pose.position.x = local_bolt_pose_angle[0];  // 螺栓X坐标
                target_pose.position.y = local_bolt_pose_angle[1];  // 螺栓Y坐标
                target_pose.position.z = current_pose.pose.position.z;  // 保持当前Z高度
                target_pose.orientation = current_pose.pose.orientation;// 保持当前姿态
                
                // ========== 打印当前位姿信息 ==========
                RCLCPP_INFO(this->get_logger(), "EE Pose:");
                RCLCPP_INFO(this->get_logger(), "Position - x: %f, y: %f, z: %f",
                        current_pose.pose.position.x,
                        current_pose.pose.position.y,
                        current_pose.pose.position.z);
                RCLCPP_INFO(this->get_logger(), "Orientation - x: %f, y: %f, z: %f, w: %f",
                        current_pose.pose.orientation.x,
                        current_pose.pose.orientation.y,
                        current_pose.pose.orientation.z,
                        current_pose.pose.orientation.w);
                
                // ========== 打印目标位姿信息 ==========
                RCLCPP_INFO(this->get_logger(), "EE Target Pose:");
                RCLCPP_INFO(this->get_logger(), "Position - x: %f, y: %f, z: %f",
                        target_pose.position.x,
                        target_pose.position.y,
                        target_pose.position.z);
                RCLCPP_INFO(this->get_logger(), "Orientation - x: %f, y: %f, z: %f, w: %f",
                        target_pose.orientation.x,
                        target_pose.orientation.y,
                        target_pose.orientation.z,
                        target_pose.orientation.w);

                // ========== 笛卡尔路径规划 ==========
                // 虽然Z不变，但仍使用笛卡尔路径确保直线运动                        
                std::vector<geometry_msgs::msg::Pose> waypoints;

                geometry_msgs::msg::Pose start = arm_move_group->getCurrentPose().pose;
                waypoints.push_back(start);

                geometry_msgs::msg::Pose target = start;
                target.position.z = target_pose.position.z; // Z保持不变
                waypoints.push_back(target);

                moveit_msgs::msg::RobotTrajectory trajectory;
                double fraction = arm_move_group->computeCartesianPath(
                    waypoints,
                    0.002,   // eef_step (meters)   末端步长2mm
                    0.0,     // jump_threshold      不检查关节跳跃
                    trajectory
                );
                
                // 如果路径规划成功(覆盖率>99%)
                if (fraction > 0.99)
                {
                    robot_trajectory::RobotTrajectory rt(
                        arm_move_group->getRobotModel(),
                        arm_move_group->getName());

                    rt.setRobotTrajectoryMsg(*arm_move_group->getCurrentState(), trajectory);

                    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
                    totg.computeTimeStamps(
                        rt,
                        0.6,   // velocity scaling factor (20%)         60%速度
                        0.3    // acceleration scaling factor (10%)     30%加速度
                    );

                    rt.getRobotTrajectoryMsg(trajectory);
                    arm_move_group->execute(trajectory);
                }

                sequence = 1;       // 进入下一序列
                planning = false;
            }

            // ========================================================================
            // 序列1: 旋转夹爪对齐螺栓角度
            // 目的: 使夹爪方向与螺栓长轴方向一致，确保准确抓取
            // ========================================================================
            else if (sequence == 1 && planning == false) //Rotating the gripper
            {
                planning = true;

                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 1: Rotating gripper" COLOR_RESET);
                
                // 获取当前末端执行器位姿
                current_pose = arm_move_group->getCurrentPose("panda_link8");
                // ========== 提取当前姿态的欧拉角 ==========
                q_current = current_pose.pose.orientation;
                tf2::Quaternion tf_quat_current(q_current.x, q_current.y, q_current.z, q_current.w);
                tf2::Matrix3x3(tf_quat_current).getRPY(roll_current, pitch_current, yaw_current);
                //RCLCPP_INFO(this->get_logger(), "Current EE Roll: %f, Pitch: %f, Yaw: %f", roll_current, pitch_current, yaw_current);
                
                // ========== 线程安全地获取螺栓角度 ==========
                std::array<float, 4> local_bolt_pose_angle;
                {
                    std::lock_guard<std::mutex> lock(pose_angle_mutex);
                    if (!bolt_pose_angle_received_) return;
                    local_bolt_pose_angle = bolt_pose_angle_;
                }

                // ========== 构造新的姿态四元数 ==========
                // 保持roll和pitch不变，只改变yaw角度为螺栓角度
                tf2::Quaternion tf_quat;
                tf_quat.setRPY(roll_current, pitch_current, local_bolt_pose_angle[3]);
                RCLCPP_INFO(this->get_logger(), "EE Yaw: %f, Bolt theta: %f", yaw_current, local_bolt_pose_angle[3]);

                // ========== 构造目标位姿 ==========
                target_pose.position.x = local_bolt_pose_angle[0];
                target_pose.position.y = local_bolt_pose_angle[1];
                target_pose.position.z = current_pose.pose.position.z;  // Z保持不变
                target_pose.orientation.x = tf_quat.x();
                target_pose.orientation.y = tf_quat.y();
                target_pose.orientation.z = tf_quat.z();
                target_pose.orientation.w = tf_quat.w();
                
                // 执行位姿目标规划和运动
                pose_target_plan_and_execute(target_pose);

                sequence = 2;
                planning = false;

            }


            // ========================================================================
            // 序列2: 张开夹爪
            // 目的: 确保夹爪完全张开，准备抓取螺栓
            // ========================================================================            
            else if (sequence == 2 && planning == false) //Opening the gripper
            {
                planning = true;

                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 2: Opening the gripper" COLOR_RESET);
                // 夹爪张开到15mm(0.015m)，适合螺栓直径
                joint_value_target_plan_and_execute(0.015f);
                sequence = 3;
                planning = false;
            }

            // ========================================================================
            // 序列3: 垂直下降到螺栓抓取高度
            // 目的: 使夹爪下降到螺栓中心位置，准备闭合抓取
            // ========================================================================
            else if (sequence == 3 && planning == false) //Moving down
            {
                planning = true;

                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 3: Moving down." COLOR_RESET);
                
                // ========== 获取螺栓Z坐标 ==========
                std::array<float, 4> local_bolt_pose_angle;
                {
                    std::lock_guard<std::mutex> lock(pose_angle_mutex);
                    if (!bolt_pose_angle_received_) return;
                    local_bolt_pose_angle = bolt_pose_angle_;
                }
                // ========== 构造目标位姿: 只改变Z ==========
                target_pose.position.x = current_pose.pose.position.x;
                target_pose.position.y = current_pose.pose.position.y;
                // const double z_offset = 0.01;  // 额外向下 10mm，根据实际差距调整
                // target_pose.position.z = local_bolt_pose_angle[2] - z_offset;  // 补偿后的Z高度
                target_pose.position.z = local_bolt_pose_angle[2];  // 螺栓的Z高度
                target_pose.orientation = current_pose.pose.orientation;
            
                RCLCPP_INFO(this->get_logger(), "EE Pose:");
                RCLCPP_INFO(this->get_logger(), "Position - x: %f, y: %f, z: %f",
                        current_pose.pose.position.x,
                        current_pose.pose.position.y,
                        current_pose.pose.position.z);
                RCLCPP_INFO(this->get_logger(), "Orientation - x: %f, y: %f, z: %f, w: %f",
                        current_pose.pose.orientation.x,
                        current_pose.pose.orientation.y,
                        current_pose.pose.orientation.z,
                        current_pose.pose.orientation.w);

                RCLCPP_INFO(this->get_logger(), "EE Target Pose:");
                RCLCPP_INFO(this->get_logger(), "Position - x: %f, y: %f, z: %f",
                        target_pose.position.x,
                        target_pose.position.y,
                        target_pose.position.z);
                RCLCPP_INFO(this->get_logger(), "Orientation - x: %f, y: %f, z: %f, w: %f",
                        target_pose.orientation.x,
                        target_pose.orientation.y,
                        target_pose.orientation.z,
                        target_pose.orientation.w);

                std::vector<geometry_msgs::msg::Pose> waypoints;

                geometry_msgs::msg::Pose start = arm_move_group->getCurrentPose().pose;
                waypoints.push_back(start);

                geometry_msgs::msg::Pose target = start;
                target.position.z = target_pose.position.z;
                waypoints.push_back(target);

                moveit_msgs::msg::RobotTrajectory trajectory;
                double fraction = arm_move_group->computeCartesianPath(
                    waypoints,
                    0.002,   // eef_step (meters)   2mm步长
                    0.0,     // jump_threshold
                    trajectory
                );

                if (fraction > 0.99)
                {
                    robot_trajectory::RobotTrajectory rt(
                        arm_move_group->getRobotModel(),
                        arm_move_group->getName());

                    rt.setRobotTrajectoryMsg(*arm_move_group->getCurrentState(), trajectory);

                    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
                    totg.computeTimeStamps(
                        rt,
                        0.6,   // velocity scaling factor (20%)
                        0.3    // acceleration scaling factor (10%)
                    );

                    rt.getRobotTrajectoryMsg(trajectory);
                    arm_move_group->execute(trajectory);
                }

                sequence = 4;
                RCLCPP_INFO(this->get_logger(), 
                    COLOR_GREEN "序列3: 已完成,等待2秒后进入序列4" COLOR_RESET);
                rclcpp::sleep_for(std::chrono::milliseconds(2000));
                planning = false;
            }
            // ========================================================================
            // 序列4: 闭合夹爪抓取螺栓
            // 目的: 夹紧螺栓，确保抓取稳固
            // ========================================================================
            else if (sequence == 4 && planning == false) //Closing the gripper
            {
                planning = true;
                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 4: Closing the gripper." COLOR_RESET);
                apply_sleep(0.5);   // 等待0.5秒，确保位置稳定
                // 夹爪闭合到3mm(0.003m)，夹紧螺栓
                joint_value_target_plan_and_execute(0.003f);
                apply_sleep(0.5);   // 等待夹爪完全闭合
                sequence = 5;
                planning = false;
            }

            // ========================================================================
            // 序列5: 垂直上升提起螺栓
            // 目的: 将螺栓从料盘中提起，准备移动到放置位置
            // ========================================================================
            else if (sequence == 5 && planning == false) //Moving up
            {
                planning = true;

                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 5: Moving up." COLOR_RESET);
                
                // 更新状态
                arm_move_group->setStartStateToCurrentState();
                current_pose = arm_move_group->getCurrentPose("panda_link8");
                
                // ========== 构造目标位姿: 向上移动10cm ==========
                target_pose.position.x = current_pose.pose.position.x;
                target_pose.position.y = current_pose.pose.position.y;
                target_pose.position.z = current_pose.pose.position.z + 0.1;    // 上升10cm
                target_pose.orientation = current_pose.pose.orientation;
            
                RCLCPP_INFO(this->get_logger(), "EE Pose:");
                RCLCPP_INFO(this->get_logger(), "Position - x: %f, y: %f, z: %f",
                        current_pose.pose.position.x,
                        current_pose.pose.position.y,
                        current_pose.pose.position.z);
                RCLCPP_INFO(this->get_logger(), "Orientation - x: %f, y: %f, z: %f, w: %f",
                        current_pose.pose.orientation.x,
                        current_pose.pose.orientation.y,
                        current_pose.pose.orientation.z,
                        current_pose.pose.orientation.w);

                RCLCPP_INFO(this->get_logger(), "EE Target Pose:");
                RCLCPP_INFO(this->get_logger(), "Position - x: %f, y: %f, z: %f",
                        target_pose.position.x,
                        target_pose.position.y,
                        target_pose.position.z);
                RCLCPP_INFO(this->get_logger(), "Orientation - x: %f, y: %f, z: %f, w: %f",
                        target_pose.orientation.x,
                        target_pose.orientation.y,
                        target_pose.orientation.z,
                        target_pose.orientation.w);
                // ========== 笛卡尔路径: 垂直直线上升 ==========
                std::vector<geometry_msgs::msg::Pose> waypoints;

                geometry_msgs::msg::Pose start = arm_move_group->getCurrentPose().pose;
                waypoints.push_back(start);

                geometry_msgs::msg::Pose target = start;
                target.position.z = target_pose.position.z;
                waypoints.push_back(target);

                moveit_msgs::msg::RobotTrajectory trajectory;
                double fraction = arm_move_group->computeCartesianPath(
                    waypoints,
                    0.002,   // eef_step (meters)
                    0.0,     // jump_threshold
                    trajectory
                );

                if (fraction > 0.99)
                {
                    robot_trajectory::RobotTrajectory rt(
                        arm_move_group->getRobotModel(),
                        arm_move_group->getName());

                    rt.setRobotTrajectoryMsg(*arm_move_group->getCurrentState(), trajectory);

                    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
                    totg.computeTimeStamps(
                        rt,
                        0.6,   // velocity scaling factor (20%)     // 60%速度
                        0.3    // acceleration scaling factor (10%)  // 30%加速度
                    );

                    rt.getRobotTrajectoryMsg(trajectory);
                    arm_move_group->execute(trajectory);
                }

                sequence = 6;
                planning = false;
            }
            // ========================================================================
            // 序列6: 移动到螺栓放置位置并旋转夹爪到-45度
            // 目的: 将螺栓移动到支架上方，并调整姿态准备插入
            // ========================================================================
            else if (sequence == 6 && planning == false) //Moving to bolt placement position and rotate gripper
            {
                planning = true;

                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 6: Moving to the bolt placement position." COLOR_RESET);

                arm_move_group->setStartStateToCurrentState();
                current_pose = arm_move_group->getCurrentPose("panda_link8");
                
                // ========== 提取当前欧拉角 ==========
                q_current = current_pose.pose.orientation;
                tf2::Quaternion tf_quat_current(q_current.x, q_current.y, q_current.z, q_current.w);
                tf2::Matrix3x3(tf_quat_current).getRPY(roll_current, pitch_current, yaw_current);       
                
                // ========== 构造新姿态: yaw旋转到-45度(-PI/4) ==========
                // 目的: 使螺栓与支架孔位对齐                
                tf2::Quaternion tf_quat;
                tf_quat.setRPY(roll_current, pitch_current, -PI/4);

                // ========== 构造目标位姿 ==========
                target_pose.position.x = bolt_places[i][0];     // 第i个放置位置的X
                target_pose.position.y = bolt_places[i][1];     // 第i个放置位置的Y
                target_pose.position.z = current_pose.pose.position.z;// 保持当前Z高度
                target_pose.orientation.x = tf_quat.x();
                target_pose.orientation.y = tf_quat.y();
                target_pose.orientation.z = tf_quat.z();
                target_pose.orientation.w = tf_quat.w();
            
                RCLCPP_INFO(this->get_logger(), "EE Pose:");
                RCLCPP_INFO(this->get_logger(), "Position - x: %f, y: %f, z: %f",
                        current_pose.pose.position.x,
                        current_pose.pose.position.y,
                        current_pose.pose.position.z);
                RCLCPP_INFO(this->get_logger(), "Orientation - x: %f, y: %f, z: %f, w: %f",
                        current_pose.pose.orientation.x,
                        current_pose.pose.orientation.y,
                        current_pose.pose.orientation.z,
                        current_pose.pose.orientation.w);

                RCLCPP_INFO(this->get_logger(), "EE Target Pose:");
                RCLCPP_INFO(this->get_logger(), "Position - x: %f, y: %f, z: %f",
                        target_pose.position.x,
                        target_pose.position.y,
                        target_pose.position.z);
                RCLCPP_INFO(this->get_logger(), "Orientation - x: %f, y: %f, z: %f, w: %f",
                        target_pose.orientation.x,
                        target_pose.orientation.y,
                        target_pose.orientation.z,
                        target_pose.orientation.w);
                // 执行运动(涉及XY平移和姿态旋转)
                pose_target_plan_and_execute(target_pose);
                
                sequence = 7;
                planning = false;
            }
            // ========================================================================
            // 序列7: 垂直下降到放置高度
            // 目的: 将螺栓精确放入支架孔位
            // ========================================================================
            else if (sequence == 7 && planning == false) //Moving down
            {
                planning = true;

                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 7: Moving down." COLOR_RESET);

                // ========== 构造目标位姿: 下降到放置高度 ==========
                target_pose.position.x = current_pose.pose.position.x;
                target_pose.position.y = current_pose.pose.position.y;
                target_pose.position.z = bolt_places[i][2]; // 第i个放置位置的Z高度
                target_pose.orientation = current_pose.pose.orientation;
            
                RCLCPP_INFO(this->get_logger(), "EE Pose:");
                RCLCPP_INFO(this->get_logger(), "Position - x: %f, y: %f, z: %f",
                        current_pose.pose.position.x,
                        current_pose.pose.position.y,
                        current_pose.pose.position.z);
                RCLCPP_INFO(this->get_logger(), "Orientation - x: %f, y: %f, z: %f, w: %f",
                        current_pose.pose.orientation.x,
                        current_pose.pose.orientation.y,
                        current_pose.pose.orientation.z,
                        current_pose.pose.orientation.w);

                RCLCPP_INFO(this->get_logger(), "EE Target Pose:");
                RCLCPP_INFO(this->get_logger(), "Position - x: %f, y: %f, z: %f",
                        target_pose.position.x,
                        target_pose.position.y,
                        target_pose.position.z);
                RCLCPP_INFO(this->get_logger(), "Orientation - x: %f, y: %f, z: %f, w: %f",
                        target_pose.orientation.x,
                        target_pose.orientation.y,
                        target_pose.orientation.z,
                        target_pose.orientation.w);
                // ========== 笛卡尔路径: 垂直直线下降 ==========
                std::vector<geometry_msgs::msg::Pose> waypoints;

                geometry_msgs::msg::Pose start = arm_move_group->getCurrentPose().pose;
                waypoints.push_back(start);

                geometry_msgs::msg::Pose target = start;
                target.position.z = target_pose.position.z;
                waypoints.push_back(target);

                moveit_msgs::msg::RobotTrajectory trajectory;
                double fraction = arm_move_group->computeCartesianPath(
                    waypoints,
                    0.002,   // eef_step (meters)
                    0.0,     // jump_threshold
                    trajectory
                );

                if (fraction > 0.99)
                {
                    robot_trajectory::RobotTrajectory rt(
                        arm_move_group->getRobotModel(),
                        arm_move_group->getName());

                    rt.setRobotTrajectoryMsg(*arm_move_group->getCurrentState(), trajectory);

                    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
                    totg.computeTimeStamps(
                        rt,
                        0.3,   // velocity scaling factor (20%)
                        0.15    // acceleration scaling factor (10%)
                    );

                    rt.getRobotTrajectoryMsg(trajectory);
                    arm_move_group->execute(trajectory);
                }

                sequence = 8;
                planning = false;
            }
            // ========================================================================
            // 序列8: 张开夹爪释放螺栓
            // 目的: 释放螺栓,完成放置
            // ========================================================================
            else if (sequence == 8 && planning == false) //Opening the gripper.
            {
                planning = true;

                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 8: Opening the gripper." COLOR_RESET);

                // 夹爪张开到20mm(0.02m)
                joint_value_target_plan_and_execute(0.02f);

                sequence = 9;
                planning = false;
            }
            
            // ========================================================================
            // 序列9: 返回准备位置
            // 目的: 机械臂返回安全的准备姿态,准备下一次抓取
            // ========================================================================            
            else if (sequence == 9 && planning == false) //Moving to the ready position
            {
                planning = true;

                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 9: Moving to the ready position." COLOR_RESET);

                named_target_plan_and_execute("ready");
                joint_value_target_plan_and_execute(0.0f);

                sequence = 10;
                planning = false;
            }

            // ========================================================================
            // 序列10: 完成本次循环，准备下一个螺栓
            // 目的: 
            //   1. 发送触发信号通知视觉系统继续检测
            //   2. 递增放置位置索引
            //   3. 检查是否完成所有螺栓
            // ========================================================================
            else if (sequence == 10 && planning == false)
            {
                planning = true;

                // ========== 发送触发信号 ==========
                // 通知视觉系统(FoundationPose)恢复物体检测
                msg.data = true;
                bool_pub->publish(msg);
                RCLCPP_INFO(this->get_logger(), COLOR_GREEN "sequence 10: Sending signal to resume foundation pose inference." COLOR_RESET);
                // 移动到下一个放置位置
                i = i+1;
                apply_sleep(1.0);

                // ========== 检查是否还有螺栓需要放置 ==========
                if(i < bolt_places.size()){
                  sequence = 0;
                  planning = false;
                }
                else{
                  RCLCPP_INFO(this->get_logger(), COLOR_GREEN "The simulation is completed!" COLOR_RESET);
                }
                // 重置位姿接收标志，等待下一个螺栓的位姿数据
                bolt_pose_angle_received_ = false;
            }

    }

    // -------------------- MOTION FUNCTIONS --------------------
    /******************************************************************************
     * 函数: pose_target_plan_and_execute
     * 参数: target_pose - 目标位姿
     * 功能: 规划并执行到目标位姿的运动
     ******************************************************************************/
    void pose_target_plan_and_execute(const geometry_msgs::msg::Pose &target_pose)
    {
        arm_move_group->setPoseTarget(target_pose);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool success = (
            arm_move_group->plan(plan) ==
            moveit::core::MoveItErrorCode::SUCCESS
        );

        if (success)
        {
            RCLCPP_INFO(this->get_logger(), "Plan OK, executing...");
            auto result = arm_move_group->execute(plan);
            if (result == moveit::core::MoveItErrorCode::SUCCESS)
                RCLCPP_INFO(this->get_logger(), "Motion done.");
            else
                RCLCPP_WARN(this->get_logger(), "Motion failed.");
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Planning failed.");
        }
    }

    /******************************************************************************
     * 函数: named_target_plan_and_execute
     * 参数: name - 命名目标名称(如"ready", "home"等)
     * 功能: 移动到预定义的命名位置
     * 要求: 命名位置需在MoveIt配置文件中定义
     ******************************************************************************/
    void named_target_plan_and_execute(const std::string &name)
    {
        bool ok = arm_move_group->setNamedTarget(name);
        if (!ok)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to set named target '%s'", name.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Moving to '%s'...", name.c_str());
        auto result = arm_move_group->move();

        if (result)
            RCLCPP_INFO(this->get_logger(), "Reached target '%s'", name.c_str());
        else
            RCLCPP_ERROR(this->get_logger(), "Failed to reach '%s'", name.c_str());
    }

    /******************************************************************************
     * 函数: joint_value_target_plan_and_execute
     * 参数: displacement - 夹爪张开距离(米)
     * 功能: 控制夹爪开闭
     * 说明: 
     *   - Panda夹爪有2个手指,但通过模仿关节实现对称运动
     *   - 只需控制一个关节(index 0)
     *   - 0.0 = 完全闭合, 0.04 = 完全张开
     ******************************************************************************/
    void joint_value_target_plan_and_execute(float displacement){
        // Get current joint values
        std::vector<double> joint_values = gripper_move_group->getCurrentJointValues();
        std::ostringstream joint_values_stream;
        for (const auto& value : joint_values) {
            joint_values_stream << value << " ";
        }
        RCLCPP_INFO(this->get_logger(), "Current joint values: %s", joint_values_stream.str().c_str());

        // Set joint value 设置目标关节值（只设置第一个关节，第二个手执会自动模仿）
        int joint_index = 0; // We have to set only one joint, because right finger mimics left finger (index:0)
        joint_values[joint_index] = displacement;
        bool success = gripper_move_group->setJointValueTarget(joint_values);   
        if (success) {
            auto execution_success = gripper_move_group->move();
            if (execution_success) {
                RCLCPP_INFO(this->get_logger(), "Motion execution successful!");
            } else {
                RCLCPP_WARN(this->get_logger(), "Motion execution failed.");
            }
        }
        else{
            RCLCPP_WARN(this->get_logger(), "Planning failed. Unable to find a valid plan.");
        }
    }

    // -------------------- MEMBER VARIABLES --------------------
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr pos_angle_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr bool_pub;
    rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr timer_collision_;

    rclcpp::CallbackGroup::SharedPtr pose_angle_group_;
    rclcpp::CallbackGroup::SharedPtr timer_group_;

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_move_group;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_move_group;
    std::mutex pose_angle_mutex;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ControlRobot>();
    node->initializeMoveGroup();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
