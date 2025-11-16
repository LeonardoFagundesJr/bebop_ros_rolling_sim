#!/usr/bin/env python3

"""
inverse_dynamic_controller.py

ROS2 rclpy implementation of the inverse-dynamic compensator.

Subscriptions:
 - /bebop1/pose        geometry_msgs/msg/PoseStamped (or Pose)
 - /goal               geometry_msgs/msg/PoseStamped (desired reference)
Publishes:
 - /bebop1/cmd_vel     geometry_msgs/msg/Twist

Parameters:
 - namespace/topic names can be remapped / set via ros2 param
 - gains: 16-element list as in MATLAB version
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, Pose
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Empty
from .cInverseDynamicController import cInvDynCtrl
import numpy as np
import time

def wrap_to_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

class Controller(Node):
    class State:
        IDLE = 0
        AUTOMATIC = 1
        TAKING_OFF = 2
        LANDING = 3
        EMERGENCY_STOP = 4

    def __init__(self):
        super().__init__('controller')
        
        # Parámetros
        self.declare_parameter('frequency', 50.0)
        self.declare_parameter('robot_name', 'bebop3')
        self.declare_parameter('goal_name', 'goal')
        self.declare_parameter('takeoff_threshold', 0.04)
        self.declare_parameter('landing_threshold', 0.08)
        self.declare_parameter('takeoff_height', 1.0)  # Nueva altura de despegue

        self.frequency = self.get_parameter('frequency').value
        self.robot_name = self.get_parameter('robot_name').value.strip()
        self.goal_name = self.get_parameter("goal_name").value.strip()
        self.takeoff_threshold = self.get_parameter('takeoff_threshold').value
        self.landing_threshold = self.get_parameter('landing_threshold').value
        self.takeoff_height = self.get_parameter('takeoff_height').value

        # Parameters para el controlador inverso dinámico
        self.declare_parameter('gains', [
            2.0, 2.0, 3.0, 1.5,    # Ksp diag
            2.0, 2.0, 1.8, 5.0,    # Ksd diag
            1.0, 1.0, 1.0, 1.5,    # Kp diag
            1.0, 1.0, 1.0, 1.0     # Kd diag
        ])
        self.declare_parameter('model_simp', [0.8417,0.18227,0.8354,0.17095,3.966,4.001,9.8524,4.7295])
        self.declare_parameter('kinematics_control', False)
        self.declare_parameter('min_dt', 1e-4)
        
        self.gains = np.array(self.get_parameter('gains').get_parameter_value().double_array_value, dtype=float)
        self.model = np.array(self.get_parameter('model_simp').get_parameter_value().double_array_value, dtype=float)
        self.kinematics_control = self.get_parameter('kinematics_control').get_parameter_value().bool_value
        self.min_dt = self.get_parameter('min_dt').get_parameter_value().double_value



        if not self.robot_name:
            self.get_logger().info('El parámetro "robot_name" está vacío. Se usará "bebop3" por defecto.')
            self.robot_name = 'bebop3'

        self.get_logger().info(f"Robot Name: {self.robot_name}")

        # Publicadores
        qos = QoSProfile(depth=10)
        self.cmd_pub = self.create_publisher(Twist, f"/{self.robot_name}/cmd_vel", qos)
        self.cmd_des = self.create_publisher(Pose, f"/{self.robot_name}/setpointG", qos)
        self.cmd_enable = self.create_publisher(Bool, f"/{self.robot_name}/enable", qos)
        
        # Suscriptores
        self.goal_sub = self.create_subscription(Pose, f"/{self.goal_name}", self.goal_changed, qos) # publica o objetivo
        self.pos_sub = self.create_subscription(Pose, f"/{self.robot_name}/pose", self.pos_changed, qos)
        self.state_sub = self.create_subscription(Int32, "/state", self.state_changed, qos) # publica o estado do robô
        
        # Servicios
        self.takeoff_srv = self.create_service(Empty, 'takeoff', self.takeoff_callback)
        self.land_srv = self.create_service(Empty, 'land', self.land_callback)
        
        # PIDs
        self.pid_x = self.create_pid('X')
        self.pid_y = self.create_pid('Y')
        self.pid_z = self.create_pid('Z')
        self.pid_yaw = self.create_pid('Yaw')
        
        # Estado
        self.state = self.State.IDLE
        self.goal = Pose()
        self.current_pose = Pose()
        self.enable = False
        self.takeoff_complete = False  # Nuevo flag para controlar despegue completado
        
        # Timer
        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop)

    def create_pid(self, axis):
        prefix = f'PIDs.{axis}.'
        kp = self.get_param_or(prefix + 'kp', 0.0)
        kd = self.get_param_or(prefix + 'kd', 0.0)
        ki = self.get_param_or(prefix + 'ki', 0.0)
        min_output = self.get_param_or(prefix + 'minOutput', -1.0)
        max_output = self.get_param_or(prefix + 'maxOutput', 1.0)
        integrator_min = self.get_param_or(prefix + 'integratorMin', -0.5)
        integrator_max = self.get_param_or(prefix + 'integratorMax', 0.5)
        dt = 1.0 / self.frequency
        
        return cInvDynCtrl(self, kp, kd, ki, min_output, max_output, integrator_min, integrator_max, dt, axis.lower())

    def get_param_or(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).get_parameter_value().double_value

    def state_changed(self, msg):
        """Maneja cambios en el estado desde el tópico /state"""
        new_state = msg.data
        
        if new_state == self.State.IDLE:
            if self.state != self.State.IDLE:
                self.get_logger().info("Cambiando a estado IDLE")
                self.state = self.State.IDLE
                self.enable = False
                
        elif new_state == self.State.AUTOMATIC:
            if self.state == self.State.TAKING_OFF and self.takeoff_complete:
                self.get_logger().info("Cambiando a estado AUTOMATIC")
                self.state = self.State.AUTOMATIC
                self.enable = True
            elif self.state != self.State.TAKING_OFF:
                self.get_logger().info("No se puede cambiar a AUTOMATIC sin completar el despegue primero")
                
        elif new_state == self.State.TAKING_OFF:
            if self.state != self.State.TAKING_OFF:
                self.get_logger().info("Cambiando a estado TAKING_OFF")
                self.state = self.State.TAKING_OFF
                self.takeoff_complete = False
                self.enable = True
                
        elif new_state == self.State.LANDING:
            if self.state != self.State.LANDING:
                self.get_logger().info("Cambiando a estado LANDING")
                self.state = self.State.LANDING
                self.enable = True
                
        elif new_state == self.State.EMERGENCY_STOP:
            if self.state != self.State.EMERGENCY_STOP:
                self.get_logger().info("¡EMERGENCY STOP activado!")
                self.state = self.State.EMERGENCY_STOP
                self.enable = False
        
        # Publicar estado de enable
        self.cmd_enable.publish(Bool(data=self.enable))

    def goal_changed(self, msg):
        self.goal = msg

    def pos_changed(self, msg):
        self.current_pose = msg

    def takeoff_callback(self, request, response):
        self.get_logger().info('Takeoff requested!')
        if self.state != self.State.TAKING_OFF:
            self.state = self.State.TAKING_OFF
            self.takeoff_complete = False
            self.enable = True
            self.cmd_enable.publish(Bool(data=self.enable))
        return response

    def land_callback(self, request, response):
        self.get_logger().info('Landing requested!')
        if self.state != self.State.LANDING:
            self.state = self.State.LANDING
            self.enable = True
            self.cmd_enable.publish(Bool(data=self.enable))
        return response

    # Loop principal de controle:
    def control_loop(self):

        # Caso o comando de takeoff esteja ativo
        if self.state == self.State.TAKING_OFF:
            current_z = self.current_pose.position.z
            
            if not self.takeoff_complete:
                if current_z < self.takeoff_height - 0.05:  
                    msg = Twist()
                    msg.linear.z = float(self.pid_z.update(current_z, self.takeoff_height))
                    self.cmd_pub.publish(msg)
                else:
                    # Altura alcanzada
                    self.get_logger().info(f"Altura de despegue alcanzada: {current_z:.2f}m")
                    self.takeoff_complete = True
            else:
                # Mantener la altura con PID
                msg = Twist()
                msg.linear.z = float(self.pid_z.update(current_z, self.takeoff_height))
                self.cmd_pub.publish(msg)

        # Caso o comando de landing esteja ativo
        elif self.state == self.State.LANDING:
            current_z = self.current_pose.position.z
            msg = Twist()
            
            if current_z > self.landing_threshold:
                # Descender controladamente
                msg.linear.z = float(self.pid_z.update(current_z, 0.0)) * 0.5  # Reducir velocidad
                self.cmd_pub.publish(msg)
                # print(current_z > self.landing_threshold)
                # print(f"Z= {current_z}; Z_d= {self.landing_threshold}")
            else:
                # Aterrizaje completado
                self.get_logger().info("¡Aterrizaje completado!")
                self.state = self.State.IDLE
                self.enable = False
                self.cmd_enable.publish(Bool(data=self.enable))
                self.cmd_pub.publish(Twist())

        # ------------------------------------------------------------------------------------
        # Caso o controle automático esteja ativo (loop de controle e definição de setpoints)
        elif self.state == self.State.AUTOMATIC and self.enable:
            try:
                # Obtener orientación actual y deseada
                euler_c = euler_from_quaternion([
                    self.current_pose.orientation.x,
                    self.current_pose.orientation.y,
                    self.current_pose.orientation.z,
                    self.current_pose.orientation.w
                ])

                euler_d = euler_from_quaternion([
                    self.goal.orientation.x,
                    self.goal.orientation.y,
                    self.goal.orientation.z,
                    self.goal.orientation.w
                ])

                # Posiciones y yaw deseado
                x_d, y_d, z_d, yaw_d = self.goal.position.x, self.goal.position.y, self.goal.position.z, euler_d[2]

                # Control PID
                msg = Twist()
                msg.linear.x = float(self.pid_x.update(self.current_pose.position.x, x_d))
                msg.linear.y = float(self.pid_y.update(self.current_pose.position.y, y_d))
                msg.linear.z = float(self.pid_z.update(self.current_pose.position.z, z_d))
                msg.angular.z = float(self.pid_yaw.update(euler_c[2], yaw_d))
                self.cmd_pub.publish(msg)

                # Publicar setpoint para visualización
                msg_goal = Pose()
                msg_goal.position = self.goal.position
                quat = quaternion_from_euler(0, 0, yaw_d)
                msg_goal.orientation.x = quat[0]
                msg_goal.orientation.y = quat[1]
                msg_goal.orientation.z = quat[2]
                msg_goal.orientation.w = quat[3]
                self.cmd_des.publish(msg_goal)

            except Exception as e:
                self.get_logger().error(f"Error en control automático: {str(e)}")
                self.enable = False
                self.cmd_enable.publish(Bool(data=self.enable))

        # Caso o comando de landing tenha terminado ou o emergência esteja ativo (desliga os motores)
        elif self.state in [self.State.IDLE, self.State.EMERGENCY_STOP]:
            self.cmd_pub.publish(Twist())
            self.enable = False
            self.cmd_enable.publish(Bool(data=self.enable))
            # print(f"Estado dos motores: IDLE o EMERGENCY STOP. Comandos de velocidad a cero. Status: {self.enable}")

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()