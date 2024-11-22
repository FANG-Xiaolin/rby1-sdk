#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <iomanip>
#include <iostream>
#include <thread>
#include "dynamixel_sdk.h"  // Uses Dynamixel SDK library
#include "network/Network.h"
#include "rby1-sdk/model.h"
#include "rby1-sdk/robot.h"
#include "rby1-sdk/robot_command_builder.h"

#include "rby1-sdk/upc/device.h"

#include <unistd.h>
#include <QApplication>
#include <algorithm>
#include <cstring>
using namespace rb;
using namespace std::chrono_literals;

// Protocol version
#define PROTOCOL_VERSION 2.0  // See which protocol version is used in the Dynamixel

// Default setting
#define BAUDRATE 2000000

#define ADDR_TORQUE_ENABLE 64
#define ADDR_PRESENT_POSITION 132
#define ADDR_GOAL_CURRENT 102
#define ADDR_GOAL_POSITION 116
#define ADDR_OPERATING_MODE 11  // Address for operating mode

#define CURRENT_CONTROL_MODE 0
#define CURRENT_BASED_POSITION_CONTROL_MODE 5

#define ADDR_PRESENT_BUTTON_STATUS 132
#define ADDR_GOAL_VIBRATION_LEVEL 102

#define TORQUE_ENABLE 1
#define TORQUE_DISABLE 0

#define D2R 0.017453288888888
#define R2D 57.29579143313326

bool redandancy_mode = false;
double m_sf = 0.4;

double init_cnt = 0.;
bool ma_master_mode = false;

using namespace rb::dyn;
using namespace rb;

#define MIN_INDEX 0
#define MAX_INDEX 1

bool is_first_init = false;
std::mutex mtx_q_joint_ma_info;
Eigen::Vector<double, 24> q_redandancy_fix = Eigen::Vector<double, 24>::Zero();
Eigen::Matrix<double, 14, 1> q_joint_ma = Eigen::Matrix<double, 14, 1>::Zero();

std::mutex mtx_hand_controller_info;
Eigen::Matrix<double, 2, 1> hand_controller_trigger = Eigen::Matrix<double, 2, 1>::Constant(0.5);
Eigen::Matrix<double, 2, 1> hand_controller_button = Eigen::Matrix<double, 2, 1>::Constant(0);

std::vector<Eigen::Matrix<double, 2, 1>> hand_controller_trigger_min_max = {Eigen::Matrix<double, 2, 1>({0, 1000}),
                                                                            Eigen::Matrix<double, 2, 1>({0, 1000})};

int gripper_direction = 0;
bool ma_info_verbose = false;

std::vector<double> torque_constant = {1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043,
                                       1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043};

const std::string kAll = ".*";

Network* network;

#include <QEventLoop>
#include <QThread>
#include <QUdpSocket>

class UdpThread : public QThread {
  Q_OBJECT
 public:
  void run() override {
    QEventLoop loop;
    loop.exec();
  }
};

Eigen::Vector<double, 24> q_joint_rby1_24x1 = Eigen::Vector<double, 24>::Zero();

std::optional<std::pair<int, std::pair<int, int>>> ReadButtonStatus(dynamixel::PortHandler* portHandler,
                                                                    dynamixel::PacketHandler* packetHandler, int id) {

  int32_t position = 0;
  uint8_t dxl_error = 0;
  int dxl_comm_result =
      packetHandler->read4ByteTxRx(portHandler, id, ADDR_PRESENT_BUTTON_STATUS, (uint32_t*)&position, &dxl_error);
  if (dxl_comm_result == COMM_SUCCESS) {
    int button = (position >> 8) & 0xff;
    int trigger = ((position >> 16) & 0xff) | (((position >> 24) & 0xff) << 8);
    return std::make_pair(id, std::make_pair(button, trigger));
  } else {
    return {};
  }
}

void TorqueEnable(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler, int id, int onoff) {
  packetHandler->write1ByteTxOnly(portHandler, id, ADDR_TORQUE_ENABLE, onoff);
  std::this_thread::sleep_for(std::chrono::microseconds(500));
}

std::optional<int> ReadTorqueEnable(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler,
                                    int id) {
  int8_t onoff = -1;
  uint8_t dxl_error = 0;
  int dxl_comm_result = packetHandler->read1ByteTxRx(portHandler, id, ADDR_TORQUE_ENABLE, (uint8_t*)&onoff, &dxl_error);

  if (dxl_comm_result == COMM_SUCCESS) {
    return onoff;
  } else {
    return {};
  }
}

std::optional<double> ReadEncoder(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler,
                                  int id) {
  int32_t position = 0;
  uint8_t dxl_error = 0;
  int dxl_comm_result =
      packetHandler->read4ByteTxRx(portHandler, id, ADDR_PRESENT_POSITION, (uint32_t*)&position, &dxl_error);

  if (dxl_comm_result == COMM_SUCCESS) {
    return (double)position / 4096. * 2. * 3.141592;  // unit [rad]
  } else {
    return {};
  }
}

std::optional<std::vector<std::pair<int, double>>> BulkReadEncoder(dynamixel::PortHandler* portHandler,
                                                                   dynamixel::PacketHandler* packetHandler,
                                                                   std::vector<int> ids) {

  std::vector<std::pair<int, double>> position_vector;
  dynamixel::GroupBulkRead groupBulkRead(portHandler, packetHandler);

  for (auto const& id : ids) {
    if (id < 0x80) {
      groupBulkRead.addParam(id, ADDR_PRESENT_POSITION, 4);
    }
  }

  groupBulkRead.txRxPacket();

  for (auto const& id : ids) {
    if (id < 0x80) {
      if (groupBulkRead.isAvailable(id, ADDR_PRESENT_POSITION, 4)) {
        int position = groupBulkRead.getData(id, ADDR_PRESENT_POSITION, 4);
        position_vector.push_back(std::make_pair(id, (double)position / 4096. * 2. * 3.141592));
      }
    }
  }

  if (position_vector.size() == 0) {
    return {};
  } else {
    return position_vector;
  }
}

std::optional<std::vector<std::pair<int, double>>> BulkReadGoalPosition(dynamixel::PortHandler* portHandler,
                                                                        dynamixel::PacketHandler* packetHandler,
                                                                        std::vector<int> ids) {

  std::vector<std::pair<int, double>> position_vector;
  dynamixel::GroupBulkRead groupBulkRead(portHandler, packetHandler);

  for (auto const& id : ids) {
    if (id < 0x80) {
      groupBulkRead.addParam(id, ADDR_GOAL_POSITION, 4);
    }
  }

  groupBulkRead.txRxPacket();

  for (auto const& id : ids) {
    if (id < 0x80) {
      if (groupBulkRead.isAvailable(id, ADDR_GOAL_POSITION, 4)) {
        int position = groupBulkRead.getData(id, ADDR_GOAL_POSITION, 4);
        position_vector.push_back(std::make_pair(id, (double)position / 4096. * 2. * 3.141592));
      }
    }
  }

  if (position_vector.size() == 0) {
    return {};
  } else {
    return position_vector;
  }
}

void SendGoalPosition(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler, int id,
                      int goal_position) {
  packetHandler->write4ByteTxOnly(portHandler, id, ADDR_GOAL_POSITION, goal_position);
  std::this_thread::sleep_for(std::chrono::microseconds(500));
}

std::optional<int> ReadOperationMode(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler,
                                     int id) {
  int8_t operation_mode = -1;
  uint8_t dxl_error = 0;
  int dxl_comm_result =
      packetHandler->read1ByteTxRx(portHandler, id, ADDR_OPERATING_MODE, (uint8_t*)&operation_mode, &dxl_error);

  if (dxl_comm_result == COMM_SUCCESS) {
    return operation_mode;
  } else {
    return {};
  }
}

std::optional<std::vector<std::pair<int, int>>> BulkReadOperationMode(dynamixel::PortHandler* portHandler,
                                                                      dynamixel::PacketHandler* packetHandler,
                                                                      std::vector<int> ids) {

  std::vector<std::pair<int, int>> operation_mode_vector;
  dynamixel::GroupBulkRead groupBulkRead(portHandler, packetHandler);

  for (auto const& id : ids) {
    if (id < 0x80) {
      groupBulkRead.addParam(id, ADDR_OPERATING_MODE, 1);
    }
  }

  groupBulkRead.txRxPacket();

  for (auto const& id : ids) {
    if (id < 0x80) {
      if (groupBulkRead.isAvailable(id, ADDR_OPERATING_MODE, 1)) {
        int operation_mode = groupBulkRead.getData(id, ADDR_OPERATING_MODE, 1);
        operation_mode_vector.push_back(std::make_pair(id, operation_mode));
      }
    }
  }

  if (operation_mode_vector.size() == 0) {
    return {};
  } else {
    return operation_mode_vector;
  }
}

std::optional<std::vector<std::pair<int, int>>> BulkReadTorqueEnable(dynamixel::PortHandler* portHandler,
                                                                     dynamixel::PacketHandler* packetHandler,
                                                                     std::vector<int> ids) {

  std::vector<std::pair<int, int>> torque_enable_vector;
  dynamixel::GroupBulkRead groupBulkRead(portHandler, packetHandler);

  for (auto const& id : ids) {
    if (id < 0x80) {
      groupBulkRead.addParam(id, ADDR_TORQUE_ENABLE, 1);
    }
  }

  groupBulkRead.txRxPacket();

  for (auto const& id : ids) {
    if (id < 0x80) {
      if (groupBulkRead.isAvailable(id, ADDR_TORQUE_ENABLE, 1)) {
        int operation_mode = groupBulkRead.getData(id, ADDR_TORQUE_ENABLE, 1);
        torque_enable_vector.push_back(std::make_pair(id, operation_mode));
      }
    }
  }

  if (torque_enable_vector.size() == 0) {
    return {};
  } else {
    return torque_enable_vector;
  }
}

void BulkWriteTorqueEnable(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler,
                           std::vector<std::pair<int, int>> id_and_enable_vector) {

  dynamixel::GroupBulkWrite groupBulkWrite(portHandler, packetHandler);

  uint8_t param[1];

  for (auto const& id_and_enable : id_and_enable_vector) {
    if (id_and_enable.first < 0x80) {
      param[0] = id_and_enable.second;
      groupBulkWrite.addParam(id_and_enable.first, ADDR_TORQUE_ENABLE, 1, param);
    }
  }

  groupBulkWrite.txPacket();
  std::this_thread::sleep_for(std::chrono::microseconds(500));
}

void BulkWriteTorqueEnable(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler,
                           std::vector<int> ids, int enable) {

  dynamixel::GroupBulkWrite groupBulkWrite(portHandler, packetHandler);

  uint8_t param[1];

  for (auto const& id : ids) {
    if (id < 0x80) {
      param[0] = enable;
      groupBulkWrite.addParam(id, ADDR_TORQUE_ENABLE, 1, param);
    }
  }

  groupBulkWrite.txPacket();
  std::this_thread::sleep_for(std::chrono::microseconds(500));
}

void BulkWriteGoalPosition(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler,
                           std::vector<std::pair<int, double>> id_and_q_vector) {

  dynamixel::GroupBulkWrite groupBulkWrite(portHandler, packetHandler);

  for (auto const& id_and_q : id_and_q_vector) {
    if (id_and_q.first < 0x80) {
      int goal_position = (int)(id_and_q.second * 4096. / 2. / 3.141592);
      uint8_t param[4];
      param[0] = DXL_LOBYTE(DXL_LOWORD(goal_position));
      param[1] = DXL_HIBYTE(DXL_LOWORD(goal_position));
      param[2] = DXL_LOBYTE(DXL_HIWORD(goal_position));
      param[3] = DXL_HIBYTE(DXL_HIWORD(goal_position));
      groupBulkWrite.addParam(id_and_q.first, ADDR_GOAL_POSITION, 4, param);
    }
  }

  groupBulkWrite.txPacket();
  std::this_thread::sleep_for(std::chrono::microseconds(500));
}

void BulkWriteOperationMode(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler,
                            std::vector<std::pair<int, int>> id_and_mode_vector) {

  dynamixel::GroupBulkWrite groupBulkWrite(portHandler, packetHandler);

  uint8_t param[1];

  for (auto const& id_and_mode : id_and_mode_vector) {
    if (id_and_mode.first < 0x80) {
      param[0] = id_and_mode.second;
      groupBulkWrite.addParam(id_and_mode.first, ADDR_OPERATING_MODE, 1, param);
    }
  }

  groupBulkWrite.txPacket();
  std::this_thread::sleep_for(std::chrono::microseconds(500));
}

void BulkWriteSendTorque(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler,
                         std::vector<std::pair<int, double>> id_and_torque_vector) {

  dynamixel::GroupBulkWrite groupBulkWrite(portHandler, packetHandler);

  uint16_t param[1];

  for (auto const& id_and_mode : id_and_torque_vector) {
    if (id_and_mode.first < 0x80) {
      param[0] = (int16_t)(id_and_mode.second / torque_constant[id_and_mode.first] * 1000. / 2.69);
      groupBulkWrite.addParam(id_and_mode.first, ADDR_GOAL_CURRENT, 2, reinterpret_cast<uint8_t*>(&param));
    }
  }

  groupBulkWrite.txPacket();
  std::this_thread::sleep_for(std::chrono::microseconds(500));
}

void SendOperationMode(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler, int id,
                       int operation_mode) {
  packetHandler->write1ByteTxOnly(portHandler, id, ADDR_OPERATING_MODE, operation_mode);
  std::this_thread::sleep_for(std::chrono::microseconds(500));
}

void SendTorque(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler, int id,
                double joint_torque) {
  int32_t torque_value = (int32_t)(joint_torque / torque_constant[id] * 1000. / 2.69);
  packetHandler->write2ByteTxOnly(portHandler, id, ADDR_GOAL_CURRENT, torque_value);
}

void SendCurrent(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler, int id, double current) {
  //current unit is [A]
  int32_t current_value = (int)(current / 2.69 * 1000.);
  packetHandler->write2ByteTxOnly(portHandler, id, ADDR_GOAL_CURRENT, current_value);
}

Eigen::Matrix<double, 14, 1> ComputeGravityTorque(std::shared_ptr<rb::dyn::Robot<14>> robot,
                                                  std::shared_ptr<State<14>> state,
                                                  Eigen::Matrix<double, 14, 1> q_joint) {

  state->SetQ(q_joint);
  robot->ComputeForwardKinematics(state);
  robot->ComputeDiffForwardKinematics(state);
  robot->Compute2ndDiffForwardKinematics(state);
  robot->ComputeInverseDynamics(state);
  return state->GetTau();  // / unit [Nm]
}

Eigen::Matrix<double, 14, 1> calc_torque_for_limit_avoid(Eigen::Matrix<double, 14, 1> q_joint) {

  Eigen::Matrix<double, 14, 1> torque_add;
  torque_add.setZero();

  int arm_dof = 7;

  int n_joint = 1;
  if (q_joint(n_joint) > -10. * D2R) {
    torque_add(n_joint) += (-10. * D2R - q_joint(n_joint)) * 4.;
  }

  n_joint = arm_dof + 1;
  if (q_joint(n_joint) < 10. * D2R) {
    torque_add(n_joint) += (10. * D2R - q_joint(n_joint)) * 4.;
  }

  n_joint = 2;
  if (q_joint(n_joint) > 90 * D2R) {
    torque_add(n_joint) += (90. * D2R - q_joint(n_joint)) * 0.5;
  }
  if (q_joint(n_joint) < 0) {
    torque_add(n_joint) += (0. - q_joint(n_joint)) * 0.5;
  }

  n_joint = arm_dof + 2;
  if (q_joint(n_joint) < -90 * D2R) {
    torque_add(n_joint) += (-90. * D2R - q_joint(n_joint)) * 0.5;
  }
  if (q_joint(n_joint) > 0) {
    torque_add(n_joint) += (0. - q_joint(n_joint)) * 0.5;
  }

  n_joint = 5;
  if (q_joint(n_joint) > 90 * D2R) {
    torque_add(n_joint) += (90. * D2R - q_joint(n_joint)) * 1.;
  }
  if (q_joint(n_joint) < 0. * D2R) {
    torque_add(n_joint) += (0. * D2R - q_joint(n_joint)) * 1.;
  }

  n_joint = arm_dof + 5;
  if (q_joint(n_joint) > 90 * D2R) {
    torque_add(n_joint) += (90. * D2R - q_joint(n_joint)) * 1.;
  }
  if (q_joint(n_joint) < 0. * D2R) {
    torque_add(n_joint) += (0. * D2R - q_joint(n_joint)) * 1.;
  }

  n_joint = 4;
  if (q_joint(n_joint) > 90 * D2R) {
    torque_add(n_joint) += (90. * D2R - q_joint(n_joint)) * 0.5;
  }
  if (q_joint(n_joint) < -90 * D2R) {
    torque_add(n_joint) += (-90. * D2R - q_joint(n_joint)) * 0.5;
  }

  n_joint = arm_dof + 4;
  if (q_joint(n_joint) > 90 * D2R) {
    torque_add(n_joint) += (90. * D2R - q_joint(n_joint)) * 0.5;
  }
  if (q_joint(n_joint) < -90 * D2R) {
    torque_add(n_joint) += (-90. * D2R - q_joint(n_joint)) * 0.5;
  }

  n_joint = 3;
  if (q_joint(n_joint) < -135. * D2R) {
    torque_add(n_joint) += (-135. * D2R - q_joint(n_joint)) * 6.;
  }

  if (q_joint(n_joint) > -20. * D2R) {
    torque_add(n_joint) += (-20. * D2R - q_joint(n_joint)) * 6.;
  }

  n_joint = arm_dof + 3;
  if (q_joint(n_joint) < -135. * D2R) {
    torque_add(n_joint) += (-135. * D2R - q_joint(n_joint)) * 6.;
  }

  if (q_joint(n_joint) > -20. * D2R) {
    torque_add(n_joint) += (-20. * D2R - q_joint(n_joint)) * 6.;
  }

  Eigen::Matrix<double, 14, 1> torque_add_limit;
  torque_add_limit.setConstant(300. / 1000.);

  torque_add_limit(5) = 500. / 1000.;
  torque_add_limit(arm_dof + 5) = 500 / 1000.;

  torque_add = torque_add.cwiseMin(torque_add_limit);
  torque_add = torque_add.cwiseMax(-torque_add_limit);

  for (int i = 0; i < 14; i++) {
    torque_add(i) *= torque_constant[i];
  }

  return torque_add;
}

void control_loop_for_robot(std::shared_ptr<rb::Robot<y1_model::A>> robot) {
  Eigen::Vector<double, 6> q_joint_waist;
  Eigen::Vector<double, 7> q_joint_right_arm;
  Eigen::Vector<double, 7> q_joint_left_arm;
  q_joint_waist.setZero();
  q_joint_right_arm.setZero();
  q_joint_left_arm.setZero();

  double minimum_time = 4.;

  double right_arm_minimum_time = 1.;
  double left_arm_minimum_time = 1.;
  double lpf_update_ratio = 0.1;
  double torso_minimum_time = 1.0;
  double wheel_minimum_time = 0.5;

  Eigen::Vector<double, 2> wheel_velocity, wheel_acceleration;
  wheel_velocity << 1 * 3.14 * 2., 1 * 3.14 * 2.;
  wheel_acceleration.setConstant(100. / 10.);

  Eigen::Vector<double, 7> arm_acc_limit, arm_vel_limit;

  arm_acc_limit.setConstant(1200.0);
  arm_acc_limit *= D2R;

  arm_vel_limit << 160, 160, 160, 160, 330, 330, 330;
  arm_vel_limit *= D2R;

  Eigen::Matrix<double, 4, 4> T_torso;
  T_torso.setIdentity();
  T_torso.block(0, 3, 3, 1) << 0, 0, 1.1;
  double stop_orientation_tracking_error = 1e-5;
  double stop_position_tracking_error = 1e-5;

  Eigen::Vector<double, 7> q_joint_right_target, q_joint_left_target;
  q_joint_right_target.setZero();
  q_joint_left_target.setZero();

  auto dyn = robot->GetDynamics();
  auto dyn_state = dyn->MakeState({"base", "link_torso_5", "ee_right", "ee_left"}, y1_model::A::kRobotJointNames);
  Eigen::Vector<double, 7> q_joint_right_lb, q_joint_left_lb, q_joint_right_ub, q_joint_left_ub;
  q_joint_right_ub = dyn->GetLimitQUpper(dyn_state).block(2 + 6, 0, 7, 1);
  q_joint_right_lb = dyn->GetLimitQLower(dyn_state).block(2 + 6, 0, 7, 1);

  q_joint_left_ub = dyn->GetLimitQUpper(dyn_state).block(2 + 6 + 7, 0, 7, 1);
  q_joint_left_lb = dyn->GetLimitQLower(dyn_state).block(2 + 6 + 7, 0, 7, 1);

  Eigen::Vector<double, 2> linear_velocity = Eigen::Vector<double, 2>::Zero();
  double angular_velocity = 0.;

  double angular_acceleration_limit = 100.;
  Eigen::Vector<double, 2> linear_acceleration_limit;
  linear_acceleration_limit << 100, 100;

  double control_hold_time = 1e6;

  {
    //go to init pos
    auto rv = robot
                  ->SendCommand(RobotCommandBuilder().SetCommand(ComponentBasedCommandBuilder().SetBodyCommand(
                      BodyComponentBasedCommandBuilder()
                          .SetTorsoCommand(JointPositionCommandBuilder().SetMinimumTime(3).SetPosition(
                              Eigen::Vector<double, 6>{0, 30, -60, 30, 0, 0} * D2R))
                          .SetRightArmCommand(JointPositionCommandBuilder().SetMinimumTime(3).SetPosition(
                              Eigen::Vector<double, 7>{45, -15, 0, -135, 0, 45, 0} * D2R))
                          .SetLeftArmCommand(JointPositionCommandBuilder().SetMinimumTime(3).SetPosition(
                              Eigen::Vector<double, 7>{45, 15, 0, -135, 0, 45, 0} * D2R)))))
                  ->Get();

    if (rv.finish_code() != RobotCommandFeedback::FinishCode::kOk) {
      std::cerr << "Error: Failed to conduct demo motion." << std::endl;
    }

    is_first_init = true;

    q_joint_right_target << 45, -15, 0, -135, 0, 45, 0;
    q_joint_left_target << 45, 15, 0, -135, 0, 45, 0;
    q_joint_right_target = q_joint_right_target * 3.141592 / 180.;
    q_joint_left_target = q_joint_left_target * 3.141592 / 180.;
  }

  std::unique_ptr<RobotCommandStreamHandler<y1_model::A>> stream;
  stream = robot->CreateCommandStream();

  bool flag_mode_change = false;

  while (1) {
    auto joy_stick_data = network->GetJoyStickData();

    if (joy_stick_data.buttons[6]) {
      if (!flag_mode_change) {
        redandancy_mode = !redandancy_mode;
        flag_mode_change = true;
        q_redandancy_fix = q_joint_rby1_24x1;
      }
    } else {
      flag_mode_change = false;
    }
    std::cout << "mode: " << redandancy_mode << std::endl;

    // std::cout << "start !" << std::endl;
    MobilityCommandBuilder mobility_command;
    CartesianCommandBuilder right_arm_command;
    CartesianCommandBuilder left_arm_command;
    CartesianCommandBuilder torso_command;

    {
      std::lock_guard<std::mutex> lg(mtx_q_joint_ma_info);
      {
        std::lock_guard<std::mutex> lg(mtx_hand_controller_info);

        Eigen::Vector<double, 24> q_joint_rby1_24x1_col;
        q_joint_rby1_24x1_col.setZero();

        q_joint_rby1_24x1_col.block(2 + 0, 0, 6, 1) = q_joint_rby1_24x1.block(2, 0, 6, 1);
        q_joint_rby1_24x1_col.block(2 + 6, 0, 14, 1) = q_joint_ma;

        dyn_state->SetQ(q_joint_rby1_24x1_col);
        dyn->ComputeForwardKinematics(dyn_state);
        auto res_col = dyn->DetectCollisionsOrNearestLinks(dyn_state, 1);
        bool is_coliision = false;

        if (res_col[0].distance < 0.02) {
          is_coliision = true;
        }

        if (hand_controller_button(0) && !is_coliision) {
          //right hand position control mode
          q_joint_right_target =
              q_joint_right_target * (1 - lpf_update_ratio) + q_joint_ma.block(0, 0, 7, 1) * lpf_update_ratio;
        } else {
          right_arm_minimum_time = 1.0;
        }

        if (hand_controller_button(1) && !is_coliision) {
          //left hand position control mode
          q_joint_left_target =
              q_joint_left_target * (1 - lpf_update_ratio) + q_joint_ma.block(7, 0, 7, 1) * lpf_update_ratio;
        } else {
          left_arm_minimum_time = 1.0;
        }
      }
    }

    // q_joint_right_target << 45, 0, 0, -135, 0, 45, 0;  //right arm command
    // q_joint_left_target << 45, 0, 0, -135, 0, 45, 0;   //left arm command

    // q_joint_right_target *= 3.141592 / 180.;
    // q_joint_left_target *= 3.141592 / 180.;

    linear_velocity << joy_stick_data.axisRightY / 2.0, 0;  // mobility command
    angular_velocity = -joy_stick_data.axisLeftX / 2.0;     // mobility command
    // linear_velocity << 0, 0;  // mobility command
    // angular_velocity = 0;     // mobility command

    dyn_state->SetQ(q_joint_rby1_24x1);
    dyn->ComputeForwardKinematics(dyn_state);
    T_torso = dyn->ComputeTransformation(dyn_state, 0, 1);

    Eigen::Vector<double, 3> torso_pos_command = Eigen::Vector<double, 3>::Zero();  // torso command
    Eigen::Vector<double, 3> torso_ori_command = Eigen::Vector<double, 3>::Zero();  // torso command

    torso_pos_command << joy_stick_data.buttons[11] - joy_stick_data.buttons[12],
        joy_stick_data.buttons[14] - joy_stick_data.buttons[13], joy_stick_data.buttons[5] - joy_stick_data.triggerLeft;
    torso_ori_command << joy_stick_data.buttons[1] - joy_stick_data.buttons[2],
        joy_stick_data.buttons[3] - joy_stick_data.buttons[0], joy_stick_data.buttons[4] - joy_stick_data.triggerRight;
    Eigen::Vector<double, 3> torso_se3v_ori = math::SO3::Log(T_torso.block(0, 0, 3, 3)) + torso_ori_command / 10.0;
    Eigen::Vector<double, 3> torso_se3v_pos = T_torso.block(0, 3, 3, 1) + torso_pos_command / 20.0;

    for (int i = 0; i < 3; i++) {
      torso_se3v_ori(i) = std::min(std::max(torso_se3v_ori(i), -0.5236), 0.5236);
      if (i == 2) {
        torso_se3v_pos(i) = std::min(std::max(torso_se3v_pos(i), 1.0), 1.2);
      } else {
        torso_se3v_pos(i) = std::min(std::max(torso_se3v_pos(i), -0.15), 0.15);
      }
    }
    // std::cout << "torso_se3v_pos: " << torso_se3v_pos << std::endl;
    // std::cout << "torso_se3v_ori: " << torso_se3v_ori << std::endl;
    T_torso.block(0, 0, 3, 3) = math::SO3::Exp(torso_se3v_ori);
    T_torso.block(0, 3, 3, 1) = torso_se3v_pos;
    if(redandancy_mode){
      mobility_command.SetCommand(SE2VelocityCommandBuilder()
                                      .SetVelocity(Eigen::Vector<double, 2>::Zero(), 0)
                                      .SetAccelerationLimit(linear_acceleration_limit, angular_acceleration_limit)
                                      .SetMinimumTime(wheel_minimum_time)
                                      .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(control_hold_time)));

    } else{
      mobility_command.SetCommand(SE2VelocityCommandBuilder()
                                      .SetVelocity(linear_velocity, angular_velocity)
                                      .SetAccelerationLimit(linear_acceleration_limit, angular_acceleration_limit)
                                      .SetMinimumTime(wheel_minimum_time)
                                      .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(control_hold_time)));
    }


    {
      //dual arm command
      for (int i = 0; i < 7; i++) {
        q_joint_right_target(i) = std::clamp(q_joint_right_target(i), q_joint_right_lb(i), q_joint_right_ub(i));
        q_joint_left_target(i) = std::clamp(q_joint_left_target(i), q_joint_left_lb(i), q_joint_left_ub(i));
      }

      right_arm_minimum_time *= 0.99;
      right_arm_minimum_time = std::max(right_arm_minimum_time, 0.05);

      left_arm_minimum_time *= 0.99;
      left_arm_minimum_time = std::max(left_arm_minimum_time, 0.05);



      if (!redandancy_mode) {
              Eigen::Vector<double, 24> q_joint_rby1_24x1_fk;
      q_joint_rby1_24x1_fk.setZero();

      q_joint_rby1_24x1_fk.block(2 + 0, 0, 6, 1) = q_joint_rby1_24x1.block(2, 0, 6, 1);
      q_joint_rby1_24x1_fk.block(2 + 6, 0, 7, 1) = q_joint_right_target;
      q_joint_rby1_24x1_fk.block(2 + 6 + 7, 0, 7, 1) = q_joint_left_target;

      dyn_state->SetQ(q_joint_rby1_24x1_fk);

      dyn->ComputeForwardKinematics(dyn_state);

      Eigen::Matrix<double, 4, 4> T_right_from_torso, T_left_from_torso;
      T_right_from_torso = dyn->ComputeTransformation(dyn_state, 1, 2);
      T_left_from_torso = dyn->ComputeTransformation(dyn_state, 1, 3);
        right_arm_command.AddTarget("link_torso_5", "ee_right", T_right_from_torso, 1, 3.141592, 100)
        .AddJointPositionTarget("right_arm_0", q_joint_right_target(0), 3.14, 100)
        .AddJointPositionTarget("right_arm_1", q_joint_right_target(1), 3.14, 100)
        .AddJointPositionTarget("right_arm_2", q_joint_right_target(2), 3.14, 100)
        .AddJointPositionTarget("right_arm_3", q_joint_right_target(3), 3.14, 100)
        .AddJointPositionTarget("right_arm_4", q_joint_right_target(4), 3.14, 100)
        .AddJointPositionTarget("right_arm_5", q_joint_right_target(5), 3.14, 100)
        .AddJointPositionTarget("right_arm_6", q_joint_right_target(6), 3.14, 100)
            .SetMinimumTime(right_arm_minimum_time)
            .SetStopOrientationTrackingError(stop_orientation_tracking_error/1000.0)
            .SetStopPositionTrackingError(stop_position_tracking_error/1000.0)
            .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(control_hold_time));

        left_arm_command.AddTarget("link_torso_5", "ee_left", T_left_from_torso, 1, 3.141592, 100)
        .AddJointPositionTarget("left_arm_0", q_joint_left_target(0), 3.14, 100)
        .AddJointPositionTarget("left_arm_1", q_joint_left_target(1), 3.14, 100)
        .AddJointPositionTarget("left_arm_2", q_joint_left_target(2), 3.14, 100)
        .AddJointPositionTarget("left_arm_3", q_joint_left_target(3), 3.14, 100)
        .AddJointPositionTarget("left_arm_4", q_joint_left_target(4), 3.14, 100)
        .AddJointPositionTarget("left_arm_5", q_joint_left_target(5), 3.14, 100)
        .AddJointPositionTarget("left_arm_6", q_joint_left_target(6), 3.14, 100)
            .SetMinimumTime(left_arm_minimum_time)
            .SetStopOrientationTrackingError(stop_orientation_tracking_error/1000.0)
            .SetStopPositionTrackingError(stop_position_tracking_error/1000.0)
            .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(control_hold_time));

      } else {
              Eigen::Vector<double, 24> q_joint_rby1_24x1_fk;
      q_joint_rby1_24x1_fk.setZero();

      q_joint_rby1_24x1_fk.block(2 + 0, 0, 6, 1) = q_joint_rby1_24x1.block(2, 0, 6, 1);
      q_joint_rby1_24x1_fk.block(2 + 6, 0, 7, 1) = q_joint_right_target;
      q_joint_rby1_24x1_fk.block(2 + 6 + 7, 0, 7, 1) = q_joint_left_target;

      dyn_state->SetQ(q_redandancy_fix);

      dyn->ComputeForwardKinematics(dyn_state);

      Eigen::Matrix<double, 4, 4> T_right_from_torso, T_left_from_torso;
      T_right_from_torso = dyn->ComputeTransformation(dyn_state, 0, 2);
      T_left_from_torso = dyn->ComputeTransformation(dyn_state, 0, 3);
        right_arm_command.AddTarget("base", "ee_right", T_right_from_torso, 1, 3.141592, 100)
            .AddJointPositionTarget("right_arm_2", q_joint_rby1_24x1(2+6+2) + joy_stick_data.axisRightX/10.0, 3.14, 100)
            .SetMinimumTime(right_arm_minimum_time)
            .SetStopOrientationTrackingError(stop_orientation_tracking_error)
            .SetStopPositionTrackingError(stop_position_tracking_error)
            .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(control_hold_time));

        left_arm_command.AddTarget("base", "ee_left", T_left_from_torso, 1, 3.141592, 100)
            .AddJointPositionTarget("left_arm_2", q_joint_rby1_24x1(2+6+2+7) + joy_stick_data.axisLeftX/10.0, 3.14, 100)
            .SetMinimumTime(left_arm_minimum_time)
            .SetStopOrientationTrackingError(stop_orientation_tracking_error)
            .SetStopPositionTrackingError(stop_position_tracking_error)
            .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(control_hold_time));
      }

      // right_arm_command.SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(control_hold_time))
      //     .SetMinimumTime(right_arm_minimum_time)
      //     .SetPosition(q_joint_right_target)
      //     .SetVelocityLimit(arm_vel_limit)
      //     .SetAccelerationLimit(arm_acc_limit);
      // // std::cout << "set right arm command\n";

      // left_arm_command.SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(control_hold_time))
      //     .SetMinimumTime(left_arm_minimum_time)
      //     .SetPosition(q_joint_left_target)
      //     .SetVelocityLimit(arm_vel_limit)
      //     .SetAccelerationLimit(arm_acc_limit);
      // // std::cout << "set left arm command\n";
    }

    {
      //torso command
      torso_command.AddTarget("base", "link_torso_5", T_torso, 1, 3.141592, 3)
          .SetMinimumTime(torso_minimum_time)
          .SetStopOrientationTrackingError(stop_orientation_tracking_error)
          .SetStopPositionTrackingError(stop_position_tracking_error)
          .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(control_hold_time));
      // std::cout << "set torso command\n";
    }

    {
      //whole body command
      RobotCommandBuilder whole_body_command;
      whole_body_command.SetCommand(ComponentBasedCommandBuilder()
                                        .SetMobilityCommand(mobility_command)
                                        .SetBodyCommand(BodyComponentBasedCommandBuilder()
                                                            .SetLeftArmCommand(left_arm_command)
                                                            .SetRightArmCommand(right_arm_command)
                                                            .SetTorsoCommand(torso_command)));

      // std::cout << "set whole body command\n";
      stream->SendCommand(whole_body_command);
    }
    // std::cout << "end !" << std::endl;
  }
}

void control_loop_for_master_arm(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler,
                                 std::vector<int> activeIDs) {

  auto robot = std::make_shared<rb::dyn::Robot<14>>(LoadRobotFromURDF(PATH "/master_arm.urdf", "Base"));
  auto state = robot->MakeState<std::vector<std::string>, std::vector<std::string>>(
      {"Base", "Link_0R", "Link_1R", "Link_2R", "Link_3R", "Link_4R", "Link_5R", "Link_6R", "Link_0L", "Link_1L",
       "Link_2L", "Link_3L", "Link_4L", "Link_5L", "Link_6L"},
      {"J0_Shoulder_Pitch_R", "J1_Shoulder_Roll_R", "J2_Shoulder_Yaw_R", "J3_Elbow_R", "J4_Wrist_Yaw1_R",
       "J5_Wrist_Pitch_R", "J6_Wrist_Yaw2_R", "J7_Shoulder_Pitch_L", "J8_Shoulder_Roll_L", "J9_Shoulder_Yaw_L",
       "J10_Elbow_L", "J11_Wrist_Yaw1_L", "J12_Wrist_Pitch_L", "J13_Wrist_Yaw2_L"});

  state->SetGravity({0, 0, 0, 0, 0, -9.81});

  Eigen::Matrix<double, 14, 1> q_joint, tau_joint, q_joint_target;
  Eigen::Matrix<int, 14, 1> operation_mode, torque_enable;
  std::vector<std::optional<std::pair<int, std::pair<int, int>>>> button_status_vector;
  q_joint.setZero();
  tau_joint.setZero();
  q_joint_target.setZero();

  Eigen::Matrix<double, 14, 1> temp_eigen;
  Eigen::Matrix<double, 2, 1> button_info, trigger_info;
  temp_eigen.setZero();
  button_info.setZero();
  trigger_info.setZero();

  Eigen::Vector<double, 14> q_joint_default;
  q_joint_default << 45, -30, 0, -135, -30, 90, 0, 45, 30, 0, -135, 30, 90, 0;
  q_joint_default = q_joint_default * 3.141592 / 180.;

  while (true) {
    auto start = std::chrono::steady_clock::now();

    button_status_vector.clear();
    operation_mode.setConstant(-1);
    torque_enable.setZero();
    temp_eigen.setConstant(-1);

    for (int id : activeIDs) {
      if (id >= 0x80) {
        //for hand board
        std::optional<std::pair<int, std::pair<int, int>>> temp_button_status =
            ReadButtonStatus(portHandler, packetHandler, id);
        button_status_vector.push_back(temp_button_status);
      }
    }

    std::optional<std::vector<std::pair<int, double>>> temp_q_joint_vector =
        BulkReadEncoder(portHandler, packetHandler, activeIDs);
    if (temp_q_joint_vector.has_value()) {
      for (auto const& ret : temp_q_joint_vector.value()) {
        q_joint(ret.first) = ret.second;
      }
    }

    tau_joint = ComputeGravityTorque(robot, state, q_joint) * m_sf;

    Eigen::Matrix<double, 14, 1> add_torque = calc_torque_for_limit_avoid(q_joint);

    tau_joint = tau_joint + add_torque;

    {
      std::lock_guard<std::mutex> lg(mtx_q_joint_ma_info);
      q_joint_ma = q_joint;
    }

    auto temp_operation_mode_vector = BulkReadOperationMode(portHandler, packetHandler, activeIDs);
    if (temp_operation_mode_vector.has_value()) {
      for (auto const& ret : temp_operation_mode_vector.value()) {
        operation_mode(ret.first) = ret.second;
      }
    }

    auto temp_torque_enable_vector = BulkReadTorqueEnable(portHandler, packetHandler, activeIDs);
    if (temp_torque_enable_vector.has_value()) {
      for (auto const& ret : temp_torque_enable_vector.value()) {
        torque_enable(ret.first) = ret.second;
      }
    }

    std::vector<std::pair<int, int>> id_and_enable_vector;
    for (auto const& id : activeIDs) {
      if (!torque_enable(id)) {
        id_and_enable_vector.push_back(std::make_pair(id, 1));
      }
    }

    BulkWriteTorqueEnable(portHandler, packetHandler, id_and_enable_vector);

    std::vector<std::pair<int, int>> id_and_mode_vector;
    std::vector<std::pair<int, double>> id_and_q_vector;
    std::vector<int> id_torque_onoff_vector;
    std::vector<std::pair<int, double>> id_send_torque_vector;
    // std::cout<<"is_first_init: "<<is_first_init<<std::endl;
    if (!is_first_init) {
      std::this_thread::sleep_for(10ms);
      continue;
    }

    int ma_master_mode_trigger = -2;
    for (auto& button_status : button_status_vector) {

      if (button_status.has_value()) {
        int id_hand_controlelr = button_status.value().first;
        std::pair<int, int> temp_button_status = button_status.value().second;
        int button = temp_button_status.first;
        ma_master_mode_trigger += button;
        // std::cout<<"eeee"<<std::endl;
      }
    }
    // std::cout<<"ma_master_mode_trigger: "<<ma_master_mode_trigger<<std::endl;
    if (ma_master_mode_trigger == 0) {
      ma_master_mode = true;
      init_cnt = 0;
    }

    if (!ma_master_mode) {
      Eigen::Vector<double, 14> temp_q;

      {
        std::lock_guard<std::mutex> lg(mtx_q_joint_ma_info);

        init_cnt += 0.005;

        if (init_cnt > 1) {
          init_cnt = 1.;
        }

        temp_q = q_joint_rby1_24x1.block(2 + 6, 0, 14, 1) * init_cnt + q_joint_ma * (1. - init_cnt);
      }
      // right arm
      for (int id = 0; id < 7; id++) {
        // position control
        if (operation_mode(id) != CURRENT_BASED_POSITION_CONTROL_MODE) {
          id_and_mode_vector.push_back(std::make_pair(id, CURRENT_BASED_POSITION_CONTROL_MODE));
          id_torque_onoff_vector.push_back(id);
          id_and_q_vector.push_back(std::make_pair(id, temp_q(id)));
        } else {
          id_and_q_vector.push_back(std::make_pair(id, temp_q(id)));
        }
      }

      //left arm
      for (int id = 7; id < 7 + 7; id++) {
        // position control
        if (operation_mode(id) != CURRENT_BASED_POSITION_CONTROL_MODE) {
          id_and_mode_vector.push_back(std::make_pair(id, CURRENT_BASED_POSITION_CONTROL_MODE));
          id_torque_onoff_vector.push_back(id);
          id_and_q_vector.push_back(std::make_pair(id, temp_q(id)));
        } else {
          id_and_q_vector.push_back(std::make_pair(id, temp_q(id)));
        }
      }

      BulkWriteTorqueEnable(portHandler, packetHandler, id_torque_onoff_vector, 0);
      BulkWriteOperationMode(portHandler, packetHandler, id_and_mode_vector);
      BulkWriteTorqueEnable(portHandler, packetHandler, id_torque_onoff_vector, 1);

      BulkWriteGoalPosition(portHandler, packetHandler, id_and_q_vector);
    } else {
      //master arm dual-arm command

      for (auto& button_status : button_status_vector) {

        if (button_status.has_value()) {
          int id_hand_controlelr = button_status.value().first;
          std::pair<int, int> temp_button_status = button_status.value().second;
          int button = temp_button_status.first;
          int trigger = temp_button_status.second;
          trigger_info(id_hand_controlelr - 0x80) = trigger;
          button_info(id_hand_controlelr - 0x80) = button;

          {
            std::lock_guard<std::mutex> lg(mtx_hand_controller_info);

            hand_controller_trigger(id_hand_controlelr - 0x80) =
                (trigger - hand_controller_trigger_min_max[id_hand_controlelr - 0x80](MIN_INDEX)) /
                (hand_controller_trigger_min_max[id_hand_controlelr - 0x80](MAX_INDEX) -
                 hand_controller_trigger_min_max[id_hand_controlelr - 0x80](MIN_INDEX));

            hand_controller_button(id_hand_controlelr - 0x80) = button;
          }

          if (id_hand_controlelr == 0x80) {
            // right arm
            for (int id = 0; id < 7; id++) {

              if (button == 0) {
                // position control
                if (operation_mode(id) != CURRENT_BASED_POSITION_CONTROL_MODE) {
                  id_and_mode_vector.push_back(std::make_pair(id, CURRENT_BASED_POSITION_CONTROL_MODE));
                  id_torque_onoff_vector.push_back(id);
                }
              } else if (button == 1) {
                // current control
                if (operation_mode(id) != CURRENT_CONTROL_MODE) {
                  id_and_mode_vector.push_back(std::make_pair(id, CURRENT_CONTROL_MODE));
                  id_torque_onoff_vector.push_back(id);
                } else {
                  id_send_torque_vector.push_back(std::make_pair(id, tau_joint(id)));
                }
              }
            }
          }

          if (id_hand_controlelr == 0x81) {
            //left arm
            for (int id = 7; id < 7 + 7; id++) {

              if (button == 0) {
                // position control
                if (operation_mode(id) != CURRENT_BASED_POSITION_CONTROL_MODE) {
                  id_and_mode_vector.push_back(std::make_pair(id, CURRENT_BASED_POSITION_CONTROL_MODE));
                  id_torque_onoff_vector.push_back(id);
                }
              } else if (button == 1) {
                // current control
                if (operation_mode(id) != CURRENT_CONTROL_MODE) {
                  id_and_mode_vector.push_back(std::make_pair(id, CURRENT_CONTROL_MODE));
                  id_torque_onoff_vector.push_back(id);
                } else {
                  id_send_torque_vector.push_back(std::make_pair(id, tau_joint(id)));
                }
              }
            }
          }
        }
      }

      BulkWriteTorqueEnable(portHandler, packetHandler, id_torque_onoff_vector, 0);
      BulkWriteOperationMode(portHandler, packetHandler, id_and_mode_vector);
      BulkWriteTorqueEnable(portHandler, packetHandler, id_torque_onoff_vector, 1);

      BulkWriteSendTorque(portHandler, packetHandler, id_send_torque_vector);
    }

    static int cnt = 0;
    if (ma_info_verbose) {
      if (cnt++ % 5 == 0) {

        std::cout << "button_info : " << button_info.transpose() << std::endl;
        std::cout << "trigger_info : " << trigger_info.transpose() << std::endl;
        std::cout << "right q_joint [deg]: " << q_joint.block(0, 0, 7, 1).transpose() * 180. / 3.141592 << std::endl;
        std::cout << "left q_joint [deg]: " << q_joint.block(7, 0, 7, 1).transpose() * 180. / 3.141592 << std::endl;

        std::cout << "Duration: "
                  << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                                  start)
                             .count() /
                         1.e6
                  << " ms" << std::endl;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void control_loop_for_gripper(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler,
                              std::vector<int> activeIDs) {

  std::vector<Eigen::Matrix<double, 2, 1>> q_min_max_vector;
  q_min_max_vector.push_back(Eigen::Matrix<double, 2, 1>::Zero());
  q_min_max_vector.push_back(Eigen::Matrix<double, 2, 1>::Zero());

  while (1) {
    static int cnt = 0;
    int is_init = true;

    if (activeIDs.size() != 2) {
      std::cout << "The number of Dynamixels for hand gripper does not match the configuration\n";
      return;
    }

    // total moving angle 540 deg

    for (auto const& id : activeIDs) {
      while (1) {

        std::optional<int> operation_mode = ReadOperationMode(portHandler, packetHandler, id);

        if (operation_mode.has_value()) {
          if (operation_mode.value() != CURRENT_CONTROL_MODE) {
            TorqueEnable(portHandler, packetHandler, id, 0);
            SendOperationMode(portHandler, packetHandler, id, CURRENT_CONTROL_MODE);
            TorqueEnable(portHandler, packetHandler, id, 1);
            std::cout << "try to change control mode, id : " << id << std::endl;
          } else {
            break;
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      while (1) {

        std::optional<int> torque_enable = ReadTorqueEnable(portHandler, packetHandler, id);

        if (torque_enable.has_value()) {
          if (!torque_enable.value()) {
            TorqueEnable(portHandler, packetHandler, id, 1);
            std::cout << "try to enable torque, id : " << id << std::endl;
          } else {
            break;
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      if (cnt % 2 == 0) {
        std::optional<double> q = ReadEncoder(portHandler, packetHandler, id);
        if (q.has_value()) {
          q_min_max_vector[id](cnt % 2) = q.value();
        }
        SendCurrent(portHandler, packetHandler, id, 0.5);
      } else {
        std::optional<double> q = ReadEncoder(portHandler, packetHandler, id);
        if (q.has_value()) {
          q_min_max_vector[id](cnt % 2) = q.value();
        }
        SendCurrent(portHandler, packetHandler, id, -0.5);
      }

      if ((double)(abs(q_min_max_vector[id](MAX_INDEX) - q_min_max_vector[id](MIN_INDEX))) * 180 / 3.141592 <
          540 * 0.9) {
        is_init = false;
      }
    }

    if (is_init) {
      for (auto const& id : activeIDs) {
        if (q_min_max_vector[id](MIN_INDEX) > q_min_max_vector[id](MAX_INDEX)) {
          double temp = q_min_max_vector[id](MIN_INDEX);
          q_min_max_vector[id](MIN_INDEX) = q_min_max_vector[id](MAX_INDEX);
          q_min_max_vector[id](MAX_INDEX) = temp;
        }

        SendCurrent(portHandler, packetHandler, id, 0.5);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(3000));
      break;
    }

    cnt++;

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }

  std::cout << "finish init\n";

  while (1) {

    for (auto const& id : activeIDs) {
      std::optional<int> operation_mode = ReadOperationMode(portHandler, packetHandler, id);
      if (operation_mode.has_value()) {
        if (operation_mode.value() != CURRENT_BASED_POSITION_CONTROL_MODE) {
          TorqueEnable(portHandler, packetHandler, id, 0);
          SendOperationMode(portHandler, packetHandler, id, CURRENT_BASED_POSITION_CONTROL_MODE);
          TorqueEnable(portHandler, packetHandler, id, 1);
        } else {
          double goal_position = 0;

          {
            std::lock_guard<std::mutex> lg(mtx_hand_controller_info);

            static Eigen::Vector<double, 2> temp_hand_controller_trigger = Eigen::Matrix<double, 2, 1>::Constant(0.5);

            temp_hand_controller_trigger(id) =
                temp_hand_controller_trigger(id) * 0.9 + hand_controller_trigger(id) * 0.1;

            temp_hand_controller_trigger = (temp_hand_controller_trigger.array().max(0.).min(1.)).matrix();

            hand_controller_trigger(id) = (double)((int)(temp_hand_controller_trigger(id) * 100)) / 100.;

            if (gripper_direction) {
              goal_position = hand_controller_trigger(id) * q_min_max_vector[id](MAX_INDEX) +
                              (1. - hand_controller_trigger(id)) * q_min_max_vector[id](MIN_INDEX);

            } else {
              goal_position = hand_controller_trigger(id) * q_min_max_vector[id](MIN_INDEX) +
                              (1. - hand_controller_trigger(id)) * q_min_max_vector[id](MAX_INDEX);
            }
          }

          SendGoalPosition(portHandler, packetHandler, id, (int)(goal_position * 4096. / 3.141592 / 2.));
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

std::string resolve_symlink(const std::string& symlink) {
  char buf[1024];
  ssize_t len = readlink(symlink.c_str(), buf, sizeof(buf) - 1);
  if (len != -1) {
    buf[len] = '\0';
    return std::string(buf);
  }
  return "";
}

int main(int argc, char** argv) {

  QApplication app(argc, argv);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <server address> [servo]" << std::endl;
    return 1;
  }

  UdpThread thread;
  thread.start();
  network = new Network();

  std::string address{argv[1]};
  std::string servo = ".*";  // 기본값

  if (argc >= 3) {
    servo = argv[2];
  }

  try {
    // Latency timer setting
    upc::InitializeDevice(upc::kGripperDeviceName);
    upc::InitializeDevice(upc::kMasterArmDeviceName);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  auto robot = rb::Robot<y1_model::A>::Create(address);

  std::cout << "Attempting to connect to the robot..." << std::endl;
  if (!robot->Connect()) {
    std::cerr << "Error: Unable to establish connection to the robot at " << address << std::endl;
    return 1;
  }
  std::cout << "Successfully connected to the robot." << std::endl;

  std::cout << "Starting state update..." << std::endl;

  robot->SetParameter("joint_position_command.cutoff_frequency", "5.0");

  robot->StartStateUpdate(
      [&](const auto& state) {
        if (ma_info_verbose) {
          std::cout << "State Update Received:" << std::endl;
          std::cout << "  Timestamp: " << state.timestamp.tv_sec << ".";
          std::cout << std::setw(9) << std::setfill('0') << state.timestamp.tv_nsec << std::endl;
          std::cout << "  wasit [deg]     : " << state.position.block(2, 0, 6, 1).transpose() * R2D << std::endl;
          std::cout << "  right arm [deg] : " << state.position.block(2 + 6, 0, 7, 1).transpose() * R2D << std::endl;
          std::cout << "  left arm [deg]  : " << state.position.block(2 + 6 + 7, 0, 7, 1).transpose() * R2D
                    << std::endl;
        }

        q_joint_rby1_24x1.block(2, 0, 20, 1) = state.position.block(2, 0, 20, 1);
      },
      100 /* Hz */);

  std::cout << "!!\n";

  std::this_thread::sleep_for(1s);

  std::cout << "Checking power status..." << std::endl;
  if (!robot->IsPowerOn(kAll)) {
    std::cout << "Power is currently OFF. Attempting to power on..." << std::endl;
    if (!robot->PowerOn(kAll)) {
      std::cerr << "Error: Failed to power on the robot." << std::endl;
      return 1;
    }
    std::cout << "Robot powered on successfully." << std::endl;
  } else {
    std::cout << "Power is already ON." << std::endl;
  }

  std::cout << "Checking servo status..." << std::endl;
  if (!robot->IsServoOn(servo)) {
    std::cout << "Servo is currently OFF. Attempting to activate servo..." << std::endl;
    if (!robot->ServoOn(servo)) {
      std::cerr << "Error: Failed to activate servo." << std::endl;
      return 1;
    }
    std::cout << "Servo activated successfully." << std::endl;
  } else {
    std::cout << "Servo is already ON." << std::endl;
  }

  const auto& control_manager_state = robot->GetControlManagerState();
  if (control_manager_state.state == ControlManagerState::State::kMajorFault ||
      control_manager_state.state == ControlManagerState::State::kMinorFault) {
    std::cerr << "Warning: Detected a "
              << (control_manager_state.state == ControlManagerState::State::kMajorFault ? "Major" : "Minor")
              << " Fault in the Control Manager." << std::endl;

    std::cout << "Attempting to reset the fault..." << std::endl;
    if (!robot->ResetFaultControlManager()) {
      std::cerr << "Error: Unable to reset the fault in the Control Manager." << std::endl;
      return 1;
    }
    std::cout << "Fault reset successfully." << std::endl;
  }
  std::cout << "Control Manager state is normal. No faults detected." << std::endl;

  std::cout << "Enabling the Control Manager..." << std::endl;
  if (!robot->EnableControlManager()) {
    std::cerr << "Error: Failed to enable the Control Manager." << std::endl;
    return 1;
  }
  std::cout << "Control Manager enabled successfully." << std::endl;

  if (robot->IsPowerOn(".*")) {
    robot->SetToolFlangeOutputVoltage("right", 12);
    robot->SetToolFlangeOutputVoltage("left", 12);
    std::cout << "Attempting to 12V power on for gripper." << std::endl;

    std::this_thread::sleep_for(1s);
  }

  const char* devicename_master_arm = "/dev/rby1_master_arm";

  dynamixel::PortHandler* portHandler = dynamixel::PortHandler::getPortHandler(devicename_master_arm);
  dynamixel::PacketHandler* packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

  if (!portHandler->openPort()) {
    std::cerr << "Failed to open the port!" << std::endl;
    return 1;
  }

  if (!portHandler->setBaudRate(BAUDRATE)) {
    std::cerr << "Failed to change the baudrate!" << std::endl;
    return 1;
  }

  std::vector<int> activeIDs;

  for (int id = 0; id < 14; ++id) {
    uint8_t dxl_error = 0;
    int dxl_comm_result = packetHandler->ping(portHandler, id, &dxl_error);
    if (dxl_comm_result == COMM_SUCCESS) {
      std::cout << "Dynamixel ID " << id << " is active." << std::endl;
      activeIDs.push_back(id);
    } else {
      std::cerr << "Dynamixel ID " << id << " is not active." << std::endl;
    }
  }

  for (int id = 0x80; id < 0x80 + 2; id++) {
    uint8_t dxl_error = 0;
    int dxl_comm_result = packetHandler->ping(portHandler, id, &dxl_error);
    if (dxl_comm_result == COMM_SUCCESS) {
      std::cout << "Dynamixel ID " << id << " is active." << std::endl;
      activeIDs.push_back(id);
    } else {
      std::cerr << "Dynamixel ID " << id << " is not active." << std::endl;
    }
  }

  if (activeIDs.size() != 16) {
    std::cerr << "Unable to ping all devices for master arm" << std::endl;
    Eigen::Map<Eigen::VectorXi> ids(activeIDs.data(), activeIDs.size());
    std::cerr << "active ids: " << ids.transpose() << std::endl;
    return 1;
  }

  for (int id : activeIDs) {
    if (id < 0x80) {
      int dxl_comm_result = packetHandler->write1ByteTxOnly(portHandler, id, ADDR_OPERATING_MODE, CURRENT_CONTROL_MODE);
      if (dxl_comm_result != COMM_SUCCESS) {
        std::cerr << "Failed to write current control mode value: " << packetHandler->getTxRxResult(dxl_comm_result)
                  << std::endl;
        return 1;
      }

      TorqueEnable(portHandler, packetHandler, id, 1);
    }
  }

  std::thread master_arm_handler(control_loop_for_master_arm, portHandler, packetHandler, activeIDs);

  const char* devicename_gripper = "/dev/rby1_gripper";

  dynamixel::PortHandler* portHandler_gripper = dynamixel::PortHandler::getPortHandler(devicename_gripper);
  dynamixel::PacketHandler* packetHandler_gripper = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

  if (!portHandler_gripper->openPort()) {
    std::cerr << "Failed to open the port!" << std::endl;
    return 1;
  }

  if (!portHandler_gripper->setBaudRate(BAUDRATE)) {
    std::cerr << "Failed to change the baudrate!" << std::endl;
    return 1;
  }

  std::vector<int> activeIDs_gripper;

  for (int id = 0; id < 2; ++id) {
    uint8_t dxl_error = 0;
    int dxl_comm_result = packetHandler_gripper->ping(portHandler_gripper, id, &dxl_error);
    if (dxl_comm_result == COMM_SUCCESS) {
      std::cout << "Dynamixel ID " << id << " is active." << std::endl;
      activeIDs_gripper.push_back(id);
    } else {
      std::cerr << "Dynamixel ID " << id << " is not active." << std::endl;
    }
  }

  if (activeIDs_gripper.size() != 2) {
    std::cerr << "Unable to ping all devices for grippers" << std::endl;
    Eigen::Map<Eigen::VectorXi> ids(activeIDs_gripper.data(), activeIDs_gripper.size());
    std::cerr << "active ids: " << ids.transpose() << std::endl;
    return 1;
  }

  for (int id : activeIDs_gripper) {
    int dxl_comm_result =
        packetHandler_gripper->write1ByteTxOnly(portHandler_gripper, id, ADDR_OPERATING_MODE, CURRENT_CONTROL_MODE);
    if (dxl_comm_result != COMM_SUCCESS) {
      std::cerr << "Failed to write current control mode value: "
                << packetHandler_gripper->getTxRxResult(dxl_comm_result) << std::endl;
      return 1;
    }

    TorqueEnable(portHandler_gripper, packetHandler_gripper, id, 1);
  }

  std::thread gripper_handler(control_loop_for_gripper, portHandler_gripper, packetHandler_gripper, activeIDs_gripper);

  std::this_thread::sleep_for(1s);

  std::thread robot_op(control_loop_for_robot, robot);

  // thread.wait();
  // return app.exec();

  return app.exec();
}

#include "whole_body_command.moc"