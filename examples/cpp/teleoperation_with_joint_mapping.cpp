#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__s
#endif

#include <Eigen/Core>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include "dynamixel_sdk.h"  // Uses Dynamixel SDK library
#include "rby1-sdk/dynamics/robot.h"
#include "rby1-sdk/dynamics/state.h"

#include "rby1-sdk/model.h"
#include "rby1-sdk/robot.h"
#include "rby1-sdk/robot_command_builder.h"

#include "rby1-sdk/upc/device.h"

#include <unistd.h>
#include <algorithm>
#include <cstring>

#include <QApplication>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

using namespace rb;
using namespace std::chrono_literals;

const std::string kAll = ".*";

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

double m_sf = 0.4;

double init_cnt = 0.;
bool init_operation = false;

using namespace rb::dyn;
using namespace rb;

#define MIN_INDEX 0
#define MAX_INDEX 1

enum class ControlMode {
  DUAL_ARM,
  TORSO,
  WHEEL,
};

ControlMode control_mode = ControlMode::DUAL_ARM;
ControlMode control_mode_old = ControlMode::DUAL_ARM;

std::mutex mtx_q_joint_ma_info;
Eigen::Matrix<double, 14, 1> q_joint_ma = Eigen::Matrix<double, 14, 1>::Zero();
Eigen::Matrix<double, 14, 1> q_joint_rby1 = Eigen::Matrix<double, 14, 1>::Zero();
Eigen::Matrix<double, 6, 1> q_joint_rby1_waist = Eigen::Matrix<double, 6, 1>::Zero();

std::mutex mtx_hand_controller_info;
Eigen::Matrix<double, 2, 1> hand_controller_trigger = Eigen::Matrix<double, 2, 1>::Constant(0.5);
Eigen::Matrix<double, 2, 1> hand_controller_button = Eigen::Matrix<double, 2, 1>::Constant(0);

Eigen::Vector<double, 3> rpy_right = Eigen::Vector<double, 3>::Zero();
Eigen::Vector<double, 3> rpy_left = Eigen::Vector<double, 3>::Zero();

//XXX: CAUTION CHECK YOUR TRIGGER FIRMWARE
std::vector<Eigen::Matrix<double, 2, 1>> hand_controller_trigger_min_max = {Eigen::Matrix<double, 2, 1>({0, 1000}),
                                                                            Eigen::Matrix<double, 2, 1>({0, 1000})};
int gripper_direction = 0;
bool ma_info_verbose = false;

std::vector<double> torque_constant = {1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043,
                                       1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043};

int main_window(int _argc, char** _argv) {

  QApplication app(_argc, _argv);

  // Create a main window widget
  QWidget window;
  window.setWindowTitle("Master Arm Control Window");

  // Create two buttons
  QPushButton* button_dual_arm_joint_mapping = new QPushButton("Dual Arm Joint Mapping");
  QPushButton* button_torso_task_mapping = new QPushButton("Torso Task Mapping");
  QPushButton* button_wheel_velocity_mapping = new QPushButton("Wheel Velocity Mapping");
  QPushButton* button_test = new QPushButton("QUIT");

  // Connect the buttons to their respective functions
  QObject::connect(button_dual_arm_joint_mapping, &QPushButton::clicked, []() {
    control_mode = ControlMode::DUAL_ARM;
    qDebug("Control Mode: Dual Arm!");
  });
  QObject::connect(button_torso_task_mapping, &QPushButton::clicked, []() {
    control_mode = ControlMode::TORSO;
    qDebug("Control Mode: Torso!");
  });
  QObject::connect(button_wheel_velocity_mapping, &QPushButton::clicked, []() {
    control_mode = ControlMode::WHEEL;
    qDebug("Control Mode: Wheel!");
  });
  QObject::connect(button_test, &QPushButton::clicked, &app, &QApplication::quit);

  // Set up a vertical layout and add the buttons
  QVBoxLayout* layout = new QVBoxLayout;
  layout->addWidget(button_dual_arm_joint_mapping);
  layout->addWidget(button_torso_task_mapping);
  layout->addWidget(button_wheel_velocity_mapping);
  layout->addWidget(button_test);

  // Set the layout for the main window and show it
  window.setLayout(layout);
  window.resize(300, 200);

  window.show();

  return app.exec();
}

void SendVibration(dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler, int id, int level) {

  if (id > 0x80) {
    packetHandler->write2ByteTxOnly(portHandler, id, ADDR_GOAL_VIBRATION_LEVEL, level);
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
}

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

std::pair<Eigen::Vector<double, 3>, Eigen::Vector<double, 3>> calc_diff(std::shared_ptr<dyn::Robot<14>> robot,
                                                                        std::shared_ptr<dyn::State<14>> state,
                                                                        Eigen::Vector<double, 14> q_joint_target,
                                                                        Eigen::Vector<double, 14> q_joint) {
  Eigen::Matrix<double, 4, 4> T_right, T_left, T_right_target, T_left_target;

  state->SetQ(q_joint_target);
  robot->ComputeForwardKinematics(state);
  T_right_target = robot->ComputeTransformation(state, 0, 7);
  T_left_target = robot->ComputeTransformation(state, 0, 14);

  state->SetQ(q_joint);
  robot->ComputeForwardKinematics(state);
  T_right = robot->ComputeTransformation(state, 0, 7);
  T_left = robot->ComputeTransformation(state, 0, 14);

  Eigen::Matrix<double, 4, 4> T_right_diff = T_right_target.inverse() * T_right;
  Eigen::Matrix<double, 4, 4> T_left_diff = T_left_target.inverse() * T_left;

  Eigen::Vector<double, 3> diff_right, diff_left;
  diff_right.topRows(3) = math::SE3::Log(T_right_diff).topRows(3);
  diff_left.topRows(3) = math::SE3::Log(T_left_diff).topRows(3);

  int maxIndex;
  Eigen::Vector<double, 3> unit_axis;

  diff_right.cwiseAbs().maxCoeff(&maxIndex);
  unit_axis.setZero();
  unit_axis(maxIndex) = 1.;
  diff_right = diff_right.cwiseProduct(unit_axis);

  diff_left.cwiseAbs().maxCoeff(&maxIndex);
  unit_axis.setZero();
  unit_axis(maxIndex) = 1.;
  diff_left = diff_left.cwiseProduct(unit_axis);

  diff_right = diff_right.unaryExpr([](double v) {
    double saturation = 0.3;
    double offset = 0.15;
    double abs_v = std::abs(v);
    if (abs_v < offset) {
      return 0.0;
    } else if (abs_v <= saturation) {
      double scaled = (abs_v - offset) / (saturation - offset);
      return v < 0 ? -scaled : scaled;
    } else {
      return v < 0 ? -1.0 : 1.0;
    }
  });

  diff_left = diff_left.unaryExpr([](double v) {
    double saturation = 0.3;
    double offset = 0.15;
    double abs_v = std::abs(v);
    if (abs_v < offset) {
      return 0.0;
    } else if (abs_v <= saturation) {
      double scaled = (abs_v - offset) / (saturation - offset);
      return v < 0 ? -scaled : scaled;
    } else {
      return v < 0 ? -1.0 : 1.0;
    }
  });

  return std::make_pair(diff_right, diff_left);
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

    if (!init_operation) {
      Eigen::Vector<double, 14> temp_q;

      {
        std::lock_guard<std::mutex> lg(mtx_q_joint_ma_info);

        init_cnt += 0.01;

        if (init_cnt > 1) {
          init_operation = true;
          init_cnt = 1.;
        }
      }

      if (control_mode == ControlMode::DUAL_ARM) {
        temp_q = q_joint_rby1 * init_cnt + q_joint_ma * (1. - init_cnt);
      } else {
        temp_q = q_joint_default * init_cnt + q_joint_ma * (1. - init_cnt);
      }
      std::cout<<"temp_q: "<<temp_q.transpose()<<std::endl;
      std::cout<<"q_joint_rby1: "<<q_joint_rby1.transpose()<<std::endl;

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
    }

    if (control_mode == ControlMode::DUAL_ARM && init_operation) {
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
    } else if (control_mode == ControlMode::TORSO && init_operation) {
      //master arm torso command

      std::optional<std::vector<std::pair<int, double>>> q_target_vector =
          BulkReadGoalPosition(portHandler, packetHandler, activeIDs);
      if (q_target_vector.has_value()) {
        for (auto const& ret : q_target_vector.value()) {
          q_joint_target(ret.first) = ret.second;
        }
      }

      auto ret = calc_diff(robot, state, q_joint_target, q_joint);

      rpy_right = ret.first;
      rpy_left = ret.second;

      std::cout << "RPY_RIGHT: " << rpy_right << std::endl;
      std::cout << "RPY_LEFT: " << rpy_left << std::endl;

      // right arm
      for (int id = 0; id < 7; id++) {
        // position control
        if (operation_mode(id) != CURRENT_BASED_POSITION_CONTROL_MODE) {
          id_and_mode_vector.push_back(std::make_pair(id, CURRENT_BASED_POSITION_CONTROL_MODE));
          id_torque_onoff_vector.push_back(id);
          id_and_q_vector.push_back(std::make_pair(id, q_joint_default(id)));
        }else{
          id_and_q_vector.push_back(std::make_pair(id, q_joint_default(id)));
        }
      }

      //left arm
      for (int id = 7; id < 7 + 7; id++) {
        // position control
        if (operation_mode(id) != CURRENT_BASED_POSITION_CONTROL_MODE) {
          id_and_mode_vector.push_back(std::make_pair(id, CURRENT_BASED_POSITION_CONTROL_MODE));
          id_torque_onoff_vector.push_back(id);
          id_and_q_vector.push_back(std::make_pair(id, q_joint_default(id)));
        }else{
          id_and_q_vector.push_back(std::make_pair(id, q_joint_default(id)));
        }
      }

      BulkWriteTorqueEnable(portHandler, packetHandler, id_torque_onoff_vector, 0);
      BulkWriteOperationMode(portHandler, packetHandler, id_and_mode_vector);
      BulkWriteTorqueEnable(portHandler, packetHandler, id_torque_onoff_vector, 1);

      BulkWriteGoalPosition(portHandler, packetHandler, id_and_q_vector);

    } else if (control_mode == ControlMode::WHEEL && init_operation) {
      //master arm wheel command

      std::optional<std::vector<std::pair<int, double>>> q_target_vector =
          BulkReadGoalPosition(portHandler, packetHandler, activeIDs);
      if (q_target_vector.has_value()) {
        for (auto const& ret : q_target_vector.value()) {
          q_joint_target(ret.first) = ret.second;
        }
      }

      auto ret = calc_diff(robot, state, q_joint_target, q_joint);

      rpy_right = ret.first;
      rpy_left = ret.second;
      std::cout << "RPY_RIGHT: " << rpy_right << std::endl;
      std::cout << "RPY_LEFT: " << rpy_left << std::endl;

      // right arm
      for (int id = 0; id < 7; id++) {
        // position control
        if (operation_mode(id) != CURRENT_BASED_POSITION_CONTROL_MODE) {
          id_and_mode_vector.push_back(std::make_pair(id, CURRENT_BASED_POSITION_CONTROL_MODE));
          id_torque_onoff_vector.push_back(id);
          id_and_q_vector.push_back(std::make_pair(id, q_joint_default(id)));
        }else{
          id_and_q_vector.push_back(std::make_pair(id, q_joint_default(id)));
        }
      }

      //left arm
      for (int id = 7; id < 7 + 7; id++) {
        // position control
        if (operation_mode(id) != CURRENT_BASED_POSITION_CONTROL_MODE) {
          id_and_mode_vector.push_back(std::make_pair(id, CURRENT_BASED_POSITION_CONTROL_MODE));
          id_torque_onoff_vector.push_back(id);
          id_and_q_vector.push_back(std::make_pair(id, q_joint_default(id)));
        }else{
          id_and_q_vector.push_back(std::make_pair(id, q_joint_default(id)));
        }
      }

      BulkWriteTorqueEnable(portHandler, packetHandler, id_torque_onoff_vector, 0);
      BulkWriteOperationMode(portHandler, packetHandler, id_and_mode_vector);
      BulkWriteTorqueEnable(portHandler, packetHandler, id_torque_onoff_vector, 1);

      BulkWriteGoalPosition(portHandler, packetHandler, id_and_q_vector);
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

void control_loop_for_qt(int argc_, char** argv_) {
  { auto ret = main_window(argc_, argv_); }
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

  try {
    // Latency timer setting
    upc::InitializeDevice(upc::kGripperDeviceName);
    upc::InitializeDevice(upc::kMasterArmDeviceName);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <server address> [servo]" << std::endl;
    return 1;
  }

  std::string address{argv[1]};
  std::string servo = ".*";  // 기본값

  if (argc >= 3) {
    servo = argv[2];
  }

  auto robot = rb::Robot<y1_model::A>::Create(address);

  std::cout << "Attempting to connect to the robot..." << std::endl;
  if (!robot->Connect()) {
    std::cerr << "Error: Unable to establish connection to the robot at " << address << std::endl;
    return 1;
  }
  std::cout << "Successfully connected to the robot." << std::endl;

  std::cout << "Starting state update..." << std::endl;

  robot->StartStateUpdate(
      [&](const auto& state) {
        static int cnt_cout = 0;
        static bool flag = false;
        if (cnt_cout++ > 10) {
          std::cout << "  waist [deg]     : " << state.position.block(2, 0, 6, 1).transpose() * R2D << std::endl;
          std::cout << "  right arm [deg] : " << state.position.block(2 + 6, 0, 7, 1).transpose() * R2D << std::endl;
          std::cout << "  left arm [deg]  : " << state.position.block(2 + 6 + 7, 0, 7, 1).transpose() * R2D
                    << std::endl;
        }
        if(!flag){
          q_joint_ma =state.position.block(2 + 6, 0, 14, 1);
          flag = true;
        }
        q_joint_rby1_waist = state.position.block(2, 0, 6, 1);
        q_joint_rby1 = state.position.block(2 + 6, 0, 14, 1);
      },
      50 /* Hz */);

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

  if (robot->IsPowerOn("48v")) {
    robot->SetToolFlangeOutputVoltage("right", 12);
    robot->SetToolFlangeOutputVoltage("left", 12);
    std::cout << "Attempting to 12V power on for gripper." << std::endl;

    std::this_thread::sleep_for(1s);
  }

  robot->SetParameter("joint_position_command.cutoff_frequency", "10.0");
  std::cout << robot->GetParameter("joint_position_command.cutoff_frequency") << std::endl;

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

  std::thread qt_handler(control_loop_for_qt, argc, argv);

  if (1) {

    robot
        ->SendCommand(RobotCommandBuilder().SetCommand(
            ComponentBasedCommandBuilder().SetBodyCommand(BodyComponentBasedCommandBuilder().SetTorsoCommand(
                JointPositionCommandBuilder()
                    .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(1.))
                    .SetMinimumTime(5)
                    .SetPosition(Eigen::Vector<double, 6>{0, 30, -60, 30, 0, 0} * D2R)))))
        ->Get();

    Eigen::Matrix<double, 14, 1> q_joint_ref;
    q_joint_ref.setZero();

    {
      std::lock_guard<std::mutex> lg(mtx_q_joint_ma_info);
      q_joint_ref = q_joint_ma;
    }

    auto dyn = robot->GetDynamics();
    auto dyn_state = dyn->MakeState({"base", "link_torso_5", "ee_right", "ee_left"}, y1_model::A::kRobotJointNames);

    Eigen::Vector<double, 14> q_lower_limit, q_upper_limit;
    q_upper_limit = dyn->GetLimitQUpper(dyn_state).block(2 + 6, 0, 14, 1);
    q_lower_limit = dyn->GetLimitQLower(dyn_state).block(2 + 6, 0, 14, 1);

    // auto stream = robot->CreateCommandStream();
    std::unique_ptr<RobotCommandStreamHandler<y1_model::A>> stream;

    std::this_thread::sleep_for(5s);

    double right_arm_minimum_time = 1.;
    double left_arm_minimum_time = 1.;
    double lpf_update_ratio = 0.1;
    double torso_minimum_time = 1.0;
    double wheel_minimum_time = 5.0;
    int control_cnt = 0;
    Eigen::Vector<double, 3> add_pos = Eigen::Vector<double, 3>::Zero();
    Eigen::Vector<double, 3> add_ori = Eigen::Vector<double, 3>::Zero();
    Eigen::Matrix<double, 4, 4> T_torso;
    T_torso.setIdentity();
    double update_ratio = 0.;

    std::cout << "Start to send command to robot." << std::endl;

    Eigen::Matrix<double, 20, 1> q_joint_ref_20x1;
    q_joint_ref_20x1.setZero();
    q_joint_ref_20x1.block(0, 0, 6, 1) << 0, 30, -60, 30, 0, 0;
    q_joint_ref_20x1 *= D2R;
    q_joint_rby1_waist = q_joint_ref_20x1.block(0,0,6,1);

    while (1) {
      //robot control loop
      std::this_thread::sleep_for(1ms);
      if (control_mode_old != control_mode) {
        update_ratio = 1.0;

        if (stream != nullptr) {
          stream.reset();
          stream = nullptr;
        }

        Eigen::Vector<double, 24> dyn_q;
            q_joint_ref_20x1.block(0, 0, 6, 1) =
        q_joint_ref_20x1.block(0, 0, 6, 1) * (1 - lpf_update_ratio) + q_joint_rby1_waist * lpf_update_ratio;

        dyn_q.setZero();
        dyn_q.block(2, 0, 20, 1) = q_joint_ref_20x1;
        dyn_state->SetQ(dyn_q);
        dyn->ComputeForwardKinematics(dyn_state);

        T_torso = dyn->ComputeTransformation(dyn_state, 0, 1);

        add_pos.setZero();
        add_ori.setZero();
        rpy_right.setZero();
        rpy_left.setZero();

        right_arm_minimum_time = 1.0;
        left_arm_minimum_time = 1.0;
        torso_minimum_time = 1.0;
        wheel_minimum_time = 5.0;
        control_cnt = 0;
        init_cnt = 0.;
        init_operation = false;
      }
      control_mode_old = control_mode;

      if (control_mode == ControlMode::DUAL_ARM) {
        std::cout << "DUAL_ARM START" << std::endl;
        if (stream == nullptr || stream->IsDone()) {
          std::cout<<"Create Stream 111"<<std::endl;
          stream = robot->CreateCommandStream();
          std::cout<<"Create Stream 222"<<std::endl;
        }

        if (control_cnt++ > 100) {
          control_cnt = 999;

          {
            std::lock_guard<std::mutex> lg(mtx_q_joint_ma_info);
            {
              std::lock_guard<std::mutex> lg(mtx_hand_controller_info);

              q_joint_ref_20x1.block(6, 0, 14, 1) =
                  q_joint_ref_20x1.block(6, 0, 14, 1) * (1 - lpf_update_ratio) + q_joint_ma * lpf_update_ratio;

              q_joint_ref_20x1.block(0, 0, 6, 1) =
                  q_joint_ref_20x1.block(0, 0, 6, 1) * (1 - lpf_update_ratio) + q_joint_rby1_waist * lpf_update_ratio;

              Eigen::Matrix<double, 24, 1> q_joint_rby1_col_model;
              q_joint_rby1_col_model.setZero();
              q_joint_rby1_col_model.block(2, 0, 20, 1) = q_joint_ref_20x1;
              std::cout << "SET Q1111" << std::endl;
              dyn_state->SetQ(q_joint_rby1_col_model);
              std::cout << "SET Q2222" << std::endl;
              std::cout << "ComputeForwardKinematics 111" << std::endl;
              dyn->ComputeForwardKinematics(dyn_state);
              std::cout << "ComputeForwardKinematics 222" << std::endl;
              auto res_col = dyn->DetectCollisionsOrNearestLinks(dyn_state, 1);
              bool is_coliision = false;

              if (res_col[0].distance < 0.02) {
                is_coliision = true;
              }

              if (hand_controller_button(0) && !is_coliision) {
                //right hand position control mode
                q_joint_ref.block(0, 0, 7, 1) = q_joint_ref.block(0, 0, 7, 1) * (1 - lpf_update_ratio) +
                                                q_joint_ma.block(0, 0, 7, 1) * lpf_update_ratio;
              } else {
                right_arm_minimum_time = 1.0;
              }

              if (hand_controller_button(1) && !is_coliision) {
                //left hand position control mode
                q_joint_ref.block(7, 0, 7, 1) = q_joint_ref.block(7, 0, 7, 1) * (1 - lpf_update_ratio) +
                                                q_joint_ma.block(7, 0, 7, 1) * lpf_update_ratio;
              } else {
                left_arm_minimum_time = 1.0;
              }
            }
          }

          for (int i = 0; i < 14; i++) {
            q_joint_ref(i) = std::clamp(q_joint_ref(i), q_lower_limit(i), q_upper_limit(i));
          }
          Eigen::Vector<double, 7> target_position_left = q_joint_ref.block(7, 0, 7, 1);
          Eigen::Vector<double, 7> target_position_right = q_joint_ref.block(0, 0, 7, 1);
          Eigen::Vector<double, 7> acc_limit, vel_limit;

          acc_limit.setConstant(1200.0);
          acc_limit *= D2R;

          vel_limit << 160, 160, 160, 160, 330, 330, 330;
          vel_limit *= D2R;

          RobotCommandBuilder command_builder;

          right_arm_minimum_time *= 0.99;
          right_arm_minimum_time = std::max(right_arm_minimum_time, 0.01);

          left_arm_minimum_time *= 0.99;
          left_arm_minimum_time = std::max(left_arm_minimum_time, 0.01);

          try {
            std::cout<<"Make Builder 111"<<std::endl;
            command_builder.SetCommand(ComponentBasedCommandBuilder().SetBodyCommand(
                BodyComponentBasedCommandBuilder()
                    .SetLeftArmCommand(JointPositionCommandBuilder()
                                           .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(4.))
                                           .SetMinimumTime(left_arm_minimum_time)
                                           .SetPosition(target_position_left)
                                           .SetVelocityLimit(vel_limit)
                                           .SetAccelerationLimit(acc_limit))
                    .SetRightArmCommand(JointPositionCommandBuilder()
                                            .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(4.))
                                            .SetMinimumTime(right_arm_minimum_time)
                                            .SetPosition(target_position_right)
                                            .SetVelocityLimit(vel_limit)
                                            .SetAccelerationLimit(acc_limit))));
            std::cout<<"Make Builder 222"<<std::endl;
            std::cout<<"SendCommand 1111"<<std::endl;

            stream->SendCommand(command_builder);
            std::cout<<"SendCommand 2222"<<std::endl;
          } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
          }
        }
        std::this_thread::sleep_for(5ms);

      }

      else if (control_mode == ControlMode::TORSO) {
        // torso control mode
        std::cout << "TORSO START" << std::endl;
        if (stream == nullptr || stream->IsDone()) {
          std::cout<<"Create Stream 111"<<std::endl;
          stream = robot->CreateCommandStream();
          std::cout<<"Create Stream 222"<<std::endl;
        }

        if (control_cnt++ > 100) {
          control_cnt = 999;

          Eigen::Vector<double, 3> temp_rpy_right;
          temp_rpy_right(0) = -rpy_right(1);
          temp_rpy_right(1) = rpy_right(0);
          temp_rpy_right(2) = rpy_right(2);

          double lpf_update_ratio2 = 0.2;
          add_pos = add_pos * (1. - lpf_update_ratio2) + temp_rpy_right / 10.0 * 0.005 * lpf_update_ratio2;
          add_ori = add_ori * (1. - lpf_update_ratio2) + rpy_left / 10.0 * 0.01 * lpf_update_ratio2;

          Eigen::Vector<double, 3> temp_ori = math::SO3::Log(T_torso.block(0, 0, 3, 3)) + add_ori;
          Eigen::Vector<double, 3> temp_pos = T_torso.block(0, 3, 3, 1) + add_pos;

          for (int i = 0; i < 3; i++) {
            temp_ori(i) = std::min(std::max(temp_ori(i), -1.), 1.);
            if (i == 2) {
              temp_pos(i) = std::min(std::max(temp_pos(i), 0.9), 1.2);
            } else {
              temp_pos(i) = std::min(std::max(temp_pos(i), -0.2), 0.2);
            }
          }

          T_torso.block(0, 0, 3, 3) = math::SO3::Exp(temp_ori);
          T_torso.block(0, 3, 3, 1) = temp_pos;

          torso_minimum_time *= 0.99;
          torso_minimum_time = std::max(torso_minimum_time, 0.01);

          try {
            RobotCommandBuilder command_builder;
            double stop_orientation_tracking_error = 1e-5;
            double stop_position_tracking_error = 1e-5;
            double minimum_time = 0.01;
            std::cout<<"Make Builder 111"<<std::endl;
            command_builder.SetCommand(ComponentBasedCommandBuilder().SetBodyCommand(
                BodyCommandBuilder().SetCommand(BodyComponentBasedCommandBuilder().SetTorsoCommand(
                    CartesianCommandBuilder()
                        .AddTarget("base", "link_torso_5", T_torso, 1, 3.141592, 3)
                        .SetMinimumTime(torso_minimum_time)
                        .SetStopOrientationTrackingError(stop_orientation_tracking_error)
                        .SetStopPositionTrackingError(stop_position_tracking_error)
                        .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(100))))));
            std::cout<<"Make Builder 222"<<std::endl;
            std::cout<<"SendCommand 1111"<<std::endl;
            stream->SendCommand(command_builder);
            std::cout<<"SendCommand 2222"<<std::endl;
          } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
          }
        }
        std::this_thread::sleep_for(5ms);

      } else if (control_mode == ControlMode::WHEEL) {
        // wheel control mode
        std::cout << "WHEEL START" << std::endl;

        if (stream == nullptr || stream->IsDone()) {
          std::cout<<"Create Stream 111"<<std::endl;
          stream = robot->CreateCommandStream();
          std::cout<<"Create Stream 222"<<std::endl;
        }

        if (control_cnt++ > 100) {
          control_cnt = 999;

          double v_ref_wheel_right = rpy_right(1) / 5.0;
          double v_ref_wheel_left = rpy_left(1) / 5.0;

          Eigen::Vector<double, 2> wheel_velocity, wheel_acceleration;
          wheel_velocity << v_ref_wheel_right * 3.14 * 2., v_ref_wheel_left * 3.14 * 2.;
          wheel_acceleration.setConstant(100. / 10.);

          wheel_minimum_time *= 0.99;
          wheel_minimum_time = std::max(wheel_minimum_time, 0.01);
          try {
            RobotCommandBuilder command_builder;
            std::cout<<"Make Builder 111"<<std::endl;
            command_builder.SetCommand(
                ComponentBasedCommandBuilder().SetMobilityCommand(MobilityCommandBuilder().SetCommand(
                    JointVelocityCommandBuilder()
                        .SetVelocity(wheel_velocity)
                        .SetAccelerationLimit(wheel_acceleration)
                        .SetMinimumTime(wheel_minimum_time)
                        .SetCommandHeader(CommandHeaderBuilder().SetControlHoldTime(100)))));

            std::cout<<"Make Builder 222"<<std::endl;
            std::cout<<"SendCommand 1111"<<std::endl;
            stream->SendCommand(command_builder);
            std::cout<<"SendCommand 2222"<<std::endl;
          } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
          }
        }
        std::this_thread::sleep_for(5ms);
      }
    }
  }

  master_arm_handler.join();
  gripper_handler.join();

  portHandler->closePort();
  portHandler_gripper->closePort();

  return 0;
}
