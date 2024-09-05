#pragma once

namespace rb {

struct BatteryInfo {};

struct PowerInfo {
  std::string name{};
};

struct JointInfo {
  std::string name{};

  bool has_brake{};
};

struct RobotInfo {
  std::string robot_version{};

  BatteryInfo battery_info;

  std::vector<PowerInfo> power_infos{};

  int degree_of_freedom{};

  std::vector<JointInfo> joint_infos{};

  std::vector<unsigned int> mobility_joint_idx{};

  std::vector<unsigned int> body_joint_idx{};

  std::vector<unsigned int> head_joint_idx{};
};

}  // namespace rb