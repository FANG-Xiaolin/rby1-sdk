#include <iomanip>
#include <iostream>
#include <thread>
#include "rby1-sdk/model.h"
#include "rby1-sdk/robot.h"
#include "rby1-sdk/robot_command_builder.h"

using namespace rb;
using namespace std::chrono_literals;

#define D2R 0.017453
#define R2D 57.296

const std::string kAll = ".*";

// const std::string kAll = "^(?!.*wheel$).*";

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <server address>" << std::endl;
    return 1;
  }

  std::string address{argv[1]};

  auto robot = Robot<y1_model::A>::Create(address);

  std::cout << "Attempting to connect to the robot..." << std::endl;
  if (!robot->Connect()) {
    std::cerr << "Error: Unable to establish connection to the robot at " << address << std::endl;
    return 1;
  }
  std::cout << "Successfully connected to the robot." << std::endl;

  std::cout << "Starting state update..." << std::endl;

  robot->StartStateUpdate(
      [](const auto& state) {
        std::cout << "State Update Received:" << std::endl;
        std::cout << "  Timestamp: " << state.timestamp.tv_sec << ".";
        std::cout << std::setw(9) << std::setfill('0') << state.timestamp.tv_nsec << std::endl;
        std::cout << "  wasit [deg]     : " << state.position.block(2, 0, 6, 1).transpose() * R2D << std::endl;
        std::cout << "  right arm [deg] : " << state.position.block(2 + 6, 0, 7, 1).transpose() * R2D << std::endl;
        std::cout << "  left arm [deg]  : " << state.position.block(2 + 6 + 7, 0, 7, 1).transpose() * R2D << std::endl;
      },
      0.1 /* Hz */);

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
  if (!robot->IsServoOn(kAll)) {
    std::cout << "Servo is currently OFF. Attempting to activate servo..." << std::endl;
    if (!robot->ServoOn(kAll)) {
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

  try {
    if (robot->IsPowerOn("48v")) {
      robot->SetToolFlangeOutputVoltage("right", 12);
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  robot->SetParameter("default.acceleration_limit_scaling", "0.8");
  robot->SetParameter("joint_position_command.cutoff_frequency", "5");
  robot->SetParameter("cartesian_command.cutoff_frequency", "5");
  robot->SetParameter("default.linear_acceleration_limit", "5");

  std::this_thread::sleep_for(1s);

  Eigen::Vector<double, 6> q_joint_torso;
  Eigen::Vector<double, 7> q_joint_right;
  Eigen::Vector<double, 7> q_joint_left;
  q_joint_torso.setZero();
  q_joint_right.setZero();
  q_joint_left.setZero();

  double minimum_time = 4.;
  double velocity_tracking_gain = 0.01;
  double stop_cost = 1e-2;
  double weight = 0.005;
  double min_delta_cost = 1e-4;
  int patience = 10;

  Eigen::Matrix<double, 4, 4> T_torso, T_right, T_left;

  T_torso.setIdentity();
  T_right.setIdentity();
  T_left.setIdentity();

  auto dyn = robot->GetDynamics();
  auto dyn_state = dyn->MakeState({"base", "link_torso_5", "ee_right", "ee_left"}, y1_model::A::kRobotJointNames);

  Eigen::Matrix<double, 24, 1> q_joint;
  q_joint.setZero();

  {

    q_joint_torso << 0, 30, -60, 30, 0, 0;
    q_joint_right << 30, -30, 0, -90, 0, 30, 0;
    q_joint_left << 30, 30, 0, -90, 0, 30, 0;

    q_joint_torso *= D2R;
    q_joint_right *= D2R;
    q_joint_left *= D2R;

    auto rv =
        robot
            ->SendCommand(RobotCommandBuilder().SetCommand(ComponentBasedCommandBuilder().SetBodyCommand(
                BodyComponentBasedCommandBuilder()
                    .SetTorsoCommand(JointPositionCommandBuilder().SetMinimumTime(5).SetPosition(q_joint_torso))
                    .SetRightArmCommand(JointPositionCommandBuilder().SetMinimumTime(5).SetPosition(q_joint_right))
                    .SetLeftArmCommand(JointPositionCommandBuilder().SetMinimumTime(5).SetPosition(q_joint_left)))))
            ->Get();
  }

  q_joint.block(2, 0, 6, 1) = q_joint_torso;
  q_joint.block(2 + 6, 0, 7, 1) = q_joint_right;
  q_joint.block(2 + 6 + 7, 0, 7, 1) = q_joint_left;

  dyn_state->SetQ(q_joint);

  dyn->ComputeForwardKinematics(dyn_state);

  Eigen::Matrix<double,4,4> T_base2torso = dyn->ComputeTransformation(dyn_state, 0, 1);
  Eigen::Matrix<double,4,4> T_torso2right = dyn->ComputeTransformation(dyn_state, 1, 2);
  Eigen::Matrix<double,4,4> T_torso2left = dyn->ComputeTransformation(dyn_state, 1, 3);

  std::cout << "T_base2torso: " << std::endl;
  std::cout << T_base2torso << std::endl;

  std::cout << "T_torso2right: " << std::endl;
  std::cout << T_torso2right << std::endl;

  std::cout << "T_torso2left: " << std::endl;
  std::cout << T_torso2left << std::endl;

  {
    std::cout << "optimal control example 1\n";

    auto rv = robot
                  ->SendCommand(RobotCommandBuilder().SetCommand(ComponentBasedCommandBuilder().SetBodyCommand(
                      OptimalControlCommandBuilder()
                          .AddCartesianTarget("base", "link_torso_5", T_base2torso, weight, weight)
                          .AddCartesianTarget("link_torso_5", "ee_right", T_torso2right, weight, weight)
                          .AddCartesianTarget("link_torso_5", "ee_left", T_torso2left, weight, weight)
                          // .AddJointPositionTarget("right_arm_2", 3.141592 / 2., weight)
                          // .AddJointPositionTarget("left_arm_2", -3.141592 / 2., weight)
                          .SetVelocityLimitScaling(1.0)
                          .SetVelocityTrackingGain(velocity_tracking_gain)
                          .SetStopCost(stop_cost)
                          .SetMinDeltaCost(min_delta_cost)
                          .SetPatience(patience))))
                  ->Get();

    if (rv.finish_code() != RobotCommandFeedback::FinishCode::kOk) {
      std::cerr << "Error: Failed to conduct demo motion." << std::endl;
      return 1;
    }

    // std::this_thread::sleep_for(1s);
  }

  {
    std::cout << "optimal control example 2\n";

    T_base2torso(2, 3) -= 0.3;

    auto rv = robot
                  ->SendCommand(RobotCommandBuilder().SetCommand(ComponentBasedCommandBuilder().SetBodyCommand(
                      OptimalControlCommandBuilder()
                          .AddCartesianTarget("base", "link_torso_5", T_base2torso, weight, weight)
                          .AddCartesianTarget("link_torso_5", "ee_right", T_torso2right, weight, weight)
                          .AddCartesianTarget("link_torso_5", "ee_left", T_torso2left, weight, weight)
                          // .AddJointPositionTarget("right_arm_2", 3.141592 / 2., weight)
                          // .AddJointPositionTarget("left_arm_2", -3.141592 / 2., weight)
                          .SetVelocityLimitScaling(1.0)
                          .SetVelocityTrackingGain(velocity_tracking_gain)
                          .SetStopCost(stop_cost)
                          .SetMinDeltaCost(min_delta_cost)
                          .SetPatience(patience))))
                  ->Get();

    if (rv.finish_code() != RobotCommandFeedback::FinishCode::kOk) {
      std::cerr << "Error: Failed to conduct demo motion." << std::endl;
      return 1;
    }

    // std::this_thread::sleep_for(1s);
  }

  {
    std::cout << "optimal control example 3\n";

    T_base2torso(2, 3) += 0.3;

    auto rv = robot
                  ->SendCommand(RobotCommandBuilder().SetCommand(ComponentBasedCommandBuilder().SetBodyCommand(
                      OptimalControlCommandBuilder()
                          .AddCartesianTarget("base", "link_torso_5", T_base2torso, weight, weight)
                          .AddCartesianTarget("link_torso_5", "ee_right", T_torso2right, weight, weight)
                          .AddCartesianTarget("link_torso_5", "ee_left", T_torso2left, weight, weight)
                          // .AddJointPositionTarget("right_arm_2", 3.141592 / 2., weight)
                          // .AddJointPositionTarget("left_arm_2", -3.141592 / 2., weight)
                          .SetVelocityLimitScaling(1.0)
                          .SetVelocityTrackingGain(velocity_tracking_gain)
                          .SetStopCost(stop_cost)
                          .SetMinDeltaCost(min_delta_cost)
                          .SetPatience(patience))))
                  ->Get();

    if (rv.finish_code() != RobotCommandFeedback::FinishCode::kOk) {
      std::cerr << "Error: Failed to conduct demo motion." << std::endl;
      return 1;
    }

    // std::this_thread::sleep_for(1s);
  }

  std::cout << "end of demo\n";

  return 0;
}
