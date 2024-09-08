#pragma once

#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "rby1-sdk/dynamics/inertial.h"
#include "rby1-sdk/dynamics/joint.h"
#include "rby1-sdk/dynamics/link.h"
#include "rby1-sdk/dynamics/state.h"
#include "rby1-sdk/math/liegroup.h"

namespace rb::dyn {

/**********************************************************
 * FORWARD DECLARATION
 **********************************************************/

//
template <int DOF>
class Robot;
class Link;
class Joint;

/**********************************************************
 * MOBILE BASE
 **********************************************************/

enum class MobileBaseType {
  kNone,          //
  kDifferential,  //
  kOmni           //
};

struct MobileBase {
  MobileBaseType type{MobileBaseType::kNone};

  math::SE3::MatrixType T;  // front = x-axis
  std::vector<std::string> joints;
  std::vector<double> params;
};

struct MobileBaseDifferential : public MobileBase {
  unsigned int right_wheel_idx{};
  unsigned int left_wheel_idx{};

  double wheel_base{0.};
  double wheel_radius{0.};
};

/**********************************************************
 * ROBOT
 **********************************************************/

struct RobotConfiguration {
  std::string name;
  std::shared_ptr<Link> base_link;
  std::shared_ptr<MobileBase> mobile_base;
};

template <int DOF>
class Robot {
 public:
  template <typename T, int N>
  using ContainerType = typename std::conditional_t<(N > 0), std::array<T, (unsigned int)N>, std::vector<T>>;

  struct Link_ {
    struct SubLink_ {
      std::shared_ptr<Link> link;
      Inertial::MatrixType J_wrt_p;
      math::SE3::MatrixType M_wrt_p;
      math::SE3::MatrixType M_wrt_base;
    };

    std::vector<SubLink_> links;
    Inertial::MatrixType J{Inertial::MatrixType::Zero()};  // Base link 기준 Link_ inertial
    math::SE3::MatrixType M{math::SE3::Identity()};        // Base link to Link_ frame

    Inertial::MatrixType I{Inertial::MatrixType::Zero()};  // Link_ frame 기준 Link_ inertial

    int depth{0};
    int parent_joint_idx{-1};
    std::vector<int> child_joint_idx{};

    void SetBaseLink(const std::shared_ptr<Link>& link, const math::SE3::MatrixType& T) {
      // Rest link information
      links.clear();

      J = Inertial::MatrixType::Zero();
      M = T;

      AddLink(link, math::SE3::Identity());
    }

    int AddLink(const std::shared_ptr<Link>& link, const math::SE3::MatrixType& M_wrt_p) {
      Inertial::MatrixType J_wrt_p = Inertial::Transform(M_wrt_p, link->I_);
      int idx = links.size();
      links.push_back({.link = link, .J_wrt_p = J_wrt_p, .M_wrt_p = M_wrt_p, .M_wrt_base = M * M_wrt_p});
      I += J_wrt_p;
      J += Inertial::Transform(M, J_wrt_p);
      return idx;
    }
  };

  struct Joint_ {
    std::shared_ptr<Joint> joint{nullptr};
    math::se3v::MatrixType S{math::se3v::MatrixType::Zero()};  // zero pose

    int parent_link_idx{};
    int child_link_idx{};
  };

  struct LinkIdx_ {
    int link_idx;
    int sub_link_idx;
  };

  explicit Robot(const RobotConfiguration& robot_configuration) { Build(robot_configuration); }

  std::shared_ptr<Link> GetBase() { return links_[0].links[0].link; }

  std::vector<std::string> GetLinkNames() const {
    std::vector<std::string> names;
    for (const auto& [n, idx] : link_idx_) {
      names.push_back(n);
    }
    return names;
  }

  std::vector<std::string> GetJointNames() const {
    std::vector<std::string> names;
    for (const auto& [n, idx] : joint_idx_) {
      names.push_back(n);
    }
    return names;
  }

  std::shared_ptr<Link> GetLink(const std::string& name) const {
    if (link_idx_.find(name) == link_idx_.end()) {
      return nullptr;
    }
    const auto& idx = link_idx_.at(name);
    return links_[idx.link_idx].links[idx.sub_link_idx].link;
  }

  std::shared_ptr<Link> GetLink(std::shared_ptr<State<DOF>> state, int index) const {
    if (index >= state->utr_link_map.size()) {
      return nullptr;
    }
    const auto& idx = state->utr_link_map[index];
    return links_[idx.link_idx].links[idx.sub_link_idx].link;
  }

  template <typename LinkContainer = std::vector<std::string>, typename JointContainer = std::vector<std::string>>
  std::shared_ptr<State<DOF>> MakeState(const LinkContainer& link_names, const JointContainer& joint_names) {
    static_assert(std::is_same_v<typename LinkContainer::value_type, std::string> ||
                      std::is_same_v<typename LinkContainer::value_type, std::string_view>,
                  "LinkContainer value_type must be std::string");
    static_assert(std::is_same_v<typename JointContainer::value_type, std::string> ||
                      std::is_same_v<typename JointContainer::value_type, std::string_view>,
                  "JointContainer value_type must be std::string");

    std::vector<bool> flag;
    flag.resize(n_joints_);
    std::fill(flag.begin(), flag.end(), false);

    auto state = std::shared_ptr<State<DOF>>(new State<DOF>(n_joints_));
    for (int i = 0; i < joint_names.size(); i++) {
      const auto& name = joint_names[i];

      auto it = std::find_if(joints_.begin(), joints_.end(), [name](const auto& j) { return j.joint->name_ == name; });
      if (it == joints_.end()) {
        throw std::runtime_error("Cannot find the joint with name");
      }
      state->utr_joint_map[i] = it - joints_.begin();
      state->joint_names[i] = it->joint->name_;
      flag[state->utr_joint_map[i]] = true;
    }
    for (unsigned int i = joint_names.size(); i < n_joints_; i++) {
      auto it = std::find(flag.begin(), flag.end(), false);
      state->utr_joint_map[i] = it - flag.begin();
      state->joint_names[i] = (joints_.begin() + (it - flag.begin()))->joint->name_;
      *it = true;
    }
    for (int i = 0; i < state->utr_joint_map.size(); i++) {
      state->rtu_joint_map[state->utr_joint_map[i]] = i;
    }

    std::unordered_map<std::string, bool> name_flag;
    state->link_names.clear();
    state->utr_link_map.clear();
    for (const auto& name : link_names) {
      auto it = link_idx_.find(name);
      if (it == link_idx_.end()) {
        throw std::runtime_error("The link with the given name does not exist");
      }
      state->utr_link_map.push_back(it->second);
      state->link_names.push_back(name);
      name_flag[name] = true;
    }
    for (const auto& [k, v] : link_idx_) {
      if (name_flag.find(k) == name_flag.end()) {
        state->utr_link_map.push_back(v);
        state->link_names.push_back(k);
      }
    }
    for (int i = 0; i < state->link_names.size(); i++) {
      const auto& n = state->link_names[i];
      auto it = link_idx_.find(n);
      if (it == link_idx_.end()) {
        throw std::runtime_error("Fatal error; link names in state should be found in link idx");
      }
      if (it->second.link_idx == 0 && it->second.sub_link_idx == 0) {
        state->base_link_user_idx = i;
      }
    }

    return state;
  }

  int GetDOF() const { return n_joints_; }

  int GetNumberOfJoints() const { return n_joints_; }

  void ComputeForwardKinematics(std::shared_ptr<State<DOF>> state) {
    for (int i = 0; i < n_joints_; i++) {
      state->E[i] = math::SE3::Exp(joints_[i].S, state->q(i));

      int parent_joint_idx = links_[joints_[i].parent_link_idx].parent_joint_idx;
      state->T[i] = GetJointT(state, parent_joint_idx) * state->E[i];

      state->S.col(i) = math::SE3::Ad(GetJointT(state, parent_joint_idx), joints_[i].S);
    }
  }

  void ComputeDiffForwardKinematics(std::shared_ptr<State<DOF>> state) {  // In Body Frame
    for (int i = 0; i < n_joints_; i++) {
      state->V.col(i) = joints_[i].S * state->qdot(i);

      int parent_joint_idx = links_[joints_[i].parent_link_idx].parent_joint_idx;
      state->V.col(i) += math::SE3::InvAd(state->E[i], GetJointV(state, parent_joint_idx));
    }
  }

  void Compute2ndDiffForwardKinematics(std::shared_ptr<State<DOF>> state) {
    for (int i = 0; i < n_joints_; i++) {
      state->Vdot.col(i) = joints_[i].S * state->qddot(i);

      int parent_joint_idx = links_[joints_[i].parent_link_idx].parent_joint_idx;
      state->Vdot.col(i) += math::SE3::InvAd(state->E[i], GetJointVdot(state, parent_joint_idx)) +
                            math::SE3::ad(state->V.col(i), joints_[i].S * state->qdot(i));
    }
  }

  void ComputeInverseDynamics(std::shared_ptr<State<DOF>> state) {
    for (int i = n_joints_ - 1; i >= 0; i--) {
      Eigen::Vector<double, 6> F;
      F.setZero();

      for (const auto& child_joint_idx : links_[joints_[i].child_link_idx].child_joint_idx) {
        F += math::SE3::InvAd(state->E[child_joint_idx]).transpose() * state->F.col(child_joint_idx);
      }
      F += links_[joints_[i].child_link_idx].J * state->Vdot.col(i) -
           math::SE3::adTranspose(state->V.col(i), links_[joints_[i].child_link_idx].J * state->V.col(i));

      state->F.col(i) = F;
      state->tau(i) = joints_[i].S.dot(F);
    }
  }

  Eigen::Vector<double, DOF> ComputeGravityTerm(std::shared_ptr<State<DOF>> state) {
    Eigen::Vector<double, DOF> gravity_term;
    gravity_term.resize(n_joints_);
    std::vector<Eigen::Vector<double, 6>> Vdot{}, Fs{};
    Vdot.resize(n_joints_);
    Fs.resize(n_joints_);

    // Calculate Vdot
    for (int i = 0; i < n_joints_; i++) {
      int parent_joint_idx = links_[joints_[i].parent_link_idx].parent_joint_idx;
      if (parent_joint_idx < 0) {
        Vdot[i] = math::SE3::InvAd(state->E[i], state->Vdot0);
      } else {
        Vdot[i] = math::SE3::InvAd(state->E[i], Vdot[parent_joint_idx]);
      }
    }

    //
    for (int i = n_joints_ - 1; i >= 0; i--) {
      Eigen::Vector<double, 6> F;
      F.setZero();

      for (const auto& child_joint_idx : links_[joints_[i].child_link_idx].child_joint_idx) {
        F += math::SE3::InvAd(state->E[child_joint_idx]).transpose() * Fs[child_joint_idx];
      }
      F += links_[joints_[i].child_link_idx].J * Vdot[i];

      Fs[i] = F;
      gravity_term(state->rtu_joint_map[i]) = joints_[i].S.dot(F);
    }

    return gravity_term;
  }

  Eigen::Matrix<double, DOF, DOF> ComputeMassMatrix(std::shared_ptr<State<DOF>> state) {
    Eigen::Matrix<double, DOF, DOF> M(n_joints_, n_joints_);
    M.setZero();

    std::vector<Eigen::Matrix<double, 6, DOF>> J;
    J.resize(n_joints_);

    for (int i = 0; i < n_joints_; i++) {
      int parent_joint_idx = links_[joints_[i].parent_link_idx].parent_joint_idx;
      if (parent_joint_idx < 0) {
        J[i].setZero();
        J[i].col(state->rtu_joint_map[i]) = joints_[i].S;
      } else {
        J[i] = math::SE3::Ad(state->E[i].inverse()) * J[parent_joint_idx];
        J[i].col(state->rtu_joint_map[i]) = joints_[i].S;
      }
    }

    for (int i = 0; i < n_joints_; i++) {
      M += J[i].transpose() * links_[joints_[i].child_link_idx].J * J[i];
    }

    return M;
  }

  Eigen::Matrix<double, 6, 6> ComputeReflectiveInertia(std::shared_ptr<State<DOF>> state, unsigned int from,
                                                       unsigned int to) {
    if (from >= state->utr_link_map.size() || to >= state->utr_link_map.size()) {
      throw std::runtime_error("Out of range state link");
    }
    const auto& J = ComputeBodyJacobian(state, from, to);
    Eigen::Matrix<double, DOF, DOF> m_inv = ComputeMassMatrix(state).completeOrthogonalDecomposition().pseudoInverse();
    return (J * m_inv * J.transpose()).completeOrthogonalDecomposition().pseudoInverse();
  }

  math::SE3::MatrixType ComputeTransformation(std::shared_ptr<State<DOF>> state, unsigned int from, unsigned int to) {
    if (from >= state->utr_link_map.size() || to >= state->utr_link_map.size()) {
      throw std::runtime_error("Out of range state link");
    }
    return GetTransformation(state, state->utr_link_map[from], state->utr_link_map[to]);
  }

  math::se3v::MatrixType ComputeBodyVelocity(std::shared_ptr<State<DOF>> state, unsigned int from, unsigned int to) {
    if (from >= state->utr_link_map.size() || to >= state->utr_link_map.size()) {
      throw std::runtime_error("Out of range state link");
    }
    math::SE3::MatrixType T_from_to = ComputeTransformation(state, from, to);
    math::se3v::MatrixType V_from = GetLinkV(state, state->utr_link_map[from]);
    math::se3v::MatrixType V_to = GetLinkV(state, state->utr_link_map[to]);
    return V_to - math::SE3::InvAd(T_from_to, V_from);
  }

  Eigen::Matrix<double, 6, DOF> ComputeSpaceJacobian(std::shared_ptr<State<DOF>> state, unsigned int from,
                                                     unsigned int to) {
    return GetSpaceJacobian(state, state->utr_link_map[from], state->utr_link_map[to]);
  }

  Eigen::Matrix<double, 6, DOF> ComputeBodyJacobian(std::shared_ptr<State<DOF>> state, unsigned int from,
                                                    unsigned int to) {
    math::SE3::MatrixType T = ComputeTransformation(state, from, to);
    return math::SE3::InvAd(T) * ComputeSpaceJacobian(state, from, to);
  }

  double ComputeMass(std::shared_ptr<State<DOF>> state, unsigned int target_link) {
    const auto& idx = state->utr_link_map[target_link];
    return Inertial::GetMass(links_[idx.link_idx].links[idx.sub_link_idx].link->I_);
  }

  Eigen::Vector3d ComputeCenterOfMass(std::shared_ptr<State<DOF>> state, unsigned int ref_link,
                                      unsigned int target_link) {
    const auto& idx = state->utr_link_map[target_link];
    return math::SE3::Multiply(ComputeTransformation(state, ref_link, target_link),
                               Inertial::GetCOM(links_[idx.link_idx].links[idx.sub_link_idx].link->I_));
  }

  Eigen::Vector3d ComputeCenterOfMass(std::shared_ptr<State<DOF>> state, unsigned int ref_link,
                                      const std::vector<unsigned int>& target_links) {
    Eigen::Vector3d com{Eigen::Vector3d ::Zero()};
    double mass = 0;
    for (auto target_link : target_links) {
      const auto& idx = state->utr_link_map[target_link];
      const auto& p = Inertial::GetCOM(links_[idx.link_idx].links[idx.sub_link_idx].link->I_);
      double m = Inertial::GetMass(links_[idx.link_idx].links[idx.sub_link_idx].link->I_);
      com += math::SE3::Multiply(ComputeTransformation(state, ref_link, target_link), p) * m;
      mass += m;
    }
    return com / mass;
  }

  Eigen::Matrix<double, 3, DOF> ComputeCenterOfMassJacobian(std::shared_ptr<State<DOF>> state, unsigned int ref_link,
                                                            unsigned int target_link) {
    using namespace math;

    const auto& idx = state->utr_link_map[target_link];
    return SE3::GetRotation(ComputeTransformation(state, ref_link, target_link)) *
           (SE3::InvAd(SE3::T(Inertial::GetCOM(links_[idx.link_idx].links[idx.sub_link_idx].link->I_))) *
            ComputeBodyJacobian(state, ref_link, target_link))
               .block(3, 0, 3, n_joints_);
  }

  /**
   * Compute total inertial of robot
   * @param state
   * @param ref_link
   * @return
   */
  Inertial::MatrixType ComputeTotalInertial(std::shared_ptr<State<DOF>> state, unsigned int ref_link) {
    Inertial::MatrixType I{Inertial::MatrixType::Zero()};
    for (const auto& link : links_) {
      I += Inertial::Transform(GetJointT(state, link.parent_joint_idx), link.J);
    }

    return Inertial::Transform(math::SE3::Inverse(GetLinkT(state, state->utr_link_map[ref_link])), I);
  }

  Eigen::Vector3d ComputeCenterOfMass(std::shared_ptr<State<DOF>> state, unsigned int ref_link) {
    Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
    double mass = 0;

    for (const auto& link : links_) {
      sum += Inertial::GetMass(link.J) *
             math::SE3::Multiply(GetJointT(state, link.parent_joint_idx), Inertial::GetCOM(link.J));
      mass += Inertial::GetMass(link.J);
    }

    return math::SE3::Multiply(math::SE3::Inverse(GetLinkT(state, state->utr_link_map[ref_link])), sum / mass);
  }

  Eigen::Matrix<double, 3, DOF> ComputeCenterOfMassJacobian(std::shared_ptr<State<DOF>> state, unsigned int ref_link) {
    using namespace math;
    Eigen::Matrix<double, 3, DOF> J;
    J.resize(3, n_joints_);
    J.setZero();
    double mass = 0;
    for (int i = 0; i < links_.size(); i++) {
      const auto& link = links_[i];
      double m = Inertial::GetMass(links_[i].I);
      J += m * SE3::GetRotation(GetTransformation(state, state->utr_link_map[ref_link], {i, 0})) *
           (SE3::InvAd(SE3::T(Inertial::GetCOM(links_[i].I))) *
            GetBodyJacobian(state, state->utr_link_map[ref_link], {i, 0}))
               .block(3, 0, 3, n_joints_);
      mass += m;
    }
    return J / mass;
  }

  // TODO: ComputeBodyJacobianDot
  // TODO: ComputeBodyAcceleration
  // TODO: InverseDiffDynamics

  Eigen::Vector<double, DOF> GetLimitQLower(const std::shared_ptr<State<DOF>>& state) {
    return GetJointProperty(state, [](auto j) { return j->GetLimitQLower(); });
  }

  Eigen::Vector<double, DOF> GetLimitQUpper(const std::shared_ptr<State<DOF>>& state) {
    return GetJointProperty(state, [](auto j) { return j->GetLimitQUpper(); });
  }

  Eigen::Vector<double, DOF> GetLimitQdotLower(const std::shared_ptr<State<DOF>>& state) {
    return GetJointProperty(state, [](auto j) { return j->GetLimitQdotLower(); });
  }

  Eigen::Vector<double, DOF> GetLimitQdotUpper(const std::shared_ptr<State<DOF>>& state) {
    return GetJointProperty(state, [](auto j) { return j->GetLimitQdotUpper(); });
  }

  Eigen::Vector<double, DOF> GetLimitQddotLower(const std::shared_ptr<State<DOF>>& state) {
    return GetJointProperty(state, [](auto j) { return j->GetLimitQddotLower(); });
  }

  Eigen::Vector<double, DOF> GetLimitQddotUpper(const std::shared_ptr<State<DOF>>& state) {
    return GetJointProperty(state, [](auto j) { return j->GetLimitQddotUpper(); });
  }

  Eigen::Vector<double, DOF> GetJointProperty(const std::shared_ptr<State<DOF>>& state,
                                              const std::function<double(std::shared_ptr<Joint>)>& getter) {
    Eigen::Vector<double, DOF> prop;
    prop.resize(n_joints_);
    for (int i = 0; i < n_joints_; i++) {
      prop(state->rtu_joint_map[i]) = getter(joints_[i].joint);
    }
    return prop;
  }

  void ComputeMobilityInverseDiffKinematics(std::shared_ptr<State<DOF>> state,       //
                                            const Eigen::Vector2d& linear_velocity,  // (m/s)
                                            double angular_velocity                  // (rad/s)
  ) {
    math::se2v::MatrixType S;
    S(0) = angular_velocity;
    S.tail<2>() = linear_velocity;

    ComputeMobilityInverseDiffKinematics(state, S);
  }

  void ComputeMobilityInverseDiffKinematics(std::shared_ptr<State<DOF>> state,           //
                                            const math::se2v::MatrixType& body_velocity  // w, x, y
  ) {
    math::se3v::MatrixType S{math::se3v::MatrixType::Zero()};
    S.block<3, 1>(2, 0) = body_velocity;

    S = math::SE3::InvAd(mobile_base_->T, S);

    switch (mobile_base_->type) {
      case MobileBaseType::kNone: {
        break;
      }
      case MobileBaseType::kDifferential: {
        auto mb = std::static_pointer_cast<MobileBaseDifferential>(mobile_base_);
        double v_right = S(3) + mb->wheel_base / 2 * S(2);
        double v_left = S(3) - mb->wheel_base / 2 * S(2);
        state->qdot(mb->right_wheel_idx) = v_right / mb->wheel_radius;
        state->qdot(mb->left_wheel_idx) = v_left / mb->wheel_radius;
        break;
      }
      case MobileBaseType::kOmni: {
        break;
      }
    }
  }

  math::se2v::MatrixType ComputeMobilityDiffKinematics(  //
      std::shared_ptr<State<DOF>> state                  //
  ) {
    math::se3v::MatrixType S{math::se3v::MatrixType::Zero()};

    switch (mobile_base_->type) {
      case MobileBaseType::kNone: {
        break;
      }
      case MobileBaseType::kDifferential: {
        auto mb = std::static_pointer_cast<MobileBaseDifferential>(mobile_base_);
        double w_r = state->qdot(mb->right_wheel_idx);
        double w_l = state->qdot(mb->left_wheel_idx);
        S.block<3, 1>(2, 0) = math::se2v::MatrixType{(w_r - w_l) * mb->wheel_radius / mb->wheel_base,
                                                     (w_r + w_l) * mb->wheel_radius / 2, 0};
      }
      case MobileBaseType::kOmni: {
        // TODO
        break;
      }
    }

    S = math::SE3::Ad(mobile_base_->T, S);

    return S.block<3, 1>(2, 0);
  }

  static int CountJoints(const std::shared_ptr<Link>& base_link, bool include_fixed = false) {
    int n_joints = 0;

    std::queue<std::shared_ptr<Link>> que;
    que.push(base_link);
    while (!que.empty()) {
      std::shared_ptr<Link> link = que.front();
      que.pop();

      for (const auto& joint : link->GetChildJointList()) {
        que.push(joint->GetChildLink());

        if (joint->IsFixed() && !include_fixed)
          ;
        else {
          n_joints++;
        }
      }
    }
    return n_joints;
  }

 protected:
  Robot() = default;

  void Build(const RobotConfiguration& rc) {
    const auto& base = rc.base_link;

    int n_joints = CountJoints(base);

    if constexpr (DOF < 0) {
      n_links_ = n_joints + 1;
      n_joints_ = n_joints;

      links_.resize(n_links_);
      joints_.resize(n_joints_);
    } else {
      if (n_joints != DOF) {
        throw std::runtime_error("DOF does not match the number of joints");
      }
      n_links_ = DOF + 1;
      n_joints_ = DOF;
    }

    struct QueueItem {
      int parent_joint_idx{};
      int parent_link_idx{};
      std::shared_ptr<Link> link;
      int depth{};
      bool merge_parent_link{};
      math::SE3::MatrixType M_wrt_base;
      math::SE3::MatrixType M_wrt_p;
    };

    std::queue<QueueItem> que;
    int link_idx = 0;
    int joint_idx = 0;
    {
      que.push({
          .parent_joint_idx = -1,               //
          .parent_link_idx = link_idx++,        //
          .link = base,                         //
          .depth = 0,                           //
          .merge_parent_link = false,           //
          .M_wrt_base = math::SE3::Identity(),  //
          .M_wrt_p = math::SE3::Identity()      //
      });
      while (!que.empty()) {
        auto e = que.front();
        que.pop();

        if (!e.merge_parent_link) {
          links_[e.parent_link_idx].SetBaseLink(e.link, e.M_wrt_base);
          links_[e.parent_link_idx].depth = e.depth;
          links_[e.parent_link_idx].parent_joint_idx = e.parent_joint_idx;
          link_idx_[e.link->name_] = {e.parent_link_idx, 0};
        } else {
          int sub_link_idx = links_[e.parent_link_idx].AddLink(e.link, e.M_wrt_p);
          link_idx_[e.link->name_] = {e.parent_link_idx, sub_link_idx};
        }

        for (const auto& joint : e.link->child_joints_) {
          if (!joint->fixed_) {
            math::SE3::MatrixType M_wrt_base = e.M_wrt_base * joint->T_pj_;
            joints_[joint_idx] = {.joint = joint,
                                  .S = math::SE3::Ad(M_wrt_base, joint->S_),
                                  .parent_link_idx = e.parent_link_idx,
                                  .child_link_idx = link_idx};
            joint_idx_[joint->name_] = joint_idx;
            links_[e.parent_link_idx].child_joint_idx.push_back(joint_idx);
            que.push({.parent_joint_idx = joint_idx++,
                      .parent_link_idx = link_idx++,
                      .link = joint->child_link_,
                      .depth = e.depth + 1,
                      .merge_parent_link = false,
                      .M_wrt_base = M_wrt_base,
                      .M_wrt_p = math::SE3::Identity()});
          } else {
            que.push({.parent_joint_idx = e.parent_joint_idx,
                      .parent_link_idx = e.parent_link_idx,
                      .link = joint->child_link_,
                      .depth = e.depth,
                      .merge_parent_link = true,
                      .M_wrt_base = e.M_wrt_base * joint->T_pj_ * joint->T_jc_,
                      .M_wrt_p = e.M_wrt_p * joint->T_pj_ * joint->T_jc_});
          }
        }
      }
    }

    if (rc.mobile_base) {
      switch (rc.mobile_base->type) {
        case MobileBaseType::kNone: {
          auto mb = std::make_shared<MobileBase>();
          *std::static_pointer_cast<MobileBase>(mb) = *rc.mobile_base;
          mobile_base_ = mb;
          break;
        }
        case MobileBaseType::kDifferential: {
          auto mb = std::make_shared<MobileBaseDifferential>();
          *std::static_pointer_cast<MobileBase>(mb) = *rc.mobile_base;
          if (rc.mobile_base->joints.size() != 2) {
            throw std::runtime_error("Differential type mobile should have two joints.");
          }
          if (joint_idx_.find(rc.mobile_base->joints[0]) == joint_idx_.end()) {
            throw std::runtime_error("Right wheel has invalid parameter.");
          }
          if (joint_idx_.find(rc.mobile_base->joints[1]) == joint_idx_.end()) {
            throw std::runtime_error("Left wheel has invalid parameter.");
          }
          if (rc.mobile_base->params.size() != 2) {
            throw std::runtime_error("Differential type mobile should have two parameters.");
          }
          mb->right_wheel_idx = joint_idx_[rc.mobile_base->joints[0]];
          mb->left_wheel_idx = joint_idx_[rc.mobile_base->joints[1]];
          mb->wheel_base = rc.mobile_base->params[0];
          mb->wheel_radius = rc.mobile_base->params[1];
          mobile_base_ = mb;
          break;
        }
        case MobileBaseType::kOmni:
          // TODO
          break;
      }
    }
  }

  math::SE3::MatrixType GetLinkT(std::shared_ptr<State<DOF>> state, const LinkIdx_& idx) {
    return GetJointT(state, links_[idx.link_idx].parent_joint_idx) *
           links_[idx.link_idx].links[idx.sub_link_idx].M_wrt_base;
  }

  math::se3v::MatrixType GetLinkV(std::shared_ptr<State<DOF>> state, const LinkIdx_& idx) {
    return math::SE3::InvAd(links_[idx.link_idx].links[idx.sub_link_idx].M_wrt_base,
                            GetJointV(state, links_[idx.link_idx].parent_joint_idx));
  }

  math::SE3::MatrixType GetJointT(std::shared_ptr<State<DOF>> state, int joint_idx) {
    if (joint_idx < 0)
      return math::SE3::Identity();
    return state->T[joint_idx];
  }

  math::se3v::MatrixType GetJointV(std::shared_ptr<State<DOF>> state, int joint_idx) {
    if (joint_idx < 0)
      return state->V0;
    return state->V.col(joint_idx);
  }

  math::se3v::MatrixType GetJointVdot(std::shared_ptr<State<DOF>> state, int joint_idx) {
    if (joint_idx < 0)
      return state->Vdot0;
    return state->Vdot.col(joint_idx);
  }

  math::SE3::MatrixType GetTransformation(std::shared_ptr<State<DOF>> state, const LinkIdx_& from, const LinkIdx_& to) {
    return math::SE3::Inverse(GetLinkT(state, from)) * GetLinkT(state, to);
  }

  Eigen::Matrix<double, 6, DOF> GetSpaceJacobian(std::shared_ptr<State<DOF>> state, const LinkIdx_& from,
                                                 const LinkIdx_& to) {
    Eigen::Matrix<double, 6, DOF> J;
    J.resize(6, n_joints_);
    J.setZero();

    // TODO: Need to optimize

    int from_link_idx = from.link_idx;
    int to_link_idx = to.link_idx;
    while (from_link_idx != to_link_idx) {
      if (links_[from_link_idx].depth > links_[to_link_idx].depth) {
        int pj = links_[from_link_idx].parent_joint_idx;
        J.col(state->rtu_joint_map[pj]) = -state->S.col(pj);
        from_link_idx = joints_[pj].parent_link_idx;
      } else {
        int pj = links_[to_link_idx].parent_joint_idx;
        J.col(state->rtu_joint_map[pj]) = state->S.col(pj);
        to_link_idx = joints_[pj].parent_link_idx;
      }
    }

    return math::SE3::InvAd(GetLinkT(state, from)) * J;
  }

  Eigen::Matrix<double, 6, DOF> GetBodyJacobian(std::shared_ptr<State<DOF>> state, const LinkIdx_& from,
                                                const LinkIdx_& to) {
    math::SE3::MatrixType T = GetTransformation(state, from, to);
    return math::SE3::InvAd(T) * GetSpaceJacobian(state, from, to);
  }

 private:
  ContainerType<Link_, DOF + 1> links_;
  ContainerType<Joint_, DOF> joints_;

  std::unordered_map<std::string, LinkIdx_> link_idx_;       // link name to (parent idx, sub link idx)
  std::unordered_map<std::string, unsigned int> joint_idx_;  // joint name to joint idx

  size_t n_links_{};
  size_t n_joints_{};

  /**
   * MOBILE BASE
   */
  std::shared_ptr<MobileBase> mobile_base_;
};

RobotConfiguration LoadRobotFromURDFData(const std::string& model, const std::string& base_link_name);

RobotConfiguration LoadRobotFromURDF(const std::string& path, const std::string& base_link_name);

inline std::string to_string(GeomType type) {
  switch (type) {
    case GeomType::kCapsule:
      return "capsule";
  }
  return "unknown";
}

}  // namespace rb::dyn