# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rb/api/basic_command.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import duration_pb2 as google_dot_protobuf_dot_duration__pb2
from google.protobuf import wrappers_pb2 as google_dot_protobuf_dot_wrappers__pb2
from rb.api import geometry_pb2 as rb_dot_api_dot_geometry__pb2
from rb.api import command_header_pb2 as rb_dot_api_dot_command__header__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1arb/api/basic_command.proto\x12\x06rb.api\x1a\x1egoogle/protobuf/duration.proto\x1a\x1egoogle/protobuf/wrappers.proto\x1a\x15rb/api/geometry.proto\x1a\x1brb/api/command_header.proto\"\x9c\x01\n\x0bStopCommand\x1a@\n\x07Request\x12\x35\n\x0e\x63ommand_header\x18\x01 \x01(\x0b\x32\x1d.rb.api.CommandHeader.Request\x1aK\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12?\n\x17\x63ommand_header_feedback\x18\x01 \x01(\x0b\x32\x1e.rb.api.CommandHeader.Feedback\"\xad\x02\n\x12SE2VelocityCommand\x1a\xc9\x01\n\x07Request\x12\x35\n\x0e\x63ommand_header\x18\x01 \x01(\x0b\x32\x1d.rb.api.CommandHeader.Request\x12/\n\x0cminimum_time\x18\x02 \x01(\x0b\x32\x19.google.protobuf.Duration\x12%\n\x08velocity\x18\x03 \x01(\x0b\x32\x13.rb.api.SE2Velocity\x12/\n\x12\x61\x63\x63\x65leration_limit\x18\x04 \x01(\x0b\x32\x13.rb.api.SE2Velocity\x1aK\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12?\n\x17\x63ommand_header_feedback\x18\x01 \x01(\x0b\x32\x1e.rb.api.CommandHeader.Feedback\"\x94\x03\n\nJogCommand\x1a\x9d\x02\n\x07Request\x12\x35\n\x0e\x63ommand_header\x18\x01 \x01(\x0b\x32\x1d.rb.api.CommandHeader.Request\x12\x12\n\njoint_name\x18\x02 \x01(\t\x12\x34\n\x0evelocity_limit\x18\x03 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x38\n\x12\x61\x63\x63\x65leration_limit\x18\x04 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x1b\n\x11\x61\x62solute_position\x18\x05 \x01(\x01H\x00\x12\x1b\n\x11relative_position\x18\x06 \x01(\x01H\x00\x12\x12\n\x08one_step\x18\x07 \x01(\x08H\x00\x42\t\n\x07\x63ommand\x1a\x66\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12?\n\x17\x63ommand_header_feedback\x18\x01 \x01(\x0b\x32\x1e.rb.api.CommandHeader.Feedback\x12\x19\n\x11target_joint_name\x18\x02 \x01(\t\"\x85\x02\n\x14JointVelocityCommand\x1a\x9f\x01\n\x07Request\x12\x35\n\x0e\x63ommand_header\x18\x01 \x01(\x0b\x32\x1d.rb.api.CommandHeader.Request\x12/\n\x0cminimum_time\x18\x02 \x01(\x0b\x32\x19.google.protobuf.Duration\x12\x10\n\x08velocity\x18\x03 \x03(\x01\x12\x1a\n\x12\x61\x63\x63\x65leration_limit\x18\x04 \x03(\x01\x1aK\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12?\n\x17\x63ommand_header_feedback\x18\x01 \x01(\x0b\x32\x1e.rb.api.CommandHeader.Feedback\"\xd5\x02\n\x14JointPositionCommand\x1a\xef\x01\n\x07Request\x12\x35\n\x0e\x63ommand_header\x18\x01 \x01(\x0b\x32\x1d.rb.api.CommandHeader.Request\x12/\n\x0cminimum_time\x18\x02 \x01(\x0b\x32\x19.google.protobuf.Duration\x12\x10\n\x08position\x18\x03 \x03(\x01\x12\x16\n\x0evelocity_limit\x18\x04 \x03(\x01\x12\x1a\n\x12\x61\x63\x63\x65leration_limit\x18\x05 \x03(\x01\x12\x36\n\x10\x63utoff_frequency\x18\n \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x1aK\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12?\n\x17\x63ommand_header_feedback\x18\x01 \x01(\x0b\x32\x1e.rb.api.CommandHeader.Feedback\"\xaf\x06\n\x10\x43\x61rtesianCommand\x1a\x92\x02\n\rSE3PoseTarget\x12\x15\n\rref_link_name\x18\x01 \x01(\t\x12\x11\n\tlink_name\x18\x02 \x01(\t\x12\x1a\n\x01T\x18\x03 \x01(\x0b\x32\x0f.rb.api.SE3Pose\x12;\n\x15linear_velocity_limit\x18\x04 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12<\n\x16\x61ngular_velocity_limit\x18\x05 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12@\n\x1a\x61\x63\x63\x65leration_limit_scaling\x18\x06 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x1a?\n\rTrackingError\x12\x16\n\x0eposition_error\x18\x01 \x01(\x01\x12\x16\n\x0erotation_error\x18\x02 \x01(\x01\x1a\xb5\x02\n\x07Request\x12\x35\n\x0e\x63ommand_header\x18\x01 \x01(\x0b\x32\x1d.rb.api.CommandHeader.Request\x12/\n\x0cminimum_time\x18\x02 \x01(\x0b\x32\x19.google.protobuf.Duration\x12\x37\n\x07targets\x18\x03 \x03(\x0b\x32&.rb.api.CartesianCommand.SE3PoseTarget\x12\x42\n\x1cstop_position_tracking_error\x18\x04 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x45\n\x1fstop_orientation_tracking_error\x18\x05 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x1a\x8c\x01\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12?\n\x17\x63ommand_header_feedback\x18\x01 \x01(\x0b\x32\x1e.rb.api.CommandHeader.Feedback\x12?\n\x0ftracking_errors\x18\x02 \x03(\x0b\x32&.rb.api.CartesianCommand.TrackingError\"\xb7\x01\n\x1aGravityCompensationCommand\x1aL\n\x07Request\x12\x35\n\x0e\x63ommand_header\x18\x01 \x01(\x0b\x32\x1d.rb.api.CommandHeader.Request\x12\n\n\x02on\x18\x02 \x01(\x08\x1aK\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12?\n\x17\x63ommand_header_feedback\x18\x01 \x01(\x0b\x32\x1e.rb.api.CommandHeader.Feedback\"\xc9\x03\n\x17ImpedanceControlCommand\x1a?\n\rTrackingError\x12\x16\n\x0eposition_error\x18\x01 \x01(\x01\x12\x16\n\x0erotation_error\x18\x02 \x01(\x01\x1a\xd7\x01\n\x07Request\x12\x35\n\x0e\x63ommand_header\x18\x01 \x01(\x0b\x32\x1d.rb.api.CommandHeader.Request\x12\x15\n\rref_link_name\x18\x03 \x01(\t\x12\x11\n\tlink_name\x18\x04 \x01(\t\x12\x1a\n\x01T\x18\x05 \x01(\x0b\x32\x0f.rb.api.SE3Pose\x12(\n\x12translation_weight\x18\x06 \x01(\x0b\x32\x0c.rb.api.Vec3\x12%\n\x0frotation_weight\x18\x07 \x01(\x0b\x32\x0c.rb.api.Vec3\x1a\x92\x01\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12?\n\x17\x63ommand_header_feedback\x18\x01 \x01(\x0b\x32\x1e.rb.api.CommandHeader.Feedback\x12\x45\n\x0etracking_error\x18\x02 \x01(\x0b\x32-.rb.api.ImpedanceControlCommand.TrackingError\"\xba\x08\n\x15OptimalControlCommand\x1a\x8a\x01\n\rCartesianCost\x12\x15\n\rref_link_name\x18\x01 \x01(\t\x12\x11\n\tlink_name\x18\x02 \x01(\t\x12\x1a\n\x01T\x18\x03 \x01(\x0b\x32\x0f.rb.api.SE3Pose\x12\x1a\n\x12translation_weight\x18\x04 \x01(\x01\x12\x17\n\x0frotation_weight\x18\x05 \x01(\x01\x1aU\n\x10\x43\x65nterOfMassCost\x12\x15\n\rref_link_name\x18\x01 \x01(\t\x12\x1a\n\x04pose\x18\x02 \x01(\x0b\x32\x0c.rb.api.Vec3\x12\x0e\n\x06weight\x18\x03 \x01(\x01\x1aP\n\x11JointPositionCost\x12\x12\n\njoint_name\x18\x01 \x01(\t\x12\x17\n\x0ftarget_position\x18\x02 \x01(\x01\x12\x0e\n\x06weight\x18\x03 \x01(\x01\x1a\xb4\x04\n\x07Request\x12\x35\n\x0e\x63ommand_header\x18\x01 \x01(\x0b\x32\x1d.rb.api.CommandHeader.Request\x12\x44\n\x0f\x63\x61rtesian_costs\x18\x02 \x03(\x0b\x32+.rb.api.OptimalControlCommand.CartesianCost\x12K\n\x13\x63\x65nter_of_mass_cost\x18\x03 \x01(\x0b\x32..rb.api.OptimalControlCommand.CenterOfMassCost\x12M\n\x14joint_position_costs\x18\x04 \x03(\x0b\x32/.rb.api.OptimalControlCommand.JointPositionCost\x12<\n\x16velocity_limit_scaling\x18\x05 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12<\n\x16velocity_tracking_gain\x18\x06 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12/\n\tstop_cost\x18\x07 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x34\n\x0emin_delta_cost\x18\x08 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12-\n\x08patience\x18\t \x01(\x0b\x32\x1b.google.protobuf.Int32Value\x1a\xb3\x01\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12?\n\x17\x63ommand_header_feedback\x18\x01 \x01(\x0b\x32\x1e.rb.api.CommandHeader.Feedback\x12\x12\n\ntotal_cost\x18\x02 \x01(\x01\x12\x17\n\x0f\x63\x61rtesian_costs\x18\x03 \x03(\x01\x12\x1b\n\x13\x63\x65nter_of_mass_cost\x18\x04 \x01(\x01\x12\x1c\n\x14joint_position_costs\x18\x05 \x03(\x01\"\xc2\x01\n\x16RealTimeControlCommand\x1a@\n\x07Request\x12\x35\n\x0e\x63ommand_header\x18\x01 \x01(\x0b\x32\x1d.rb.api.CommandHeader.Request\x1a\x66\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12?\n\x17\x63ommand_header_feedback\x18\x01 \x01(\x0b\x32\x1e.rb.api.CommandHeader.Feedback\x12\x0c\n\x04port\x18\x02 \x01(\x05\x12\x0b\n\x03uid\x18\x03 \x01(\x04\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'rb.api.basic_command_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_STOPCOMMAND']._serialized_start=155
  _globals['_STOPCOMMAND']._serialized_end=311
  _globals['_STOPCOMMAND_REQUEST']._serialized_start=170
  _globals['_STOPCOMMAND_REQUEST']._serialized_end=234
  _globals['_STOPCOMMAND_FEEDBACK']._serialized_start=236
  _globals['_STOPCOMMAND_FEEDBACK']._serialized_end=311
  _globals['_SE2VELOCITYCOMMAND']._serialized_start=314
  _globals['_SE2VELOCITYCOMMAND']._serialized_end=615
  _globals['_SE2VELOCITYCOMMAND_REQUEST']._serialized_start=337
  _globals['_SE2VELOCITYCOMMAND_REQUEST']._serialized_end=538
  _globals['_SE2VELOCITYCOMMAND_FEEDBACK']._serialized_start=236
  _globals['_SE2VELOCITYCOMMAND_FEEDBACK']._serialized_end=311
  _globals['_JOGCOMMAND']._serialized_start=618
  _globals['_JOGCOMMAND']._serialized_end=1022
  _globals['_JOGCOMMAND_REQUEST']._serialized_start=633
  _globals['_JOGCOMMAND_REQUEST']._serialized_end=918
  _globals['_JOGCOMMAND_FEEDBACK']._serialized_start=920
  _globals['_JOGCOMMAND_FEEDBACK']._serialized_end=1022
  _globals['_JOINTVELOCITYCOMMAND']._serialized_start=1025
  _globals['_JOINTVELOCITYCOMMAND']._serialized_end=1286
  _globals['_JOINTVELOCITYCOMMAND_REQUEST']._serialized_start=1050
  _globals['_JOINTVELOCITYCOMMAND_REQUEST']._serialized_end=1209
  _globals['_JOINTVELOCITYCOMMAND_FEEDBACK']._serialized_start=236
  _globals['_JOINTVELOCITYCOMMAND_FEEDBACK']._serialized_end=311
  _globals['_JOINTPOSITIONCOMMAND']._serialized_start=1289
  _globals['_JOINTPOSITIONCOMMAND']._serialized_end=1630
  _globals['_JOINTPOSITIONCOMMAND_REQUEST']._serialized_start=1314
  _globals['_JOINTPOSITIONCOMMAND_REQUEST']._serialized_end=1553
  _globals['_JOINTPOSITIONCOMMAND_FEEDBACK']._serialized_start=236
  _globals['_JOINTPOSITIONCOMMAND_FEEDBACK']._serialized_end=311
  _globals['_CARTESIANCOMMAND']._serialized_start=1633
  _globals['_CARTESIANCOMMAND']._serialized_end=2448
  _globals['_CARTESIANCOMMAND_SE3POSETARGET']._serialized_start=1654
  _globals['_CARTESIANCOMMAND_SE3POSETARGET']._serialized_end=1928
  _globals['_CARTESIANCOMMAND_TRACKINGERROR']._serialized_start=1930
  _globals['_CARTESIANCOMMAND_TRACKINGERROR']._serialized_end=1993
  _globals['_CARTESIANCOMMAND_REQUEST']._serialized_start=1996
  _globals['_CARTESIANCOMMAND_REQUEST']._serialized_end=2305
  _globals['_CARTESIANCOMMAND_FEEDBACK']._serialized_start=2308
  _globals['_CARTESIANCOMMAND_FEEDBACK']._serialized_end=2448
  _globals['_GRAVITYCOMPENSATIONCOMMAND']._serialized_start=2451
  _globals['_GRAVITYCOMPENSATIONCOMMAND']._serialized_end=2634
  _globals['_GRAVITYCOMPENSATIONCOMMAND_REQUEST']._serialized_start=2481
  _globals['_GRAVITYCOMPENSATIONCOMMAND_REQUEST']._serialized_end=2557
  _globals['_GRAVITYCOMPENSATIONCOMMAND_FEEDBACK']._serialized_start=236
  _globals['_GRAVITYCOMPENSATIONCOMMAND_FEEDBACK']._serialized_end=311
  _globals['_IMPEDANCECONTROLCOMMAND']._serialized_start=2637
  _globals['_IMPEDANCECONTROLCOMMAND']._serialized_end=3094
  _globals['_IMPEDANCECONTROLCOMMAND_TRACKINGERROR']._serialized_start=1930
  _globals['_IMPEDANCECONTROLCOMMAND_TRACKINGERROR']._serialized_end=1993
  _globals['_IMPEDANCECONTROLCOMMAND_REQUEST']._serialized_start=2730
  _globals['_IMPEDANCECONTROLCOMMAND_REQUEST']._serialized_end=2945
  _globals['_IMPEDANCECONTROLCOMMAND_FEEDBACK']._serialized_start=2948
  _globals['_IMPEDANCECONTROLCOMMAND_FEEDBACK']._serialized_end=3094
  _globals['_OPTIMALCONTROLCOMMAND']._serialized_start=3097
  _globals['_OPTIMALCONTROLCOMMAND']._serialized_end=4179
  _globals['_OPTIMALCONTROLCOMMAND_CARTESIANCOST']._serialized_start=3123
  _globals['_OPTIMALCONTROLCOMMAND_CARTESIANCOST']._serialized_end=3261
  _globals['_OPTIMALCONTROLCOMMAND_CENTEROFMASSCOST']._serialized_start=3263
  _globals['_OPTIMALCONTROLCOMMAND_CENTEROFMASSCOST']._serialized_end=3348
  _globals['_OPTIMALCONTROLCOMMAND_JOINTPOSITIONCOST']._serialized_start=3350
  _globals['_OPTIMALCONTROLCOMMAND_JOINTPOSITIONCOST']._serialized_end=3430
  _globals['_OPTIMALCONTROLCOMMAND_REQUEST']._serialized_start=3433
  _globals['_OPTIMALCONTROLCOMMAND_REQUEST']._serialized_end=3997
  _globals['_OPTIMALCONTROLCOMMAND_FEEDBACK']._serialized_start=4000
  _globals['_OPTIMALCONTROLCOMMAND_FEEDBACK']._serialized_end=4179
  _globals['_REALTIMECONTROLCOMMAND']._serialized_start=4182
  _globals['_REALTIMECONTROLCOMMAND']._serialized_end=4376
  _globals['_REALTIMECONTROLCOMMAND_REQUEST']._serialized_start=170
  _globals['_REALTIMECONTROLCOMMAND_REQUEST']._serialized_end=234
  _globals['_REALTIMECONTROLCOMMAND_FEEDBACK']._serialized_start=4274
  _globals['_REALTIMECONTROLCOMMAND_FEEDBACK']._serialized_end=4376
# @@protoc_insertion_point(module_scope)
