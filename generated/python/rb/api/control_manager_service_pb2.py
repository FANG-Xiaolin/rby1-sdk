# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rb/api/control_manager_service.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from rb.api import control_manager_pb2 as rb_dot_api_dot_control__manager__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n$rb/api/control_manager_service.proto\x12\x06rb.api\x1a\x1crb/api/control_manager.proto2\xcc\x01\n\x15\x43ontrolManagerService\x12\x66\n\x15\x43ontrolManagerCommand\x12$.rb.api.ControlManagerCommandRequest\x1a%.rb.api.ControlManagerCommandResponse\"\x00\x12K\n\x0cSetTimeScale\x12\x1b.rb.api.SetTimeScaleRequest\x1a\x1c.rb.api.SetTimeScaleResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'rb.api.control_manager_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_CONTROLMANAGERSERVICE']._serialized_start=79
  _globals['_CONTROLMANAGERSERVICE']._serialized_end=283
# @@protoc_insertion_point(module_scope)
