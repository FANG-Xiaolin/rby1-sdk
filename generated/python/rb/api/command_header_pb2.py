# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rb/api/command_header.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import duration_pb2 as google_dot_protobuf_dot_duration__pb2
from rb.api import geometry_pb2 as rb_dot_api_dot_geometry__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1brb/api/command_header.proto\x12\x06rb.api\x1a\x1egoogle/protobuf/duration.proto\x1a\x15rb/api/geometry.proto\"z\n\rCommandHeader\x1aK\n\x07Request\x12\x34\n\x11\x63ontrol_hold_time\x18\x01 \x01(\x0b\x32\x19.google.protobuf.DurationJ\x04\x08\x02\x10\x03J\x04\x08\x03\x10\x04\x1a\x1c\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12\x10\n\x08\x66inished\x18\x01 \x01(\x08\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'rb.api.command_header_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_COMMANDHEADER']._serialized_start=94
  _globals['_COMMANDHEADER']._serialized_end=216
  _globals['_COMMANDHEADER_REQUEST']._serialized_start=111
  _globals['_COMMANDHEADER_REQUEST']._serialized_end=186
  _globals['_COMMANDHEADER_FEEDBACK']._serialized_start=188
  _globals['_COMMANDHEADER_FEEDBACK']._serialized_end=216
# @@protoc_insertion_point(module_scope)
