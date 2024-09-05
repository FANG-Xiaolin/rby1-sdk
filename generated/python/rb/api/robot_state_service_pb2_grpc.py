# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from rb.api import robot_state_pb2 as rb_dot_api_dot_robot__state__pb2

GRPC_GENERATED_VERSION = '1.65.2'
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = '1.66.0'
SCHEDULED_RELEASE_DATE = 'August 6, 2024'
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    warnings.warn(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in rb/api/robot_state_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
        + f' This warning will become an error in {EXPECTED_ERROR_RELEASE},'
        + f' scheduled for release on {SCHEDULED_RELEASE_DATE}.',
        RuntimeWarning
    )


class RobotStateServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetRobotState = channel.unary_unary(
                '/rb.api.RobotStateService/GetRobotState',
                request_serializer=rb_dot_api_dot_robot__state__pb2.GetRobotStateRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_robot__state__pb2.GetRobotStateResponse.FromString,
                _registered_method=True)
        self.GetRobotStateStream = channel.unary_stream(
                '/rb.api.RobotStateService/GetRobotStateStream',
                request_serializer=rb_dot_api_dot_robot__state__pb2.GetRobotStateStreamRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_robot__state__pb2.GetRobotStateStreamResponse.FromString,
                _registered_method=True)
        self.GetControlManagerState = channel.unary_unary(
                '/rb.api.RobotStateService/GetControlManagerState',
                request_serializer=rb_dot_api_dot_robot__state__pb2.GetControlManagerStateRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_robot__state__pb2.GetControlManagerStateResponse.FromString,
                _registered_method=True)
        self.ResetOdometry = channel.unary_unary(
                '/rb.api.RobotStateService/ResetOdometry',
                request_serializer=rb_dot_api_dot_robot__state__pb2.ResetOdometryRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_robot__state__pb2.ResetOdometryResponse.FromString,
                _registered_method=True)


class RobotStateServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetRobotState(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetRobotStateStream(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetControlManagerState(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ResetOdometry(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_RobotStateServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetRobotState': grpc.unary_unary_rpc_method_handler(
                    servicer.GetRobotState,
                    request_deserializer=rb_dot_api_dot_robot__state__pb2.GetRobotStateRequest.FromString,
                    response_serializer=rb_dot_api_dot_robot__state__pb2.GetRobotStateResponse.SerializeToString,
            ),
            'GetRobotStateStream': grpc.unary_stream_rpc_method_handler(
                    servicer.GetRobotStateStream,
                    request_deserializer=rb_dot_api_dot_robot__state__pb2.GetRobotStateStreamRequest.FromString,
                    response_serializer=rb_dot_api_dot_robot__state__pb2.GetRobotStateStreamResponse.SerializeToString,
            ),
            'GetControlManagerState': grpc.unary_unary_rpc_method_handler(
                    servicer.GetControlManagerState,
                    request_deserializer=rb_dot_api_dot_robot__state__pb2.GetControlManagerStateRequest.FromString,
                    response_serializer=rb_dot_api_dot_robot__state__pb2.GetControlManagerStateResponse.SerializeToString,
            ),
            'ResetOdometry': grpc.unary_unary_rpc_method_handler(
                    servicer.ResetOdometry,
                    request_deserializer=rb_dot_api_dot_robot__state__pb2.ResetOdometryRequest.FromString,
                    response_serializer=rb_dot_api_dot_robot__state__pb2.ResetOdometryResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'rb.api.RobotStateService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('rb.api.RobotStateService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class RobotStateService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetRobotState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/rb.api.RobotStateService/GetRobotState',
            rb_dot_api_dot_robot__state__pb2.GetRobotStateRequest.SerializeToString,
            rb_dot_api_dot_robot__state__pb2.GetRobotStateResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetRobotStateStream(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(
            request,
            target,
            '/rb.api.RobotStateService/GetRobotStateStream',
            rb_dot_api_dot_robot__state__pb2.GetRobotStateStreamRequest.SerializeToString,
            rb_dot_api_dot_robot__state__pb2.GetRobotStateStreamResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetControlManagerState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/rb.api.RobotStateService/GetControlManagerState',
            rb_dot_api_dot_robot__state__pb2.GetControlManagerStateRequest.SerializeToString,
            rb_dot_api_dot_robot__state__pb2.GetControlManagerStateResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ResetOdometry(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/rb.api.RobotStateService/ResetOdometry',
            rb_dot_api_dot_robot__state__pb2.ResetOdometryRequest.SerializeToString,
            rb_dot_api_dot_robot__state__pb2.ResetOdometryResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
