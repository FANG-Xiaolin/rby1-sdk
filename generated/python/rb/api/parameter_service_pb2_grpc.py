# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from rb.api import parameter_pb2 as rb_dot_api_dot_parameter__pb2

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
        + f' but the generated code in rb/api/parameter_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
        + f' This warning will become an error in {EXPECTED_ERROR_RELEASE},'
        + f' scheduled for release on {SCHEDULED_RELEASE_DATE}.',
        RuntimeWarning
    )


class ParameterServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.FactoryResetAllParameters = channel.unary_unary(
                '/rb.api.ParameterService/FactoryResetAllParameters',
                request_serializer=rb_dot_api_dot_parameter__pb2.FactoryResetAllParametersRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_parameter__pb2.FactoryResetAllParametersResponse.FromString,
                _registered_method=True)
        self.FactoryResetParameter = channel.unary_unary(
                '/rb.api.ParameterService/FactoryResetParameter',
                request_serializer=rb_dot_api_dot_parameter__pb2.FactoryResetParameterRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_parameter__pb2.FactoryResetParameterResponse.FromString,
                _registered_method=True)
        self.ResetAllParameters = channel.unary_unary(
                '/rb.api.ParameterService/ResetAllParameters',
                request_serializer=rb_dot_api_dot_parameter__pb2.ResetAllParametersRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_parameter__pb2.ResetAllParametersResponse.FromString,
                _registered_method=True)
        self.ResetParameter = channel.unary_unary(
                '/rb.api.ParameterService/ResetParameter',
                request_serializer=rb_dot_api_dot_parameter__pb2.ResetParameterRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_parameter__pb2.ResetParameterResponse.FromString,
                _registered_method=True)
        self.GetParameter = channel.unary_unary(
                '/rb.api.ParameterService/GetParameter',
                request_serializer=rb_dot_api_dot_parameter__pb2.GetParameterRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_parameter__pb2.GetParameterResponse.FromString,
                _registered_method=True)
        self.SetParameter = channel.unary_unary(
                '/rb.api.ParameterService/SetParameter',
                request_serializer=rb_dot_api_dot_parameter__pb2.SetParameterRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_parameter__pb2.SetParameterResponse.FromString,
                _registered_method=True)
        self.GetParameterList = channel.unary_unary(
                '/rb.api.ParameterService/GetParameterList',
                request_serializer=rb_dot_api_dot_parameter__pb2.GetParameterListRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_parameter__pb2.GetParameterListResponse.FromString,
                _registered_method=True)
        self.ResetAllParametersToDefault = channel.unary_unary(
                '/rb.api.ParameterService/ResetAllParametersToDefault',
                request_serializer=rb_dot_api_dot_parameter__pb2.ResetAllParametersToDefaultRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_parameter__pb2.ResetAllParametersToDefaultResponse.FromString,
                _registered_method=True)
        self.ResetParameterToDefault = channel.unary_unary(
                '/rb.api.ParameterService/ResetParameterToDefault',
                request_serializer=rb_dot_api_dot_parameter__pb2.ResetParameterToDefaultRequest.SerializeToString,
                response_deserializer=rb_dot_api_dot_parameter__pb2.ResetParameterToDefaultResponse.FromString,
                _registered_method=True)


class ParameterServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def FactoryResetAllParameters(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FactoryResetParameter(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ResetAllParameters(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ResetParameter(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetParameter(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetParameter(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetParameterList(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ResetAllParametersToDefault(self, request, context):
        """Deprecated
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ResetParameterToDefault(self, request, context):
        """Deprecated
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ParameterServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'FactoryResetAllParameters': grpc.unary_unary_rpc_method_handler(
                    servicer.FactoryResetAllParameters,
                    request_deserializer=rb_dot_api_dot_parameter__pb2.FactoryResetAllParametersRequest.FromString,
                    response_serializer=rb_dot_api_dot_parameter__pb2.FactoryResetAllParametersResponse.SerializeToString,
            ),
            'FactoryResetParameter': grpc.unary_unary_rpc_method_handler(
                    servicer.FactoryResetParameter,
                    request_deserializer=rb_dot_api_dot_parameter__pb2.FactoryResetParameterRequest.FromString,
                    response_serializer=rb_dot_api_dot_parameter__pb2.FactoryResetParameterResponse.SerializeToString,
            ),
            'ResetAllParameters': grpc.unary_unary_rpc_method_handler(
                    servicer.ResetAllParameters,
                    request_deserializer=rb_dot_api_dot_parameter__pb2.ResetAllParametersRequest.FromString,
                    response_serializer=rb_dot_api_dot_parameter__pb2.ResetAllParametersResponse.SerializeToString,
            ),
            'ResetParameter': grpc.unary_unary_rpc_method_handler(
                    servicer.ResetParameter,
                    request_deserializer=rb_dot_api_dot_parameter__pb2.ResetParameterRequest.FromString,
                    response_serializer=rb_dot_api_dot_parameter__pb2.ResetParameterResponse.SerializeToString,
            ),
            'GetParameter': grpc.unary_unary_rpc_method_handler(
                    servicer.GetParameter,
                    request_deserializer=rb_dot_api_dot_parameter__pb2.GetParameterRequest.FromString,
                    response_serializer=rb_dot_api_dot_parameter__pb2.GetParameterResponse.SerializeToString,
            ),
            'SetParameter': grpc.unary_unary_rpc_method_handler(
                    servicer.SetParameter,
                    request_deserializer=rb_dot_api_dot_parameter__pb2.SetParameterRequest.FromString,
                    response_serializer=rb_dot_api_dot_parameter__pb2.SetParameterResponse.SerializeToString,
            ),
            'GetParameterList': grpc.unary_unary_rpc_method_handler(
                    servicer.GetParameterList,
                    request_deserializer=rb_dot_api_dot_parameter__pb2.GetParameterListRequest.FromString,
                    response_serializer=rb_dot_api_dot_parameter__pb2.GetParameterListResponse.SerializeToString,
            ),
            'ResetAllParametersToDefault': grpc.unary_unary_rpc_method_handler(
                    servicer.ResetAllParametersToDefault,
                    request_deserializer=rb_dot_api_dot_parameter__pb2.ResetAllParametersToDefaultRequest.FromString,
                    response_serializer=rb_dot_api_dot_parameter__pb2.ResetAllParametersToDefaultResponse.SerializeToString,
            ),
            'ResetParameterToDefault': grpc.unary_unary_rpc_method_handler(
                    servicer.ResetParameterToDefault,
                    request_deserializer=rb_dot_api_dot_parameter__pb2.ResetParameterToDefaultRequest.FromString,
                    response_serializer=rb_dot_api_dot_parameter__pb2.ResetParameterToDefaultResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'rb.api.ParameterService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('rb.api.ParameterService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class ParameterService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def FactoryResetAllParameters(request,
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
            '/rb.api.ParameterService/FactoryResetAllParameters',
            rb_dot_api_dot_parameter__pb2.FactoryResetAllParametersRequest.SerializeToString,
            rb_dot_api_dot_parameter__pb2.FactoryResetAllParametersResponse.FromString,
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
    def FactoryResetParameter(request,
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
            '/rb.api.ParameterService/FactoryResetParameter',
            rb_dot_api_dot_parameter__pb2.FactoryResetParameterRequest.SerializeToString,
            rb_dot_api_dot_parameter__pb2.FactoryResetParameterResponse.FromString,
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
    def ResetAllParameters(request,
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
            '/rb.api.ParameterService/ResetAllParameters',
            rb_dot_api_dot_parameter__pb2.ResetAllParametersRequest.SerializeToString,
            rb_dot_api_dot_parameter__pb2.ResetAllParametersResponse.FromString,
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
    def ResetParameter(request,
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
            '/rb.api.ParameterService/ResetParameter',
            rb_dot_api_dot_parameter__pb2.ResetParameterRequest.SerializeToString,
            rb_dot_api_dot_parameter__pb2.ResetParameterResponse.FromString,
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
    def GetParameter(request,
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
            '/rb.api.ParameterService/GetParameter',
            rb_dot_api_dot_parameter__pb2.GetParameterRequest.SerializeToString,
            rb_dot_api_dot_parameter__pb2.GetParameterResponse.FromString,
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
    def SetParameter(request,
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
            '/rb.api.ParameterService/SetParameter',
            rb_dot_api_dot_parameter__pb2.SetParameterRequest.SerializeToString,
            rb_dot_api_dot_parameter__pb2.SetParameterResponse.FromString,
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
    def GetParameterList(request,
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
            '/rb.api.ParameterService/GetParameterList',
            rb_dot_api_dot_parameter__pb2.GetParameterListRequest.SerializeToString,
            rb_dot_api_dot_parameter__pb2.GetParameterListResponse.FromString,
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
    def ResetAllParametersToDefault(request,
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
            '/rb.api.ParameterService/ResetAllParametersToDefault',
            rb_dot_api_dot_parameter__pb2.ResetAllParametersToDefaultRequest.SerializeToString,
            rb_dot_api_dot_parameter__pb2.ResetAllParametersToDefaultResponse.FromString,
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
    def ResetParameterToDefault(request,
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
            '/rb.api.ParameterService/ResetParameterToDefault',
            rb_dot_api_dot_parameter__pb2.ResetParameterToDefaultRequest.SerializeToString,
            rb_dot_api_dot_parameter__pb2.ResetParameterToDefaultResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
