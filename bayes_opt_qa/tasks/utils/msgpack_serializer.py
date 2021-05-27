from typing import Any

import msgpack
from prefect.engine.serializers import Serializer


class MsgPackSerializer(Serializer):
    """A Serializer that uses MsgPack to serialize objects"""

    def serialize(self, value: Any) -> bytes:
        """
        Serialize an object to MsgPack
        Args:
            - value (Any): the value to serialize
        Returns:
            - bytes: the serialized value
        """
        return msgpack.packb(value, use_bin_type=True)

    def deserialize(self, value: bytes) -> Any:
        """
        Deserialize an object from MsgPack
        Args:
            - value (bytes): the value to deserialize
        Returns:
            - Any: the deserialized value
        """
        return msgpack.unpackb(value, raw=False, use_list=False)
