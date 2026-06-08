from typing import Type, Any

from pydantic import BaseModel
from llama_index.core.program.function_program import get_function_tool


# TODO: It will redefine behavior or code that uses fn_schema's model_json_schema
#       but not sure whether that could lead to a real problem.
def _make_tool_schema_model(model_cls: Type[BaseModel]) -> Type[BaseModel]:
    """
    Monky patch to be comaptible with google genai, when creating tool from pydantic
    model. The cause arises from the fact that we're using 'model_config' of extra is
    forbidden.

    Return a subclass of `model_cls` whose model_json_schema()
    drops all `additionalProperties` occurrences.

    Runtime validation behavior (extra='forbid') is preserved,
    only the exported JSON schema is sanitized.
    """

    def _drop_additional(node: Any) -> Any:
        if isinstance(node, dict):
            # remove additionalProperties at this level
            node.pop("additionalProperties", None)
            # recurse into values
            for v in node.values():
                _drop_additional(v)
        elif isinstance(node, list):
            for item in node:
                _drop_additional(item)
        return node

    name = f"{model_cls.__name__}ToolSchema"

    @classmethod  # type: ignore[override]
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict:
        schema = model_cls.model_json_schema(*args, **kwargs)
        return _drop_additional(schema)

    attrs = {
        "__doc__": model_cls.__doc__,
        "model_json_schema": model_json_schema,
    }

    ToolModel = type(name, (model_cls,), attrs)
    return ToolModel


def create_model_tool(model_cls):
    """
    Create a safe FunctionTool from a Pydantic model, with JSON schema sanitization
    Replaces llamaindex's get_function_tool utility function
    """
    model_cls_patched = _make_tool_schema_model(model_cls)
    return get_function_tool(model_cls_patched)
