"""yaml_codec.py
Utility functions to serialize and deserialize ExecutableGraph objects
from and to YAML.  The primary public helpers are:

- graph_to_yaml(graph)  -> dict  (ready for yaml.safe_dump)
- yaml_to_graph(data)   -> ExecutableGraph

The YAML representation is intentionally human-readable and aims to be
stable under round-tripping.  For callables we try a dotted-path first; if
that fails we fall back to a base64-encoded cloudpickle blob.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Union
import base64
import importlib

import yaml  # PyYAML is already declared as a dependency
import cloudpickle

from pyautocausal.orchestration.graph import ExecutableGraph  # noqa: I001  (local import to avoid cycles)
from pyautocausal.orchestration.nodes import Node, InputNode, DecisionNode  # noqa: E402  (after sys.path)
from pyautocausal.persistence.output_config import OutputConfig
from pyautocausal.persistence.output_types import OutputType
from pyautocausal.persistence.parameter_mapper import TransformedFunctionWrapper

# ---------------------------------------------------------------------------
# Helper functions for (de)serialising callables
# ---------------------------------------------------------------------------

def _callable_to_spec(func: Callable) -> Dict[str, str]:
    """Turn *func* into a serialisable mapping.

    Strategy:
    1.  If the function can be re-imported via ``module:qualname`` we emit:
        {type: dotted, value: "module:qualname"}
    2.  If it's our special transformed wrapper, we deconstruct it into a
        portable spec so it can be rebuilt after deserialisation.
    3.  Otherwise we emit a base64-encoded cloudpickle dump, patching the
        code object's filename to be portable if possible.
    """
    # Is it our special wrapper? If so, deconstruct it.
    if isinstance(func, TransformedFunctionWrapper):
        # Recursively create a spec for the inner function, which ensures
        # that our co_filename patching logic below gets applied to it.
        inner_func_spec = _callable_to_spec(func.func)
        return {
            "type": "transformed_wrapper",
            "func_spec": inner_func_spec,
            "arg_mapping": func.arg_mapping,
        }

    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)

    # ------------------------------------------------------------------
    # 1.  First preference: dotted import path (portable & no pickling)
    # ------------------------------------------------------------------
    if module and qualname:
        try:
            mod = importlib.import_module(module)
            resolved = eval(qualname, mod.__dict__)  # noqa: S307 – controlled input
            if resolved is func:
                return {"type": "dotted", "value": f"{module}:{qualname}"}
        except Exception:  # pragma: no cover – fallback path
            pass

    # ------------------------------------------------------------------
    # 2.  Fallback: pickle – but first normalise *co_filename* so that
    #     inspect.getsource works after deserialisation regardless of the
    #     absolute path of the developer machine on which the graph was
    #     created.  We rewrite it to a package-relative path such as
    #     "pyautocausal/pipelines/library/estimators.py".
    # ------------------------------------------------------------------
    try:
        import types, os, pathlib

        code = func.__code__

        # Only patch if the filename is an absolute path that is unlikely to
        # exist on another machine / inside Docker.
        if os.path.isabs(code.co_filename):
            if module:
                # Derive a portable path from the module name.
                portable_filename = module.replace(".", os.sep) + ".py"
            else:
                # Best effort – just strip leading slashes to make it relative
                portable_filename = os.path.basename(code.co_filename)

            # On Python ≥3.8 we can use CodeType.replace; fall back otherwise.
            try:
                new_code = code.replace(co_filename=portable_filename)  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover – <3.8 not supported in project CI
                new_code = types.CodeType(
                    code.co_argcount,
                    code.co_posonlyargcount if hasattr(code, "co_posonlyargcount") else 0,
                    code.co_kwonlyargcount,
                    code.co_nlocals,
                    code.co_stacksize,
                    code.co_flags,
                    code.co_code,
                    code.co_consts,
                    code.co_names,
                    code.co_varnames,
                    portable_filename,
                    code.co_name,
                    code.co_firstlineno,
                    code.co_lnotab,
                    code.co_freevars,
                    code.co_cellvars,
                )

            func = types.FunctionType(
                new_code,
                func.__globals__,
                func.__name__,
                func.__defaults__,
                func.__closure__,
            )

    except Exception:  # pragma: no cover – defensive; if anything goes wrong just pickle as-is
        pass

    # Finally pickle the (possibly patched) function
    pickled = cloudpickle.dumps(func)
    encoded = base64.b64encode(pickled).decode("utf-8")
    return {"type": "pickle", "value": encoded}


def _spec_to_callable(spec: Dict[str, str]) -> Callable:
    """Reconstruct function from the mapping produced by *_callable_to_spec*."""
    spec_type = spec["type"]
    
    if spec_type == "dotted":
        module, qualname = spec["value"].split(":", 1)
        mod = importlib.import_module(module)
        return eval(qualname, mod.__dict__)  # noqa: S307 – controlled input

    elif spec_type == "transformed_wrapper":
        from .parameter_mapper import make_transformable  # local import to avoid cycle
        
        # Recursively reconstruct the inner function first
        inner_func = _spec_to_callable(spec["func_spec"])
        
        # Now, re-apply the transformation
        return make_transformable(inner_func).transform(spec["arg_mapping"])

    elif spec_type == "pickle":
        data = base64.b64decode(spec["value"].encode("utf-8"))
        return cloudpickle.loads(data)
    else:  # pragma: no cover – defensive
        raise ValueError(f"Unknown callable spec type: {spec['type']}")


# ---------------------------------------------------------------------------
# Graph <-> YAML
# ---------------------------------------------------------------------------

def graph_to_mapping(graph: ExecutableGraph) -> Dict[str, Any]:
    """Serialise *graph* to a plain-Python mapping that yaml.safe_dump can emit."""
    nodes_block: List[Dict[str, Any]] = []

    # First gather node information
    for node in graph._nodes_by_name.values():  # type: ignore[attr-defined]
        info: Dict[str, Any] = {"name": node.name, "kind": node.__class__.__name__}

        if isinstance(node, InputNode):
            dtype = node.input_dtype
            if dtype is not Any:
                info["input_dtype"] = f"{dtype.__module__}:{dtype.__qualname__}"

        if isinstance(node, DecisionNode):
            info["condition"] = _callable_to_spec(node.condition)
            info["ewt"] = [n.name for n in node._ewt_nodes]  # type: ignore[attr-defined]
            info["ewf"] = [n.name for n in node._ewf_nodes]  # type: ignore[attr-defined]

        if isinstance(node, Node) and not isinstance(node, InputNode):
            info["action_function"] = _callable_to_spec(node.action_function)

        # Optional output config
        if getattr(node, "output_config", None):
            oc: OutputConfig = node.output_config  # type: ignore[assignment]
            info["output_config"] = {
                "output_type": oc.output_type.name if isinstance(oc.output_type, OutputType) else str(oc.output_type),
                "output_filename": oc.output_filename,
            }

        nodes_block.append(info)

    edges_block = [
        {"from": u.name, "to": v.name}
        for u, v in graph.edges()
    ]

    return {"nodes": nodes_block, "edges": edges_block}


def graph_to_yaml(graph: ExecutableGraph, filepath: Union[str, Path]) -> Path:
    mapping = graph_to_mapping(graph)
    path = Path(filepath).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(mapping, fh, sort_keys=False)
    return path


def mapping_to_graph(mapping: Dict[str, Any]) -> ExecutableGraph:
    """Recreate an ExecutableGraph from a mapping previously produced."""
    from pyautocausal.orchestration.nodes import Node as NodeCls, InputNode as InputNodeCls, DecisionNode as DecisionNodeCls  # local import to avoid cycles

    g = ExecutableGraph()

    # First pass: create nodes without edges
    tmp: Dict[str, Any] = {}
    for nd in mapping["nodes"]:
        kind = nd["kind"]
        name = nd["name"]
        if kind == "InputNode":
            dtype_spec = nd.get("input_dtype")
            if dtype_spec and dtype_spec != "typing.Any":
                mod, qual = dtype_spec.split(":", 1)
                dtype = getattr(importlib.import_module(mod), qual)
            else:
                from typing import Any as _Any
                dtype = _Any
            g.create_input_node(name, input_dtype=dtype)
            node_obj = g.get(name)
        elif kind == "DecisionNode":
            condition = _spec_to_callable(nd["condition"])
            node_obj = DecisionNodeCls(name=name, condition=condition)
            g.add_node_to_graph(node_obj)
        elif kind == "Node":
            action = _spec_to_callable(nd["action_function"])
            oc = None
            if "output_config" in nd:
                oc_map = nd["output_config"]
                ot_name = oc_map["output_type"]
                ot = getattr(OutputType, ot_name, ot_name)
                oc = OutputConfig(output_type=ot, output_filename=oc_map["output_filename"])
            node_obj = NodeCls(name=name, action_function=action, output_config=oc, save_node=bool(oc))
            g.add_node_to_graph(node_obj)
        else:
            raise ValueError(f"Unknown node kind: {kind}")
        tmp[name] = node_obj

    # Second pass: add edges
    for edge in mapping["edges"]:
        g.add_edge(tmp[edge["from"]], tmp[edge["to"]])

    # Third pass: rebuild decision branch metadata
    for nd in mapping["nodes"]:
        if nd["kind"] == "DecisionNode":
            node_obj: DecisionNode = tmp[nd["name"]]  # type: ignore[assignment]
            for n in nd.get("ewt", []):
                node_obj.add_execute_when_true(tmp[n])
            for n in nd.get("ewf", []):
                node_obj.add_execute_when_false(tmp[n])

    return g


def yaml_to_graph(filepath: Union[str, Path]) -> ExecutableGraph:
    path = Path(filepath).expanduser().resolve()
    with path.open("r", encoding="utf-8") as fh:
        mapping = yaml.safe_load(fh)
    return mapping_to_graph(mapping) 