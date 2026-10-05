"""Graph evaluation with per-node caching. Independent of bpy."""

from dataclasses import dataclass, field

from .ops import OPS, COLOR, GRAY, SAME


@dataclass
class NodeSpec:
    id: str
    op: str
    params: dict = field(default_factory=dict)
    # input name -> upstream node id (or None when unconnected)
    inputs: dict = field(default_factory=dict)


class GraphError(Exception):
    pass


def _freeze(value):
    if isinstance(value, (list, tuple)):
        return tuple(round(float(v), 6) for v in value)
    if isinstance(value, float):
        return round(value, 6)
    return value


class Evaluator:
    """Evaluates a graph through a backend, re-running only nodes whose
    parameters, inputs or resolution changed since the last call."""

    def __init__(self, backend):
        self.backend = backend
        # node id -> (key, buffer, kind)
        self.cache = {}
        self.ran = []

    def evaluate(self, nodes, targets, size):
        """Return {target id: (buffer, kind)} for the requested node ids."""
        self.ran = []
        keys = {}
        results = {}
        visiting = set()

        def visit(node_id):
            if node_id in results:
                return results[node_id]
            if node_id in visiting:
                raise GraphError("The graph has a cycle")
            node = nodes.get(node_id)
            if node is None:
                raise GraphError(f"Missing node {node_id}")
            op = OPS.get(node.op)
            if op is None:
                raise GraphError(f"Unknown operation {node.op}")
            visiting.add(node_id)
            upstream = {}
            for name in op.inputs:
                src = node.inputs.get(name)
                upstream[name] = visit(src) if src is not None else None
            visiting.discard(node_id)

            params = dict(op.defaults)
            params.update({k: v for k, v in node.params.items() if k in op.defaults})
            key = (
                node.op,
                tuple(size),
                tuple(sorted((k, _freeze(v)) for k, v in params.items())),
                tuple((name, keys[node.inputs[name]] if upstream[name] else None) for name in op.inputs),
            )
            keys[node_id] = hash(key)

            cached = self.cache.get(node_id)
            if cached is not None and cached[0] == keys[node_id]:
                results[node_id] = (cached[1], cached[2])
                return results[node_id]

            kind = op.output
            if kind == SAME:
                kinds = [u[1] for u in upstream.values() if u is not None]
                kind = COLOR if COLOR in kinds else GRAY
            buffers = {name: (u[0] if u else None) for name, u in upstream.items()}
            buf = self.backend.run(op, buffers, params, tuple(size), kind)
            if cached is not None:
                self.backend.free(cached[1])
            self.cache[node_id] = (keys[node_id], buf, kind)
            self.ran.append(node_id)
            results[node_id] = (buf, kind)
            return results[node_id]

        out = {t: visit(t) for t in targets}

        for node_id in list(self.cache):
            if node_id not in nodes:
                self.backend.free(self.cache.pop(node_id)[1])
        return out

    def clear(self):
        for _, buf, _ in self.cache.values():
            self.backend.free(buf)
        self.cache.clear()
