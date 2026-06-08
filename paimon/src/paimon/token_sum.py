import csv
import json
import warnings
from dataclasses import dataclass, asdict

from paimon import cfg

_cost_map: dict[str, dict[str, float]] | None = None

COST_UNIT_FACTOR: float = 1 / 1_000_000
_default_cost_map = {"input": 0.0, "cached": 0.0, "output": 0.0}


def safe_get(obj, *path, default=None):
    """Traverse nested attributes or dict keys safely.
    Returns default if anything is missing."""
    for key in path:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(key, default)
            continue
        obj = getattr(obj, key, default)
    return obj


def get_cost_map() -> dict[str, dict[str, float]]:
    global _cost_map
    if not cfg.cost_file:
        raise ValueError("Token cost csv file is not set")
    if _cost_map is None:
        _cost_map = {}
        with open(cfg.cost_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row["Model"]
                _cost_map[model] = {
                    "input": float(row["Input"]),
                    "cached": float(row["CachedInput"]),
                    "output": float(row["Output"]),
                }
    return _cost_map.copy()


@dataclass
class TokenUsageEntry:
    # auxiliary information
    name: str | None
    llm_model: str | None
    tool_call: str | list[str]

    input_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    output_tokens: int
    total_tokens: int

    role: str | None = None

    @property
    def cost_map(self) -> dict[str, float]:
        """Get cost map for this model"""
        if self.llm_model is None:
            return _default_cost_map.copy()
        return get_cost_map().get(
            self.llm_model,
            _default_cost_map.copy(),
        )

    @property
    def uncached_input_cost(self) -> float:
        """Calculate uncached input token cost"""
        cost_map = self.cost_map
        uncached_tokens = self.input_tokens - self.cached_tokens
        return uncached_tokens * cost_map.get("input", 0.0) * COST_UNIT_FACTOR

    @property
    def cached_cost(self) -> float:
        """Calculate cached token cost"""
        cost_map = self.cost_map
        return self.cached_tokens * cost_map.get("cached", 0.0) * COST_UNIT_FACTOR

    @property
    def output_cost(self) -> float:
        """Calculate output token cost"""
        cost_map = self.cost_map
        return self.output_tokens * cost_map.get("output", 0.0) * COST_UNIT_FACTOR

    @property
    def total_cost(self) -> float:
        """Calculate total cost (reasoning tokens not charged separately)"""
        return self.uncached_input_cost + self.cached_cost + self.output_cost

    def get_cost_breakdown(self) -> dict[str, float]:
        """Get cost breakdown as dictionary"""
        return {
            "uncached_input_cost": self.uncached_input_cost,
            "cached_cost": self.cached_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenUsageEntry":
        try:
            return cls(
                name=data["name"],
                llm_model=data["llm_model"],
                tool_call=data.get("tool_call", []),
                input_tokens=int(data["input_tokens"]),
                reasoning_tokens=int(data["reasoning_tokens"]),
                cached_tokens=int(data["cached_tokens"]),
                output_tokens=int(data["output_tokens"]),
                total_tokens=int(data["total_tokens"]),
                role=data.get("role"),
            )
        except Exception as e:
            warnings.warn(f"TokenUsageEntry.from_dict failed; init zeros: {e}")
            return cls(
                name=None,
                llm_model=None,
                tool_call=[],
                input_tokens=0,
                reasoning_tokens=0,
                cached_tokens=0,
                output_tokens=0,
                total_tokens=0,
                role=None,
            )

    def to_dict(self) -> dict:
        """Return JSON dump string with all attributes and computed properties."""
        data = asdict(self)
        data.update(
            {
                "uncached_input_cost": self.uncached_input_cost,
                "cached_cost": self.cached_cost,
                "output_cost": self.output_cost,
                "total_cost": self.total_cost,
                "cost_file": cfg.cost_file,
            }
        )
        return data

    def to_json(self, indent: int | None = None) -> str:
        """Return JSON dump string with all attributes and computed properties."""
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        """String representation"""
        cost_str = f"${self.total_cost:.4f}" if self.llm_model else "N/A"
        return (
            f"Token(agent={self.name}, model={self.llm_model}, "
            f"total={self.total_tokens}, cost={cost_str})"
        )

    def __repr__(self) -> str:
        """Detailed representation"""
        return (
            f"Token(name={self.name!r} "
            f"llm_model={self.llm_model!r}, "
            f"tool_call={self.tool_call!r}, "
            f"input_tokens={self.input_tokens}, "
            f"cached_tokens={self.cached_tokens}, "
            f"output_tokens={self.output_tokens}, "
            f"reasoning_tokens={self.reasoning_tokens}, "
            f"total_tokens={self.total_tokens})"
        )


@dataclass
class TokenSum:
    items: list[TokenUsageEntry]

    def __add__(self, other) -> "TokenSum":
        if isinstance(other, TokenSum):
            return TokenSum(self.items + other.items)
        elif isinstance(other, TokenUsageEntry):
            self.items.append(other)
            return TokenSum(self.items)
        return NotImplemented

    def __iadd__(self, other) -> "TokenSum":
        if not isinstance(other, TokenSum):
            self.items.extend(other.items)
            return self
        elif isinstance(other, TokenUsageEntry):
            self.items.append(other)
            return self
        return NotImplemented

    @classmethod
    def from_dict(cls, data: dict) -> "TokenSum":
        try:
            items_data = data["items"]
            items = [TokenUsageEntry.from_dict(x) for x in items_data]
            return cls(items=items)
        except Exception as e:
            warnings.warn(f"TokenSum.from_dict failed; init empty: {e}")
            return cls(items=[])

    def to_dict(self) -> dict:
        data = {
            "items": [t.to_dict() for t in self.items],
            "total_input_tokens": sum(t.input_tokens for t in self.items),
            "total_cached_tokens": sum(t.cached_tokens for t in self.items),
            "total_output_tokens": sum(t.output_tokens for t in self.items),
            "total_reasoning_tokens": sum(t.reasoning_tokens for t in self.items),
            "total_tokens": sum(t.total_tokens for t in self.items),
            "uncached_input_cost": sum(t.uncached_input_cost for t in self.items),
            "cached_cost": sum(t.cached_cost for t in self.items),
            "output_cost": sum(t.output_cost for t in self.items),
            "total_cost": sum(t.total_cost for t in self.items),
            "cost_file": cfg.cost_file,
        }
        return data

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return f"TokenSum(items={len(self.items)})"
