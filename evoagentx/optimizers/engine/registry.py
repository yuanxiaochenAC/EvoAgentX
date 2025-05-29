import re
from typing import Any, Callable, Dict, List
from ...prompts.template import PromptTemplate

_INDEX_RE = re.compile(r'^(.*?)\[(.*?)\]$')

class OptimizableField:
    """
    Represents a parameter that can be optimized.

    This class wraps a runtime attribute with dynamic getter and setter
    functions, allowing it to be exposed to the optimizer as a tunable field.
    """

    def __init__(self,
                 name: str,
                 getter: Callable[[], Any],
                 setter: Callable[[Any], None]):
        """
        Parameters:
        - name (str): The alias used to register the field in the registry.
        - getter (Callable): A function that returns the current value.
        - setter (Callable): A function that updates the value.
        """
        self.name = name        # Registry key
        self._get = getter      # Function to retrieve current value
        self._set = setter      # Function to update value

    def get(self) -> Any:
        """Return the current value of the field."""
        return self._get()

    def set(self, value: Any) -> None:
        """Update the field to the given value."""
        self._set(value)


class ParamRegistry:
    """
    Central registry for all parameters that can be exposed to optimization.

    Allows dynamic binding and tracking of runtime attributes via dot-paths,
    dictionary keys, or list indices. Provides getter/setter access to all
    registered parameters for optimizers.
    """
    def __init__(self) -> None:
        """Initialize an empty registry of optimizable fields."""
        self.fields: Dict[str, OptimizableField] = {}

    def register_field(self, field: OptimizableField):
        """Manually register an OptimizableField with its alias name."""
        self.fields[field.name] = field

    def get(self, name: str) -> Any:
        """Retrieve the current value of a registered field by name."""
        return self.fields[name].get()

    def set(self, name: str, value: Any):
        """Set the value of a registered field by name."""
        self.fields[name].set(value)

    def names(self) -> List[str]:
        """Return a list of all registered field names (aliases)."""
        return list(self.fields.keys())

    def track(self, root_or_obj: Any, path_or_attr: str, *, name: str | None = None):
        """
        Register a parameter to be optimized. Supports both nested paths and direct attributes.

        Parameters:
        - root_or_obj (Any): the base object or container
        - path_or_attr (str): a path like 'prompt.template' or a direct attribute like 'template'
        - name (str | None): optional alias for this parameter

        Supported formats:
        - registry.track(program, "prompt.template")              # nested attribute
        - registry.track(program, "metadata['style']")           # dictionary key
        - registry.track(program, "components[2].prefix")        # list index
        - registry.track(program.prompt, "template")             # direct object + attribute
        - registry.track([
            (program, "prompt.template"),
            (program, "metadata['style']", "style"),
            (program.prompt, "prefix", "prompt_prefix")
          ])                                                    # batch registration
        - registry.track(program, "prompt.template").track(program, "prompt.prefix")  # chained calls

        Returns:
        - self (PromptRegistry): for chaining
        """
        if isinstance(root_or_obj, list | tuple):
            # batch mode: track([(obj, path), (obj, path, alias), ...])
            # Example:
            # registry.track([
            #     (program, "prompt.template"),
            #     (program, "metadata['style']", "style"),
            #     (program.prompt, "prefix", "prompt_prefix")
            # ])
            for item in root_or_obj:
                if len(item) == 2:
                    self.track(item[0], item[1])
                elif len(item) == 3:
                    self.track(item[0], item[1], name=item[2])
            return self

        if "." in path_or_attr or "[" in path_or_attr:
            return self._track_path(root_or_obj, path_or_attr, name)
        else:
            key = name or path_or_attr

            def getter():
                return getattr(root_or_obj, path_or_attr)

            def setter(v):
                setattr(root_or_obj, path_or_attr, v)

            field = OptimizableField(key, getter, setter)
            if key in self.fields:
                import warnings
                warnings.warn(f"Field '{key}' is already registered. Overwriting.")
            self.register_field(field)
            return self

    def _track_path(self, root: Any, path: str, name: str | None = None):
        """
        Internal helper that registers a nested field (via dot path, index, or key)
        as an OptimizableField by dynamically creating getter and setter functions.

        Parameters:
        - root (Any): the root object to start walking from
        - path (str): dot-separated path supporting list/dict access
        - name (Optional[str]): alias for the parameter (defaults to last path segment)

        Returns:
        - self
        """
        key = name or path.split(".")[-1]
        parent, leaf = self._walk(root, path)

        def getter():
            return parent[leaf] if isinstance(parent, (list, dict)) else getattr(parent, leaf)

        def setter(v):
            if isinstance(parent, (list, dict)):
                parent[leaf] = v
            else:
                setattr(parent, leaf, v)

        field = OptimizableField(key, getter, setter)
        self.register_field(field)
        return self

    def _walk(self, root, path: str):
        """
        Internal helper to resolve a dot-separated path string into its parent container
        and the leaf attribute/key/index for assignment or retrieval.

        Supports:
        - Nested attributes: e.g. "a.b.c"
        - Dict key access: e.g. "config['key']"
        - List index access: e.g. "layers[0]"

        Parameters:
        - root (Any): root object to walk from
        - path (str): path string to resolve
        - create_missing (bool): unused placeholder for future extensions

        Returns:
        - (parent, leaf): where parent[leaf] or getattr(parent, leaf) is the target
        """
        cur = root
        parts = path.split(".")
        for part in parts[:-1]:
            m = _INDEX_RE.match(part)
            if m:
                attr, idx = m.groups()
                cur = getattr(cur, attr) if attr else cur
                idx = idx.strip()
                if (idx.startswith("'") and idx.endswith("'")) or (idx.startswith('"') and idx.endswith('"')):
                    idx = idx[1:-1]
                elif idx.isdigit():
                    idx = int(idx)
                cur = cur[idx]
            else:
                cur = getattr(cur, part)

        leaf = parts[-1]
        m = _INDEX_RE.match(leaf)
        if m:
            attr, idx = m.groups()
            parent = getattr(cur, attr) if attr else cur
            idx = idx.strip()
            if (idx.startswith("'") and idx.endswith("'")) or (idx.startswith('"') and idx.endswith('"')):
                idx = idx[1:-1]
            elif idx.isdigit():
                idx = int(idx)
            return parent, idx
        return cur, leaf
    

class PromptTemplateRegister(ParamRegistry):
    """
    Enhanced parameter registry that supports directly registering PromptTemplate instances
    or prompt strings as a single optimizable object.
    """

    def track(self, root_or_obj: Any, path_or_attr: str, *, name: str | None = None):
        if isinstance(root_or_obj, (list, tuple)):
            for item in root_or_obj:
                if len(item) == 2:
                    self.track(item[0], item[1])
                elif len(item) == 3:
                    self.track(item[0], item[1], name=item[2])
            return self

        key = name or path_or_attr

        try:
            value = getattr(root_or_obj, path_or_attr)
        except AttributeError:
            return super().track(root_or_obj, path_or_attr, name=name)

        if isinstance(value, (str, PromptTemplate)):
            # Register the entire prompt or PromptTemplate object
            field = OptimizableField(
                key,
                getter=lambda: getattr(root_or_obj, path_or_attr),
                setter=lambda v: setattr(root_or_obj, path_or_attr, v)
            )
            self.register_field(field)
            return self

        # Fall back to original path-based tracking if not str/template
        return super().track(root_or_obj, path_or_attr, name=name)