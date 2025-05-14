from __future__ import annotations
import abc, asyncio, inspect, random, re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Sequence, Union


# Regular expression to match indexing expressions like foo[0] or bar["key"]
_INDEX_RE = re.compile(r'^(.*?)\[(.*?)\]$')


# ─────────────────────────────────────────────────────────────
# 1.  Runtime field helpers
# ─────────────────────────────────────────────────────────────
class OptimizableField:
    """Expose a concrete runtime attribute via get/set."""
    def __init__(self,
                 name: str,
                 getter: Callable[[], Any],
                 setter: Callable[[Any], None]):
        self.name, self._get, self._set = name, getter, setter
    def get(self) -> Any:            return self._get()
    def set(self, value: Any) -> None: self._set(value)


class PromptRegistry:
    """Central registry for all runtime-patchable fields."""
    def __init__(self) -> None:
        self.fields: Dict[str, OptimizableField] = {}
    def register_field(self, field: OptimizableField):
        self.fields[field.name] = field
    # convenience
    def get(self, name: str) -> Any:
        return self.fields[name].get()
    def set(self, name: str, value: Any):
        self.fields[name].set(value)
    def names(self) -> List[str]:
        return list(self.fields.keys())

    # -- 新增 API ----------------------------------------------
    def register_path(self, root: Any, path: str, *, name: str|None=None):
        """用类似 'encoder.layers[3].dropout_p' 的字符串一次性注册。"""
        key = name or path.split(".")[-1]          # 建议让用户自起更短 alias
        parent, leaf = self._walk(root, path)

        def getter():                       # 读
            return parent[leaf] if isinstance(parent, (list, dict)) else getattr(parent, leaf)

        def setter(v):                      # 写
            if isinstance(parent, (list, dict)):
                parent[leaf] = v
            else:
                setattr(parent, leaf, v)

        field = OptimizableField(key, getter, setter)
        self.register_field(field)
        return field

    def _walk(self, root, path: str, create_missing=False):
        """
        Navigates through nested object attributes and indices using a dotted path syntax.
        
        This method traverses a nested object structure following a path string like 'a.b.c' 
        or 'items[0].name' to access deeply nested attributes or indexed elements.
        
        Parameters
        ----------
        root : Any
            The root object to start traversal from
        path : str
            Dotted path notation for accessing nested attributes. Supports indexing with 
            square brackets for lists/dicts, e.g. 'users[0].profile.name'
        create_missing : bool, default=False
            If True, automatically creates missing intermediate dictionary objects
            
        Returns
        -------
        tuple
            A tuple containing (parent_object, leaf_key_or_index) where:
            - parent_object is the object containing the final attribute
            - leaf_key_or_index is the key or index for the final attribute
        
        Examples
        --------
        For obj = {'users': [{'name': 'Alice'}]}:
        _walk(obj, 'users[0].name') -> (obj['users'][0], 'name')
        """
        
        cur = root
        parts = path.split(".")
        for part in parts[:-1]:
            m = _INDEX_RE.match(part)
            if m:  # list / dict 取下标
                attr, idx = m.groups()
                cur = getattr(cur, attr) if attr else cur
                idx = int(idx) if idx.isdigit() else idx.strip("\"'")
                cur = cur[idx]
            else:
                cur = getattr(cur, part)
        leaf = parts[-1]
        m = _INDEX_RE.match(leaf)
        if m:
            attr, idx = m.groups()
            parent = getattr(cur, attr) if attr else cur
            leaf   = int(idx) if idx.isdigit() else idx.strip("\"'")
            return parent, leaf
        return cur, leaf


# ─────────────────────────────────────────────────────────────
# 2.  CodeBlock  (sync / async dual‑compatible)
# ─────────────────────────────────────────────────────────────
# result = await block.run(cfg)
class CodeBlock:
    """
    封装"一段代码"——可以是 async 也可以是普通函数。

    Parameters
    ----------
    name : str
        逻辑名（日志、调试友好）
    func : Callable[[dict], Any | Awaitable[Any]]
        · async def          → 框架会 await
        · 普通   def         → 框架同步执行
    """

    def __init__(self, name: str, func: Callable[[Dict[str, Any]], Any]):
        self.name = name
        self._func = func

    async def run(self, cfg: Dict[str, Any]) -> Any:
        """在异步环境里统一调用。"""
        if inspect.iscoroutinefunction(self._func):
            # 原本就是 async
            return await self._func(cfg)
        # 同步函数：直接执行，结果原样返回
        return self._func(cfg)

    # 若在纯同步脚本里想临时跑一下 block(cfg)
    def __call__(self, cfg: Dict[str, Any]) -> Any:
        # 如果已经在 event‑loop 中，创建任务；否则临时跑一次 loop
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(self.run(cfg))
        except RuntimeError:  # no running loop
            return asyncio.run(self.run(cfg))

    def __repr__(self):
        kind = "async" if inspect.iscoroutinefunction(self._func) else "sync"
        return f"<CodeBlock {self.name} ({kind})>"




# ─────────────────────────────────────────────────────────────
# 3.  BaseCodeBlockOptimizer
# ─────────────────────────────────────────────────────────────
class BaseCodeBlockOptimizer(abc.ABC):
    """
    Abstract optimiser that
      • performs parallel trials
      • writes sampled cfg back to runtime via PromptRegistry
      • validates that registered names appear in CodeBlock signature
    """

    def __init__(self,
                 registry: PromptRegistry,
                 metric: str,
                 maximize: bool = True,
                 max_trials: int = 30,
                 parallel: int = 4):
        self.registry   = registry
        self.metric     = metric
        self.maximize   = maximize
        self.max_trials = max_trials
        self.parallel   = parallel

    # -------- Subclasses need to implement-------------------------------
    @abc.abstractmethod
    def sample_cfg(self) -> Dict[str, Any]:
        """Return a cfg dict (may include subset of registry names)."""

    @abc.abstractmethod
    def update(self, cfg: Dict[str, Any], score: float):
        """Update internal optimiser state."""

    # -------- Internal: apply cfg to runtime --------------------
    async def _apply_cfg(self, cfg: Dict[str, Any]):
        for k, v in cfg.items():
            if k in self.registry.fields:
                self.registry.set(k, v)

    # -------- Confirm that CodeBlock accepts these fields -------------
    def _check_codeblock_compat(self, code_block: CodeBlock):
        sig = inspect.signature(code_block._func)
        params = sig.parameters.values()

        # if function has a **kwargs or a single cfg dict parameter, skip strict check
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        accepts_cfg_dict = "cfg" in sig.parameters

        if has_kwargs or accepts_cfg_dict:
            return  # compatible by design

        allowed_keys = set(sig.parameters)  # explicit arg names
        unknown = set(self.registry.names()) - allowed_keys
        if unknown:
            import warnings
            warnings.warn(f"PromptRegistry fields {unknown} are not present in "
                          f"{code_block.name}() signature; they will be ignored.")

    # -------- Main loop -----------------------------------
    async def run(self,
                  code_block: CodeBlock,
                  evaluator: Callable[[Dict[str, Any], Any], float]
                  ) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], float]]]:

        # once‑off compatibility check
        self._check_codeblock_compat(code_block)

        best_cfg, best_score = None, -float("inf") if self.maximize else float("inf")
        history: List[Tuple[Dict[str, Any], float]] = []
        sem = asyncio.Semaphore(self.parallel)

        async def trial():
            cfg = self.sample_cfg()               # Subclass sample
            await self._apply_cfg(cfg)            # Write back to runtime

            async with sem:                       # Concurrency limit
                result = await code_block.run(cfg)
                score  = evaluator(cfg, result)
                self.update(cfg, score)

            nonlocal best_cfg, best_score
            history.append((cfg, score))
            better = score > best_score if self.maximize else score < best_score
            if better:
                best_cfg, best_score = cfg, score

        await asyncio.gather(*(trial() for _ in range(self.max_trials)))
        return best_cfg, history



# ────────────────────────────────────────────────────────────
# Other  Helper: bind_cfg – write cfg into nested attributes
# ────────────────────────────────────────────────────────────
def bind_cfg(obj: Any, cfg: Dict[str, Any]) -> None:
    """Recursively write *cfg* values into (potentially nested) attributes
    of *obj*.  Key like "a.b.c" becomes obj.a.b.c = value.
    """
    for key, val in cfg.items():
        parts = key.split(".")
        cur = obj
        for part in parts[:-1]:
            cur = getattr(cur, part)
        setattr(cur, parts[-1], val)



# Demo
# ───────────────────── ───────────────────── ──────────────────── #
# ───────────────────── ───────────────────── ──────────────────── #
# ───────────────────── Demo: 业务对象 & 工作流 ──────────────────── #
@dataclass
class Sampler:
    temperature: float = 0.7
    top_p: float = 0.9

class Workflow:
    def __init__(self):
        self.system_prompt = "You are a helpful assistant."
        self.few_shot      = "Q: 1+1=?\nA: 2"
        self.sampler       = Sampler()

    async def run(self, cfg):
        # 真实业务里应该是调用 LLM
        prompt = f"{self.system_prompt}\n{self.few_shot}\nUser: {cfg.get('query','Hi')}"
        # 对于 demo，返回随机“质量分”
        return {"prompt": prompt, "score": random.uniform(0, 1)}

flow = Workflow()

# ─────────────────────── 注册需要调的字段 ───────────────────────── #
registry = PromptRegistry()
registry.register_path(flow, "system_prompt", name="sys_prompt")
registry.register_path(flow, "sampler.temperature")     # key = sampler_temperature
registry.register_path(flow, "sampler.top_p")           # key = sampler_top_p

# ─────────────────────── Optimizer 子类 (Random) ────────────────── #
class RandomSearchOptimizer(BaseCodeBlockOptimizer):
    def sample_cfg(self) -> Dict[str, Any]:
        return {
            # 随机温度 / top_p
            "sampler_temperature": random.uniform(0.3, 1.3),
            "sampler_top_p":       random.uniform(0.5, 1.0),
            # system_prompt 做两版随机切换
            "sys_prompt": random.choice([
                "You are a helpful assistant.",
                "You are a super-concise assistant."
            ]),
        }
    def update(self, cfg, score):  # 这里我们不用任何复杂算法
        pass

# ─────────────────────── CodeBlock & evaluator ──────────────────── #
code_block = CodeBlock("run_workflow", flow.run)

def evaluator(cfg, result) -> float:
    # 直接用 flow.run 返回的 score
    return result["score"]

# ─────────────────────────── 跑起来! ────────────────────────────── #
async def main():
    opt = RandomSearchOptimizer(registry, metric="score", max_trials=10)
    best_cfg, history = await opt.run(code_block, evaluator)

    print("\n=== Trial history ===")
    for i, (cfg, score) in enumerate(history, 1):
        print(f"{i:02d}: score={score:.3f}, cfg={{k:v for k,v in cfg.items()}}")

    print("\n=== Best ===")
    print(best_cfg)

if __name__ == "__main__":
    asyncio.run(main())
