"""Harbor external agent for vanilla LFM2.5-1.2B — no augmentation.

Register as: benchmarks.harbor.agents:VanillaAgent
Usage: harbor run -d <dataset> --agent benchmarks.harbor.agents:VanillaAgent
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ImportError:
    BaseAgent = object  # type: ignore[assignment,misc]
    BaseEnvironment = object  # type: ignore[assignment,misc]
    AgentContext = object  # type: ignore[assignment,misc]


class VanillaAgent(BaseAgent):
    """Vanilla LFM2.5-1.2B agent — raw model without augmentation."""

    @staticmethod
    def name() -> str:
        return "vanilla-lfm"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(  # type: ignore[attr-defined]
            "pip install llama-cpp-python 2>&1 | tail -1"
        )
        agent_script = Path(__file__).resolve().parent / "_agent_runner.py"
        await environment.exec("mkdir -p /agent")  # type: ignore[attr-defined]
        script_content = agent_script.read_text()
        await environment.exec(  # type: ignore[attr-defined]
            f"cat > /agent/runner.py << 'HEREDOC_END'\n{script_content}\nHEREDOC_END"
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model_path = os.environ.get(
            "OCZY_MODEL_PATH", "/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
        )
        cmd = (
            "python /agent/runner.py "
            f"--model '{model_path}' "
            "--mode vanilla "
            f"--instruction '{instruction}'"
        )
        result = await environment.exec(cmd)  # type: ignore[attr-defined]
        if result.exit_code == 0:
            context.stdout = result.stdout  # type: ignore[attr-defined]
        else:
            context.stderr = result.stderr  # type: ignore[attr-defined]
