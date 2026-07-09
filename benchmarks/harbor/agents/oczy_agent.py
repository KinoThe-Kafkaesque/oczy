"""Harbor external agent for Oczy — knowledge-augmented LFM2.5-1.2B.

Register as: benchmarks.harbor.agents:OczyAgent
Usage: harbor run -d <dataset> --agent benchmarks.harbor.agents:OczyAgent
"""

from __future__ import annotations

import json
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


class OczyAgent(BaseAgent):
    """Oczy knowledge-augmented agent.

    Before executing, recalls relevant facts from the Oczy codebase
    knowledge store and prepends them to the instruction.  The model
    then generates bash commands to solve the task.
    """

    @staticmethod
    def name() -> str:
        return "oczy-lfm"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install the model client inside the sandbox."""
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
        """Execute the Oczy-augmented agent loop."""
        model_path = os.environ.get(
            "OCZY_MODEL_PATH", "/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
        )
        facts_path = os.environ.get("OCZY_FACTS_PATH", "")
        cmd = (
            "python /agent/runner.py "
            f"--model '{model_path}' "
            "--mode oczy "
            f"--facts '{facts_path}' "
            f"--instruction '{instruction}'"
        )
        result = await environment.exec(cmd)  # type: ignore[attr-defined]
        if result.exit_code == 0:
            context.stdout = result.stdout  # type: ignore[attr-defined]
        else:
            context.stderr = result.stderr  # type: ignore[attr-defined]
