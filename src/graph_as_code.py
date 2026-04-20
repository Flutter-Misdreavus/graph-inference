"""Graph-as-Code interactive node classification pipeline."""
import re
from typing import Optional

import pandas as pd

from .prompt_template import build_graph_as_code_prompt
from .code_executor import SecureExecutor
from .llm_client import LLMClient


MAX_STEPS = 15
ANSWER_PATTERN = re.compile(r'Answer[\s:]+(\d+)', re.IGNORECASE)


class GraphAsCodeAgent:
    """ReAct-style agent that lets the LLM iteratively query a graph DataFrame."""

    def __init__(self, df: pd.DataFrame, llm: LLMClient, class_labels: dict[int, str]):
        self.df = df
        self.llm = llm
        self.class_labels = class_labels
        self.executor = SecureExecutor(df)

    def classify(self, node_id: int, verbose: bool = False) -> Optional[int]:
        """Run the Graph-as-Code loop for a single node.

        Returns predicted label id or None if no valid answer was produced.
        """
        system_prompt = build_graph_as_code_prompt(node_id, self.class_labels)
        history: list[dict] = []
        step = 0

        while step < MAX_STEPS:
            step += 1
            # The LLM sees the full history (previous reasoning + execution results)
            user_msg = self._build_user_message(step, history, node_id)
            response = self.llm.chat(system_prompt, user_msg, history=history)

            if verbose:
                print(f"\n[Step {step}]\n{response}\n")

            # Check for termination
            answer_match = ANSWER_PATTERN.search(response)
            if answer_match:
                pred = int(answer_match.group(1))
                if pred in self.class_labels:
                    return pred
                # Invalid class id – continue
                history.append({'role': 'assistant', 'content': response})
                history.append({'role': 'user', 'content': f"Invalid class id {pred}. Valid ids are 0-{max(self.class_labels)}. Please try again."})
                continue

            # Extract code line (last non-empty line after reasoning)
            code = self._extract_code(response)
            if code is None:
                history.append({'role': 'assistant', 'content': response})
                history.append({'role': 'user', 'content': "No valid pandas command found. Please provide reasoning followed by a single pandas expression on the last line, or end with 'Answer: [class_id]'."})
                continue

            # Execute code
            success, result = self.executor.execute(code)
            if verbose:
                print(f"  -> {'OK' if success else 'ERR'}: {result[:200]}...")

            history.append({'role': 'assistant', 'content': response})
            history.append({'role': 'user', 'content': f"Result:\n{result}"})

        # Max steps reached without answer
        return None

    @staticmethod
    def _build_user_message(step: int, history: list, node_id: int) -> str:
        """Build the user message for the current step.

        For step 1 the LLM starts fresh; for later steps the history
        carries the context.
        """
        if step == 1:
            return f"Please classify node {node_id}."
        return "Continue."

    @staticmethod
    def _extract_code(response: str) -> Optional[str]:
        """Try to extract a single pandas expression from the response.

        Heuristic: scan from bottom up; skip pure numeric Answer lines;
        prefer lines that start with ``df.`` or ``pd.``.
        """
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        for line in reversed(lines):
            lower = line.lower()
            if lower.startswith('answer:'):
                rest = line[7:].strip()
                if rest.isdigit():
                    continue
                # "Answer: df.loc[x]" → treat rest as code
                return rest.strip('`')
            if line.startswith('df.') or line.startswith('pd.'):
                return line.strip('`')
            if 'df.' in line or 'pd.' in line:
                if any(c in line for c in ['(', '[', '=']):
                    return line.strip('`')
        return None
