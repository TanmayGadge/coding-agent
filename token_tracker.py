# token_tracker.py
from typing import Dict, Any, Optional, Generator, AsyncGenerator
from contextlib import contextmanager, asynccontextmanager

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class TokenTracker:
    """
    Real-time token and cost tracker for LangGraph workflows.
    Tracks total usage as well as per-sub-agent breakdown using stream_events metadata.
    """

    def __init__(
        self,
        input_cost_per_1m: float = 2.50,   # Default GPT-4o input rate ($/1M tokens)
        output_cost_per_1m: float = 10.00, # Default GPT-4o output rate ($/1M tokens)
        currency_symbol: str = "$"
    ):
        self.input_cost_per_token = input_cost_per_1m / 1_000_000
        self.output_cost_per_token = output_cost_per_1m / 1_000_000
        self.currency = currency_symbol

        # Global counters
        self.total_input = 0
        self.total_output = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # Sub-agent breakdown: { "node_name": {"input": 0, "output": 0, "total": 0, "cost": 0.0} }
        self.agent_breakdown: Dict[str, Dict[str, Any]] = {}
        self.console = Console()
        self._live_instance: Optional[Live] = None

    def process_event(self, event: Dict[str, Any]) -> None:
        """Parses a single LangGraph stream event and updates counters if an LLM completion ended."""
        if event.get("event") == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            if not output:
                return

            # Extract standardized usage metadata
            usage = getattr(output, "usage_metadata", None)
            
            # Fallback for LLMResult objects
            if not usage and hasattr(output, "generations"):
                try:
                    usage = output.generations[0][0].message.usage_metadata
                except (AttributeError, IndexError):
                    pass

            if usage:
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                tot = usage.get("total_tokens", inp + out)
                cost = (inp * self.input_cost_per_token) + (out * self.output_cost_per_token)

                # Automatically extract sub-agent/node name from event metadata
                metadata = event.get("metadata", {})
                node_name = metadata.get("langgraph_node", "orchestrator")

                # Update global totals
                self.total_input += inp
                self.total_output += out
                self.total_tokens += tot
                self.total_cost += cost

                # Update per-agent totals
                if node_name not in self.agent_breakdown:
                    self.agent_breakdown[node_name] = {"input": 0, "output": 0, "total": 0, "cost": 0.0}

                self.agent_breakdown[node_name]["input"] += inp
                self.agent_breakdown[node_name]["output"] += out
                self.agent_breakdown[node_name]["total"] += tot
                self.agent_breakdown[node_name]["cost"] += cost

                # Trigger CLI UI update if running inside live context
                if self._live_instance:
                    self._live_instance.update(self.render_panel())

    def render_panel(self) -> Panel:
        """Generates a Rich Panel displaying overall usage and per-sub-agent breakdown table."""
        table = Table(box=None, expand=True, show_header=True, header_style="bold magenta")
        table.add_column("Sub-Agent / Node", style="cyan", ratio=2)
        table.add_column("Input", justify="right", style="dim")
        table.add_column("Output", justify="right", style="dim")
        table.add_column("Total Tokens", justify="right", style="bold white")
        table.add_column("Cost", justify="right", style="bold green")

        for agent, stats in self.agent_breakdown.items():
            table.add_row(
                agent,
                f"{stats['input']:,}",
                f"{stats['output']:,}",
                f"{stats['total']:,}",
                f"{self.currency}{stats['cost']:.5f}"
            )

        # Header summary line
        summary_text = (
            f"[bold cyan]Total Tokens:[/] [bold white]{self.total_tokens:,}[/] "
            f"([dim]In: {self.total_input:,} | Out: {self.total_output:,}[/])  │  "
            f"[bold green]Total Cost:[/] [bold green]{self.currency}{self.total_cost:.5f}[/]"
        )

        # Combine summary and breakdown table
        layout = Table.grid(expand=True)
        layout.add_row(summary_text)
        if self.agent_breakdown:
            layout.add_row("─" * 55)
            layout.add_row(table)

        return Panel(
            layout,
            title="[bold yellow]⚡ Real-Time Token & Cost Tracker[/bold yellow]",
            border_style="blue",
            expand=True
        )

    @contextmanager
    def live(self) -> Generator["TokenTracker", None, None]:
        """Context manager to attach a fixed CLI live status bar during stream execution."""
        with Live(self.render_panel(), console=self.console, refresh_per_second=10) as live_bar:
            self._live_instance = live_bar
            try:
                yield self
            finally:
                self._live_instance = None

    def stream(
        self,
        graph: Any,
        input_data: Any,
        config: Optional[Dict[str, Any]] = None,
        version: str = "v3"
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Convenience wrapper around graph.stream_events().
        Tracks usage, renders the CLI status bar, and yields stream events.
        """
        with self.live():
            for event in graph.stream_events(input_data, config=config, version=version):
                self.process_event(event)
                yield event