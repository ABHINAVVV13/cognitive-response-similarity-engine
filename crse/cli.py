"""
Command-line interface for the Cognitive Response Similarity Engine.

Usage::

    crse compare video_a.mp4 video_b.mp4
    crse compare video_a.mp4 video_b.mp4 --output results.json --plot
    crse compare video_a.mp4 video_b.mp4 --device cuda --regions emotional_limbic visual_cortex
    crse compare --runpod URL_A URL_B      # run on RunPod serverless GPU
    crse regions                           # list available brain regions
    crse runpod health                     # check RunPod endpoint status
"""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ───────────────────────────────────────────────────────────────────────────
# CLI group
# ───────────────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="crse")
def main():
    """🧠 Cognitive Response Similarity Engine (CRSE)

    Compare neural response patterns between videos using Meta TRIBE v2.
    """


# ───────────────────────────────────────────────────────────────────────────
# compare command
# ───────────────────────────────────────────────────────────────────────────


@main.command()
@click.argument("video_a", type=click.Path())
@click.argument("video_b", type=click.Path())
@click.option("--model", default="facebook/tribev2", help="HuggingFace model ID or local path.")
@click.option("--cache", default="./cache", help="Feature cache directory.")
@click.option("--device", default="auto", help="PyTorch device (auto/cpu/cuda).")
@click.option("--output", "-o", default=None, help="Save results to a JSON file.")
@click.option(
    "--regions", "-r", multiple=True,
    help="Brain regions to analyse (repeat for multiple). Omit for all.",
)
@click.option("--plot", is_flag=True, help="Generate and save visualization plots.")
@click.option("--plot-dir", default="./crse_plots", help="Directory for plot images.")
@click.option("--runpod", is_flag=True, help="Run on a RunPod serverless GPU endpoint instead of locally.")
@click.option("--runpod-api-key", envvar="RUNPOD_API_KEY", default=None, help="RunPod API key (or set RUNPOD_API_KEY).")
@click.option("--runpod-endpoint-id", envvar="CRSE_ENDPOINT_ID", default=None, help="RunPod endpoint ID (or set CRSE_ENDPOINT_ID).")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def compare(video_a, video_b, model, cache, device, output, regions, plot, plot_dir, runpod, runpod_api_key, runpod_endpoint_id, verbose):
    """Compare two videos and report neural response similarity.

    With --runpod, VIDEO_A and VIDEO_B should be publicly accessible URLs.
    Without --runpod, they should be local file paths.
    """
    _setup_logging(verbose)

    # Header
    console.print()
    mode_label = "[bright_magenta]☁️  RunPod Serverless GPU[/]" if runpod else "[dim]Local Inference[/]"
    console.print(
        Panel(
            f"[bold bright_cyan]🧠  Cognitive Response Similarity Engine[/]\n"
            f"[dim]Powered by Meta TRIBE v2[/]  ·  {mode_label}",
            border_style="bright_blue",
            padding=(1, 4),
        )
    )
    console.print()

    region_list = list(regions) if regions else None

    console.print(f"  [dim]Video A:[/]  {video_a}")
    console.print(f"  [dim]Video B:[/]  {video_b}")
    console.print()

    if runpod:
        # ── RunPod remote execution ────────────────────────────────────
        from crse.runpod_client import CRSERunPodClient

        try:
            client = CRSERunPodClient(
                api_key=runpod_api_key,
                endpoint_id=runpod_endpoint_id,
            )
        except ValueError as e:
            console.print(f"[bright_red]Error:[/] {e}")
            raise SystemExit(1)

        with console.status("[bold magenta]Sending job to RunPod GPU worker...", spinner="dots12"):
            raw_result = client.compare(
                video_a_url=video_a,
                video_b_url=video_b,
                regions=region_list,
            )

        if "error" in raw_result:
            console.print(f"[bright_red]RunPod Error:[/] {raw_result['error']}")
            raise SystemExit(1)

        # Reconstruct a ComparisonResult-like object for display
        from crse.engine import ComparisonResult, RegionScore
        result = ComparisonResult(
            video_a=raw_result.get("video_a", video_a),
            video_b=raw_result.get("video_b", video_b),
            whole_brain=raw_result.get("whole_brain", {}),
            regions=[
                RegionScore(
                    name=r["name"],
                    description=r.get("description", ""),
                    n_vertices=r.get("n_vertices", 0),
                    metrics=r.get("metrics", {}),
                )
                for r in raw_result.get("regions", [])
            ],
            prediction_shape_a=tuple(raw_result.get("prediction_shape_a", [])),
            prediction_shape_b=tuple(raw_result.get("prediction_shape_b", [])),
            elapsed_seconds=raw_result.get("elapsed_seconds", 0),
            metadata=raw_result.get("metadata", {}),
        )
    else:
        # ── Local execution ────────────────────────────────────────────
        from crse.engine import CRSEngine

        with console.status("[bold cyan]Loading TRIBE v2 model...", spinner="dots"):
            engine = CRSEngine(
                model_id=model,
                cache_folder=cache,
                device=device,
                regions=region_list,
            )

        with console.status("[bold cyan]Running brain predictions & comparison...", spinner="dots12"):
            result = engine.compare(video_a, video_b)

    # ── Display results ────────────────────────────────────────────────

    # Whole brain table
    wb_table = Table(
        title="🌐 Whole-Brain Similarity",
        title_style="bold bright_cyan",
        border_style="bright_blue",
        show_header=True,
        header_style="bold bright_white",
    )
    wb_table.add_column("Metric", style="bright_white", min_width=28)
    wb_table.add_column("Score", justify="right", style="bold")
    wb_table.add_column("Indicator", min_width=22)

    for metric, score in result.whole_brain.items():
        color = "bright_green" if score >= 0.3 else ("yellow" if score >= 0 else "bright_red")
        bar = _rich_bar(score)
        wb_table.add_row(
            metric.replace("_", " ").title(),
            f"[{color}]{score:+.4f}[/]",
            bar,
        )

    console.print(wb_table)
    console.print()

    # Per-region table
    if result.regions:
        rg_table = Table(
            title="🧩 Per-Region Breakdown",
            title_style="bold bright_cyan",
            border_style="bright_blue",
            show_header=True,
            header_style="bold bright_white",
        )
        rg_table.add_column("Region", style="bright_white", min_width=22)
        rg_table.add_column("Vertices", justify="right", style="dim")
        rg_table.add_column("Cosine", justify="right")
        rg_table.add_column("Pearson", justify="right")
        rg_table.add_column("Temporal", justify="right")
        rg_table.add_column("RSA", justify="right")
        rg_table.add_column("Mean", justify="right", style="bold")

        for r in result.regions:
            def _fmt(val):
                color = "bright_green" if val >= 0.3 else ("yellow" if val >= 0 else "bright_red")
                return f"[{color}]{val:+.3f}[/]"

            rg_table.add_row(
                r.name.replace("_", " ").title(),
                f"{r.n_vertices:,}",
                _fmt(r.metrics.get("cosine_similarity", 0)),
                _fmt(r.metrics.get("pearson_correlation", 0)),
                _fmt(r.metrics.get("temporal_correlation", 0)),
                _fmt(r.metrics.get("representational_similarity", 0)),
                _fmt(r.mean_score),
            )

        console.print(rg_table)
        console.print()

    # Timing
    console.print(
        f"  [dim]Completed in[/] [bold bright_cyan]{result.elapsed_seconds:.1f}s[/]"
    )
    console.print()

    # ── Save outputs ───────────────────────────────────────────────────

    if output:
        result.save(output)
        console.print(f"  [dim]Results saved to[/] [bright_green]{output}[/]")

    if plot:
        from pathlib import Path

        from crse.visualization import (
            plot_metric_bars,
            plot_similarity_radar,
            plot_whole_brain_summary,
        )

        plot_path = Path(plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)

        plot_whole_brain_summary(result, save_path=plot_path / "whole_brain.png")
        plot_similarity_radar(result, save_path=plot_path / "radar.png")
        plot_metric_bars(result, save_path=plot_path / "regions_bars.png")
        console.print(f"  [dim]Plots saved to[/] [bright_green]{plot_dir}/[/]")

    console.print()


# ───────────────────────────────────────────────────────────────────────────
# runpod command group
# ───────────────────────────────────────────────────────────────────────────


@main.group()
def runpod():
    """☁️  Manage RunPod serverless endpoint."""


@runpod.command()
@click.option("--api-key", envvar="RUNPOD_API_KEY", default=None)
@click.option("--endpoint-id", envvar="CRSE_ENDPOINT_ID", default=None)
def health(api_key, endpoint_id):
    """Check the health of your CRSE RunPod endpoint."""
    from crse.runpod_client import CRSERunPodClient

    try:
        client = CRSERunPodClient(api_key=api_key, endpoint_id=endpoint_id)
    except ValueError as e:
        console.print(f"[bright_red]Error:[/] {e}")
        raise SystemExit(1)

    with console.status("[bold magenta]Checking endpoint health...", spinner="dots"):
        data = client.health()

    table = Table(
        title="☁️  RunPod Endpoint Health",
        title_style="bold bright_magenta",
        border_style="bright_magenta",
    )
    table.add_column("Field", style="bright_white")
    table.add_column("Value", style="bright_cyan")

    for key, val in data.items():
        table.add_row(str(key), str(val))

    console.print()
    console.print(table)
    console.print()


@runpod.command()
@click.argument("job_id")
@click.option("--api-key", envvar="RUNPOD_API_KEY", default=None)
@click.option("--endpoint-id", envvar="CRSE_ENDPOINT_ID", default=None)
def status(job_id, api_key, endpoint_id):
    """Check the status of a RunPod async job."""
    from crse.runpod_client import CRSERunPodClient

    try:
        client = CRSERunPodClient(api_key=api_key, endpoint_id=endpoint_id)
    except ValueError as e:
        console.print(f"[bright_red]Error:[/] {e}")
        raise SystemExit(1)

    data = client.get_status(job_id)
    console.print(f"  [dim]Job:[/]     {job_id}")
    console.print(f"  [dim]Status:[/]  [bold]{data.get('status', 'UNKNOWN')}[/]")
    if data.get("output"):
        import json
        console.print(json.dumps(data["output"], indent=2))
    console.print()


# ───────────────────────────────────────────────────────────────────────────
# regions command
# ───────────────────────────────────────────────────────────────────────────


@main.command()
def regions():
    """List all available brain regions and their descriptions."""
    from crse.brain_regions import BrainRegionManager, REGION_LABEL_PATTERNS

    mgr = BrainRegionManager()

    table = Table(
        title="🧩 Available Brain Regions",
        title_style="bold bright_cyan",
        border_style="bright_blue",
        show_header=True,
        header_style="bold bright_white",
    )
    table.add_column("Region Key", style="bright_white", min_width=24)
    table.add_column("Description", style="dim", max_width=60)
    table.add_column("Atlas Labels", style="cyan", max_width=40)

    for name, patterns in REGION_LABEL_PATTERNS.items():
        desc = mgr.get_region_description(name)
        table.add_row(
            name,
            desc[:80] + "..." if len(desc) > 80 else desc,
            ", ".join(patterns[:4]) + ("..." if len(patterns) > 4 else ""),
        )

    console.print()
    console.print(table)
    console.print()


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def _rich_bar(score: float, width: int = 20) -> str:
    """Render a coloured bar for Rich."""
    import numpy as np

    if not np.isfinite(score):
        return "[dim]n/a[/]"
    norm = (score + 1.0) / 2.0
    norm = max(0.0, min(1.0, norm))
    filled = int(round(norm * width))
    if score >= 0.3:
        color = "bright_green"
    elif score >= 0:
        color = "yellow"
    else:
        color = "bright_red"
    return f"[{color}]{'█' * filled}[/][dim]{'░' * (width - filled)}[/]"


if __name__ == "__main__":
    main()
