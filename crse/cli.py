"""
Command-line interface for the Cognitive Response Similarity Engine.

Usage::

    crse compare video_a.mp4 video_b.mp4
    crse compare --runpod URL_A URL_B --render-brain
    crse regions
    crse runpod health
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(package_name="crse")
def main():
    """Cognitive Response Similarity Engine — TRIBE v2 video comparison."""


@main.command()
@click.argument("video_a", type=click.Path())
@click.argument("video_b", type=click.Path())
@click.option("--model", default="facebook/tribev2", help="HuggingFace model ID or local path.")
@click.option("--cache", default="./cache", help="Feature cache directory.")
@click.option("--device", default="auto", help="PyTorch device (auto/cpu/cuda).")
@click.option("--output", "-o", default=None, help="Save lean scores JSON (no base64 blobs).")
@click.option(
    "--regions", "-r", multiple=True,
    help="Brain regions (repeat for multiple). Omit for all.",
)
@click.option("--runpod", is_flag=True, help="Run on RunPod (VIDEO_* must be public URLs).")
@click.option("--runpod-api-key", envvar="RUNPOD_API_KEY", default=None)
@click.option("--runpod-endpoint-id", envvar="CRSE_ENDPOINT_ID", default=None)
@click.option(
    "--runpod-timeout",
    type=int,
    default=None,
    help="Max seconds to wait (default 1800 or CRSE_RUNPOD_TIMEOUT).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default="crse_out",
    help="Output root: brain PNGs in brain/, viewer in viewer/ when --render-brain.",
)
@click.option(
    "--render-brain",
    is_flag=True,
    help="Also export full (T×V) predictions + WebGL viewer (large RunPod response).",
)
@click.option("--verbose", "-v", is_flag=True)
def compare(
    video_a,
    video_b,
    model,
    cache,
    device,
    output,
    regions,
    runpod,
    runpod_api_key,
    runpod_endpoint_id,
    runpod_timeout,
    out_dir,
    render_brain,
    verbose,
):
    """Compare two videos (local paths or URLs). Always writes cortical PNGs under OUT_DIR/brain/."""
    _setup_logging(verbose)
    out_path = Path(out_dir)
    region_list = list(regions) if regions else None

    console.print()
    mode_label = "[bright_magenta]RunPod[/]" if runpod else "[dim]Local[/]"
    console.print(
        Panel(
            f"[bold bright_cyan]Cognitive Response Similarity Engine[/]\n"
            f"[dim]Meta TRIBE v2[/]  ·  {mode_label}",
            border_style="bright_blue",
            padding=(1, 4),
        )
    )
    console.print()
    console.print(f"  [dim]Video A:[/]  {video_a}")
    console.print(f"  [dim]Video B:[/]  {video_b}")
    console.print(f"  [dim]Out:[/]      {out_path.resolve()}")
    console.print()

    if runpod:
        from crse.brain_viewer_export import export_interactive_viewer, load_predictions_from_visualization_dict
        from crse.runpod_client import CRSERunPodClient, save_surface_pngs_from_result

        try:
            client = CRSERunPodClient(
                api_key=runpod_api_key,
                endpoint_id=runpod_endpoint_id,
                timeout=runpod_timeout,
            )
        except ValueError as e:
            console.print(f"[bright_red]Error:[/] {e}")
            raise SystemExit(1)

        with console.status("[bold magenta]RunPod job...", spinner="dots12"):
            raw = client.compare(
                video_a_url=video_a,
                video_b_url=video_b,
                regions=region_list,
                render_brain=render_brain,
            )

        if "error" in raw:
            console.print(f"[bright_red]RunPod Error:[/] {raw['error']}")
            raise SystemExit(1)

        brain_dir = out_path / "brain"
        b64_map = raw.get("surface_pngs_base64")
        err_msg = b64_map.get("_error") if isinstance(b64_map, dict) else None
        written = save_surface_pngs_from_result(raw, brain_dir)
        if err_msg:
            console.print(f"[yellow]Warning:[/] [dim]Brain PNGs: {err_msg}[/]")
        elif not written:
            console.print(
                "[yellow]Warning:[/] [dim]No PNGs in response — rebuild worker with tribe-plot (runpod/Dockerfile).[/]"
            )

        if render_brain:
            viz = raw.get("visualization")
            pair = load_predictions_from_visualization_dict(viz) if isinstance(viz, dict) else None
            if pair is None:
                console.print(
                    "[yellow]Warning:[/] [dim]No prediction arrays in response (need render_brain on worker).[/]"
                )
            else:
                try:
                    export_interactive_viewer(out_path / "viewer", pair[0], pair[1])
                except Exception as e:
                    console.print(f"[yellow]Warning:[/] [dim]Viewer export failed: {e}[/]")

        from crse.engine import ComparisonResult, RegionScore

        result = ComparisonResult(
            video_a=raw.get("video_a", video_a),
            video_b=raw.get("video_b", video_b),
            whole_brain=raw.get("whole_brain", {}),
            regions=[
                RegionScore(
                    name=r["name"],
                    description=r.get("description", ""),
                    n_vertices=r.get("n_vertices", 0),
                    metrics=r.get("metrics", {}),
                )
                for r in raw.get("regions", [])
            ],
            prediction_shape_a=tuple(raw.get("prediction_shape_a", [])),
            prediction_shape_b=tuple(raw.get("prediction_shape_b", [])),
            elapsed_seconds=raw.get("elapsed_seconds", 0),
            metadata=raw.get("metadata", {}),
        )
    else:
        from crse.engine import CRSEngine

        with console.status("[bold cyan]Loading TRIBE...", spinner="dots"):
            engine = CRSEngine(
                model_id=model,
                cache_folder=cache,
                device=device,
                regions=region_list,
            )

        with console.status("[bold cyan]Predicting & comparing...", spinner="dots12"):
            result = engine.compare(
                video_a,
                video_b,
                out_dir=out_path,
                render_brain=render_brain,
            )

    wb_table = Table(
        title="Whole-Brain Similarity",
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

    if result.regions:
        rg_table = Table(
            title="Per-Region",
            title_style="bold bright_cyan",
            border_style="bright_blue",
            show_header=True,
            header_style="bold bright_white",
        )
        rg_table.add_column("Region", style="bright_white", min_width=22)
        rg_table.add_column("Vertices", justify="right", style="dim")
        rg_table.add_column("Cosine", justify="right")
        rg_table.add_column("Pearson", justify="right")
        rg_table.add_column("Mean", justify="right", style="bold")

        for r in result.regions:
            def _fmt(val):
                c = "bright_green" if val >= 0.3 else ("yellow" if val >= 0 else "bright_red")
                return f"[{c}]{val:+.3f}[/]"

            rg_table.add_row(
                r.name.replace("_", " ").title(),
                f"{r.n_vertices:,}",
                _fmt(r.metrics.get("cosine_similarity", 0)),
                _fmt(r.metrics.get("pearson_correlation", 0)),
                _fmt(r.mean_score),
            )

        console.print(rg_table)
        console.print()

    console.print(f"  [dim]Time[/] [bold bright_cyan]{result.elapsed_seconds:.1f}s[/]")
    console.print(f"  [dim]Brain PNGs[/] [bright_green]{out_path / 'brain'}[/]")
    if render_brain:
        console.print(
            f"  [dim]3D viewer[/] [bright_green]{out_path / 'viewer'}[/] "
            "[dim]→ cd viewer && python -m http.server 8765[/]"
        )
    console.print()

    if output:
        result.save(output)
        console.print(f"  [dim]JSON[/] [bright_green]{output}[/]")

    console.print()


@main.group()
def runpod():
    """RunPod endpoint helpers."""


@runpod.command()
@click.option("--api-key", envvar="RUNPOD_API_KEY", default=None)
@click.option("--endpoint-id", envvar="CRSE_ENDPOINT_ID", default=None)
def health(api_key, endpoint_id):
    """Check RunPod endpoint health."""
    from crse.runpod_client import CRSERunPodClient

    try:
        client = CRSERunPodClient(api_key=api_key, endpoint_id=endpoint_id)
    except ValueError as e:
        console.print(f"[bright_red]Error:[/] {e}")
        raise SystemExit(1)

    with console.status("[bold magenta]Checking...", spinner="dots"):
        data = client.health()

    table = Table(title="RunPod Health", border_style="bright_magenta")
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
    """Async job status."""
    import json

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
        console.print(json.dumps(data["output"], indent=2))
    console.print()


@main.command()
def regions():
    """List brain regions."""
    from crse.brain_regions import BrainRegionManager, REGION_LABEL_PATTERNS

    mgr = BrainRegionManager()
    table = Table(title="Brain Regions", border_style="bright_blue", show_header=True)
    table.add_column("Key", style="bright_white", min_width=24)
    table.add_column("Description", style="dim", max_width=60)
    table.add_column("Atlas", style="cyan", max_width=40)

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


def _rich_bar(score: float, width: int = 20) -> str:
    import numpy as np

    if not np.isfinite(score):
        return "[dim]n/a[/]"
    norm = max(0.0, min(1.0, (score + 1.0) / 2.0))
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
