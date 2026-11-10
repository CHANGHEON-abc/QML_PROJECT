"""Generate edge and per-shot logical-error datasets for a chosen memory size.

Example (from the project directory)::

    python export_edge_dataset.py --n 100 --distance 5 --rounds 20 --p 0.001 --p0 0.001 \
        --alpha 0.0005 --period-range 10 50 --period-seed-start 0 \
        --shots 100000 --seed 2026 --batch-size 10000 --output-dir edge_datasets

The default edge output file is oracle_edges.pkl. It contains
EXACTLY list[list[dict]], with n inner lists (one oracle graph per list) and no wrapper,
parameters, detector coordinates, or other metadata. Each edge dict has:

    edge_id, source, target, is_boundary, weight, error_probability, fault_ids

The separate oracle_logical_error_rates.pkl contains EXACTLY list[float],
also of length n. rates[i] is the per-shot logical error probability measured
using graphs[i], with the same period_seed_start+i circuit. Values lie in
[0, 1], are not percentages, and are not divided by the number of QEC rounds.
There is no metadata wrapper in either file. --shots is the number of memory
shots PER graph; --n is the number of graphs.

--distance and --rounds select one memory geometry for all n graphs in a run.
Their defaults are distance=3 and rounds=4. Use separate output directories
for runs with different geometries to keep each edge/LER pair aligned.

The boundary target is None. fault_ids is a list of logical observable IDs,
consistent with export_dem_weights.py. Edge IDs are local to each graph.

Graph i uses period_seed_start+i (i=0,...,n-1). The phase seed, when supplied,
is held fixed for every graph; omitting it sets all phases to zero. Distinct
period seeds affect the oracle graph. Optional --decoder static or both also
supports the static baseline. Static ignores drift, so all n static graphs
have equal values. They are independent lists/dicts, so editing one
after loading does not modify another. Each selected decoder also gets its
own <decoder>_logical_error_rates.pkl with the same index order.

Shots are sampled from the original noisy oracle CIRCUIT, preserving physical
noise correlations, and decoded using its corresponding oracle DEM (or the
static assumed DEM when explicitly requested). With --decoder both, both
decoders receive the same sampled shots. A master --seed derives one Stim
sampling seed per graph using default_rng(seed).bit_generator.random_raw().
Reproducibility requires the same arguments, batch size, and library versions.
If a selected decoder cannot match a sampled syndrome, export stops rather
than silently writing a conditional error rate over a reduced denominator.

Python usage::

    from export_edge_dataset import export_edge_dataset
    export_edge_dataset(100, "edge_datasets", distance=5, rounds=20,
                        period_range=(10, 50))

    import pickle
    with open("edge_datasets/oracle_edges.pkl", "rb") as stream:
        graphs = pickle.load(stream)
    with open("edge_datasets/oracle_logical_error_rates.pkl", "rb") as stream:
        rates = pickle.load(stream)
    first_edge = graphs[0][0]
    first_logical_error_rate = rates[0]
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm
import numpy as np

from export_dem_weights import graph_record
from surface_code_drift import Configuration, build_circuits, decode_outcomes, make_matching


def export_edge_dataset(
    n: int,
    output_dir: str | Path = ".",
    *,
    distance: int = 3,
    rounds: int = 4,
    p: float = 0.001,
    p0: float = 0.001,
    alpha: float = 0.0005,
    period_range: tuple[int, int] = (10, 50),
    period_seed_start: int = 0,
    phase_seed: int | None = None,
    basis: str = "z",
    decoder: str = "oracle",
    shots: int = 10_000,
    seed: int = 0,
    batch_size: int = 10_000,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save aligned edge lists and per-shot LERs for n physical circuits.

    Return keys are decoder names for edge files and <decoder>_ler for LER
    files, e.g. paths['oracle'] and paths['oracle_ler']. All payloads are fully
    generated and serialized before publishing files. Each file is published
    atomically; a publishing error can leave earlier files updated. The
    returned path mapping is not stored inside any pickle.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(distance, int) or distance < 2:
        raise ValueError("distance must be an integer >= 2")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not isinstance(period_seed_start, int) or not 0 <= period_seed_start < 2**64:
        raise ValueError("period_seed_start must be an integer in [0, 2**64)")
    if period_seed_start + n > 2**64:
        raise ValueError("period_seed_start + n exceeds the available distinct uint64 seeds")
    if not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    if decoder not in ("oracle", "static", "both"):
        raise ValueError("decoder must be oracle, static, or both")
    selected = ("oracle", "static") if decoder == "both" else (decoder,)
    output_dir = Path(output_dir)
    paths = {}
    for name in selected:
        paths[name] = output_dir / f"{name}_edges.pkl"
        paths[f"{name}_ler"] = output_dir / f"{name}_logical_error_rates.pkl"
    for path in paths.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}; choose another directory or use --overwrite")

    datasets: dict[str, list[list[dict]]] = {name: [] for name in selected}
    rates: dict[str, list[float]] = {name: [] for name in selected}
    sampling_seeds = np.random.default_rng(seed)
    static_edges = None
    static_matching = None
    for index in tqdm(range(n), desc="Generating datasets"):
        config = Configuration(
            distance=distance, rounds=rounds, p=p, p0=p0, alpha=alpha, R=None,
            period_range=period_range, period_seed=period_seed_start + index,
            phase_seed=phase_seed, basis=basis,
        )
        static, oracle, _ = build_circuits(config)
        matchings = {}
        if "oracle" in datasets:
            datasets["oracle"].append(graph_record(oracle)["edges"])
            matchings["oracle"] = make_matching(oracle)
        if "static" in datasets:
            if static_edges is None:
                static_edges = graph_record(static)["edges"]
                static_matching = make_matching(static)
            datasets["static"].append(copy.deepcopy(static_edges))
            matchings["static"] = static_matching

        sampling_seed = int(sampling_seeds.bit_generator.random_raw())
        sampler = oracle.compile_detector_sampler(seed=sampling_seed)
        logical_errors = {name: 0 for name in selected}
        completed = 0
        while completed < shots:
            count = min(batch_size, shots - completed)
            detectors, observables = sampler.sample(shots=count, separate_observables=True)
            for name, matching in matchings.items():
                logical, undecodable = decode_outcomes(matching, detectors, observables)
                if np.any(undecodable):
                    raise ValueError(
                        f"Cannot compute per-shot {name} logical error rate for graph {index}: "
                        f"period_seed={config.period_seed} has undecodable syndromes"
                    )
                logical_errors[name] += int(np.count_nonzero(logical))
            completed += count
        for name in selected:
            rates[name].append(float(logical_errors[name] / shots))

    payloads = {**datasets, **{f"{name}_ler": rates[name] for name in selected}}
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for name, path in paths.items():
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".tmp", prefix=f".{path.name}.",
                                             dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                staged.append((temporary, path))
                pickle.dump(payloads[name], stream, protocol=pickle.HIGHEST_PROTOCOL)
        if not overwrite:
            for _, path in staged:
                if path.exists():
                    raise FileExistsError(f"Output appeared during export: {path}")
        for temporary, path in staged:
            if overwrite:
                os.replace(temporary, path)
            elif os.name == "nt":
                temporary.rename(path)
            else:
                os.link(temporary, path)
                temporary.unlink()
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return {name: path.resolve() for name, path in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Save n oracle edge lists and aligned per-shot logical error rates for a chosen distance and round count.",
        epilog="Graph i uses period_seed_start+i. Static graphs are equal because static ignores drift.",
    )
    parser.add_argument("--n", type=int, required=True, help="Number of graphs in each PKL")
    parser.add_argument("--distance", type=int, default=3, help="Code distance, integer >= 2 (default: 3)")
    parser.add_argument("--rounds", type=int, default=4, help="QEC rounds per memory shot (default: 4)")
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--p0", "--p_0", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--period-range", nargs=2, type=int, default=(10, 50), metavar=("R_MIN", "R_MAX"))
    parser.add_argument("--period-seed-start", type=int, default=0, help="First seed; subsequent graphs increment it by one")
    parser.add_argument("--phase-seed", type=int, help="One phase seed shared by all graphs; omitted means phase zero")
    parser.add_argument("--basis", choices=["x", "z"], default="z")
    parser.add_argument("--decoder", choices=["oracle", "static", "both"], default="oracle",
                        help="Decoder to export (default: oracle)")
    parser.add_argument("--shots", type=int, default=10_000, help="Memory shots per graph (default: 10000)")
    parser.add_argument("--seed", type=int, default=0, help="Master sampling seed, separate from period seeds (default: 0)")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Shots sampled and decoded per batch (default: 10000)")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = export_edge_dataset(
            args.n, args.output_dir, distance=args.distance, rounds=args.rounds,
            p=args.p, p0=args.p0, alpha=args.alpha,
            period_range=tuple(args.period_range), period_seed_start=args.period_seed_start,
            phase_seed=args.phase_seed, basis=args.basis, decoder=args.decoder,
            shots=args.shots, seed=args.seed, batch_size=args.batch_size, overwrite=args.overwrite,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    for name, path in paths.items():
        if name.endswith("_ler"):
            print(f"Saved {args.n} {name[:-4]} per-shot logical error rates to {path}")
        else:
            print(f"Saved {args.n} {name} edge lists to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
