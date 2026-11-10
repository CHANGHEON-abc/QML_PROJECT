"""Export static/oracle MWPM edge weights to both JSON and pickle (.pkl).

Run this file in the same environment as surface_code_drift.py. No Monte Carlo
shots are generated: weights depend on the circuit noise model and the fixed
period/phase assignments, not on the sampling seed or number of shots.

Examples (from the project directory)::

    python export_dem_weights.py --distance 5 --p 0.001 --p0 0.001 \
        --alpha 0.0005 --R 20 --phase-seed 42 --rounds 100 \
        --output dem_edge_weights.json

    python export_dem_weights.py --distance 3 5 --p 0.001 --p0 0.001 \
        --alpha 0.0005 --period-range 10 50 --period-seed 17 \
        --phase-seed 42 --rounds 100 --include-dem --output random_weights.json

Each run saves the JSON path given by --output and a .pkl file with the same
stem. --pkl-output can override the pickle path. Both formats contain the same
plain Python dictionary/list structure; the pickle does not store live Stim or
PyMatching objects.

The data contains one entry per configuration. ``decoders.static.edges`` and
``decoders.oracle.edges`` contain the actual PyMatching edge attributes after
Stim DEM decomposition and PyMatching parallel-edge merging. These are not
weights independently assigned to each raw DEM error instruction. Each edge
has source/target detector IDs (NOT physical qubit IDs), its weight, probability,
and fault_ids (logical observable IDs). A null target denotes the virtual
boundary. Nodes' Stim coordinates are listed separately under ``detectors``.
The edge list is undirected, sorted by its endpoints, and numbered locally.

The optional --include-dem saves the decomposed DEM as text in both formats.
Either existing output file requires --overwrite. All numeric values are finite;
an unavailable PyMatching edge probability is null in JSON and None in pickle.

Python API::

    from surface_code_drift import Configuration
    from export_dem_weights import export_weights
    export_weights([Configuration(3, .001, .001, .0005, 20, 100)],
                   "dem_edge_weights.json")

    import pickle
    with open("dem_edge_weights.pkl", "rb") as stream:
        data = pickle.load(stream)
    edges = data["configurations"][0]["decoders"]["oracle"]["edges"]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import platform
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pymatching
import stim

from surface_code_drift import (
    Configuration,
    PERIOD_ASSIGNMENT_METHOD,
    PHASE_ASSIGNMENT_METHOD,
    build_circuits,
    parse_values,
    physical_qubits,
)


def graph_record(circuit: stim.Circuit, *, include_dem: bool = False) -> dict:
    """Use the same DEM and ordinary MWPM construction as the experiment."""
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    matching.ensure_num_fault_ids(circuit.num_observables)
    coordinates = circuit.get_detector_coordinates()
    edges = []
    for source, target, attributes in matching.edges():
        source = int(source)
        target = int(target) if target is not None else None
        if target is not None and source > target:
            source, target = target, source
        weight = float(attributes["weight"])
        if not math.isfinite(weight):
            raise ValueError(f"Nonfinite edge weight for ({source}, {target})")
        probability = float(attributes.get("error_probability", -1))
        if probability == -1:
            probability = None
        elif not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Invalid edge probability for ({source}, {target}): {probability}")
        edges.append({
            "source": source,
            "target": target,
            "is_boundary": target is None,
            "weight": weight,
            "error_probability": probability,
            "fault_ids": sorted(int(fault) for fault in attributes["fault_ids"]),
        })
    edges.sort(key=lambda edge: (edge["source"], -1 if edge["target"] is None else edge["target"]))
    edges = [{"edge_id": index, **edge} for index, edge in enumerate(edges)]
    record = {
        "num_detectors": int(circuit.num_detectors),
        "num_observables": int(circuit.num_observables),
        "num_edges": len(edges),
        "detectors": [{"detector_id": detector, "coordinates": [float(value) for value in coordinates.get(detector, [])]}
                      for detector in range(circuit.num_detectors)],
        "edges": edges,
    }
    if include_dem:
        record["detector_error_model"] = str(dem)
    return record


def configuration_record(config: Configuration, *, index: int, decoder: str = "both", include_dem: bool = False) -> dict:
    static, oracle, _ = build_circuits(config)
    periods, phases = config.qubit_periods(), config.qubit_phases()
    decoders = {}
    for name, circuit in (("static", static), ("oracle", oracle)):
        if decoder not in ("both", name):
            continue
        graph = graph_record(circuit, include_dem=include_dem)
        graph["assumed_noise"] = {
            "after_clifford_depolarization": config.p,
            "round_start_drift": "none" if name == "static" else "p0 + alpha*sin(2*pi*n/R_i + phase_i)",
        }
        decoders[name] = graph
    parameters = asdict(config)
    if parameters["period_range"] is not None:
        # Use the same data types after JSON and pickle deserialization.
        parameters["period_range"] = list(parameters["period_range"])
    return {
        "configuration_index": index,
        "parameters": {**parameters, "period_mode": config.period_mode, "phase_mode": config.phase_mode},
        "assignment_methods": {
            "period": PERIOD_ASSIGNMENT_METHOD if config.period_seed is not None else "shared",
            "phase": PHASE_ASSIGNMENT_METHOD if config.phase_seed is not None else "zero",
        },
        "qubits": [{"qubit_id": qubit.qubit, "coordinates": [qubit.x, qubit.y], "role": qubit.role,
                    "R_i": periods[qubit.qubit], "phase_radians": phases[qubit.qubit]}
                   for qubit in physical_qubits(config.distance)],
        "decoders": decoders,
    }


def export_weights(
    configurations: Iterable[Configuration], output: str | Path, *, decoder: str = "both",
    include_dem: bool = False, overwrite: bool = False, pkl_output: str | Path | None = None,
) -> Path:
    """Export all configurations to JSON and pickle; return the JSON path.

    The pickle path defaults to output.with_suffix('.pkl'). Both files are
    checked before construction and both serializations finish before publishing.
    Each file is published with an atomic filesystem operation. Serialization
    failures preserve existing files. A failure during publishing can leave only
    the first file updated. Temporary files are created beside each destination.
    """
    if decoder not in ("static", "oracle", "both"):
        raise ValueError("decoder must be static, oracle, or both")
    configs = list(configurations)
    if not configs:
        raise ValueError("At least one configuration is required")
    for config in configs:
        config.validate()
    output = Path(output)
    pkl_output = Path(pkl_output) if pkl_output is not None else output.with_suffix(".pkl")
    if output.resolve() == pkl_output.resolve():
        raise ValueError("JSON and pickle outputs must have different paths; --output is the JSON path")
    for path in (output, pkl_output):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}; choose another path or use --overwrite")
    document = {
        "schema_version": 1,
        "graph_type": "undirected PyMatching graph from decomposed Stim DEM",
        "weight_definition": "natural log((1-p_edge)/p_edge), after PyMatching parallel-edge merging",
        "edge_probability_definition": "Matching graph edge probability; distinct from the Clifford gate probability p",
        "boundary_representation": "target=null denotes the virtual boundary",
        "node_definition": "source/target are detector IDs Dk, not physical qubit IDs; detector coordinates are supplied by Stim",
        "fault_ids_definition": "Logical observable IDs flipped by the matching edge",
        "cycle_index_origin": 0,
        "correlated_matching": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "stim": stim.__version__, "pymatching": pymatching.__version__},
        "configurations": [configuration_record(config, index=index, decoder=decoder, include_dem=include_dem)
                           for index, config in enumerate(configs, start=1)],
    }
    staged = []
    try:
        for path, binary in ((output, False), (pkl_output, True)):
            path.parent.mkdir(parents=True, exist_ok=True)
            options = {} if binary else {"encoding": "utf-8", "newline": "\n"}
            with tempfile.NamedTemporaryFile(mode="wb" if binary else "w", suffix=".tmp",
                                             prefix=f".{path.name}.", dir=path.parent,
                                             delete=False, **options) as stream:
                temporary = Path(stream.name)
                staged.append((temporary, path))
                if binary:
                    pickle.dump(document, stream, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
        # Check both paths again after the potentially lengthy construction.
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
                # Hard-linking a completed temporary file creates the final
                # path exclusively on POSIX, avoiding rename's overwrite rule.
                os.link(temporary, path)
                temporary.unlink()
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return output.resolve()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild the memory-experiment circuits and export their MWPM DEM edge weights to JSON and pickle.",
        epilog="Sweep options accept space/comma-separated values. All combinations are exported. No shots are sampled.",
    )
    parser.add_argument("--distance", nargs="+", default=["3"])
    parser.add_argument("--p", nargs="+", default=["0.001"], help="Fixed after-Clifford probability")
    parser.add_argument("--p0", "--p_0", nargs="+", default=["0.002"])
    parser.add_argument("--alpha", nargs="+", default=["0.001"])
    periods = parser.add_mutually_exclusive_group()
    periods.add_argument("--R", "--period", dest="R", nargs="+", help="Shared period; required without --period-seed")
    periods.add_argument("--period-range", nargs=2, type=int, action="append", metavar=("R_MIN", "R_MAX"))
    parser.add_argument("--period-seed", nargs="+", help="Random per-qubit period seeds; requires --period-range")
    parser.add_argument("--phase-seed", nargs="+", help="Random per-qubit phase seeds; omitted means phase zero")
    parser.add_argument("--rounds", nargs="+", default=["100"])
    parser.add_argument("--basis", nargs="+", choices=["x", "z"], default=["z"])
    parser.add_argument("--decoder", choices=["static", "oracle", "both"], default="both")
    parser.add_argument("--include-dem", action="store_true", help="Include the decomposed DEM text in each decoder record")
    parser.add_argument("--output", type=Path, default=Path("dem_edge_weights.json"), help="JSON output path")
    parser.add_argument("--pkl-output", type=Path, help="Pickle output path (default: JSON path with .pkl suffix)")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.period_seed is not None:
            if args.period_range is None:
                raise ValueError("--period-seed requires --period-range R_MIN R_MAX")
            period_options = list(dict.fromkeys(tuple(pair) for pair in args.period_range))
            period_seeds = parse_values(args.period_seed, int, "period-seed")
        else:
            if args.period_range is not None:
                raise ValueError("--period-range requires an explicit --period-seed")
            if args.R is None:
                raise ValueError("--R is required when --period-seed is absent")
            period_options = parse_values(args.R, int, "R")
            period_seeds = [None]
        axes = [parse_values(args.distance, int, "distance"), parse_values(args.p, float, "p"),
                parse_values(args.p0, float, "p0"), parse_values(args.alpha, float, "alpha"),
                period_options, parse_values(args.rounds, int, "rounds"), list(dict.fromkeys(args.basis)),
                period_seeds, parse_values(args.phase_seed, int, "phase-seed") if args.phase_seed is not None else [None]]
        configs = [Configuration(distance, p, p0, alpha, period if period_seed is None else None, rounds, basis,
                                 period_range=period if period_seed is not None else None,
                                 period_seed=period_seed, phase_seed=phase_seed)
                   for distance, p, p0, alpha, period, rounds, basis, period_seed, phase_seed in itertools.product(*axes)]
        pkl_path = args.pkl_output if args.pkl_output is not None else args.output.with_suffix(".pkl")
        path = export_weights(configs, args.output, decoder=args.decoder, include_dem=args.include_dem,
                              overwrite=args.overwrite, pkl_output=pkl_path)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {len(configs)} configuration(s), decoder={args.decoder}, to {path}")
    print(f"Saved the same data to {pkl_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
