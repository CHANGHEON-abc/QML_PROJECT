"""Compare static and oracle MWPM on identical drifting-noise memory shots.

Run ``python surface_code_drift.py --help`` for the parameter-sweep CLI.
No plotting functions are used. See README.md for the model conventions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import platform
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pymatching
import stim


PERIOD_ASSIGNMENT_METHOD = "coordinate_blake2b_pcg64_v1"
PHASE_ASSIGNMENT_METHOD = "coordinate_phase_blake2b_pcg64_v1"


@dataclass(frozen=True)
class PhysicalQubit:
    qubit: int
    x: float
    y: float
    role: str


def coordinate_rng(method: str, seed: int, qubit: PhysicalQubit) -> np.random.Generator:
    """Separate reproducible streams for each coordinate and parameter kind."""
    key = f"{method}|{seed}|{qubit.x:.17g}|{qubit.y:.17g}"
    coordinate_seed = int.from_bytes(hashlib.blake2b(key.encode("ascii"), digest_size=16).digest(), "little")
    return np.random.Generator(np.random.PCG64(coordinate_seed))


@lru_cache(maxsize=32)
def physical_qubits(distance: int) -> tuple[PhysicalQubit, ...]:
    """Stable physical coordinates; Stim's integer IDs depend on distance."""
    layout = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=1)
    data = {target.value for op in layout if op.name == "M" for target in op.targets_copy()}
    return tuple(PhysicalQubit(qubit, coordinates[0], coordinates[1], "data" if qubit in data else "ancilla")
                 for qubit, coordinates in sorted(layout.get_final_qubit_coordinates().items()))


@dataclass(frozen=True)
class Configuration:
    distance: int
    p: float
    p0: float
    alpha: float
    R: int | None
    rounds: int
    basis: str = "z"
    period_range: tuple[int, int] | None = None
    period_seed: int | None = None
    phase_seed: int | None = None

    @property
    def period_mode(self) -> str:
        return "per_qubit" if self.period_seed is not None else "uniform"

    @property
    def phase_mode(self) -> str:
        return "per_qubit" if self.phase_seed is not None else "zero"

    @property
    def has_random_assignments(self) -> bool:
        return self.period_seed is not None or self.phase_seed is not None

    def validate(self) -> None:
        if self.distance < 2:
            raise ValueError("distance must be at least 2")
        if self.rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if self.period_seed is None:
            if self.period_range is not None:
                raise ValueError("period_range requires an explicit period_seed")
            if self.R is None or self.R < 1:
                raise ValueError("R must be supplied as a positive integer when period_seed is absent")
        else:
            if self.period_range is None:
                raise ValueError("period_seed requires period_range=(R_min, R_max)")
            if self.R is not None:
                raise ValueError("Specify either R or period_range, not both")
            if len(self.period_range) != 2:
                raise ValueError("period_range must be (R_min, R_max)")
            low, high = self.period_range
            if not (isinstance(low, int) and isinstance(high, int) and 1 <= low <= high < 2**63):
                raise ValueError("period_range requires integers 1 <= R_min <= R_max < 2**63")
        for name, seed in (("period_seed", self.period_seed), ("phase_seed", self.phase_seed)):
            if seed is not None and (not isinstance(seed, int) or not 0 <= seed < 2**64):
                raise ValueError(f"{name} must be an integer in [0, 2**64)")
        if self.basis not in ("x", "z"):
            raise ValueError("basis must be x or z")
        for name, value in (("p", self.p), ("p0", self.p0)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if self.p > 0.75:
            raise ValueError("p must be <= 0.75: Stim's DEM analysis cannot handle over-mixing DEPOLARIZE1")
        self.drift_probabilities()

    def qubit_periods(self) -> dict[int, int]:
        """Draw once per coordinate, independently of sampling seed/sweep order.

        For a fixed range and period_seed, overlapping coordinates retain the
        same periods across distances, bases, and other physical parameters.
        Each coordinate has its own PCG64 stream, seeded via a stable digest;
        Python's process-randomized hash() is deliberately not used.
        """
        qubits = physical_qubits(self.distance)
        if self.period_seed is None:
            return {qubit.qubit: self.R for qubit in qubits}
        low, high = self.period_range
        periods = {}
        for qubit in qubits:
            rng = coordinate_rng(PERIOD_ASSIGNMENT_METHOD, self.period_seed, qubit)
            periods[qubit.qubit] = int(rng.integers(low, high, endpoint=True))
        return periods

    def qubit_phases(self) -> dict[int, float]:
        """Fixed per-qubit phases in radians, uniform on [0, 2*pi), or zero."""
        if self.phase_seed is None:
            return {qubit.qubit: 0.0 for qubit in physical_qubits(self.distance)}
        return {
            qubit.qubit: float(coordinate_rng(PHASE_ASSIGNMENT_METHOD, self.phase_seed, qubit).random() * math.tau)
            for qubit in physical_qubits(self.distance)
        }

    def drift_probabilities(self) -> np.ndarray:
        """Probability schedule q_i(n)=p0+alpha*sin(2*pi*n/R_i+phi_i).

        With both period and phase fixed, return shape (rounds,), preserving
        the original API. If either is random per qubit, return shape
        (rounds, num_physical_qubits), with columns in
        physical_qubits(distance) order (ascending Stim qubit ID).
        Each shot reuses the assignment and starts at n=0.
        """
        if self.rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if not self.has_random_assignments:
            if self.R is None or self.R < 1:
                raise ValueError("R must be a positive integer")
            periods = self.R
            phase = np.arange(self.rounds, dtype=np.int64) % periods
        else:
            periods = np.array(list(self.qubit_periods().values()), dtype=np.int64)[None, :]
            phase = np.arange(self.rounds, dtype=np.int64)[:, None] % periods
        # Handle the exact zeros of sin explicitly, including the R=1,2 cases.
        angles = 2 * np.pi * (phase / periods)
        if self.phase_seed is not None:
            angles += np.array(list(self.qubit_phases().values()), dtype=float)[None, :]
        sine = np.sin(angles)
        if self.phase_seed is None:
            sine[(phase == 0) | (phase == periods / 2)] = 0
        probabilities = self.p0 + self.alpha * sine
        tolerance = 1e-12
        invalid = (~np.isfinite(probabilities)) | (probabilities < -tolerance) | (probabilities > 1 + tolerance)
        if np.any(invalid):
            location = tuple(np.argwhere(invalid)[0])
            n = int(location[0])
            qubit_info = f", qubit={physical_qubits(self.distance)[location[1]].qubit}" if len(location) > 1 else ""
            raise ValueError(f"drift probability at cycle n={n}{qubit_info} is {probabilities[location]:g}, outside [0, 1]")
        # Only remove floating-point roundoff at probability boundaries.
        probabilities = np.clip(probabilities, 0, 1)
        if np.any(probabilities > 0.75 + tolerance):
            raise ValueError("drift probabilities must be <= 0.75 for Stim's DEM analysis")
        return np.minimum(probabilities, 0.75)


def build_circuits(config: Configuration) -> tuple[stim.Circuit, stim.Circuit, np.ndarray]:
    """Return the static assumed circuit, the true circuit, and its drift schedule.

    Use Stim's round-start data-noise locations as structural markers, replacing
    them with drift on *all active physical qubits*. After-Clifford noise is
    generated by Stim itself and is never replaced. Qubit IDs have gaps, so
    range(circuit.num_qubits) would incorrectly include unused qubits.
    """
    config.validate()
    probabilities = config.drift_probabilities()
    kwargs = dict(
        distance=config.distance,
        rounds=config.rounds,
        after_clifford_depolarization=config.p,
    )
    task = f"surface_code:rotated_memory_{config.basis}"
    static = stim.Circuit.generated(task, **kwargs)
    # Markers are removed before any simulation or DEM analysis.
    marked = stim.Circuit.generated(task, before_round_data_depolarization=1, **kwargs).flattened()
    active_qubits = sorted(marked.get_final_qubit_coordinates())
    data_measurement = "MX" if config.basis == "x" else "M"
    data_layers = [op for op in marked if op.name == data_measurement]
    if len(data_layers) != 1:
        raise RuntimeError("Unexpected Stim surface-code structure: final data measurement")
    data_targets = data_layers[0].targets_copy()
    drift_groups = {}
    if probabilities.ndim == 2:
        phases = config.qubit_phases()
        for column, (qubit, period) in enumerate(config.qubit_periods().items()):
            group = drift_groups.setdefault((period, phases[qubit]), {"column": column, "qubits": []})
            group["qubits"].append(qubit)
    physical = stim.Circuit()
    n = 0
    for op in marked:
        if op.name == "DEPOLARIZE1" and op.gate_args_copy() == [1.0] and op.targets_copy() == data_targets:
            if n >= config.rounds:
                raise RuntimeError("Unexpected extra round-start marker in Stim circuit")
            if probabilities.ndim == 1:
                physical.append("DEPOLARIZE1", active_qubits, float(probabilities[n]), tag="round_start_drift")
            else:
                # One independent channel per physical qubit. Group only
                # qubits with BOTH the same period and phase.
                for key in sorted(drift_groups):
                    group = drift_groups[key]
                    physical.append("DEPOLARIZE1", group["qubits"], float(probabilities[n, group["column"]]), tag="round_start_drift")
            n += 1
        else:
            physical.append(op)
    if n != config.rounds:
        raise RuntimeError(f"Found {n} round-start markers, expected {config.rounds}")
    if physical.without_noise() != static.flattened().without_noise():
        raise RuntimeError("Drift insertion changed the noiseless memory circuit")
    return static, physical, probabilities


def make_matching(circuit: stim.Circuit) -> pymatching.Matching:
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    # Keep the logical output present even in a completely noiseless graph.
    matching.ensure_num_fault_ids(circuit.num_observables)
    return matching


def decode_outcomes(
    matching: pymatching.Matching, detectors: np.ndarray, actual_observables: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate masks for logical mistakes and undecodable syndromes.

    An assumed zero-noise graph can have no explanation for an actual syndrome.
    Keep these erasures explicit instead of silently changing the static model
    by adding a noise floor. Use fast batch decoding for the ordinary case.
    """
    predictions = np.zeros_like(actual_observables, dtype=np.uint8)
    undecodable = np.zeros(len(detectors), dtype=bool)
    if matching.num_edges == 0:
        undecodable[:] = np.any(detectors, axis=1)
    else:
        def decode_slice(start: int, stop: int) -> None:
            try:
                predictions[start:stop] = matching.decode_batch(detectors[start:stop])
            except ValueError as exc:
                if "No perfect matching" not in str(exc):
                    raise
                if stop - start == 1:
                    undecodable[start] = True
                else:
                    middle = (start + stop) // 2
                    decode_slice(start, middle)
                    decode_slice(middle, stop)

        decode_slice(0, len(detectors))
    logical_errors = np.any(predictions != actual_observables, axis=1) & ~undecodable
    return logical_errors, undecodable


def wilson_interval(failures: int, shots: int) -> tuple[float, float]:
    """Two-sided 95% Wilson interval, including zero observed failures."""
    z = 1.959963984540054
    rate = failures / shots
    denominator = 1 + z * z / shots
    center = (rate + z * z / (2 * shots)) / denominator
    half_width = z * math.sqrt(rate * (1 - rate) / shots + z * z / (4 * shots * shots)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def run_configuration(
    config: Configuration, *, shots: int, seed: int = 0, batch_size: int = 10_000
) -> dict:
    if shots < 1 or batch_size < 1:
        raise ValueError("shots and batch_size must be positive")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    started = time.perf_counter()
    static_circuit, physical_circuit, probabilities = build_circuits(config)
    # The oracle knows the full schedule, but uses the same ordinary MWPM
    # approximation as static (no correlated-matching second pass).
    matchings = {"static": make_matching(static_circuit), "oracle": make_matching(physical_circuit)}
    sampler = physical_circuit.compile_detector_sampler(seed=seed)
    setup_seconds = time.perf_counter() - started
    logical_counts = {name: 0 for name in matchings}
    undecodable_counts = {name: 0 for name in matchings}
    decode_seconds = {name: 0.0 for name in matchings}
    sampling_seconds = 0.0
    static_only = oracle_only = both_failed = 0
    completed = 0
    while completed < shots:
        count = min(batch_size, shots - completed)
        before_sampling = time.perf_counter()
        detectors, observables = sampler.sample(shots=count, separate_observables=True)
        sampling_seconds += time.perf_counter() - before_sampling
        failed_masks = {}
        for name, matching in matchings.items():
            before_decoding = time.perf_counter()
            logical, undecodable = decode_outcomes(matching, detectors, observables)
            decode_seconds[name] += time.perf_counter() - before_decoding
            logical_counts[name] += int(np.count_nonzero(logical))
            undecodable_counts[name] += int(np.count_nonzero(undecodable))
            failed_masks[name] = logical | undecodable
        static_fail, oracle_fail = failed_masks["static"], failed_masks["oracle"]
        static_only += int(np.count_nonzero(static_fail & ~oracle_fail))
        oracle_only += int(np.count_nonzero(oracle_fail & ~static_fail))
        both_failed += int(np.count_nonzero(static_fail & oracle_fail))
        completed += count

    row = {
        "distance": config.distance, "p": config.p, "p0": config.p0,
        "alpha": config.alpha, "R": config.R, "shots": shots,
        "rounds": config.rounds, "basis": config.basis, "seed": seed,
        "batch_size": batch_size, "first_cycle_index": 0,
        "static_assumed_drift_probability": 0.0,
        "drift_probability_min": float(np.min(probabilities)),
        "drift_probability_max": float(np.max(probabilities)),
        "drift_probability_mean": float(np.mean(probabilities)),
        "num_physical_qubits": len(physical_circuit.get_final_qubit_coordinates()),
        "num_detectors": physical_circuit.num_detectors,
    }
    periods = list(config.qubit_periods().values())
    phases = list(config.qubit_phases().values())
    time_averages = np.mean(probabilities, axis=0) if probabilities.ndim == 2 else np.array([np.mean(probabilities)])
    row.update({
        "period_mode": config.period_mode,
        "R_min": config.period_range[0] if config.period_range else "",
        "R_max": config.period_range[1] if config.period_range else "",
        "period_seed": config.period_seed if config.period_range else "",
        "period_assignment_method": PERIOD_ASSIGNMENT_METHOD if config.period_range else "shared",
        "phase_mode": config.phase_mode,
        "phase_seed": config.phase_seed if config.phase_seed is not None else "",
        "phase_assignment_method": PHASE_ASSIGNMENT_METHOD if config.phase_seed is not None else "zero",
        "phase_radians": "" if config.phase_seed is not None else 0.0,
        "assigned_phase_min": min(phases), "assigned_phase_max": max(phases),
        "assigned_period_min": min(periods), "assigned_period_max": max(periods),
        "assigned_period_mean": float(np.mean(periods)),
        "qubit_time_average_probability_min": float(np.min(time_averages)),
        "qubit_time_average_probability_max": float(np.max(time_averages)),
    })
    for name in matchings:
        errors = logical_counts[name]
        undecodable = undecodable_counts[name]
        total_failures = errors + undecodable
        low, high = wilson_interval(total_failures, shots)
        row.update({
            f"{name}_logical_errors": errors,
            f"{name}_decode_failures": undecodable,
            f"{name}_logical_error_rate": errors / (shots - undecodable) if shots > undecodable else "",
            f"{name}_total_failures": total_failures,
            f"{name}_failure_rate": total_failures / shots,
            f"{name}_failure_rate_ci95_low": low,
            f"{name}_failure_rate_ci95_high": high,
        })
    difference = (static_only - oracle_only) / shots
    # Sample standard error for paired per-shot differences in {-1, 0, +1}.
    difference_se = math.sqrt(max(0.0, (static_only + oracle_only - shots * difference**2) / (shots * (shots - 1)))) if shots > 1 else ""
    row.update({
        "static_only_failures": static_only, "oracle_only_failures": oracle_only,
        "both_failed": both_failed, "both_succeeded": shots - static_only - oracle_only - both_failed,
        "failure_rate_difference_static_minus_oracle": difference,
        "paired_difference_std_error": difference_se,
        "setup_seconds": setup_seconds, "sampling_seconds": sampling_seconds,
        "static_decode_seconds": decode_seconds["static"],
        "oracle_decode_seconds": decode_seconds["oracle"],
        "total_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "stim_version": stim.__version__, "pymatching_version": pymatching.__version__,
    })
    return row


def parse_values(tokens: Iterable[str], cast, name: str) -> list:
    values = []
    for token in tokens:
        for part in token.split(","):
            if not part.strip():
                raise ValueError(f"Empty value in --{name}")
            try:
                value = cast(part.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid --{name} value: {part!r}") from exc
            if value not in values:
                values.append(value)
    return values


def period_assignment_rows(config: Configuration, *, configuration_index: int, sampling_seed: int) -> Iterable[dict]:
    """Actual period/phase assignment and finite-memory probability statistics."""
    periods = config.qubit_periods()
    phases = config.qubit_phases()
    probabilities = config.drift_probabilities()
    for column, qubit in enumerate(physical_qubits(config.distance)):
        schedule = probabilities[:, column] if probabilities.ndim == 2 else probabilities
        yield {
            "configuration_index": configuration_index,
            "distance": config.distance, "basis": config.basis, "p": config.p,
            "p0": config.p0, "alpha": config.alpha, "rounds": config.rounds,
            "period_mode": config.period_mode,
            "R_min": config.period_range[0] if config.period_range else config.R,
            "R_max": config.period_range[1] if config.period_range else config.R,
            "period_seed": config.period_seed, "sampling_seed": sampling_seed,
            "period_assignment_method": PERIOD_ASSIGNMENT_METHOD if config.period_seed is not None else "shared",
            "phase_mode": config.phase_mode, "phase_seed": config.phase_seed,
            "phase_assignment_method": PHASE_ASSIGNMENT_METHOD if config.phase_seed is not None else "zero",
            "qubit": qubit.qubit, "x": qubit.x, "y": qubit.y, "role": qubit.role,
            "R_i": periods[qubit.qubit], "phase_radians": phases[qubit.qubit],
            "drift_probability_min": float(np.min(schedule)),
            "drift_probability_max": float(np.max(schedule)),
            "drift_probability_mean": float(np.mean(schedule)),
        }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rotated surface-code memory: compare drift-blind static and schedule-aware oracle MWPM; save CSV only.",
        epilog="Sweep options accept spaces or commas. All combinations are run (Cartesian product). Cycle n starts at 0.",
    )
    parser.add_argument("--distance", nargs="+", default=["3"], help="Code distances, >= 2 (default: 3)")
    parser.add_argument("--p", nargs="+", default=["0.001"], help="Fixed after-Clifford depolarization probabilities (default: 0.001)")
    parser.add_argument("--p0", "--p_0", nargs="+", default=["0.002"], help="Drift offsets (default: 0.002)")
    parser.add_argument("--alpha", nargs="+", default=["0.001"], help="Nonnegative drift amplitudes (default: 0.001)")
    periods = parser.add_mutually_exclusive_group()
    periods.add_argument("--R", "--period", dest="R", nargs="+", help="Shared drift periods; required when --period-seed is absent")
    periods.add_argument("--period-range", nargs=2, type=int, action="append", metavar=("R_MIN", "R_MAX"),
                         help="Inclusive integer range for independent per-qubit periods; repeat to sweep ranges")
    parser.add_argument("--period-seed", nargs="+", help="Enable random per-qubit periods with these seeds; requires --period-range")
    parser.add_argument("--phase-seed", nargs="+", help="Random per-qubit phase seeds, uniform in [0, 2*pi); omitted means phase 0")
    parser.add_argument("--rounds", nargs="+", default=["100"], help="QEC cycles per memory shot (default: 100)")
    parser.add_argument("--shots", type=int, default=10_000, help="One shared shot count per configuration (default: 10000)")
    parser.add_argument("--basis", nargs="+", choices=["x", "z"], default=["z"], help="Logical memory basis (default: z)")
    parser.add_argument("--seed", type=int, default=0, help="Master seed; each configuration gets a derived seed (default: 0)")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Maximum shots in memory at once (default: 10000)")
    parser.add_argument("--output", type=Path, default=Path("results.csv"), help="Output CSV (default: results.csv)")
    parser.add_argument("--period-output", "--assignment-output", type=Path,
                        help="Per-qubit period/phase CSV (default: <output stem>_qubit_periods.csv)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.shots < 1 or args.batch_size < 1:
            raise ValueError("shots and batch-size must be positive")
        if not 0 <= args.seed < 2**64:
            raise ValueError("seed must be in [0, 2**64)")
        random_periods = args.period_seed is not None
        random_phases = args.phase_seed is not None
        if random_periods:
            if args.period_range is None:
                raise ValueError("--period-seed requires --period-range R_MIN R_MAX")
            period_options = list(dict.fromkeys(tuple(pair) for pair in args.period_range))
            assignment_seeds = parse_values(args.period_seed, int, "period-seed")
        else:
            if args.period_range is not None:
                raise ValueError("--period-range requires an explicit --period-seed")
            if args.R is None:
                raise ValueError("--R is required when --period-seed is absent")
            period_options = parse_values(args.R, int, "R")
            assignment_seeds = [None]
        phase_seeds = parse_values(args.phase_seed, int, "phase-seed") if random_phases else [None]
        if random_periods or random_phases:
            period_output = args.period_output or args.output.with_name(args.output.stem + "_qubit_periods.csv")
            if period_output.resolve() == args.output.resolve():
                raise ValueError("Results CSV and period/phase assignment CSV must have different paths")
        else:
            if args.period_output is not None:
                raise ValueError("--period-output/--assignment-output requires --period-seed or --phase-seed")
            period_output = None
        axes = [
            parse_values(args.distance, int, "distance"), parse_values(args.p, float, "p"),
            parse_values(args.p0, float, "p0"), parse_values(args.alpha, float, "alpha"),
            period_options, parse_values(args.rounds, int, "rounds"),
            list(dict.fromkeys(args.basis)),
            assignment_seeds,
            phase_seeds,
        ]

        def configurations():
            for distance, p, p0, alpha, period, rounds, basis, period_seed, phase_seed in itertools.product(*axes):
                yield Configuration(distance, p, p0, alpha, None if random_periods else period, rounds, basis,
                                    period_range=period if random_periods else None, period_seed=period_seed, phase_seed=phase_seed)

        # Validate every combination before creating/truncating the results file.
        count = math.prod(len(axis) for axis in axes)
        for config in configurations():
            try:
                config.validate()
            except ValueError as exc:
                raise ValueError(f"Invalid configuration {config}: {exc}") from exc
        for path in [args.output] + ([period_output] if period_output else []):
            if path.exists() and not args.overwrite:
                raise ValueError(f"Output already exists: {path}; choose another path or use --overwrite")
    except ValueError as exc:
        parser.error(str(exc))

    # Stable derivation by configuration index; save each actual Stim seed in CSV.
    seeds = np.random.default_rng(args.seed)
    writer = None
    period_writer = None
    try:
        with ExitStack() as stack:
            mode = "w" if args.overwrite else "x"
            for path in [args.output] + ([period_output] if period_output else []):
                path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(args.output.open(mode, encoding="utf-8-sig", newline=""))
            period_stream = stack.enter_context(period_output.open(mode, encoding="utf-8-sig", newline="")) if period_output else None
            for index, config in enumerate(configurations(), start=1):
                seed = int(seeds.bit_generator.random_raw())
                print(f"[{index}/{count}] {config}", flush=True)
                row = run_configuration(config, shots=args.shots, seed=seed, batch_size=args.batch_size)
                row = {"configuration_index": index, "master_seed": args.seed, **row,
                       "period_assignment_file": str(period_output.resolve()) if period_output else ""}
                if period_stream is not None:
                    for assignment in period_assignment_rows(config, configuration_index=index, sampling_seed=seed):
                        if period_writer is None:
                            period_writer = csv.DictWriter(period_stream, fieldnames=list(assignment))
                            period_writer.writeheader()
                        period_writer.writerow(assignment)
                    period_stream.flush()
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                print(f"  failure rate: static={row['static_failure_rate']:.6g}, oracle={row['oracle_failure_rate']:.6g}", flush=True)
                if row["static_decode_failures"] or row["oracle_decode_failures"]:
                    print("  Undecodable syndromes occurred; inspect *_decode_failures (included in failure rates).", file=sys.stderr)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Experiment stopped: {exc}\nCompleted rows, if any, are saved in {args.output.resolve()}", file=sys.stderr)
        return 1
    print(f"Saved {count} configurations to {args.output.resolve()}")
    if period_output:
        print(f"Saved qubit period/phase assignments to {period_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
