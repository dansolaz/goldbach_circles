# Installation:
# pip install matplotlib numpy

"""Minimal rolling-circle animation for prime discovery on a number line.

Each circle has circumference equal to p, so its radius is p / (2 * pi).
The circle rolls without slipping, and a red point on the circumference marks
integer contact positions that are divisible by p.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_NUMBER = 50

PRIME_MODE = "auto"  # "auto" or "manual"
MANUAL_CIRCLES = [2, 3, 5, 7, 11]

ANIMATION_SPEED = 0.35
SHOW_NUMBER_LABELS = True
SHOW_ALL_PREVIOUS_CIRCLES = True
KEEP_COMPOSITES_RED = True

MARK_FROM = "p_squared"  # "2p" or "p_squared"
START_ROLLING_FROM = "zero"  # "zero" or "p"

SHOW_CONTACT_MARKS = True
ROLL_BACK_ENABLED = True
SHOW_PHASE_DATA = False
SHOW_GOLDBACH_RESULT = True
FLIP_FRAME_COUNT = 40

EXPORT_ANIMATION = False
EXPORT_FORMAT = "mp4"  # "mp4" or "gif"


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
BACKGROUND_COLOR = "white"
NUMBER_LINE_COLOR = "#444444"
CANDIDATE_COLOR = "#2b6cb0"
COMPOSITE_COLOR = "#d62728"
ACTIVE_PRIME_COLOR = "#d4a017"
ACTIVE_CIRCLE_COLOR = "#1f3f5b"
PREVIOUS_CIRCLE_COLOR = "#9fb3c8"
CONTACT_MARK_COLOR = "#c61c1c"
INACTIVE_NUMBER_COLOR = "#bbbbbb"

BASE_MARKER_SIZE = 42
ACTIVE_MARKER_SIZE = 150
LINE_WIDTH = 1.6
FPS = 30
HOLD_FRAMES = 16


def is_prime(n: int) -> bool:
    """Return True if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(math.isqrt(n))
    for divisor in range(3, limit + 1, 2):
        if n % divisor == 0:
            return False
    return True


def get_primes_up_to(limit: int) -> list[int]:
    """Return all primes up to limit using the primality helper above."""
    return [value for value in range(2, limit + 1) if is_prime(value)]


def validate_configuration() -> None:
    if PRIME_MODE not in {"auto", "manual"}:
        raise ValueError("PRIME_MODE must be 'auto' or 'manual'.")
    if MARK_FROM not in {"2p", "p_squared"}:
        raise ValueError("MARK_FROM must be '2p' or 'p_squared'.")
    if START_ROLLING_FROM not in {"zero", "p"}:
        raise ValueError("START_ROLLING_FROM must be 'zero' or 'p'.")
    if EXPORT_FORMAT not in {"mp4", "gif"}:
        raise ValueError("EXPORT_FORMAT must be 'mp4' or 'gif'.")
    if ANIMATION_SPEED <= 0:
        raise ValueError("ANIMATION_SPEED must be positive.")
    if FLIP_FRAME_COUNT < 0:
        raise ValueError("FLIP_FRAME_COUNT must be >= 0.")


def get_goldbach_left_primes(max_number: int) -> set[int]:
    """Return left-side primes p where p + q = 2*max_number for some prime q."""
    target = 2 * max_number
    left_primes: set[int] = set()
    for p in range(2, max_number + 1):
        if is_prime(p) and is_prime(target - p):
            left_primes.add(p)
    return left_primes


def get_circle_start_x(p: int) -> float:
    return 0.0 if START_ROLLING_FROM == "zero" else float(p)


def get_mark_start(p: int) -> int:
    if MARK_FROM == "2p":
        return 2 * p
    return p * p


def get_mark_targets(p: int, max_number: int) -> list[int]:
    """Return the integer positions that should be marked for the circle p."""
    if p < 2:
        return []

    first_target = max(get_mark_start(p), p + 1)
    if first_target > max_number:
        return []

    first_multiple = ((first_target + p - 1) // p) * p
    if first_multiple == p:
        first_multiple += p

    return list(range(first_multiple, max_number + 1, p))


def get_backward_targets(p: int, start_x: float, end_x: float) -> list[int]:
    """Return-pass targets for a folded number line Goldbach sweep.

    With MAX at the fold midpoint, a left-side value x maps to the right-side
    value (2*MAX - x). For circle p, we mark x whenever (2*MAX - x) is a
    multiple of p:

        x = 2*MAX - k*p

    This makes the backward pass act like sieving complements toward the target
    even number 2*MAX.
    """
    if p <= 0:
        return []

    high = max(start_x, end_x)
    low = min(start_x, end_x)
    folded_sum = 2.0 * high

    targets: list[int] = []
    k_start = max(1, int(math.floor((folded_sum - high) / p)))
    k_end = int(math.floor((folded_sum - low) / p))

    for k in range(k_start, k_end + 1):
        value = folded_sum - k * p
        rounded = int(round(value))
        if abs(value - rounded) < 1e-9 and low - 1e-9 <= rounded <= high + 1e-9:
            targets.append(rounded)

    return sorted(set(targets), reverse=True)


def build_auto_circle_sequence(max_number: int) -> list[int]:
    """Discover circles by repeatedly taking the next unmarked number."""
    discovered_circles: list[int] = []
    composite_flags = np.zeros(max_number + 1, dtype=bool)

    for candidate in range(2, max_number + 1):
        if composite_flags[candidate]:
            continue

        discovered_circles.append(candidate)
        for marked_value in get_mark_targets(candidate, max_number):
            composite_flags[marked_value] = True

    return discovered_circles


def build_manual_circle_sequence(max_number: int, circle_values: Iterable[int]) -> list[int]:
    sequence: list[int] = []
    seen: set[int] = set()
    for value in circle_values:
        candidate = int(value)
        if candidate < 2 or candidate > max_number or candidate in seen:
            continue
        seen.add(candidate)
        sequence.append(candidate)
    return sequence


def get_circle_sequence(max_number: int) -> list[int]:
    if PRIME_MODE == "auto":
        return build_auto_circle_sequence(max_number)
    return build_manual_circle_sequence(max_number, MANUAL_CIRCLES)


@dataclass
class CircleRuntime:
    circle: "RollingCircle"
    forward_targets: list[int]
    backward_targets: list[int]
    next_forward_target_index: int = 0
    next_backward_target_index: int = 0


@dataclass
class FrameCircleSnapshot:
    p: int
    radius: float
    center_x: float
    center_y: float
    red_x: float
    red_y: float
    is_moving: bool


@dataclass
class FrameState:
    circle_snapshots: list[FrameCircleSnapshot]
    composite_flags: np.ndarray
    contact_positions: list[int]
    active_prime: int | None
    motion_mode: str


def make_snapshots(runtimes: list[CircleRuntime]) -> list[FrameCircleSnapshot]:
    snapshots: list[FrameCircleSnapshot] = []
    for runtime in runtimes:
        c = runtime.circle
        snapshots.append(
            FrameCircleSnapshot(
                p=c.p,
                radius=c.radius,
                center_x=c.center_x,
                center_y=c.center_y,
                red_x=c.red_point[0],
                red_y=c.red_point[1],
                is_moving=(c.distance_traveled < c.total_distance - 1e-12),
            )
        )
    return snapshots


class RollingCircle:
    """Rolling circle with phase placeholders for later extensions."""

    def __init__(
        self,
        p: int,
        start_x: float,
        end_x: float,
        direction: int = 1,
        initial_angle: float = math.pi,
        angular_velocity_sign: float = 1.0,
    ) -> None:
        self.p = int(p)
        self.radius = self.p / (2.0 * math.pi)
        self.start_x = float(start_x)
        self.end_x = float(end_x)
        self.direction = 1 if direction >= 0 else -1
        self.initial_angle = float(initial_angle)
        self.angular_velocity_sign = float(angular_velocity_sign)

        self.center_x = self.start_x
        self.center_y = self.radius
        self.distance_traveled = 0.0
        self.rotation_angle = self.initial_angle
        self.red_point = (self.center_x, 0.0)

        # Placeholder for future phase-comparison features.
        self.phase = 0.0
        self.angular_phase = 0.0
        self.phase_info: dict[str, float] = {}

        self.total_distance = abs(self.end_x - self.start_x)
        self.update_from_distance(0.0)

    def update_from_distance(self, distance: float) -> None:
        """Advance the circle while preserving the no-slip relation.

        The rolling angle obeys theta = -distance / radius. A phase offset of pi
        places the red point at the bottom initially so contacts occur every p
        units along the line, matching the desired multiple-marking behavior.
        """
        bounded_distance = min(max(distance, 0.0), self.total_distance)
        signed_distance = self.direction * bounded_distance

        self.distance_traveled = bounded_distance
        self.center_x = self.start_x + signed_distance
        self.center_y = self.radius
        self.rotation_angle = self.initial_angle + self.angular_velocity_sign * (bounded_distance / self.radius)

        self.phase = bounded_distance % self.p
        self.angular_phase = self.rotation_angle % (2.0 * math.pi)

        point_x = self.center_x + self.radius * math.sin(self.rotation_angle)
        point_y = self.center_y + self.radius * math.cos(self.rotation_angle)
        self.red_point = (point_x, point_y)

    @property
    def final_center_x(self) -> float:
        return self.end_x


def create_number_colors(visible_composites: np.ndarray) -> list[tuple[float, float, float, float]]:
    colors = [to_rgba(INACTIVE_NUMBER_COLOR), to_rgba(INACTIVE_NUMBER_COLOR)]
    for value in range(2, MAX_NUMBER + 1):
        colors.append(to_rgba(COMPOSITE_COLOR if visible_composites[value] else CANDIDATE_COLOR))
    return colors


def initialize_number_line_markers(ax: plt.Axes, max_number: int) -> tuple:
    x_values = np.arange(0, max_number + 1)
    y_values = np.zeros_like(x_values, dtype=float)
    colors = create_number_colors(np.zeros(max_number + 1, dtype=bool))

    markers = ax.scatter(
        x_values,
        y_values,
        s=BASE_MARKER_SIZE,
        c=colors,
        zorder=3,
    )

    active_prime_marker = ax.scatter(
        [],
        [],
        s=ACTIVE_MARKER_SIZE,
        facecolors="none",
        edgecolors=ACTIVE_PRIME_COLOR,
        linewidths=2.0,
        zorder=4,
    )

    label_artists: list[plt.Text] = []
    if SHOW_NUMBER_LABELS:
        label_offset = -0.22 * max(1.0, max_number / 25.0)
        for value in x_values:
            color = INACTIVE_NUMBER_COLOR if value < 2 else CANDIDATE_COLOR
            label_artists.append(
                ax.text(
                    value,
                    label_offset,
                    str(value),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=color,
                )
            )

    return markers, active_prime_marker, label_artists


def build_simulation_frames(circle_values: list[int], max_number: int) -> list[FrameState]:
    """Simulate all frames with concurrent rolling circles.

    A new circle starts as soon as its number is still unmarked at the moment
    we pass that number in the global sweep, so 3 can start while 2 is still
    moving, and 5 can start once both 2 and 3 have passed and not marked it.
    """
    runtimes: list[CircleRuntime] = []
    all_frames: list[FrameState] = []

    composite_flags = np.zeros(max_number + 1, dtype=bool)
    contact_positions: list[int] = []
    contact_seen: set[int] = set()

    start_lookup = {p: get_circle_start_x(p) for p in circle_values}
    targets_lookup = {p: get_mark_targets(p, max_number) for p in circle_values}
    spawned_primes: list[int] = []

    next_circle_index = 0
    global_sweep_x = 2.0
    hold_counter = 0
    motion_mode = "forward"

    # Safety cap to avoid accidental infinite loops when tuning parameters.
    max_frames = max(2000, int(math.ceil(max_number / ANIMATION_SPEED)) * 30)

    while len(all_frames) < max_frames:
        while motion_mode == "forward" and next_circle_index < len(circle_values):
            candidate = circle_values[next_circle_index]
            if candidate > int(math.floor(global_sweep_x + 1e-9)):
                break
            if not composite_flags[candidate]:
                start_x = start_lookup[candidate]
                runtimes.append(
                    CircleRuntime(
                        circle=RollingCircle(
                            p=candidate,
                            start_x=start_x,
                            end_x=float(max_number),
                            direction=1,
                            initial_angle=math.pi,
                            angular_velocity_sign=1.0,
                        ),
                        forward_targets=targets_lookup[candidate],
                        backward_targets=[],
                        next_forward_target_index=0,
                        next_backward_target_index=0,
                    )
                )
                spawned_primes.append(candidate)
            next_circle_index += 1

        any_moving = False
        active_prime: int | None = None

        for runtime in runtimes:
            circle = runtime.circle
            prev_distance = circle.distance_traveled
            new_distance = min(circle.total_distance, prev_distance + ANIMATION_SPEED)
            if new_distance > prev_distance + 1e-12:
                any_moving = True
            circle.update_from_distance(new_distance)

            if new_distance < circle.total_distance - 1e-12 and active_prime is None:
                active_prime = circle.p

            if motion_mode == "forward":
                while runtime.next_forward_target_index < len(runtime.forward_targets):
                    target = runtime.forward_targets[runtime.next_forward_target_index]
                    contact_distance = abs(target - circle.start_x)
                    if contact_distance <= circle.distance_traveled + 1e-9:
                        composite_flags[target] = True
                        if target not in contact_seen:
                            contact_seen.add(target)
                            contact_positions.append(target)
                        runtime.next_forward_target_index += 1
                    else:
                        break
            else:
                while runtime.next_backward_target_index < len(runtime.backward_targets):
                    target = runtime.backward_targets[runtime.next_backward_target_index]
                    contact_distance = abs(target - circle.start_x)
                    if contact_distance <= circle.distance_traveled + 1e-9:
                        # Keep prime self-protection rule during return pass too.
                        if target != circle.p:
                            composite_flags[target] = True
                            if target not in contact_seen:
                                contact_seen.add(target)
                                contact_positions.append(target)
                        runtime.next_backward_target_index += 1
                    else:
                        break

        snapshots: list[FrameCircleSnapshot] = []
        for runtime in runtimes:
            c = runtime.circle
            snapshots.append(
                FrameCircleSnapshot(
                    p=c.p,
                    radius=c.radius,
                    center_x=c.center_x,
                    center_y=c.center_y,
                    red_x=c.red_point[0],
                    red_y=c.red_point[1],
                    is_moving=(c.distance_traveled < c.total_distance - 1e-12),
                )
            )

        all_frames.append(
            FrameState(
                circle_snapshots=snapshots,
                composite_flags=composite_flags.copy(),
                contact_positions=contact_positions.copy(),
                active_prime=active_prime,
                motion_mode=motion_mode,
            )
        )

        if motion_mode == "forward":
            global_sweep_x = min(float(max_number), global_sweep_x + ANIMATION_SPEED)
            done_discovering = next_circle_index >= len(circle_values)
            if not any_moving and done_discovering:
                if ROLL_BACK_ENABLED and spawned_primes:
                    # Flip pass: re-spawn discovered circles on the right and roll back left.
                    runtimes = []
                    for p in spawned_primes:
                        right_start = float(max_number)
                        left_end = start_lookup[p]
                        backward_circle = RollingCircle(
                            p=p,
                            start_x=right_start,
                            end_x=left_end,
                            direction=-1,
                            # Visible left-right flip: start on the opposite side.
                            initial_angle=0.0,
                            # Opposite spin progression for the return pass.
                            angular_velocity_sign=-1.0,
                        )
                        runtimes.append(
                            CircleRuntime(
                                circle=backward_circle,
                                forward_targets=[],
                                backward_targets=get_backward_targets(p, right_start, left_end),
                                next_forward_target_index=0,
                                next_backward_target_index=0,
                            )
                        )
                    motion_mode = "backward"
                    hold_counter = 0
                else:
                    hold_counter += 1
            else:
                hold_counter = 0
        else:
            if not any_moving:
                hold_counter += 1
            else:
                hold_counter = 0

        if hold_counter >= HOLD_FRAMES:
            break

    return all_frames


def build_figure(max_radius: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.axhline(0.0, color=NUMBER_LINE_COLOR, linewidth=LINE_WIDTH, zorder=1)

    horizontal_margin = max_radius + 1.0
    lower_margin = max(0.7, 0.55 * max_radius)
    upper_margin = max(1.5, 2.4 * max_radius)

    ax.set_xlim(-horizontal_margin, MAX_NUMBER + horizontal_margin)
    ax.set_ylim(-lower_margin, upper_margin)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax


def create_animation() -> tuple[plt.Figure, animation.FuncAnimation]:
    validate_configuration()

    if PRIME_MODE == "auto":
        # Keep the explicit helper available for quick inspection or extension.
        _ = get_primes_up_to(MAX_NUMBER)

    circle_values = get_circle_sequence(MAX_NUMBER)
    if not circle_values:
        raise ValueError("No circles were generated. Check PRIME_MODE and MANUAL_CIRCLES.")

    goldbach_left_primes = get_goldbach_left_primes(MAX_NUMBER)

    simulation_frames = build_simulation_frames(circle_values, MAX_NUMBER)
    max_radius = max(p / (2.0 * math.pi) for p in circle_values)

    fig, ax = build_figure(max_radius)
    number_markers, active_prime_marker, label_artists = initialize_number_line_markers(ax, MAX_NUMBER)

    circle_patches: dict[int, Circle] = {}
    red_point_artists: dict[int, plt.Line2D] = {}
    for p in circle_values:
        patch = Circle((0.0, 0.0), radius=0.0, fill=False, linewidth=1.6, edgecolor=PREVIOUS_CIRCLE_COLOR, alpha=0.7, zorder=2)
        patch.set_visible(False)
        ax.add_patch(patch)
        circle_patches[p] = patch

        red_point_artist, = ax.plot([], [], "o", color=CONTACT_MARK_COLOR, markersize=4, zorder=6)
        red_point_artist.set_visible(False)
        red_point_artists[p] = red_point_artist

    contact_tick_height = -0.07 * max(1.0, max_radius)
    contact_markers = ax.scatter([], [], s=70, marker="|", c=CONTACT_MARK_COLOR, zorder=4)

    title_artist = ax.set_title("", fontsize=14, pad=18)
    info_artist = ax.text(
        0.02,
        0.96,
        "Circle circumference = p\nRed contact points mark multiples of p\nUnmarked blue numbers are prime candidates",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.95},
        zorder=10,
    )
    status_artist = ax.text(0.02, 0.76, "", transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#333333")
    phase_artist = ax.text(0.02, 0.66, "", transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#333333")

    def init() -> list:
        number_markers.set_facecolors(create_number_colors(np.zeros(MAX_NUMBER + 1, dtype=bool)))
        active_prime_marker.set_offsets(np.empty((0, 2)))
        contact_markers.set_offsets(np.empty((0, 2)))
        status_artist.set_text("")
        phase_artist.set_text("")

        for patch in circle_patches.values():
            patch.set_visible(False)
        for red_point in red_point_artists.values():
            red_point.set_visible(False)

        return [
            number_markers,
            active_prime_marker,
            contact_markers,
            title_artist,
            info_artist,
            status_artist,
            phase_artist,
            *label_artists,
            *circle_patches.values(),
            *red_point_artists.values(),
        ]

    def update(frame_index: int) -> list:
        frame_state = simulation_frames[frame_index]
        visible_composites = frame_state.composite_flags if KEEP_COMPOSITES_RED else np.zeros(MAX_NUMBER + 1, dtype=bool)

        is_last_frame = frame_index == len(simulation_frames) - 1
        if SHOW_GOLDBACH_RESULT and is_last_frame:
            # Final frame: show only Goldbach-valid left primes as blue.
            visible_composites = np.ones(MAX_NUMBER + 1, dtype=bool)
            visible_composites[0] = False
            visible_composites[1] = False
            for p in goldbach_left_primes:
                visible_composites[p] = False

        number_markers.set_facecolors(create_number_colors(visible_composites))

        if frame_state.active_prime is None:
            active_prime_marker.set_offsets(np.empty((0, 2)))
        else:
            active_prime_marker.set_offsets(np.array([[frame_state.active_prime, 0.0]]))

        if SHOW_NUMBER_LABELS:
            for value, label in enumerate(label_artists):
                if value < 2:
                    label.set_color(INACTIVE_NUMBER_COLOR)
                    label.set_fontweight("normal")
                elif frame_state.active_prime is not None and value == frame_state.active_prime:
                    label.set_color(ACTIVE_PRIME_COLOR)
                    label.set_fontweight("bold")
                elif visible_composites[value]:
                    label.set_color(COMPOSITE_COLOR)
                    label.set_fontweight("normal")
                else:
                    label.set_color(CANDIDATE_COLOR)
                    label.set_fontweight("normal")

        snapshot_by_p = {snapshot.p: snapshot for snapshot in frame_state.circle_snapshots}
        for p in circle_values:
            patch = circle_patches[p]
            point_artist = red_point_artists[p]
            snapshot = snapshot_by_p.get(p)
            if snapshot is None:
                patch.set_visible(False)
                point_artist.set_visible(False)
                continue

            patch.center = (snapshot.center_x, snapshot.center_y)
            patch.set_radius(snapshot.radius)
            is_active = snapshot.is_moving
            if is_active:
                patch.set_edgecolor(ACTIVE_CIRCLE_COLOR)
                patch.set_linewidth(2.2)
                patch.set_alpha(0.95)
            else:
                patch.set_edgecolor(PREVIOUS_CIRCLE_COLOR)
                patch.set_linewidth(1.4)
                patch.set_alpha(0.7)

            if SHOW_ALL_PREVIOUS_CIRCLES or is_active:
                patch.set_visible(True)
            else:
                patch.set_visible(False)

            if patch.get_visible():
                point_artist.set_data([snapshot.red_x], [snapshot.red_y])
                point_artist.set_visible(True)
            else:
                point_artist.set_visible(False)

        if SHOW_CONTACT_MARKS:
            if KEEP_COMPOSITES_RED:
                visible_contacts = frame_state.contact_positions
            else:
                visible_contacts = [x for x in frame_state.contact_positions if visible_composites[x]]

            if visible_contacts:
                contact_offsets = np.column_stack(
                    [np.array(visible_contacts, dtype=float), np.full(len(visible_contacts), contact_tick_height)]
                )
            else:
                contact_offsets = np.empty((0, 2))
            contact_markers.set_offsets(contact_offsets)
        else:
            contact_markers.set_offsets(np.empty((0, 2)))

        active_count = sum(1 for s in frame_state.circle_snapshots if s.is_moving)
        direction_label = "rightward (clockwise)" if frame_state.motion_mode == "forward" else "leftward (anticlockwise)"
        title_artist.set_text(
            f"Rolling circles on number line | active circles: {active_count} | mode: {direction_label}"
        )
        status_artist.set_text(
            f"Circles spawned: {len(frame_state.circle_snapshots)}/{len(circle_values)}   "
            f"Marks from {MARK_FROM}   "
            f"Start at x = {START_ROLLING_FROM}   "
            f"Pass: {frame_state.motion_mode}"
        )

        if SHOW_PHASE_DATA:
            moving = [s for s in frame_state.circle_snapshots if s.is_moving]
            if moving:
                lead = moving[0]
                phase_artist.set_text(
                    f"Lead p = {lead.p} (phase panel placeholder)\n"
                    "Future: show pairwise phase differences here"
                )
            else:
                phase_artist.set_text("")
        else:
            phase_artist.set_text("")

        if SHOW_GOLDBACH_RESULT and is_last_frame:
            target = 2 * MAX_NUMBER
            title_artist.set_text(
                f"Goldbach target {target}: left-side pair primes = {len(goldbach_left_primes)}"
            )
            status_artist.set_text(
                "Blue points are primes p such that p + q = 2*MAX_NUMBER with q prime"
            )

        # Placeholder for future roll-back support. The class already stores
        # direction, so a return path can later be added by extending timeline
        # generation and toggling the direction here.
        if ROLL_BACK_ENABLED and not active_count:
            pass

        return [
            number_markers,
            active_prime_marker,
            contact_markers,
            title_artist,
            info_artist,
            status_artist,
            phase_artist,
            *label_artists,
            *circle_patches.values(),
            *red_point_artists.values(),
        ]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(simulation_frames),
        init_func=init,
        interval=1000 / FPS,
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    return fig, anim


def export_animation(anim: animation.FuncAnimation) -> None:
    output_path = Path(__file__).with_suffix(f".{EXPORT_FORMAT}")

    if EXPORT_FORMAT == "mp4":
        if not animation.writers.is_available("ffmpeg"):
            print("MP4 export skipped: ffmpeg writer is not available in this environment.")
            return

        writer = animation.FFMpegWriter(fps=FPS, bitrate=1800)
        anim.save(output_path, writer=writer)
        print(f"Saved animation to {output_path}")
        return

    try:
        writer = animation.PillowWriter(fps=FPS)
        anim.save(output_path, writer=writer)
        print(f"Saved animation to {output_path}")
    except Exception as exc:  # pragma: no cover - depends on local writer setup
        print(f"GIF export skipped: {exc}")


def main() -> None:
    _, anim = create_animation()

    if EXPORT_ANIMATION:
        export_animation(anim)

    plt.show()


if __name__ == "__main__":
    main()