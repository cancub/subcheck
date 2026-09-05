"""Stage 2 — sample non-overlapping snippet windows on the timeline.

Windows have a fixed duration and must not overlap; consecutive windows are
kept at least ``min_gap`` seconds apart. When the media is too short to honour
``min_gap`` for the requested number of windows, the gap shrinks automatically
(the smallest possible spacing that still fits everything) so the check still
runs.

Re-rolls (music interludes etc.) reserve their rejected window so a later
re-roll never re-picks it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence


class SamplingError(ValueError):
    """Raised when sampling is impossible for the given media."""


@dataclass(frozen=True)
class Snippet:
    """One window [start, start + duration) on the media timeline."""

    index: int
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"#{self.index} [{self.start:.2f}s -> {self.end:.2f}s]"


class SnippetSampler:
    """Draws windows of a fixed duration from a media timeline."""

    def __init__(
        self,
        total: float,
        duration: float = 10.0,
        count: int = 3,
        min_gap: float = 30.0,
        seed: Optional[int] = None,
    ) -> None:
        if duration <= 0:
            raise SamplingError(f"snippet duration must be > 0 (got {duration})")
        if count < 1:
            raise SamplingError("need at least one snippet")
        if total < duration:
            raise SamplingError(
                f"media is only {total:.2f}s long — shorter than the {duration:.1f}s "
                "snippet duration; nothing can be sampled."
            )
        self.total = float(total)
        self.duration = float(duration)
        self.count = count
        self.min_gap = float(min_gap)
        self.rng = random.Random(seed)
        self._used: List[Snippet] = []

    # ------------------------------------------------------------------ gaps
    def _feasible_gap(self) -> float:
        """Largest gap we can honour while still fitting ``count`` windows.

        Windows are non-overlapping and at least ``gap`` apart, so we need
        ``count*duration + (count-1)*gap <= total``.
        """
        if self.count <= 1:
            return self.min_gap
        room = (self.total - self.count * self.duration) / (self.count - 1)
        return max(0.0, min(self.min_gap, room))

    def _fits(self, cand: Snippet, gap: float) -> bool:
        """True when ``cand`` keeps at least ``gap`` seconds from every used window."""
        for other in self._used:
            if other.start - gap < cand.end and cand.start - gap < other.end:
                return False
        return True

    def _try_pick(self, gap: float, budget: int = 400) -> Optional[Snippet]:
        """Uniformly sample a start time that satisfies the gap constraint."""
        last_start = self.total - self.duration
        for _ in range(budget):
            start = self.rng.uniform(0.0, last_start)
            cand = Snippet(index=self._next_index(), start=start, duration=self.duration)
            if self._fits(cand, gap):
                return cand
        return None

    def _pick_with_degrade(self, start_gap: float) -> Optional[Snippet]:
        """Pick a window; relax the gap (down to plain non-overlap) until one
        fits. A plain non-overlapping window always exists while at least one
        free span of ``duration`` remains, so this terminates reliably."""
        gap = start_gap
        for _ in range(64):
            cand = self._try_pick(gap, budget=250)
            if cand is not None:
                return cand
            if gap <= 1e-6:
                break
            gap = max(0.0, gap / 2.0)
        return None

    def _fallback_grid(self) -> Optional[List[Snippet]]:
        """Deterministic escape hatch: lay *all* windows out evenly.

        Random sequential placement can fragment a tight timeline so that no
        further window fits even though a valid global layout exists (e.g. three
        10 s windows in a 40 s media: random picks may leave every free gap
        shorter than 10 s). Even spacing start_i = i*(total-duration)/(n-1)
        always leaves >= duration between neighbours whenever
        total >= n*duration, so it terminates reliably.
        """
        if self.count <= 0:
            return None
        last_start = self.total - self.duration
        if self.count == 1:
            starts = [last_start / 2.0]
        else:
            starts = [last_start * i / (self.count - 1) for i in range(self.count)]
        return [
            Snippet(index=i, start=start, duration=self.duration)
            for i, start in enumerate(starts)
        ]

    # ------------------------------------------------------------- bookkeeping
    def _next_index(self) -> int:
        used_idx = {s.index for s in self._used}
        i = 0
        while i in used_idx:
            i += 1
        return i

    def reserve(self, snippet: Snippet) -> None:
        """Permanently mark a window as taken (kept *or* rejected by re-roll)."""
        self._used.append(snippet)

    @property
    def used(self) -> Sequence[Snippet]:
        return tuple(self._used)

    # ----------------------------------------------------------------- public
    def sample_all(self) -> List[Snippet]:
        """Draw the initial set of windows."""
        picked: List[Snippet] = []
        for _ in range(self.count):
            cand = self._pick_with_degrade(self._feasible_gap())
            if cand is None:
                # Random placement got stuck on a tight timeline: restart with
                # a deterministic even spread of all windows.
                grid = self._fallback_grid()
                if grid is None:
                    raise SamplingError(
                        f"Cannot fit {self.count} windows of {self.duration:.1f}s into "
                        f"{self.total:.1f}s of media."
                    )
                self._used = []
                self._used.extend(grid)
                return sorted(grid, key=lambda s: s.start)
            self.reserve(cand)
            picked.append(cand)
        picked.sort(key=lambda s: s.start)
        return picked

    def draw_extra(self) -> Optional[Snippet]:
        """Draw one additional window avoiding every reserved one (for re-rolls).

        Returns None when no further window fits (e.g. the movie is fully
        covered or too many re-rolls already happened).
        """
        if self._feasible_gap() < 0.0:
            return None
        # With each reservation the usable room shrinks; degrade the gap in a
        # few steps until a candidate fits.
        cand = self._pick_with_degrade(self._feasible_gap())
        if cand is None:
            # last resort: scan linearly for the first free window
            step = 0.25
            t = 0.0
            while t + self.duration <= self.total:
                c = Snippet(index=self._next_index(), start=t, duration=self.duration)
                if self._fits(c, 0.0):
                    cand = c
                    break
                t += step
        if cand is None:
            return None
        self.reserve(cand)
        return cand
