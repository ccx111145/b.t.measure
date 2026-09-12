# Where the guarantee stops

The manuscript reports its own failure boundaries rather than leaving them for a reader to discover.
Each one below is reproduced by the script named beside it, so the negative results can be verified
as readily as the positive ones.

## 1. Bearing error beyond the assumed bound

The outer approximation is sound only if the true angular error stays within the assumed bound `δ`.
`exp/generalize.py` sweeps `δ` over a factor of eight and shows that the cost of over-estimating it
is below 1.5 % in the directional scenario: erring on the safe side is cheap.

## 2. Multipath and spurious bearings

A 2 % spurious-bearing rate produces false absence certificates under the hard outer approximation.
The `q`-relaxed intersection removes them — but sound **and complete** certification under `q`
outliers additionally requires a *detection redundancy* `r > q`, which the covering networks used
here do not possess. This is stated in the paper as an **open network-design problem**.

Reproduce with `exp/outlier.py`, `exp/redundancy.py`, `exp/reliability.py`.

> A related, harder case: a fixed reflector produces a **coherent ghost**, a spatially correlated
> error that the `q`-relaxed framework handles structurally poorly — no defence we tested repairs
> it. Reported as open in Section 6.9. See `exp/outlier.py` and `sim/engine.py` (ghost mode).

## 3. Antenna aperture, not SNR

With realistic antenna patterns the ideal half-plane is **not** the worst case. A narrow main lobe
with low sidelobes is harder to guarantee, because the boresight can point *between* network points:
the guaranteed fraction for the 25-point network collapses from 1.000 (ideal half-plane) to 0.842
(Yagi, −13 dB sidelobes) and 0.004 (Yagi, 40° HPBW).

The certificate-reliability map in `exp/reliability.py` shows the same effect as a function of array
aperture and per-path multipath level: **raising the per-element SNR from 6 to 20 dB leaves the
false-certificate rate essentially unchanged**, because the bearing error is dominated by multipath
*bias* rather than by noise. Aperture is the control variable.

Reproduce with `exp/antenna_pattern.py` and `exp/reliability.py`.

## 4. Intermittent transmission

A duty-cycled emitter means absence can only be certified probabilistically: the guarantee degrades
to `Pr[false absence] ≤ (1 − p_on)^R`. See `exp/reliability.py` (duty slice) and Section 6.10.

## 5. No hardware validation

The study is **entirely simulation-based**. The sensor and timing models follow a documented
specification, and the direction-finding chain is benchmarked against the Cramér–Rao bound as an
external consistency check (`exp/crlb_check.py`), but no physical experiment was performed. A bench
demonstration with a software-defined radio, or a small wheeled platform in an outdoor field, would
test the modelling assumptions directly and is the single most valuable extension of this work.
