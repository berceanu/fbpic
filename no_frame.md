# A Single Frame-Agnostic Synchrotron Radiation Diagnostic

## Current state

The current branch has one public `SynchrotronRadiationDiagnostic`, backed by
`ObserverFrameRadiationAccumulator` and `ObserverRadiationWriter`. It can
already describe both relevant cases:

- `observer_frame="laboratory"` transforms simulation-frame events using the
  configured longitudinal boost.
- `observer_frame="simulation"` replaces that transform with the identity.
- In an ordinary non-boosted simulation, the inferred boost is already the
  identity (`gamma=1`, `beta=0`), so laboratory and simulation frames coincide.

The diagnostic supports the same configurable channels in either case:
spectral-angular output, observer-time detectors, source projections, source
moments, accounting, interval/cumulative output, sampling, and selections.
The working tree also captures radiation from the momentum immediately before
and after a completed particle push, rather than maintaining a separate
field-derived boosted calculation.

Despite this, some frame-related ownership remains unnecessarily distributed:

- The boost can be supplied both to `activate_synchrotron` and to the
  diagnostic, creating two possible sources of truth.
- `gamma_cutoff` is accepted at activation even though it is an observer-frame
  selection.
- A radiator stores only one `observer_accumulator`, so a species can be
  attached to only one SR diagnostic.
- Frame names and boost parameters reach relatively far into the accumulator;
  this makes future frame-specific branches tempting.

The older non-boosted implementation used different grids, output semantics,
and instantaneous gathered fields, and also mixed passive diagnostics with
radiation reaction. Since backward compatibility is not required, retaining
that implementation or a legacy dispatch path would only preserve divergence.

## Recommended design

Use one event pipeline and treat a boost as an ordinary coordinate transform:

```text
completed pusher impulse in simulation coordinates
                     |
                     v
          simulation-to-observer transform
             (identity when appropriate)
                     |
                     v
       shared cuts, products, reduction, and writer
```

The simulation-side event should contain the integer-centered source position
and time, particle weight and identity, and momentum before and after the push.
It should have no boosted/non-boosted variant. At diagnostic construction,
resolve the requested observer frame into one concrete transform object. All
downstream code should consume the transformed event and should not branch on
whether the simulation was boosted.

The public API can remain explicit and readable:

```python
electrons.activate_synchrotron()

radiation = SynchrotronRadiationDiagnostic(
    species={"electrons": electrons},
    observer_frame="laboratory",
    boost=sim.boost,  # None means the identity transform
    # product configuration ...
)
```

Internally, `observer_frame` and `boost` should immediately become a resolved
simulation-to-observer transform. For a non-boosted laboratory simulation, or
for `observer_frame="simulation"`, this is the identity. Passing a nonidentity
boost with `observer_frame="simulation"` should remain an error.

`activate_synchrotron` should only enable event capture and configure genuinely
shared numerical machinery such as the spectral table accuracy. Observer-frame
choices—including `gamma_cutoff`, grids, detectors, cuts, and output
channels—belong exclusively to `SynchrotronRadiationDiagnostic`. Radiation
reaction should be a separate state-changing particle operator, not a mode of
the passive diagnostic.

It is also preferable for the radiator to fan one captured impulse out to a
list of diagnostic consumers. The normal interface can still encourage one
comprehensive diagnostic, while permitting simultaneous laboratory- and
simulation-frame diagnostics without repeating the momentum capture.

## Required invariants

The design should be protected by tests that establish:

- An identity transform produces the same result in an ordinary laboratory
  simulation for both supported frame descriptions.
- The same physical trajectory represented in laboratory and boosted
  simulation coordinates produces equivalent laboratory-frame products.
- Every channel uses the same transformed event and observer-frame cuts.
- CPU/GPU and serial/MPI paths preserve accounting and deterministic sampling
  within their documented tolerances.
- Diagnostic cadence and final flushing never omit or double-count a completed
  pusher impulse.

This yields one SR diagnostic, one physical event definition, one product
implementation, and one output schema. “Boosted” is then only a property of the
simulation-to-observer transform, not a second diagnostic mode.
