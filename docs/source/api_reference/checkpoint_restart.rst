Checkpoints and restarts
=========================

For very long simulations, it is good to set
**checkpoints**. Checkpoints are files that contain all the simulation
data at one given iteration, so that the simulation can be later
**restarted** from this iteration.

Checkpoints are useful when there is a risk that the simulation
crashes before the end (e.g. because of the finite walltime on HPC
clusters). In this case, thanks to checkpoints, the simulation can be restarted
without having to run it again from the beginning.

Committed checkpoint manifests
------------------------------

``set_periodic_checkpoint`` writes the established per-rank field and particle
payloads, then invokes the small generic segment lifecycle on diagnostics that
provide it. A checkpoint is accepted only when
``<checkpoint_dir>/manifests/checkpoint<iteration>.json`` has been atomically
published with status ``committed``. The manifest identifies the run,
checkpoint, parent checkpoint, exact completed iteration, payload files, and
closed diagnostic segments. ``active.json`` selects the default restart head;
restarting an older explicit iteration moves that pointer to the selected
history, so files from a later abandoned branch are not selected implicitly.
Segment paths are stored relative to the manifest that contains them. Existing
absolute references remain readable. A checkpoint/radiation output bundle can
therefore be relocated when its internal directory layout is preserved.

A ``preparing`` manifest is published before any payload is mutated, and its
``committed`` replacement is written last. A crash can therefore leave a
prepared payload or an orphaned diagnostic file, but cannot make either appear
committed or fall back silently to legacy-checkpoint handling. Restart verifies
the exact requested iteration and the checkpoint identity in each rank's
payload. Large diagnostic accumulator arrays are not added to the normal
simulation checkpoint.

Diagnostics may implement ``close_segment(context)`` and
``start_segment(context)`` without exposing their internal state to checkpoint
code. Ordinary scheduled diagnostic writes retain their own cadence. Explicit
run finalization remains a diagnostic operation and is never implied by
returning from a Python ``Simulation.step`` call.

Setting checkpoints
-----------------------

.. autofunction:: fbpic.openpmd_diag.set_periodic_checkpoint

Restarting a simulation
--------------------------
		  
.. autofunction:: fbpic.openpmd_diag.restart_from_checkpoint

