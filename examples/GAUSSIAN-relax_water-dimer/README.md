### Relaxing water dimer
In this example a water-dimer is relaxed until
ML forces are less than 0.01.
Due to the ML errors the actual ab initio forces
are slightly larger (~0.05).
In the first trial only about 10 ab initio
calculations are invoked.
The ML model is automatically saved and reused
(via `model.pckl/`) in the next trials.
Therefore as the model matures, less ab initio
calculations are needed.
