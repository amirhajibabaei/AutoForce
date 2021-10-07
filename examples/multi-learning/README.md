### Learning two ab initio PES simultaneously

In this example, a ML model is generated on-the-fly for the gold crystal
using the EMT-PES (see `calc_emt.py`) and simultanously another model is 
generated for the DFT-PES (see `calc_vasp.py`).
Note that EMT is the dynamics driver and DFT is learned based on cofigurations
reached by EMT-MD.
