### LAMMPS

#### Coupled with on-the-fly ML-IAP using a Airebo potential (https://docs.lammps.org/pair_airebo.html).


# The LAMMPS related settings:


#pair_style                 airebo  6 0 0
#pair_coeff                 * * CH.airebo C

#or

#pair_style                 airebo/morse 6 0 0 
#pair_coeff                 * * CH.airebo-m C

#or

#pair_style                 rebo         
#pair_coeff                 * * CH.rebo C

```
python run_md.py
```



