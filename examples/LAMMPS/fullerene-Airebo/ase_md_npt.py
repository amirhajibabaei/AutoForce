'''Constant pressure/stress and temperature dynamics.

Combined Nose-Hoover and Parrinello-Rahman dynamics, creating an NPT
(or N,stress,T) ensemble.

The method is the one proposed by Melchionna et al. [1] and later
modified by Melchionna [2].  The differential equations are integrated
using a centered difference method [3].

 1. S. Melchionna, G. Ciccotti and B. L. Holian, "Hoover NPT dynamics
    for systems varying in shape and size", Molecular Physics 78, p. 533
    (1993).

 2. S. Melchionna, "Constrained systems and statistical distribution",
    Physical Review E, 61, p. 6165 (2000).

 3. B. L. Holian, A. J. De Groot, W. G. Hoover, and C. G. Hoover,
    "Time-reversible equilibrium and nonequilibrium isothermal-isobaric
    simulations with centered-difference Stoermer algorithms.", Physical
    Review A, 41, p. 4552 (1990).
'''

import sys
import weakref

import numpy as np

from ase.md.md import MolecularDynamics
from ase import units

linalg = np.linalg

# Delayed imports:  If the trajectory object is reading a special ASAP version
# of HooverNPT, that class is imported from Asap.Dynamics.NPTDynamics.


class NPT3(MolecularDynamics):

    classname = "NPT3"  # Used by the trajectory.
    _npt_version = 3   # Version number, used for Asap compatibility.

    def __init__(self, atoms,
                 timestep, 
                 temperature=None, 
                 externalstress=None,
                 ttime=None, 
                 pfactor=None,
                 anisotropic=True,
                 trajectory=None, 
                 logfile=None, 
                 loginterval=1,
                 append_trajectory=False):
        '''Constant pressure/stress and temperature dynamics.

        Combined Nose-Hoover and Parrinello-Rahman dynamics, creating an
        NPT (or N,stress,T) ensemble.

        The method is the one proposed by Melchionna et al. [1] and later
        modified by Melchionna [2].  The differential equations are integrated
        using a centered difference method [3].  See also NPTdynamics.tex

        The dynamics object is called with the following parameters:

        atoms: Atoms object
            The list of atoms.

        timestep: float
            The timestep in units matching eV, Å, u.

        temperature: float (deprecated)
            The desired temperature in eV.

        temperature_K: float
            The desired temperature in K.

        externalstress: float or nparray
            The external stress in eV/A^3.  Either a symmetric
            3x3 tensor, a 6-vector representing the same, or a
            scalar representing the pressure.  Note that the
            stress is positive in tension whereas the pressure is
            positive in compression: giving a scalar p is
            equivalent to giving the tensor (-p, -p, -p, 0, 0, 0).

        ttime: float
            Characteristic timescale of the thermostat, in ASE internal units
            Set to None to disable the thermostat.

            WARNING: Not specifying ttime sets it to None, disabling the
            thermostat.            

        pfactor: float
            A constant in the barostat differential equation.  If
            a characteristic barostat timescale of ptime is
            desired, set pfactor to ptime^2 * B (where ptime is in units matching
            eV, Å, u; and B is the Bulk Modulus, given in eV/Å^3).
            Set to None to disable the barostat.
            Typical metallic bulk moduli are of the order of
            100 GPa or 0.6 eV/A^3.  

            WARNING: Not specifying pfactor sets it to None, disabling the
            barostat.

        mask: None or 3-tuple or 3x3 nparray (optional)
            Optional argument.  A tuple of three integers (0 or 1),
            indicating if the system can change size along the
            three Cartesian axes.  Set to (1,1,1) or None to allow
            a fully flexible computational box.  Set to (1,1,0)
            to disallow elongations along the z-axis etc.
            mask may also be specified as a symmetric 3x3 array
            indicating which strain values may change.

        Useful parameter values:

        * The same timestep can be used as in Verlet dynamics, i.e. 5 fs is fine
          for bulk copper.

        * The ttime and pfactor are quite critical[4], too small values may
          cause instabilites and/or wrong fluctuations in T / p.  Too
          large values cause an oscillation which is slow to die.  Good
          values for the characteristic times seem to be 25 fs for ttime,
          and 75 fs for ptime (used to calculate pfactor), at least for
          bulk copper with 15000-200000 atoms.  But this is not well
          tested, it is IMPORTANT to monitor the temperature and
          stress/pressure fluctuations.


        References:

        1) S. Melchionna, G. Ciccotti and B. L. Holian, Molecular
           Physics 78, p. 533 (1993).

        2) S. Melchionna, Physical
           Review E 61, p. 6165 (2000).

        3) B. L. Holian, A. J. De Groot, W. G. Hoover, and C. G. Hoover,
           Physical Review A 41, p. 4552 (1990).

        4) F. D. Di Tolla and M. Ronchetti, Physical
           Review E 48, p. 1726 (1993).

        '''

        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval,
                                   append_trajectory=append_trajectory)
                                   #stress=True)
        # self.atoms = atoms
        # self.timestep = timestep
        if externalstress is None and pfactor is not None:
            raise TypeError("Missing 'externalstress' argument.")
        self.zero_center_of_mass_momentum(verbose=1)
        self.temperature = temperature 
        self.set_stress(externalstress)
        
        self.eta = np.zeros((3), float)
        self.zeta = 0.0
        self.zeta_integrated = 0.0
        self.initialized = 0
        self.ttime = ttime
        self.pfactor_given = pfactor
        self.anisotropic = anisotropic
        self._calculateconstants()
        self.timeelapsed = 0.0
        self.frac_traceless = 1


    def set_stress(self, stress):
        """Set the applied stress.

        Must be a symmetric 3x3 tensor, a 6-vector representing a symmetric
        3x3 tensor, or a number representing the pressure.

        Use with care, it is better to set the correct stress when creating
        the object.
        """

        if np.isscalar(stress):
            stress = np.array([-stress, -stress, -stress])
        else:
            stress = np.array(stress)
            if stress.shape != (3,):
                raise ValueError("The external stress has the wrong shape.")
        self.externalstress = stress


    def set_fraction_traceless(self, fracTraceless):
        """set what fraction of the traceless part of the force
        on eta is kept.

        By setting this to zero, the volume may change but the shape may not.
        """
        self.frac_traceless = fracTraceless

    def get_strain_rate(self):
        """Get the strain rate as an upper-triangular 3x3 matrix.
        This includes the fluctuations in the shape of the computational box.
        """
        return np.array(self.eta, copy=1)

    def set_strain_rate(self, rate):
        """Set the strain rate.  Must be an upper triangular 3x3 matrix.
        If you set a strain rate along a direction that is "masked out"
        (see ``set_mask``), the strain rate along that direction will be
        maintained constantly.
        """
        if not (rate.shape == (3,) and self._isuppertriangular(rate)):
            raise ValueError("Strain rate must be an upper triangular matrix.")
        self.eta = rate
        if self.initialized:
            # Recalculate h_past and eta_past so they match the current value.
            self._initialize_eta_h()

    def get_time(self):
        "Get the elapsed time."
        return self.timeelapsed

    def run(self, steps):
        """Perform a number of time steps."""
        if not self.initialized:
            self.initialize()
        else:
            if self.have_the_atoms_been_changed():
                raise NotImplementedError(
                    "You have modified the atoms since the last timestep.")

        for i in range(steps):
            self.step()
            self.nsteps += 1
            self.call_observers()

    def have_the_atoms_been_changed(self):
        "Checks if the user has modified the positions or momenta of the atoms"
        limit = 1e-10
        h = self._getbox()
        if max(abs((h - self.h).ravel())) > limit:
            self._warning("The computational box has been modified.")
            return 1
        expected_r = np.dot(self.q + 0.5, h)
        err = max(abs((expected_r - self.atoms.get_positions()).ravel()))
        if err > limit:
            self._warning("The atomic positions have been modified: " +
                          str(err))
            return 1
        return 0

    def step(self):
        """Perform a single time step.

        Assumes that the forces and stresses are up to date, and that
        the positions and momenta have not been changed since last
        timestep.
        """

        # Assumes the following variables are OK
        # q_past, q, q_future, p, eta, eta_past, zeta, zeta_past, h, h_past
        #
        # q corresponds to the current positions
        # p must be equal to self.atoms.GetCartesianMomenta()
        # h must be equal to self.atoms.GetUnitCell()
        #
        # print "Making a timestep"
        dt = self.dt
        h_future = self.h_past + 2 * dt * np.einsum('i,i->i', self.h, self.eta)
        
        if self.pfactor_given is None:
            deltaeta = np.zeros(3, float)
        elif self.anisotropic:
            stress = self.stresscalculator()
            volbox = self.atoms.get_volume()
            deltaeta = -2 * dt * (self.pfact * volbox *
                                  (stress - self.externalstress))
        else: # isotropic: 
            aniso_stress = self.stresscalculator()
            stress = aniso_stress.mean()*np.ones (3, float)
            volbox = self.atoms.get_volume()
            deltaeta = -2 * dt * (self.pfact * volbox *
                                  (stress - self.externalstress))

        
        eta_future = self.eta_past + deltaeta
        
        deltazeta = 2 * dt * self.tfact * (self.atoms.get_kinetic_energy() -
                                           self.desiredEkin)
        zeta_future = self.zeta_past + deltazeta
        # Advance time
        # print "Max change in scaled positions:", max(abs(self.q_future.flat - self.q.flat))
        # print "Max change in basis set", max(abs((h_future - self.h).flat))
        self.timeelapsed += dt
        self.h_past = self.h
        self.h = h_future
        self.q_past = self.q
        self.q = self.q_future
        self._setbox_and_positions(self.h, self.q)
        self.eta_past = self.eta
        self.eta = eta_future
        self.zeta_past = self.zeta
        self.zeta = zeta_future
        self._synchronize()  # for parallel simulations.
        self.zeta_integrated += dt * self.zeta
        force = self.forcecalculator()
        self._calculate_q_future(force)
        p0 = self.atoms.cell.cartesian_positions (self.q_future - self.q_past)*self._getmasses()/(2.0*dt)
        self.atoms.set_momenta(p0)

        # self.stresscalculator()

    def forcecalculator(self):
        return self.atoms.get_forces(md=True)

    def stresscalculator(self):
        return self.atoms.get_stress(include_ideal_gas=True)[:3]

    def initialize(self):
        """Initialize the dynamics.

        The dynamics requires positions etc for the two last times to
        do a timestep, so the algorithm is not self-starting.  This
        method performs a 'backwards' timestep to generate a
        configuration before the current.

        This is called automatically the first time ``run()`` is called.
        """
        # print "Initializing the NPT dynamics."
        dt = self.dt
        atoms = self.atoms
        xbox, ybox, zbox, alpha, beta, gamma = atoms.cell.cellpar()
        self.h = np.array([xbox, ybox, zbox]) 
        self.h_angle = np.array ([alpha, beta, gamma])

        #scaled coordinates
        crds = self.atoms.get_positions()
        self.q = self.atoms.cell.scaled_positions(crds)
        
        self._initialize_eta_h()
        deltazeta = dt * self.tfact * (atoms.get_kinetic_energy() -
                                       self.desiredEkin)
        self.zeta_past = self.zeta - deltazeta
        self._calculate_q_past_and_future()
        self.initialized = 1

    def get_gibbs_free_energy(self):
        """Return the Gibb's free energy, which is supposed to be conserved.

        Requires that the energies of the atoms are up to date.

        This is mainly intended as a diagnostic tool.  If called before the
        first timestep, Initialize will be called.
        """
        if not self.initialized:
            self.initialize()
        n = self._getnatoms()
        # tretaTeta = sum(diagonal(matrixmultiply(transpose(self.eta),
        #                                        self.eta)))
        contractedeta = np.sum((self.eta * self.eta).ravel())
        volbox = self.atoms.get_volume()
        gibbs = (self.atoms.get_potential_energy() +
                 self.atoms.get_kinetic_energy()
                 - np.sum(self.externalstress[0:3]) * volbox / 3.0)
        if self.ttime is not None:
            gibbs += (1.5 * n * self.temperature *
                      (self.ttime * self.zeta)**2 +
                      3 * self.temperature * (n - 1) * self.zeta_integrated)
        else:
            assert self.zeta == 0.0
        if self.pfactor_given is not None:
            gibbs += 0.5 / self.pfact * contractedeta
        else:
            assert contractedeta == 0.0
        return gibbs

    def get_center_of_mass_momentum(self):
        "Get the center of mass momentum."
        return self.atoms.get_momenta().sum(0)

    def zero_center_of_mass_momentum(self, verbose=0):
        "Set the center of mass momentum to zero."
        cm = self.get_center_of_mass_momentum()
        abscm = np.sqrt(np.sum(cm * cm))
        if verbose and abscm > 1e-4:
            self._warning(
                self.classname +
                ": Setting the center-of-mass momentum to zero "
                "(was %.6g %.6g %.6g)" % tuple(cm))
        self.atoms.set_momenta(self.atoms.get_momenta() -
                               cm / self._getnatoms())


    def attach(self, function, interval=1, *args, **kwargs):
        """Attach callback function or trajectory.

        At every *interval* steps, call *function* with arguments
        *args* and keyword arguments *kwargs*.

        If *function* is a trajectory object, its write() method is
        attached, but if *function* is a BundleTrajectory (or another
        trajectory supporting set_extra_data(), said method is first
        used to instruct the trajectory to also save internal
        data from the NPT dynamics object.
        """
        if hasattr(function, "set_extra_data"):
            # We are attaching a BundleTrajectory or similar
            function.set_extra_data("npt_init",
                                    WeakMethodWrapper(self, "get_init_data"),
                                    once=True)
            function.set_extra_data("npt_dynamics",
                                    WeakMethodWrapper(self, "get_data"))
        MolecularDynamics.attach(self, function, interval, *args, **kwargs)

    def get_init_data(self):
        "Return the data needed to initialize a new NPT dynamics."
        return {'dt': self.dt,
                'temperature': self.temperature,
                'desiredEkin': self.desiredEkin,
                'externalstress': self.externalstress,
                'mask': self.mask,
                'ttime': self.ttime,
                'tfact': self.tfact,
                'pfactor_given': self.pfactor_given,
                'pfact': self.pfact,
                'frac_traceless': self.frac_traceless}

    def get_data(self):
        "Return data needed to restore the state."
        return {'eta': self.eta,
                'eta_past': self.eta_past,
                'zeta': self.zeta,
                'zeta_past': self.zeta_past,
                'zeta_integrated': self.zeta_integrated,
                'h': self.h,
                'h_past': self.h_past,
                'timeelapsed': self.timeelapsed}

    
    def _getbox(self):
        "Get the computational box."
        return self.atoms.get_cell()

    def _getmasses(self):
        "Get the masses as an Nx1 array."
        return np.reshape(self.atoms.get_masses(), (-1, 1))

    def _separatetrace(self, mat):
        """return two matrices, one proportional to the identity
        the other traceless, which sum to the given matrix
        """
        tracePart = ((mat[0][0] + mat[1][1] + mat[2][2]) / 3.) * np.identity(3)
        return tracePart, mat - tracePart

    # A number of convenient helper methods
    def _warning(self, text):
        "Emit a warning."
        sys.stderr.write("WARNING: " + text + "\n")
        sys.stderr.flush()

    def _calculate_q_future(self, force):
        """Calculate future q.  
            Needed in Timestep and Initialization."""
        dt = self.dt
        
        alpha = (dt*dt)*self.atoms.cell.scaled_positions (force/self._getmasses())
        beta = dt *(self.eta + 0.5*self.zeta) # (x,y,z)
        inv_b = 1.0/(beta + 1.0) # linalg.inv(beta + id3)
        q0 = np.einsum('ij,j->ij', self.q_past,beta-1.0)
        self.q_future = np.einsum('ij,j->ij', 2*self.q + q0 + alpha,inv_b)
        

    def _calculate_q_past_and_future(self):
        def ekin(p, m=self.atoms.get_masses()):
            p2 = np.sum(p * p, -1)
            return 0.5 * np.sum(p2 / m) / len(m)
        p0 = self.atoms.get_momenta()
        m = self._getmasses()
        p = np.array(p0, copy=1)
        dt = self.dt
        pos = self.atoms.cell.cartesian_positions(self.q)
        for i in range(2):
            self.q_past = self.atoms.cell.scaled_positions(pos-dt*p/m)
            self._calculate_q_future(self.atoms.get_forces(md=True))
            dpos = self.atoms.cell.cartesian_positions(self.q_future-self.q_past)
            p = dpos/ (2 * dt) * m
            e = ekin(p)
            if e < 1e-5:
                # The kinetic energy and momenta are virtually zero
                return
            p = (p0 - p) + p0

    def _initialize_eta_h(self):
        #self.h_past = self.h - self.dt * np.dot(self.h, self.eta)
        self.h_past = self.h - self.dt * np.einsum('i,i->i', self.h, self.eta)
        if self.pfactor_given is None:
            deltaeta = np.zeros(3, float)
        elif self.anisotropic:
            # volbox = linalg.det(self.h)
            volbox = self.atoms.get_volume()
            deltaeta = (-self.dt * self.pfact * volbox
                        * (self.stresscalculator() - self.externalstress))
        else:
            volbox = self.atoms.get_volume()
            aniso_stress = self.stresscalculator()
            stress = aniso_stress.mean()*np.ones (3, float)
            deltaeta = (-self.dt * self.pfact * volbox
                        * (stress - self.externalstress))
        self.eta_past = self.eta - deltaeta 
        

    @staticmethod
    def _isuppertriangular(m) -> bool:
        "Check that a matrix is on upper triangular form."
        return m[1, 0] == m[2, 0] == m[2, 1] == 0.0

    def _calculateconstants(self):
        "(Re)calculate some constants when pfactor, ttime or temperature have been changed."
        n = self._getnatoms()
        if self.ttime is None:
            self.tfact = 0.0
        else:
            self.tfact = 2.0 / (3 * n * self.temperature *
                                self.ttime * self.ttime)
        if self.pfactor_given is None:
            self.pfact = 0.0
        else:
            volbox = self.atoms.get_volume()
            self.pfact = 1.0 / (self.pfactor_given * volbox)
            # self.pfact = 1.0/(n * self.temperature * self.ptime * self.ptime)
        self.desiredEkin = 1.5 * (n - 1) * self.temperature

    def _setbox_and_positions(self, h, q):
        """Set the computational box and the positions."""
        cell = np.array (list(h)+list(self.h_angle))
        self.atoms.set_cell(cell)
        r = self.atoms.cell.cartesian_positions(q)
        self.atoms.set_positions(r)

    # A few helper methods, which have been placed in separate methods
    # so they can be replaced in the parallel version.
    def _synchronize(self):
        """Synchronizes eta, h and zeta on all processors in a parallel simulation.

        In a parallel simulation, eta, h and zeta are communicated
        from the master to all slaves, to prevent numerical noise from
        causing them to diverge.

        In a serial simulation, do nothing.
        """
        pass  # This is a serial simulation object.  Do nothing.

    def _getnatoms(self):
        """Get the number of atoms.

        In a parallel simulation, this is the total number of atoms on all
        processors.
        """
        return len(self.atoms)

    def _make_special_q_arrays(self):
        """Make the arrays used to store data about the atoms.

        In a parallel simulation, these are migrating arrays.  In a
        serial simulation they are ordinary Numeric arrays.
        """
        natoms = len(self.atoms)
        self.q = np.zeros((natoms, 3), float)
        self.q_past = np.zeros((natoms, 3), float)
        self.q_future = np.zeros((natoms, 3), float)


class WeakMethodWrapper:
    """A weak reference to a method.

    Create an object storing a weak reference to an instance and
    the name of the method to call.  When called, calls the method.

    Just storing a weak reference to a bound method would not work,
    as the bound method object would go away immediately.
    """

    def __init__(self, obj, method):
        self.obj = weakref.proxy(obj)
        self.method = method

    def __call__(self, *args, **kwargs):
        m = getattr(self.obj, self.method)
        return m(*args, **kwargs)



if __name__ == "__main__":
    import sys
    from ase.io import read 
    from theforce.util.parallel import mpi_init
    from theforce.util.aseutil import init_velocities 
    from ase_md_logger import MDLogger3
    from ase_mbx import MBX
    from ase.optimize import BFGS 

    process_group = mpi_init()

    '''
    #atoms = read ('Ice2_Liq_Code/md_restart.traj', index=-1)
    atoms = read ('Ice2_Liq_Code/md_nph_T_m08.traj', index=-1)
    cell = atoms.get_cell ()
    print (cell)
    xbox, ybox, zbox, alpha, beta, gamma = atoms.cell.cellpar()
    print (xbox, ybox, zbox, alpha, beta, gamma)
    atoms.set_cell ([xbox, ybox, zbox, alpha, beta, gamma])
    cell2 = atoms.get_cell()
    print (cell2)
    '''
   
    '''
    atoms = read ('Ice2_Liq/hfixed_ice2_half.xyz')
    cell = np.array ([[23.34, 0.0, 0.0], 
                      [-9.157, 21.468, 0.0], 
                      [1.9735, 2.987, 21.589]])
    '''
    atoms = read ('Ice2_Liq/ice2_liq_small.xyz')
    cell = np.array ([[23.34, 0.0, 0.0], 
                      [-9.157, 21.468, 0.0], 
                      [2.7629, 4.1818, 30.2246]])

    '''
    atoms = read ('water.xyz')
    cell = np.array ([24.84, 24.84, 24.84, 90.0, 90.0, 90.0])
    '''
    atoms.set_cell (cell)
    atoms.pbc = [1,1,1]
    
    #atoms = read ('md.traj', index=-1)
    #xbox, ybox, zbox, alpha, beta, gamma = atoms.cell.cellpar()

    atoms.calc = MBX()

    opt = BFGS (atoms, logfile='bfgs_opt.out')
    opt.run (fmax=1.0)

    dt = 0.5*units.fs 
    ttime = 25.0*units.fs
    ptime = 100.0*units.fs 
    bulk_modulus = 137.0
    pfactor = (ptime**2) * bulk_modulus * units.GPa
    temperature_K = 250
    temperature = temperature_K * units.kB
    external_stress = 0.5 * units.GPa
    init_velocities(atoms, temperature_K)

    dyn = NPT3 (atoms,
                dt,
                temperature=temperature, #temperature_K,
                externalstress=external_stress,
                ttime=ttime,
                pfactor=pfactor,
                anisotropic=True,
                trajectory='bomd_ice2_liq_T250_P05.traj',
                logfile=None,
                append_trajectory=False,
                loginterval=100)
    
    logger = MDLogger3 (dyn=dyn, atoms=atoms, logfile='bomd_ice2_liq_T250_P05.dat', stress=True)
    dyn.attach (logger, 5)
    dyn.run (10000)
