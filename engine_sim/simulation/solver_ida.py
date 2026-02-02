"""
CVODE/SUNDIALS solver interface for multizone engine simulation.

This module provides an interface to the CVODE solver from SUNDIALS via scikits.odes.
CVODE is much more robust than scipy's BDF for extremely stiff ODE systems like
multizone combustion with detailed chemistry.
"""

import numpy as np
from typing import Dict, Callable

# Try to import scikits.odes
try:
    from scikits.odes import ode
    from scikits.odes.sundials.cvode import StatusEnum
    HAS_CVODE = True
except ImportError:
    HAS_CVODE = False
    print("Warning: scikits.odes not available. Install with: conda install -c conda-forge scikits.odes sundials")


class MultizoneCVODE:
    """
    Wrapper for CVODE solver to solve multizone engine ODE system.

    CVODE uses variable-order, variable-step BDF methods, similar to Matlab's ode15s.
    It's specifically designed for stiff ODE systems.
    """

    def __init__(self, ode_function: Callable, n_states: int, verbose: bool = True):
        """
        Initialize CVODE solver wrapper.

        Parameters
        ----------
        ode_function : callable
            Function with signature f(t, y) returning dy/dt as array
        n_states : int
            Number of state variables
        verbose : bool
            Print solver progress
        """
        if not HAS_CVODE:
            raise ImportError("scikits.odes is required for CVODE solver. "
                            "Install with: conda install -c conda-forge scikits.odes sundials")

        self.ode_function = ode_function
        self.n_states = n_states
        self.verbose = verbose
        self.nfev = 0

    def rhs_function(self, t: float, y: np.ndarray, ydot: np.ndarray) -> int:
        """
        Right-hand side function for CVODE.

        CVODE expects the RHS function to compute ydot in-place and return a status code.

        Parameters
        ----------
        t : float
            Current time
        y : ndarray
            Current state vector
        ydot : ndarray
            Output: derivative vector (computed in-place)

        Returns
        -------
        int
            0 for success, negative for error
        """
        self.nfev += 1
        try:
            # Call the ODE function
            result = self.ode_function(t, y)
            ydot[:] = result
            return 0  # Success
        except Exception as e:
            if self.verbose:
                print(f"RHS evaluation failed at t={t:.6e}: {e}")
            return -1  # Error

    def solve(self,
              y0: np.ndarray,
              t_span: tuple,
              t_eval: np.ndarray = None,
              rtol: float = 1.0e-4,
              atol: float = 1.0e-10,
              max_steps: int = 50000,
              max_order: int = 5,
              lmm_type: str = 'cvode') -> Dict:
        """
        Solve the ODE system using CVODE.

        Parameters
        ----------
        y0 : ndarray
            Initial state vector
        t_span : tuple
            (t_start, t_end)
        t_eval : ndarray, optional
            Times at which to store solution
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance (can be scalar or array)
        max_steps : int
            Maximum number of internal steps
        max_order : int
            Maximum BDF order (1-5, default 5)
        lmm_type : str
            Integrator name: 'cvode' for CVODE

        Returns
        -------
        solution : dict
            Dictionary with keys:
            - 't': time points
            - 'y': state values (2D array, each column is y at one time)
            - 'message': solver message
            - 'success': True if successful
            - 'nfev': number of function evaluations
        """
        if not HAS_CVODE:
            raise ImportError("scikits.odes not available")

        self.nfev = 0

        # Set up time points
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)
        else:
            # Ensure t_eval is a list starting with t_span[0]
            t_eval = np.array(t_eval)
            if t_eval[0] != t_span[0]:
                t_eval = np.concatenate([[t_span[0]], t_eval])

        if self.verbose:
            print(f"\nCVODE Solver (SUNDIALS) initialized:")
            print(f"  State dimension: {len(y0)}")
            print(f"  Time span: {t_span[0]:.6e} to {t_span[1]:.6e}")
            print(f"  Output points: {len(t_eval)}")
            print(f"  Method: CVODE/BDF (max order {max_order})")
            print(f"  Tolerances: rtol={rtol:.1e}, atol={atol:.1e}")
            print(f"  Max steps: {max_steps}")

        try:
            # Create CVODE solver
            solver = ode('cvode', self.rhs_function, old_api=False)

            # Set options
            solver.set_options(
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
                lband=None,  # Use dense Jacobian
                uband=None,
            )

            if self.verbose:
                print(f"  Calling solver.solve()...")

            # Solve using the solve() method
            sol = solver.solve(t_eval, y0)

            # Extract results
            if sol.flag == StatusEnum.SUCCESS:
                t_array = sol.values.t
                y_array = sol.values.y.T  # Transpose: each column is y at one time
                success = True
                message = sol.message
            else:
                # Partial success - get what we have
                t_array = sol.values.t
                y_array = sol.values.y.T
                success = False
                message = f"CVODE stopped: {sol.message} (flag={sol.flag})"

            solution = {
                't': t_array,
                'y': y_array,
                'message': message,
                'success': success,
                'nfev': self.nfev
            }

            if self.verbose:
                if success:
                    print(f"\n✓ CVODE solver completed successfully")
                else:
                    print(f"\n⚠ CVODE solver: {message}")
                print(f"  Function evaluations: {self.nfev}")
                print(f"  Output points: {len(t_array)}")

            return solution

        except Exception as e:
            if self.verbose:
                print(f"\n✗ CVODE solver failed: {e}")
                import traceback
                traceback.print_exc()

            # Return minimal solution
            solution = {
                't': np.array([t_span[0]]),
                'y': np.array([y0]).T,
                'message': f'CVODE solver failed: {str(e)}',
                'success': False,
                'nfev': self.nfev
            }
            return solution


def check_cvode_available() -> bool:
    """Check if CVODE solver is available."""
    return HAS_CVODE
