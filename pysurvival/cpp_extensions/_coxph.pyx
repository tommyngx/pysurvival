# %%cython --a
# distutils: language = c++

# Importing Numpy
# -----------------
import numpy as np
cimport numpy as cnp

# Importing Cython and C++ libraries
# -----------------------------------
import cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map
from libcpp cimport bool
from libcpp.algorithm cimport sort

# Importing C++ specific functions
# ---------------------------------
cdef extern from "functions.h" nogil:
    cdef void fit_coxph_model(
        vector[double] times,
        vector[double] events,
        vector[vector[double]] covariates,
        vector[double]& coefficients,
        double& log_likelihood
    )

# Define NumPy array types for Cython
ctypedef cnp.float64_t DTYPE_t  # 64-bit floating point

@cython.boundscheck(False)  # Turn off bounds-checking for speed
@cython.wraparound(False)   # Turn off negative indexing for speed
def fit_model(
    np.ndarray[DTYPE_t, ndim=1] times,
    np.ndarray[DTYPE_t, ndim=1] events,
    np.ndarray[DTYPE_t, ndim=2] covariates
):
    """
    Fit Cox proportional hazards model.

    Parameters:
        times: 1D NumPy array of event times
        events: 1D NumPy array of event indicators (1 = event, 0 = censoring)
        covariates: 2D NumPy array of covariates

    Returns:
        coefficients: 1D NumPy array of estimated coefficients
        log_likelihood: Log-likelihood of the fitted model
    """
    cdef vector[double] c_times = times.tolist()
    cdef vector[double] c_events = events.tolist()
    cdef vector[vector[double]] c_covariates = [row.tolist() for row in covariates]
    cdef vector[double] c_coefficients
    cdef double c_log_likelihood

    # Call C++ function
    fit_coxph_model(c_times, c_events, c_covariates, c_coefficients, c_log_likelihood)

    return np.array(c_coefficients), c_log_likelihood