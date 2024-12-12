# %%cython --a
# distutils: language = c++

# Importing Numpy
# -----------------
import numpy as np  # Standard Python import for NumPy
cimport numpy as cnp  # Cython import for NumPy

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
    cnp.ndarray[DTYPE_t, ndim=1] times,       # Change np.ndarray to cnp.ndarray
    cnp.ndarray[DTYPE_t, ndim=1] events,      # Change np.ndarray to cnp.ndarray
    cnp.ndarray[DTYPE_t, ndim=2] covariates   # Change np.ndarray to cnp.ndarray
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
    cdef double c_log_likelihood = 0.0  # Initialize variable

    # Call C++ function
    fit_coxph_model(c_times, c_events, c_covariates, c_coefficients, c_log_likelihood)

    return np.array(c_coefficients), c_log_likelihood


cdef extern from "_coxph.cpp":
    cdef cppclass _CoxPHModel:
        _CoxPHModel()
        void fit(vector[double] times, vector[double] events, vector[vector[double]] covariates)
        vector[double] predict(vector[vector[double]] covariates)

cdef class CoxPHModel:
    cdef _CoxPHModel* cpp_model

    def __cinit__(self):
        self.cpp_model = new _CoxPHModel()

    def __dealloc__(self):
        del self.cpp_model

    def fit(self, np.ndarray[double, ndim=1] times, np.ndarray[double, ndim=1] events, np.ndarray[double, ndim=2] covariates):
        cdef vector[double] c_times = times.tolist()
        cdef vector[double] c_events = events.tolist()
        cdef vector[vector[double]] c_covariates = [[cov for cov in row] for row in covariates.tolist()]
        self.cpp_model.fit(c_times, c_events, c_covariates)

    def predict(self, np.ndarray[double, ndim=2] covariates):
        cdef vector[vector[double]] c_covariates = [[cov for cov in row] for row in covariates.tolist()]
        cdef vector[double] predictions = self.cpp_model.predict(c_covariates)
        return list(predictions)