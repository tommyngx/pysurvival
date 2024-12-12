# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.utility cimport pair

# Declare the external C++ class
cdef extern from "_coxph.cpp":
    cdef cppclass _CoxPHModel:
        _CoxPHModel()                 # Constructor
        void fit(vector[double] times, vector[double] events, vector[vector[double]] covariates)
        vector[double] predict(vector[vector[double]] covariates)

# Python wrapper class for `_CoxPHModel`
cdef class CoxPHModel:
    cdef _CoxPHModel* cpp_model

    def __cinit__(self):
        """
        Initialize the C++ model.
        """
        self.cpp_model = new _CoxPHModel()

    def __dealloc__(self):
        """
        Clean up the C++ model.
        """
        del self.cpp_model

    def fit(self, cnp.ndarray[cnp.float64_t, ndim=1] times, 
                  cnp.ndarray[cnp.float64_t, ndim=1] events, 
                  cnp.ndarray[cnp.float64_t, ndim=2] covariates):
        """
        Fit the Cox Proportional Hazards model.

        Parameters:
        - times: Array of survival times.
        - events: Array of event indicators (1 if event occurred, 0 otherwise).
        - covariates: 2D array of covariates.
        """
        cdef vector[double] c_times = times.tolist()
        cdef vector[double] c_events = events.tolist()
        cdef vector[vector[double]] c_covariates = [[cov for cov in row] for row in covariates.tolist()]
        self.cpp_model.fit(c_times, c_events, c_covariates)

    def predict(self, cnp.ndarray[cnp.float64_t, ndim=2] covariates):
        """
        Predict using the fitted model.

        Parameters:
        - covariates: 2D array of covariates.

        Returns:
        - A list of predicted values.
        """
        cdef vector[vector[double]] c_covariates = [[cov for cov in row] for row in covariates.tolist()]
        cdef vector[double] predictions = self.cpp_model.predict(c_covariates)
        return list(predictions)