# _coxph.pyx
# Updated for Python 3.10 and NumPy compatibility
# cython: language_level=3

# Define NumPy version to avoid deprecated API warnings
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from numpy cimport ndarray, float64_t, int32_t
import numpy as np
cimport numpy as cnp

# Include headers for the C++ implementation
cdef extern from "pysurvival/cpp_extensions/functions.h":
    cdef cppclass _CoxPHModel:
        _CoxPHModel()
        void fit_model(const vector[double] &times, const vector[double] &events,
                       const vector[vector[double]] &covariates, vector[double] &coefficients,
                       double &log_likelihood)
        void get_baseline_hazard(vector[double] &hazards)
        void get_baseline_survival(vector[double] &survivals)
        void predict(vector[vector[double]] &data, vector[double] &predictions)

# Python class wrapping the C++ Cox Proportional Hazards Model
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
        if self.cpp_model is not NULL:
            del self.cpp_model

    def fit_model(self, times, events, covariates):
        """
        Fit the Cox Proportional Hazards model.
        :param times: Array of time-to-event data
        :param events: Array of event indicators (1 for event, 0 for censored)
        :param covariates: 2D array of covariates
        """
        cdef vector[double] c_times, c_events
        cdef vector[vector[double]] c_covariates
        cdef vector[double] c_coefficients
        cdef double c_log_likelihood

        # Convert inputs to C++ vectors
        c_times = <vector[double]>times
        c_events = <vector[double]>events
        for row in covariates:
            c_covariates.push_back(<vector[double]>row)

        # Call the C++ fit_model function
        self.cpp_model.fit_model(c_times, c_events, c_covariates, c_coefficients, c_log_likelihood)

        return c_coefficients, c_log_likelihood

    def get_baseline_hazard(self):
        """
        Get the baseline hazard function.
        :return: Array of baseline hazard values
        """
        cdef vector[double] c_hazards
        self.cpp_model.get_baseline_hazard(c_hazards)
        return np.array(c_hazards)

    def get_baseline_survival(self):
        """
        Get the baseline survival function.
        :return: Array of baseline survival values
        """
        cdef vector[double] c_survivals
        self.cpp_model.get_baseline_survival(c_survivals)
        return np.array(c_survivals)

    def predict(self, data):
        """
        Predict survival probabilities for new data.
        :param data: 2D array of covariates
        :return: Array of survival probabilities
        """
        cdef vector[vector[double]] c_data
        cdef vector[double] c_predictions

        # Convert input to C++ vector
        for row in data:
            c_data.push_back(<vector[double]>row)

        self.cpp_model.predict(c_data, c_predictions)
        return np.array(c_predictions)