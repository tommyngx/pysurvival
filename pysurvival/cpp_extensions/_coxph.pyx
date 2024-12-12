# _coxph.pyx
# Updated for Python 3.10 and Cython compatibility
# cython: language_level=3

# Define NumPy version to avoid deprecated API warnings
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from numpy cimport ndarray, float64_t
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
        self.cpp_model = <_CoxPHModel*>malloc(sizeof(_CoxPHModel))
        if self.cpp_model is NULL:
            raise MemoryError("Failed to allocate memory for _CoxPHModel")
        self.cpp_model = _CoxPHModel()

    def __dealloc__(self):
        """
        Clean up the C++ model.
        """
        if self.cpp_model is not NULL:
            del self.cpp_model

    # Fit the model
    def fit_model(self, ndarray[float64_t, ndim=1] times, 
                ndarray[float64_t, ndim=1] events, 
                ndarray[float64_t, ndim=2] covariates):
        """
        Fit the Cox Proportional Hazards model.
        """
        cdef vector[double] c_times, c_events, c_coefficients
        cdef vector[vector[double]] c_covariates
        cdef vector[double] row  # Declare the vector here
        cdef double c_log_likelihood

        # Convert 1D NumPy arrays to C++ vectors
        for i in range(times.shape[0]):
            c_times.push_back(times[i])
        for i in range(events.shape[0]):
            c_events.push_back(events[i])

        # Convert 2D NumPy array to C++ vector of vectors
        for i in range(covariates.shape[0]):
            row.clear()  # Clear the vector before reuse
            for j in range(covariates.shape[1]):
                row.push_back(covariates[i, j])
            c_covariates.push_back(row)

        # Call the C++ fit_model function
        self.cpp_model.fit_model(c_times, c_events, c_covariates, c_coefficients, c_log_likelihood)

        return np.array(c_coefficients), c_log_likelihood

    def get_baseline_hazard(self):
        """
        Get the baseline hazard function.
        :return: Array of baseline hazard values
        """
        cdef vector[double] c_hazards
        self.cpp_model.get_baseline_hazard(c_hazards)
        return np.array([c_hazards[i] for i in range(len(c_hazards))])

    def get_baseline_survival(self):
        """
        Get the baseline survival function.
        :return: Array of baseline survival values
        """
        cdef vector[double] c_survivals
        self.cpp_model.get_baseline_survival(c_survivals)
        return np.array([c_survivals[i] for i in range(len(c_survivals))])

    def predict(self, ndarray[float64_t, ndim=2] data):
        """
        Predict survival probabilities for new data.
        :param data: 2D array of covariates
        :return: Array of survival probabilities
        """
        cdef vector[vector[double]] c_data
        cdef vector[double] c_predictions

        # Convert 2D NumPy array to C++ vector of vectors
        cdef vector[double] row  # Declare the vector outside the loop
        for i in range(covariates.shape[0]):
            row.clear()  # Clear the vector for each row
            for j in range(covariates.shape[1]):
                row.push_back(covariates[i, j])
            c_covariates.push_back(row)

        self.cpp_model.predict(c_data, c_predictions)
        return np.array([c_predictions[i] for i in range(len(c_predictions))])