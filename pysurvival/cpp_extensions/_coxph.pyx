from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t
import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t  # Define data type

cdef extern from "functions.h":
    cdef cppclass _CoxPHModel:
        _CoxPHModel()  # Constructor
        void fit_model(
            vector[double] times,
            vector[double] events,
            vector[vector[double]] covariates,
            vector[double] coefficients,
            double& log_likelihood
        )
        void predict_risk(
            vector[vector[double]] data,
            vector[double]& predictions
        )

cdef class CoxPHModel:
    cdef _CoxPHModel* cpp_model

    def __cinit__(self):
        """
        Initialize the C++ model.
        """
        self.cpp_model = new _CoxPHModel()
        if self.cpp_model is NULL:
            raise MemoryError("Failed to allocate memory for _CoxPHModel")

    def __dealloc__(self):
        """
        Clean up the C++ model.
        """
        if self.cpp_model is not NULL:
            del self.cpp_model

    def fit_model(self, 
                  cnp.ndarray[DTYPE_t, ndim=1] times,
                  cnp.ndarray[DTYPE_t, ndim=1] events,
                  cnp.ndarray[DTYPE_t, ndim=2] covariates):
        """
        Fit the Cox Proportional Hazards model.
        """
        cdef vector[double] c_times, c_events, c_coefficients
        cdef vector[vector[double]] c_covariates
        cdef double c_log_likelihood
        cdef vector[double] row

        # Convert 1D NumPy arrays to C++ vectors
        for i in range(times.shape[0]):
            c_times.push_back(times[i])
        for i in range(events.shape[0]):
            c_events.push_back(events[i])

        # Convert 2D NumPy array to C++ vector of vectors
        for i in range(covariates.shape[0]):
            row.clear()
            for j in range(covariates.shape[1]):
                row.push_back(covariates[i, j])
            c_covariates.push_back(row)

        # Call the C++ fit_model function
        self.cpp_model.fit_model(c_times, c_events, c_covariates, c_coefficients, c_log_likelihood)

        return np.array(c_coefficients), c_log_likelihood

    def predict_risk(self, 
                     cnp.ndarray[DTYPE_t, ndim=2] data):
        """
        Predict risk scores for new data.
        """
        cdef vector[vector[double]] c_data
        cdef vector[double] c_predictions
        cdef vector[double] row
        cdef int i, j

        # Convert 2D NumPy array to C++ vector of vectors
        for i in range(data.shape[0]):
            row.clear()
            for j in range(data.shape[1]):
                row.push_back(data[i, j])
            c_data.push_back(row)

        # Call the C++ predict_risk function
        self.cpp_model.predict_risk(c_data, c_predictions)

        # Convert predictions back to NumPy array
        return np.array(c_predictions)