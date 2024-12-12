# Updated _coxph.pyx for proper initialization and memory handling

from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.utility cimport pair
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from numpy cimport ndarray
import numpy as np
cdef extern from "functions.h":
    cdef cppclass _CoxPHModel:
        _CoxPHModel() except +
        void fit_model(vector[double], vector[double], vector[vector[double]])
        double compute_log_likelihood()
        vector[double] predict(vector[vector[double]])
        void clear()

cdef class CoxPHModel:
    cdef _CoxPHModel* cpp_model

    def __cinit__(self):
        self.cpp_model = new _CoxPHModel()
        if not self.cpp_model:
            raise MemoryError("Failed to allocate memory for _CoxPHModel")

    def __dealloc__(self):
        if self.cpp_model:
            del self.cpp_model

    def fit(self, 
            ndarray[np.float64_t, ndim=1] times, 
            ndarray[np.float64_t, ndim=1] events, 
            ndarray[np.float64_t, ndim=2] covariates):
        """
        Fit the Cox proportional hazards model using the provided data.
        """
        cdef vector[double] c_times = vector[double](times.shape[0])
        cdef vector[double] c_events = vector[double](events.shape[0])
        cdef vector[vector[double]] c_covariates
        cdef vector[double] row  # Declare the row vector here
        cdef int i, j

        # Convert 1D NumPy arrays to C++ vectors
        for i in range(times.shape[0]):
            c_times.push_back(times[i])
            c_events.push_back(events[i])

        # Convert 2D NumPy array to C++ vector of vectors
        for i in range(covariates.shape[0]):
            row.clear()  # Clear the vector to reuse it
            for j in range(covariates.shape[1]):
                row.push_back(covariates[i, j])
            c_covariates.push_back(row)

        # Fit the model using the C++ implementation
        self.cpp_model.fit_model(c_times, c_events, c_covariates)

    def compute_log_likelihood(self) -> float:
        """
        Compute the log-likelihood of the model.
        """
        return self.cpp_model.compute_log_likelihood()

    def predict(self, ndarray[np.float64_t, ndim=2] data):
        """
        Predict using the Cox proportional hazards model.
        """
        cdef vector[vector[double]] c_data
        cdef vector[double] c_predictions
        cdef ndarray[np.float64_t, ndim=1] predictions
        cdef int i, j

        # Convert NumPy array to C++ vector of vectors
        for i in range(data.shape[0]):
            cdef vector[double] row
            for j in range(data.shape[1]):
                row.push_back(data[i, j])
            c_data.push_back(row)

        c_predictions = self.cpp_model.predict(c_data)

        # Convert C++ vector back to NumPy array
        predictions = np.empty(len(c_predictions), dtype=np.float64)
        for i in range(len(c_predictions)):
            predictions[i] = c_predictions[i]

        return predictions