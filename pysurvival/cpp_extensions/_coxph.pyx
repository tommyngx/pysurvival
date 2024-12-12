# Import required modules
cimport numpy as np
from libcpp.vector cimport vector

# Declare C++ class for Cox Proportional Hazards Model
cdef extern from "functions.h":
    cdef cppclass _CoxPHModel:
        void fit_model(vector[double] times, vector[double] events, vector[vector[double]] covariates)
        vector[double] predict(vector[vector[double]] data)

# Define the Python wrapper class
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

    def fit(self, ndarray[np.float64_t, ndim=1] times, 
            ndarray[np.float64_t, ndim=1] events, 
            ndarray[np.float64_t, ndim=2] covariates):
        """
        Fit the Cox proportional hazards model using the provided data.
        """
        cdef vector[double] c_times
        cdef vector[double] c_events
        cdef vector[vector[double]] c_covariates
        cdef vector[double] row  # Declare vector outside the loop
        cdef int i, j

        # Convert NumPy arrays to C++ vectors
        for i in range(times.shape[0]):
            c_times.push_back(times[i])
        for i in range(events.shape[0]):
            c_events.push_back(events[i])
        for i in range(covariates.shape[0]):
            row.clear()  # Clear the vector to reuse it
            for j in range(covariates.shape[1]):
                row.push_back(covariates[i, j])
            c_covariates.push_back(row)

        self.cpp_model.fit_model(c_times, c_events, c_covariates)

    def predict(self, ndarray[np.float64_t, ndim=2] data):
        """
        Predict risk scores for the given data.
        """
        cdef vector[vector[double]] c_data
        cdef vector[double] row  # Declare vector outside the loop
        cdef vector[double] c_predictions
        cdef ndarray[np.float64_t, ndim=1] predictions
        cdef int i, j

        # Convert NumPy array to C++ vector of vectors
        for i in range(data.shape[0]):
            row.clear()  # Clear the vector to reuse it
            for j in range(data.shape[1]):
                row.push_back(data[i, j])
            c_data.push_back(row)

        c_predictions = self.cpp_model.predict(c_data)

        # Convert C++ vector to NumPy array
        predictions = np.empty(len(c_predictions), dtype=np.float64)
        for i in range(len(c_predictions)):
            predictions[i] = c_predictions[i]

        return predictions