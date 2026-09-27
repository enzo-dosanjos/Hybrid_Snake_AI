/*************************************************************************
Test - Provides assertions and named test cases for the component suites
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//------------- Interface of the module <Test> (file Test.h) -------------

#ifndef TEST_H
#define TEST_H

//------------------------------------------------------------------------
// Role of the <Test> module
// Provides assertions and named test cases for the component suites
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

//------------------------------------------------------------------ Types
struct TestCase {
    std::string name;
    void (*run)();
};


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
inline void check(bool condition, const std::string &message)
// Algorithm : Reports a failed boolean assertion by throwing an exception with the supplied explanation.
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}  //----- end of check


inline void checkNear(float actual, float expected, const std::string &message,
                      float absoluteTolerance = 1e-5f, float relativeTolerance = 1e-4f)
// Algorithm : Checks finite numerical values using combined absolute and relative error tolerances.
{
    // Absolute tolerance handles values near zero; relative tolerance scales with larger expected values.
    float tolerance = absoluteTolerance + relativeTolerance * std::abs(expected);

    check(std::isfinite(actual) && std::isfinite(expected) &&
          std::abs(actual - expected) <= tolerance,
          message + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
}  //----- end of checkNear


std::vector<TestCase> getTests();
// Contract : Declares the component-provided list of named tests consumed by the common runner.

#endif //TEST_H
