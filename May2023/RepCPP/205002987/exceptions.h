#pragma once
#include <exception>

namespace numath {


struct IterException : public std::exception {

const char* what() const noexcept {
return "Could not find anything with the specified number of iterations";
}

};


struct IntervalException : public std::exception {

const char* what() const noexcept {
return "Invalid interval entered";
}

};


struct DerivativeException : public std::exception {

const char* what() const noexcept {
return "Derivative equals 0. Possible multiple roots found";
}

};


struct DenominatorException : public std::exception {

const char* what() const noexcept {
return "Denominator equals 0";
}

};


struct SolutionException : public std::exception {

const char* what() const noexcept {
return "The system does not have an unique solution";
}

};


struct MethodException : public std::exception {

const char* what() const noexcept {
return "The required methos does not exist";
}

};

}