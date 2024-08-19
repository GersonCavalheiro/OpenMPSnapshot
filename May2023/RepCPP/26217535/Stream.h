

#pragma once

#include <string>
#include <vector>

#define startA (0.1)
#define startB (0.2)
#define startC (0.0)
#define startScalar (0.4)

template<class T>
class Stream
{
public:
virtual ~Stream()
{
}

virtual void copy() = 0;
virtual void mul() = 0;
virtual void add() = 0;
virtual void triad() = 0;
virtual void nstream() = 0;
virtual T dot() = 0;

virtual void init_arrays(T initA, T initB, T initC) = 0;
virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) = 0;
};


void listDevices(void);
std::string getDeviceName(int const);
std::string getDeviceDriver(int const);
