#pragma once

#include <iostream>
#include <string>
#include "omp.h"

using namespace std;

class LogDuration {
public:
explicit LogDuration(const string& msg = "")
: message(msg + ": "), start(omp_get_wtime())
{
}

~LogDuration() {
auto finish = omp_get_wtime();
auto dur = finish - start;
cerr << message << static_cast<int>(dur * 1000) << " ms" << endl;
}
private:
string message;
double start;
};

#define UNIQ_ID_IMPL(lineno) _a_local_var_##lineno
#define UNIQ_ID(lineno) UNIQ_ID_IMPL(lineno)

#define LOG_DURATION(message) \
LogDuration UNIQ_ID(__LINE__){message};
