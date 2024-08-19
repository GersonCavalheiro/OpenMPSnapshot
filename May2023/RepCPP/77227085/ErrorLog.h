

#pragma once

#include "ThreadSafetyAnalysis.h" 
#include "adt/Mutex.h"            
#include <string>                 
#include <vector>                 

namespace rawspeed {

class ErrorLog {
Mutex mutex;
std::vector<std::string> errors GUARDED_BY(mutex);

public:
void setError(const std::string& err) REQUIRES(!mutex);
bool isTooManyErrors(unsigned many, std::string* firstErr = nullptr)
REQUIRES(!mutex);
std::vector<std::string>&& getErrors() REQUIRES(!mutex);
};

} 
