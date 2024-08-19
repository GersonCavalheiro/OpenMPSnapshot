#ifndef _STRING_OPERATIONS_
#define _STRING_OPERATIONS_

#include <string>

using namespace std;

class StringOperations{
public:
static inline string first20char(string a){
if(a.size() <= 17){
return a;
}
return a.substr(0, 17) + "...";
}

static inline string last20char(string a){
if(a.size() <= 17){
return a;
}
return  "..." + a.substr(a.size()-17, 17);
}

static inline void copyStringVector(string* v1, string* v2, int size){
#pragma omp parallel
{
#pragma omp for
for(int i = 0; i < size; i++){
v2[i] = v1[i];
}
}
}
};

#endif