#include <typeinfo>
#pragma GCC visibility push(hidden)
const std::type_info* t = &(typeid(int **));
struct A { };
#pragma GCC visibility pop
const std::type_info* t2 = &(typeid(A *));
