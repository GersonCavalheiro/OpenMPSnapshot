#pragma GCC visibility push (hidden)
#include <typeinfo>
const std::type_info& info1 = typeid(int []);
const std::type_info& info2 = typeid(int);
enum E { e = 0 };
const std::type_info& info3 = typeid(E);
struct S { S (); };
const std::type_info& info4 = typeid(S);
const std::type_info& info5 = typeid(int *);
