#pragma once
#include "pybind11_tests.h"

template <int> class LocalBase {
public:
LocalBase(int i) : i(i) { }
int i = -1;
};

using LocalType = LocalBase<0>;
using NonLocalType = LocalBase<1>;
using NonLocal2 = LocalBase<2>;
using LocalExternal = LocalBase<3>;
using MixedLocalGlobal = LocalBase<4>;
using MixedGlobalLocal = LocalBase<5>;

using ExternalType1 = LocalBase<6>;
using ExternalType2 = LocalBase<7>;

using LocalVec = std::vector<LocalType>;
using LocalVec2 = std::vector<NonLocal2>;
using LocalMap = std::unordered_map<std::string, LocalType>;
using NonLocalVec = std::vector<NonLocalType>;
using NonLocalVec2 = std::vector<NonLocal2>;
using NonLocalMap = std::unordered_map<std::string, NonLocalType>;
using NonLocalMap2 = std::unordered_map<std::string, uint8_t>;

PYBIND11_MAKE_OPAQUE(LocalVec);
PYBIND11_MAKE_OPAQUE(LocalVec2);
PYBIND11_MAKE_OPAQUE(LocalMap);
PYBIND11_MAKE_OPAQUE(NonLocalVec);
PYBIND11_MAKE_OPAQUE(NonLocalMap);
PYBIND11_MAKE_OPAQUE(NonLocalMap2);


template <typename T, int Adjust = 0, typename... Args>
py::class_<T> bind_local(Args && ...args) {
return py::class_<T>(std::forward<Args>(args)...)
.def(py::init<int>())
.def("get", [](T &i) { return i.i + Adjust; });
};

namespace pets {
class Pet {
public:
Pet(std::string name) : name_(name) {}
std::string name_;
const std::string &name() { return name_; }
};
}

struct MixGL { int i; MixGL(int i) : i{i} {} };
struct MixGL2 { int i; MixGL2(int i) : i{i} {} };
