

#pragma once

#include <cstdio>
#include <cstdlib>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

#include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
inline void erase_all(std::string &string, const std::string &search) {
for (size_t pos = 0;;) {
pos = string.find(search, pos);
if (pos == std::string::npos) break;
string.erase(pos, search.length());
}
}

PYBIND11_NOINLINE inline void clean_type_id(std::string &name) {
#if defined(__GNUG__)
int status = 0;
std::unique_ptr<char, void (*)(void *)> res {
abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), std::free };
if (status == 0)
name = res.get();
#else
detail::erase_all(name, "class ");
detail::erase_all(name, "struct ");
detail::erase_all(name, "enum ");
#endif
detail::erase_all(name, "pybind11::");
}
PYBIND11_NAMESPACE_END(detail)

template <typename T> static std::string type_id() {
std::string name(typeid(T).name());
detail::clean_type_id(name);
return name;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
