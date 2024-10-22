
#pragma once

#include "../pybind11.h"
#include "../detail/common.h"
#include "../detail/descr.h"
#include "../cast.h"
#include "../pytypes.h"

#include <string>

#ifdef __has_include
#    if defined(PYBIND11_CPP17)
#        if __has_include(<filesystem>) && \
PY_VERSION_HEX >= 0x03060000
#            include <filesystem>
#            define PYBIND11_HAS_FILESYSTEM 1
#        elif __has_include(<experimental/filesystem>)
#            include <experimental/filesystem>
#            define PYBIND11_HAS_EXPERIMENTAL_FILESYSTEM 1
#        endif
#    endif
#endif

#if !defined(PYBIND11_HAS_FILESYSTEM) && !defined(PYBIND11_HAS_EXPERIMENTAL_FILESYSTEM)           \
&& !defined(PYBIND11_HAS_FILESYSTEM_IS_OPTIONAL)
#    error                                                                                        \
"Neither #include <filesystem> nor #include <experimental/filesystem is available. (Use -DPYBIND11_HAS_FILESYSTEM_IS_OPTIONAL to ignore.)"
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

#if defined(PYBIND11_HAS_FILESYSTEM) || defined(PYBIND11_HAS_EXPERIMENTAL_FILESYSTEM)
template <typename T>
struct path_caster {

private:
static PyObject *unicode_from_fs_native(const std::string &w) {
#    if !defined(PYPY_VERSION)
return PyUnicode_DecodeFSDefaultAndSize(w.c_str(), ssize_t(w.size()));
#    else
return PyUnicode_DecodeFSDefaultAndSize(const_cast<char *>(w.c_str()), ssize_t(w.size()));
#    endif
}

static PyObject *unicode_from_fs_native(const std::wstring &w) {
return PyUnicode_FromWideChar(w.c_str(), ssize_t(w.size()));
}

public:
static handle cast(const T &path, return_value_policy, handle) {
if (auto py_str = unicode_from_fs_native(path.native())) {
return module_::import("pathlib")
.attr("Path")(reinterpret_steal<object>(py_str))
.release();
}
return nullptr;
}

bool load(handle handle, bool) {
PyObject *buf = PyOS_FSPath(handle.ptr());
if (!buf) {
PyErr_Clear();
return false;
}
PyObject *native = nullptr;
if constexpr (std::is_same_v<typename T::value_type, char>) {
if (PyUnicode_FSConverter(buf, &native) != 0) {
if (auto *c_str = PyBytes_AsString(native)) {
value = c_str;
}
}
} else if constexpr (std::is_same_v<typename T::value_type, wchar_t>) {
if (PyUnicode_FSDecoder(buf, &native) != 0) {
if (auto *c_str = PyUnicode_AsWideCharString(native, nullptr)) {
value = c_str; 
PyMem_Free(c_str);
}
}
}
Py_XDECREF(native);
Py_DECREF(buf);
if (PyErr_Occurred()) {
PyErr_Clear();
return false;
}
return true;
}

PYBIND11_TYPE_CASTER(T, const_name("os.PathLike"));
};

#endif 

#if defined(PYBIND11_HAS_FILESYSTEM)
template <>
struct type_caster<std::filesystem::path> : public path_caster<std::filesystem::path> {};
#elif defined(PYBIND11_HAS_EXPERIMENTAL_FILESYSTEM)
template <>
struct type_caster<std::experimental::filesystem::path>
: public path_caster<std::experimental::filesystem::path> {};
#endif

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
