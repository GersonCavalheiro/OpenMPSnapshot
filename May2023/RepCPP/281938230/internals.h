

#pragma once

#include "../pytypes.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
inline PyTypeObject *make_static_property_type();
inline PyTypeObject *make_default_metaclass();
inline PyObject *make_object_base_type(PyTypeObject *metaclass);

#if PY_VERSION_HEX >= 0x03070000
#    define PYBIND11_TLS_KEY_INIT(var) Py_tss_t *var = nullptr
#    define PYBIND11_TLS_GET_VALUE(key) PyThread_tss_get((key))
#    define PYBIND11_TLS_REPLACE_VALUE(key, value) PyThread_tss_set((key), (value))
#    define PYBIND11_TLS_DELETE_VALUE(key) PyThread_tss_set((key), nullptr)
#    define PYBIND11_TLS_FREE(key) PyThread_tss_free(key)
#else
#    define PYBIND11_TLS_KEY_INIT(var) decltype(PyThread_create_key()) var = 0
#    define PYBIND11_TLS_GET_VALUE(key) PyThread_get_key_value((key))
#    if PY_MAJOR_VERSION < 3
#        define PYBIND11_TLS_DELETE_VALUE(key)                               \
PyThread_delete_key_value(key)
#        define PYBIND11_TLS_REPLACE_VALUE(key, value)                       \
do {                                                            \
PyThread_delete_key_value((key));                           \
PyThread_set_key_value((key), (value));                     \
} while (false)
#    else
#        define PYBIND11_TLS_DELETE_VALUE(key)                               \
PyThread_set_key_value((key), nullptr)
#        define PYBIND11_TLS_REPLACE_VALUE(key, value)                       \
PyThread_set_key_value((key), (value))
#    endif
#    define PYBIND11_TLS_FREE(key) (void)key
#endif

#if defined(__GLIBCXX__)
inline bool same_type(const std::type_info &lhs, const std::type_info &rhs) { return lhs == rhs; }
using type_hash = std::hash<std::type_index>;
using type_equal_to = std::equal_to<std::type_index>;
#else
inline bool same_type(const std::type_info &lhs, const std::type_info &rhs) {
return lhs.name() == rhs.name() || std::strcmp(lhs.name(), rhs.name()) == 0;
}

struct type_hash {
size_t operator()(const std::type_index &t) const {
size_t hash = 5381;
const char *ptr = t.name();
while (auto c = static_cast<unsigned char>(*ptr++))
hash = (hash * 33) ^ c;
return hash;
}
};

struct type_equal_to {
bool operator()(const std::type_index &lhs, const std::type_index &rhs) const {
return lhs.name() == rhs.name() || std::strcmp(lhs.name(), rhs.name()) == 0;
}
};
#endif

template <typename value_type>
using type_map = std::unordered_map<std::type_index, value_type, type_hash, type_equal_to>;

struct overload_hash {
inline size_t operator()(const std::pair<const PyObject *, const char *>& v) const {
size_t value = std::hash<const void *>()(v.first);
value ^= std::hash<const void *>()(v.second)  + 0x9e3779b9 + (value<<6) + (value>>2);
return value;
}
};

struct internals {
type_map<type_info *> registered_types_cpp; 
std::unordered_map<PyTypeObject *, std::vector<type_info *>> registered_types_py; 
std::unordered_multimap<const void *, instance*> registered_instances; 
std::unordered_set<std::pair<const PyObject *, const char *>, overload_hash> inactive_overload_cache;
type_map<std::vector<bool (*)(PyObject *, void *&)>> direct_conversions;
std::unordered_map<const PyObject *, std::vector<PyObject *>> patients;
std::forward_list<void (*) (std::exception_ptr)> registered_exception_translators;
std::unordered_map<std::string, void *> shared_data; 
std::vector<PyObject *> loader_patient_stack; 
std::forward_list<std::string> static_strings; 
PyTypeObject *static_property_type;
PyTypeObject *default_metaclass;
PyObject *instance_base;
#if defined(WITH_THREAD)
PYBIND11_TLS_KEY_INIT(tstate);
PyInterpreterState *istate = nullptr;
~internals() {
PYBIND11_TLS_FREE(tstate);
}
#endif
};

struct type_info {
PyTypeObject *type;
const std::type_info *cpptype;
size_t type_size, type_align, holder_size_in_ptrs;
void *(*operator_new)(size_t);
void (*init_instance)(instance *, const void *);
void (*dealloc)(value_and_holder &v_h);
std::vector<PyObject *(*)(PyObject *, PyTypeObject *)> implicit_conversions;
std::vector<std::pair<const std::type_info *, void *(*)(void *)>> implicit_casts;
std::vector<bool (*)(PyObject *, void *&)> *direct_conversions;
buffer_info *(*get_buffer)(PyObject *, void *) = nullptr;
void *get_buffer_data = nullptr;
void *(*module_local_load)(PyObject *, const type_info *) = nullptr;

bool simple_type : 1;

bool simple_ancestors : 1;

bool default_holder : 1;

bool module_local : 1;
};

#define PYBIND11_INTERNALS_VERSION 4

#if defined(_MSC_VER) && defined(_DEBUG)
#   define PYBIND11_BUILD_TYPE "_debug"
#else
#   define PYBIND11_BUILD_TYPE ""
#endif

#if defined(_MSC_VER)
#   define PYBIND11_COMPILER_TYPE "_msvc"
#elif defined(__INTEL_COMPILER)
#   define PYBIND11_COMPILER_TYPE "_icc"
#elif defined(__clang__)
#   define PYBIND11_COMPILER_TYPE "_clang"
#elif defined(__PGI)
#   define PYBIND11_COMPILER_TYPE "_pgi"
#elif defined(__MINGW32__)
#   define PYBIND11_COMPILER_TYPE "_mingw"
#elif defined(__CYGWIN__)
#   define PYBIND11_COMPILER_TYPE "_gcc_cygwin"
#elif defined(__GNUC__)
#   define PYBIND11_COMPILER_TYPE "_gcc"
#else
#   define PYBIND11_COMPILER_TYPE "_unknown"
#endif

#if defined(_LIBCPP_VERSION)
#  define PYBIND11_STDLIB "_libcpp"
#elif defined(__GLIBCXX__) || defined(__GLIBCPP__)
#  define PYBIND11_STDLIB "_libstdcpp"
#else
#  define PYBIND11_STDLIB ""
#endif

#if defined(__GXX_ABI_VERSION)
#  define PYBIND11_BUILD_ABI "_cxxabi" PYBIND11_TOSTRING(__GXX_ABI_VERSION)
#else
#  define PYBIND11_BUILD_ABI ""
#endif

#if defined(WITH_THREAD)
#  define PYBIND11_INTERNALS_KIND ""
#else
#  define PYBIND11_INTERNALS_KIND "_without_thread"
#endif

#define PYBIND11_INTERNALS_ID "__pybind11_internals_v" \
PYBIND11_TOSTRING(PYBIND11_INTERNALS_VERSION) PYBIND11_INTERNALS_KIND PYBIND11_COMPILER_TYPE PYBIND11_STDLIB PYBIND11_BUILD_ABI PYBIND11_BUILD_TYPE "__"

#define PYBIND11_MODULE_LOCAL_ID "__pybind11_module_local_v" \
PYBIND11_TOSTRING(PYBIND11_INTERNALS_VERSION) PYBIND11_INTERNALS_KIND PYBIND11_COMPILER_TYPE PYBIND11_STDLIB PYBIND11_BUILD_ABI PYBIND11_BUILD_TYPE "__"

inline internals **&get_internals_pp() {
static internals **internals_pp = nullptr;
return internals_pp;
}

inline void translate_exception(std::exception_ptr p) {
try {
if (p) std::rethrow_exception(p);
} catch (error_already_set &e)           { e.restore();                                    return;
} catch (const builtin_exception &e)     { e.set_error();                                  return;
} catch (const std::bad_alloc &e)        { PyErr_SetString(PyExc_MemoryError,   e.what()); return;
} catch (const std::domain_error &e)     { PyErr_SetString(PyExc_ValueError,    e.what()); return;
} catch (const std::invalid_argument &e) { PyErr_SetString(PyExc_ValueError,    e.what()); return;
} catch (const std::length_error &e)     { PyErr_SetString(PyExc_ValueError,    e.what()); return;
} catch (const std::out_of_range &e)     { PyErr_SetString(PyExc_IndexError,    e.what()); return;
} catch (const std::range_error &e)      { PyErr_SetString(PyExc_ValueError,    e.what()); return;
} catch (const std::overflow_error &e)   { PyErr_SetString(PyExc_OverflowError, e.what()); return;
} catch (const std::exception &e)        { PyErr_SetString(PyExc_RuntimeError,  e.what()); return;
} catch (...) {
PyErr_SetString(PyExc_RuntimeError, "Caught an unknown exception!");
return;
}
}

#if !defined(__GLIBCXX__)
inline void translate_local_exception(std::exception_ptr p) {
try {
if (p) std::rethrow_exception(p);
} catch (error_already_set &e)       { e.restore();   return;
} catch (const builtin_exception &e) { e.set_error(); return;
}
}
#endif

PYBIND11_NOINLINE inline internals &get_internals() {
auto **&internals_pp = get_internals_pp();
if (internals_pp && *internals_pp)
return **internals_pp;

struct gil_scoped_acquire_local {
gil_scoped_acquire_local() : state (PyGILState_Ensure()) {}
~gil_scoped_acquire_local() { PyGILState_Release(state); }
const PyGILState_STATE state;
} gil;

constexpr auto *id = PYBIND11_INTERNALS_ID;
auto builtins = handle(PyEval_GetBuiltins());
if (builtins.contains(id) && isinstance<capsule>(builtins[id])) {
internals_pp = static_cast<internals **>(capsule(builtins[id]));

#if !defined(__GLIBCXX__)
(*internals_pp)->registered_exception_translators.push_front(&translate_local_exception);
#endif
} else {
if (!internals_pp) internals_pp = new internals*();
auto *&internals_ptr = *internals_pp;
internals_ptr = new internals();
#if defined(WITH_THREAD)

#if PY_VERSION_HEX < 0x03090000
PyEval_InitThreads();
#endif
PyThreadState *tstate = PyThreadState_Get();
#if PY_VERSION_HEX >= 0x03070000
internals_ptr->tstate = PyThread_tss_alloc();
if (!internals_ptr->tstate || PyThread_tss_create(internals_ptr->tstate))
pybind11_fail("get_internals: could not successfully initialize the TSS key!");
PyThread_tss_set(internals_ptr->tstate, tstate);
#else
internals_ptr->tstate = PyThread_create_key();
if (internals_ptr->tstate == -1)
pybind11_fail("get_internals: could not successfully initialize the TLS key!");
PyThread_set_key_value(internals_ptr->tstate, tstate);
#endif
internals_ptr->istate = tstate->interp;
#endif
builtins[id] = capsule(internals_pp);
internals_ptr->registered_exception_translators.push_front(&translate_exception);
internals_ptr->static_property_type = make_static_property_type();
internals_ptr->default_metaclass = make_default_metaclass();
internals_ptr->instance_base = make_object_base_type(internals_ptr->default_metaclass);
}
return **internals_pp;
}

inline type_map<type_info *> &registered_local_types_cpp() {
static type_map<type_info *> locals{};
return locals;
}

template <typename... Args>
const char *c_str(Args &&...args) {
auto &strings = get_internals().static_strings;
strings.emplace_front(std::forward<Args>(args)...);
return strings.front().c_str();
}

PYBIND11_NAMESPACE_END(detail)

inline PYBIND11_NOINLINE void *get_shared_data(const std::string &name) {
auto &internals = detail::get_internals();
auto it = internals.shared_data.find(name);
return it != internals.shared_data.end() ? it->second : nullptr;
}

inline PYBIND11_NOINLINE void *set_shared_data(const std::string &name, void *data) {
detail::get_internals().shared_data[name] = data;
return data;
}

template<typename T>
T &get_or_create_shared_data(const std::string &name) {
auto &internals = detail::get_internals();
auto it = internals.shared_data.find(name);
T *ptr = (T *) (it != internals.shared_data.end() ? it->second : nullptr);
if (!ptr) {
ptr = new T();
internals.shared_data[name] = ptr;
}
return *ptr;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
