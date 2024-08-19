#pragma once


#include "pybind11_tests.h"
#include <unordered_map>
#include <list>
#include <typeindex>
#include <sstream>

class ConstructorStats {
protected:
std::unordered_map<void*, int> _instances; 
std::list<std::string> _values; 
public:
int default_constructions = 0;
int copy_constructions = 0;
int move_constructions = 0;
int copy_assignments = 0;
int move_assignments = 0;

void copy_created(void *inst) {
created(inst);
copy_constructions++;
}

void move_created(void *inst) {
created(inst);
move_constructions++;
}

void default_created(void *inst) {
created(inst);
default_constructions++;
}

void created(void *inst) {
++_instances[inst];
}

void destroyed(void *inst) {
if (--_instances[inst] < 0)
throw std::runtime_error("cstats.destroyed() called with unknown "
"instance; potential double-destruction "
"or a missing cstats.created()");
}

static void gc() {
#if defined(PYPY_VERSION)
PyObject *globals = PyEval_GetGlobals();
PyObject *result = PyRun_String(
"import gc\n"
"for i in range(2):"
"    gc.collect()\n",
Py_file_input, globals, globals);
if (result == nullptr)
throw py::error_already_set();
Py_DECREF(result);
#else
py::module::import("gc").attr("collect")();
#endif
}

int alive() {
gc();
int total = 0;
for (const auto &p : _instances)
if (p.second > 0)
total += p.second;
return total;
}

void value() {} 
template <typename T, typename... Tmore> void value(const T &v, Tmore &&...args) {
std::ostringstream oss;
oss << v;
_values.push_back(oss.str());
value(std::forward<Tmore>(args)...);
}

py::list values() {
py::list l;
for (const auto &v : _values) l.append(py::cast(v));
_values.clear();
return l;
}

static ConstructorStats& get(std::type_index type) {
static std::unordered_map<std::type_index, ConstructorStats> all_cstats;
return all_cstats[type];
}

template <typename T> static ConstructorStats& get() {
#if defined(PYPY_VERSION)
gc();
#endif
return get(typeid(T));
}

static ConstructorStats& get(py::object class_) {
auto &internals = py::detail::get_internals();
const std::type_index *t1 = nullptr, *t2 = nullptr;
try {
auto *type_info = internals.registered_types_py.at((PyTypeObject *) class_.ptr()).at(0);
for (auto &p : internals.registered_types_cpp) {
if (p.second == type_info) {
if (t1) {
t2 = &p.first;
break;
}
t1 = &p.first;
}
}
}
catch (const std::out_of_range &) {}
if (!t1) throw std::runtime_error("Unknown class passed to ConstructorStats::get()");
auto &cs1 = get(*t1);
if (t2) {
auto &cs2 = get(*t2);
int cs1_total = cs1.default_constructions + cs1.copy_constructions + cs1.move_constructions + (int) cs1._values.size();
int cs2_total = cs2.default_constructions + cs2.copy_constructions + cs2.move_constructions + (int) cs2._values.size();
if (cs2_total > cs1_total) return cs2;
}
return cs1;
}
};

template <class T> void track_copy_created(T *inst) { ConstructorStats::get<T>().copy_created(inst); }
template <class T> void track_move_created(T *inst) { ConstructorStats::get<T>().move_created(inst); }
template <class T, typename... Values> void track_copy_assigned(T *, Values &&...values) {
auto &cst = ConstructorStats::get<T>();
cst.copy_assignments++;
cst.value(std::forward<Values>(values)...);
}
template <class T, typename... Values> void track_move_assigned(T *, Values &&...values) {
auto &cst = ConstructorStats::get<T>();
cst.move_assignments++;
cst.value(std::forward<Values>(values)...);
}
template <class T, typename... Values> void track_default_created(T *inst, Values &&...values) {
auto &cst = ConstructorStats::get<T>();
cst.default_created(inst);
cst.value(std::forward<Values>(values)...);
}
template <class T, typename... Values> void track_created(T *inst, Values &&...values) {
auto &cst = ConstructorStats::get<T>();
cst.created(inst);
cst.value(std::forward<Values>(values)...);
}
template <class T, typename... Values> void track_destroyed(T *inst) {
ConstructorStats::get<T>().destroyed(inst);
}
template <class T, typename... Values> void track_values(T *, Values &&...values) {
ConstructorStats::get<T>().value(std::forward<Values>(values)...);
}

inline const char *format_ptrs(const char *p) { return p; }
template <typename T>
py::str format_ptrs(T *p) { return "{:#x}"_s.format(reinterpret_cast<std::uintptr_t>(p)); }
template <typename T>
auto format_ptrs(T &&x) -> decltype(std::forward<T>(x)) { return std::forward<T>(x); }

template <class T, typename... Output>
void print_constr_details(T *inst, const std::string &action, Output &&...output) {
py::print("###", py::type_id<T>(), "@", format_ptrs(inst), action,
format_ptrs(std::forward<Output>(output))...);
}

template <class T, typename... Values> void print_copy_created(T *inst, Values &&...values) { 
print_constr_details(inst, "created via copy constructor", values...);
track_copy_created(inst);
}
template <class T, typename... Values> void print_move_created(T *inst, Values &&...values) { 
print_constr_details(inst, "created via move constructor", values...);
track_move_created(inst);
}
template <class T, typename... Values> void print_copy_assigned(T *inst, Values &&...values) {
print_constr_details(inst, "assigned via copy assignment", values...);
track_copy_assigned(inst, values...);
}
template <class T, typename... Values> void print_move_assigned(T *inst, Values &&...values) {
print_constr_details(inst, "assigned via move assignment", values...);
track_move_assigned(inst, values...);
}
template <class T, typename... Values> void print_default_created(T *inst, Values &&...values) {
print_constr_details(inst, "created via default constructor", values...);
track_default_created(inst, values...);
}
template <class T, typename... Values> void print_created(T *inst, Values &&...values) {
print_constr_details(inst, "created", values...);
track_created(inst, values...);
}
template <class T, typename... Values> void print_destroyed(T *inst, Values &&...values) { 
print_constr_details(inst, "destroyed", values...);
track_destroyed(inst);
}
template <class T, typename... Values> void print_values(T *inst, Values &&...values) {
print_constr_details(inst, ":", values...);
track_values(inst, values...);
}

