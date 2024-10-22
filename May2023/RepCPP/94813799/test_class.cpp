

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include "local_bindings.h"
#include <pybind11/stl.h>

#if defined(_MSC_VER)
#  pragma warning(disable: 4324) 
#endif

struct NoBraceInitialization {
NoBraceInitialization(std::vector<int> v) : vec{std::move(v)} {}
template <typename T>
NoBraceInitialization(std::initializer_list<T> l) : vec(l) {}

std::vector<int> vec;
};

TEST_SUBMODULE(class_, m) {
struct NoConstructor {
NoConstructor() = default;
NoConstructor(const NoConstructor &) = default;
NoConstructor(NoConstructor &&) = default;
static NoConstructor *new_instance() {
auto *ptr = new NoConstructor();
print_created(ptr, "via new_instance");
return ptr;
}
~NoConstructor() { print_destroyed(this); }
};

py::class_<NoConstructor>(m, "NoConstructor")
.def_static("new_instance", &NoConstructor::new_instance, "Return an instance");

class Pet {
public:
Pet(const std::string &name, const std::string &species)
: m_name(name), m_species(species) {}
std::string name() const { return m_name; }
std::string species() const { return m_species; }
private:
std::string m_name;
std::string m_species;
};

class Dog : public Pet {
public:
Dog(const std::string &name) : Pet(name, "dog") {}
std::string bark() const { return "Woof!"; }
};

class Rabbit : public Pet {
public:
Rabbit(const std::string &name) : Pet(name, "parrot") {}
};

class Hamster : public Pet {
public:
Hamster(const std::string &name) : Pet(name, "rodent") {}
};

class Chimera : public Pet {
Chimera() : Pet("Kimmy", "chimera") {}
};

py::class_<Pet> pet_class(m, "Pet");
pet_class
.def(py::init<std::string, std::string>())
.def("name", &Pet::name)
.def("species", &Pet::species);


py::class_<Dog>(m, "Dog", pet_class)
.def(py::init<std::string>());


py::class_<Rabbit, Pet>(m, "Rabbit")
.def(py::init<std::string>());


py::class_<Hamster, Pet>(m, "Hamster")
.def(py::init<std::string>());


py::class_<Chimera, Pet>(m, "Chimera");

m.def("pet_name_species", [](const Pet &pet) { return pet.name() + " is a " + pet.species(); });
m.def("dog_bark", [](const Dog &dog) { return dog.bark(); });

struct BaseClass {
BaseClass() = default;
BaseClass(const BaseClass &) = default;
BaseClass(BaseClass &&) = default;
virtual ~BaseClass() {}
};
struct DerivedClass1 : BaseClass { };
struct DerivedClass2 : BaseClass { };

py::class_<BaseClass>(m, "BaseClass").def(py::init<>());
py::class_<DerivedClass1>(m, "DerivedClass1").def(py::init<>());
py::class_<DerivedClass2>(m, "DerivedClass2").def(py::init<>());

m.def("return_class_1", []() -> BaseClass* { return new DerivedClass1(); });
m.def("return_class_2", []() -> BaseClass* { return new DerivedClass2(); });
m.def("return_class_n", [](int n) -> BaseClass* {
if (n == 1) return new DerivedClass1();
if (n == 2) return new DerivedClass2();
return new BaseClass();
});
m.def("return_none", []() -> BaseClass* { return nullptr; });

m.def("check_instances", [](py::list l) {
return py::make_tuple(
py::isinstance<py::tuple>(l[0]),
py::isinstance<py::dict>(l[1]),
py::isinstance<Pet>(l[2]),
py::isinstance<Pet>(l[3]),
py::isinstance<Dog>(l[4]),
py::isinstance<Rabbit>(l[5]),
py::isinstance<UnregisteredType>(l[6])
);
});

struct MismatchBase1 { };
struct MismatchDerived1 : MismatchBase1 { };

struct MismatchBase2 { };
struct MismatchDerived2 : MismatchBase2 { };

m.def("mismatched_holder_1", []() {
auto mod = py::module::import("__main__");
py::class_<MismatchBase1, std::shared_ptr<MismatchBase1>>(mod, "MismatchBase1");
py::class_<MismatchDerived1, MismatchBase1>(mod, "MismatchDerived1");
});
m.def("mismatched_holder_2", []() {
auto mod = py::module::import("__main__");
py::class_<MismatchBase2>(mod, "MismatchBase2");
py::class_<MismatchDerived2, std::shared_ptr<MismatchDerived2>,
MismatchBase2>(mod, "MismatchDerived2");
});

struct MyBase {
static std::unique_ptr<MyBase> make() {
return std::unique_ptr<MyBase>(new MyBase());
}
};

struct MyDerived : MyBase {
static std::unique_ptr<MyDerived> make() {
return std::unique_ptr<MyDerived>(new MyDerived());
}
};

py::class_<MyBase>(m, "MyBase")
.def_static("make", &MyBase::make);

py::class_<MyDerived, MyBase>(m, "MyDerived")
.def_static("make", &MyDerived::make)
.def_static("make2", &MyDerived::make);

struct ConvertibleFromUserType {
int i;

ConvertibleFromUserType(UserType u) : i(u.value()) { }
};

py::class_<ConvertibleFromUserType>(m, "AcceptsUserType")
.def(py::init<UserType>());
py::implicitly_convertible<UserType, ConvertibleFromUserType>();

m.def("implicitly_convert_argument", [](const ConvertibleFromUserType &r) { return r.i; });
m.def("implicitly_convert_variable", [](py::object o) {
const auto &r = o.cast<const ConvertibleFromUserType &>();
return r.i;
});
m.add_object("implicitly_convert_variable_fail", [&] {
auto f = [](PyObject *, PyObject *args) -> PyObject * {
auto o = py::reinterpret_borrow<py::tuple>(args)[0];
try { 
o.cast<const ConvertibleFromUserType &>();
} catch (const py::cast_error &e) {
return py::str(e.what()).release().ptr();
}
return py::str().release().ptr();
};

auto def = new PyMethodDef{"f", f, METH_VARARGS, nullptr};
return py::reinterpret_steal<py::object>(PyCFunction_NewEx(def, nullptr, m.ptr()));
}());

struct HasOpNewDel {
std::uint64_t i;
static void *operator new(size_t s) { py::print("A new", s); return ::operator new(s); }
static void *operator new(size_t s, void *ptr) { py::print("A placement-new", s); return ptr; }
static void operator delete(void *p) { py::print("A delete"); return ::operator delete(p); }
};
struct HasOpNewDelSize {
std::uint32_t i;
static void *operator new(size_t s) { py::print("B new", s); return ::operator new(s); }
static void *operator new(size_t s, void *ptr) { py::print("B placement-new", s); return ptr; }
static void operator delete(void *p, size_t s) { py::print("B delete", s); return ::operator delete(p); }
};
struct AliasedHasOpNewDelSize {
std::uint64_t i;
static void *operator new(size_t s) { py::print("C new", s); return ::operator new(s); }
static void *operator new(size_t s, void *ptr) { py::print("C placement-new", s); return ptr; }
static void operator delete(void *p, size_t s) { py::print("C delete", s); return ::operator delete(p); }
virtual ~AliasedHasOpNewDelSize() = default;
};
struct PyAliasedHasOpNewDelSize : AliasedHasOpNewDelSize {
PyAliasedHasOpNewDelSize() = default;
PyAliasedHasOpNewDelSize(int) { }
std::uint64_t j;
};
struct HasOpNewDelBoth {
std::uint32_t i[8];
static void *operator new(size_t s) { py::print("D new", s); return ::operator new(s); }
static void *operator new(size_t s, void *ptr) { py::print("D placement-new", s); return ptr; }
static void operator delete(void *p) { py::print("D delete"); return ::operator delete(p); }
static void operator delete(void *p, size_t s) { py::print("D wrong delete", s); return ::operator delete(p); }
};
py::class_<HasOpNewDel>(m, "HasOpNewDel").def(py::init<>());
py::class_<HasOpNewDelSize>(m, "HasOpNewDelSize").def(py::init<>());
py::class_<HasOpNewDelBoth>(m, "HasOpNewDelBoth").def(py::init<>());
py::class_<AliasedHasOpNewDelSize, PyAliasedHasOpNewDelSize> aliased(m, "AliasedHasOpNewDelSize");
aliased.def(py::init<>());
aliased.attr("size_noalias") = py::int_(sizeof(AliasedHasOpNewDelSize));
aliased.attr("size_alias") = py::int_(sizeof(PyAliasedHasOpNewDelSize));

bind_local<LocalExternal, 17>(m, "LocalExternal", py::module_local());

class ProtectedA {
protected:
int foo() const { return value; }

private:
int value = 42;
};

class PublicistA : public ProtectedA {
public:
using ProtectedA::foo;
};

py::class_<ProtectedA>(m, "ProtectedA")
.def(py::init<>())
#if !defined(_MSC_VER) || _MSC_VER >= 1910
.def("foo", &PublicistA::foo);
#else
.def("foo", static_cast<int (ProtectedA::*)() const>(&PublicistA::foo));
#endif

class ProtectedB {
public:
virtual ~ProtectedB() = default;

protected:
virtual int foo() const { return value; }

private:
int value = 42;
};

class TrampolineB : public ProtectedB {
public:
int foo() const override { PYBIND11_OVERLOAD(int, ProtectedB, foo, ); }
};

class PublicistB : public ProtectedB {
public:
using ProtectedB::foo;
};

py::class_<ProtectedB, TrampolineB>(m, "ProtectedB")
.def(py::init<>())
#if !defined(_MSC_VER) || _MSC_VER >= 1910
.def("foo", &PublicistB::foo);
#else
.def("foo", static_cast<int (ProtectedB::*)() const>(&PublicistB::foo));
#endif

struct BraceInitialization {
int field1;
std::string field2;
};

py::class_<BraceInitialization>(m, "BraceInitialization")
.def(py::init<int, const std::string &>())
.def_readwrite("field1", &BraceInitialization::field1)
.def_readwrite("field2", &BraceInitialization::field2);
py::class_<NoBraceInitialization>(m, "NoBraceInitialization")
.def(py::init<std::vector<int>>())
.def_readonly("vec", &NoBraceInitialization::vec);

struct BogusImplicitConversion {
BogusImplicitConversion(const BogusImplicitConversion &) { }
};

py::class_<BogusImplicitConversion>(m, "BogusImplicitConversion")
.def(py::init<const BogusImplicitConversion &>());

py::implicitly_convertible<int, BogusImplicitConversion>();

struct NestBase {};
struct Nested {};
py::class_<NestBase> base(m, "NestBase");
base.def(py::init<>());
py::class_<Nested>(base, "Nested")
.def(py::init<>())
.def("fn", [](Nested &, int, NestBase &, Nested &) {})
.def("fa", [](Nested &, int, NestBase &, Nested &) {},
"a"_a, "b"_a, "c"_a);
base.def("g", [](NestBase &, Nested &) {});
base.def("h", []() { return NestBase(); });


struct NotRegistered {};
struct StringWrapper { std::string str; };
m.def("test_error_after_conversions", [](int) {});
m.def("test_error_after_conversions",
[](StringWrapper) -> NotRegistered { return {}; });
py::class_<StringWrapper>(m, "StringWrapper").def(py::init<std::string>());
py::implicitly_convertible<std::string, StringWrapper>();

#if defined(PYBIND11_CPP17)
struct alignas(1024) Aligned {
std::uintptr_t ptr() const { return (uintptr_t) this; }
};
py::class_<Aligned>(m, "Aligned")
.def(py::init<>())
.def("ptr", &Aligned::ptr);
#endif
}

template <int N> class BreaksBase { public: virtual ~BreaksBase() = default; };
template <int N> class BreaksTramp : public BreaksBase<N> {};
typedef py::class_<BreaksBase<1>, std::unique_ptr<BreaksBase<1>>, BreaksTramp<1>> DoesntBreak1;
typedef py::class_<BreaksBase<2>, BreaksTramp<2>, std::unique_ptr<BreaksBase<2>>> DoesntBreak2;
typedef py::class_<BreaksBase<3>, std::unique_ptr<BreaksBase<3>>> DoesntBreak3;
typedef py::class_<BreaksBase<4>, BreaksTramp<4>> DoesntBreak4;
typedef py::class_<BreaksBase<5>> DoesntBreak5;
typedef py::class_<BreaksBase<6>, std::shared_ptr<BreaksBase<6>>, BreaksTramp<6>> DoesntBreak6;
typedef py::class_<BreaksBase<7>, BreaksTramp<7>, std::shared_ptr<BreaksBase<7>>> DoesntBreak7;
typedef py::class_<BreaksBase<8>, std::shared_ptr<BreaksBase<8>>> DoesntBreak8;
#define CHECK_BASE(N) static_assert(std::is_same<typename DoesntBreak##N::type, BreaksBase<N>>::value, \
"DoesntBreak" #N " has wrong type!")
CHECK_BASE(1); CHECK_BASE(2); CHECK_BASE(3); CHECK_BASE(4); CHECK_BASE(5); CHECK_BASE(6); CHECK_BASE(7); CHECK_BASE(8);
#define CHECK_ALIAS(N) static_assert(DoesntBreak##N::has_alias && std::is_same<typename DoesntBreak##N::type_alias, BreaksTramp<N>>::value, \
"DoesntBreak" #N " has wrong type_alias!")
#define CHECK_NOALIAS(N) static_assert(!DoesntBreak##N::has_alias && std::is_void<typename DoesntBreak##N::type_alias>::value, \
"DoesntBreak" #N " has type alias, but shouldn't!")
CHECK_ALIAS(1); CHECK_ALIAS(2); CHECK_NOALIAS(3); CHECK_ALIAS(4); CHECK_NOALIAS(5); CHECK_ALIAS(6); CHECK_ALIAS(7); CHECK_NOALIAS(8);
#define CHECK_HOLDER(N, TYPE) static_assert(std::is_same<typename DoesntBreak##N::holder_type, std::TYPE##_ptr<BreaksBase<N>>>::value, \
"DoesntBreak" #N " has wrong holder_type!")
CHECK_HOLDER(1, unique); CHECK_HOLDER(2, unique); CHECK_HOLDER(3, unique); CHECK_HOLDER(4, unique); CHECK_HOLDER(5, unique);
CHECK_HOLDER(6, shared); CHECK_HOLDER(7, shared); CHECK_HOLDER(8, shared);


#define CHECK_BROKEN(N) static_assert(std::is_same<typename Breaks##N::type, BreaksBase<-N>>::value, \
"Breaks1 has wrong type!");

