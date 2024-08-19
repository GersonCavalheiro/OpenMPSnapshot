

#ifndef BOOST_DLL_DETAIL_CTOR_DTOR_HPP_
#define BOOST_DLL_DETAIL_CTOR_DTOR_HPP_

#include <boost/dll/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

#include <boost/dll/detail/aggressive_ptr_cast.hpp>
#include <boost/dll/detail/get_mem_fn_type.hpp>

#if defined(_MSC_VER) 
#   include <boost/dll/detail/demangling/msvc.hpp>
#else
#   include <boost/dll/detail/demangling/itanium.hpp>
#endif


namespace boost { namespace dll { namespace detail {


template<typename Signature>
struct constructor;

template<typename Class, typename ...Args>
struct constructor<Class(Args...)> {
typedef typename detail::get_mem_fn_type<Class, void(Args...)>::mem_fn standard_t;
typedef Class*(*allocating_t)(Args...);


standard_t standard;
allocating_t allocating;

void call_standard  (Class * const ptr, Args...args){ (ptr->*standard)(static_cast<Args>(args)...); }

Class * call_allocating(Args...args){ return allocating(static_cast<Args>(args)...); }


bool has_allocating() const { return allocating != nullptr; }

bool has_standard() const { return standard != nullptr; }

bool is_empty() const { return (allocating == nullptr) && (standard == nullptr) ; }

constructor() = delete;
constructor(const constructor &) = default;

explicit constructor(standard_t standard, allocating_t allocating = nullptr)
: standard(standard)
, allocating(allocating)
{}
};



template <typename Class>
struct destructor {
#if !defined(_WIN32)
typedef void(*type)(Class* const);
#elif !defined(_WIN64)
typedef void(__thiscall * type)(Class* const);
#else
typedef void(__cdecl * type)(Class* const);
#endif

typedef type standard_t;
typedef type deleting_t;

standard_t standard;
deleting_t deleting;

void call_standard(Class * const ptr){ standard(ptr); }

void call_deleting(Class * const ptr){ deleting(ptr); }

bool has_deleting() const { return deleting != nullptr; }

bool has_standard() const { return standard != nullptr; }

bool is_empty() const { return (deleting == nullptr) && (standard == nullptr) ; }
destructor() = delete;

destructor(const destructor &) = default;

explicit destructor(const standard_t &standard, const deleting_t &deleting = nullptr)
: standard(standard)
, deleting(deleting)
{}
};

#if defined(_MSC_VER) 

template<typename Signature, typename Lib>
constructor<Signature> load_ctor(Lib & lib, const mangled_storage_impl::ctor_sym & ct) {
typedef typename constructor<Signature>::standard_t standard_t;
standard_t ctor = lib.template get<standard_t>(ct);
return constructor<Signature>(ctor);
}

template<typename Class, typename Lib>
destructor<Class> load_dtor(Lib & lib, const mangled_storage_impl::dtor_sym & dt) {
typedef typename destructor<Class>::standard_t standard_t;
void * buf = &lib.template get<unsigned char>(dt);
standard_t dtor;
std::memcpy(&dtor, &buf, sizeof(dtor));
return destructor<Class>(dtor);
}

#else

template<typename Signature, typename Lib>
constructor<Signature> load_ctor(Lib & lib, const mangled_storage_impl::ctor_sym & ct) {
typedef typename constructor<Signature>::standard_t   stand;
typedef typename constructor<Signature>::allocating_t alloc;

stand s = nullptr;
alloc a = nullptr;


if (!ct.C1.empty())
{
void *buf = &lib.template get<unsigned char>(ct.C1);
std::memcpy(&s, &buf, sizeof(void*));
}
if (!ct.C3.empty())
{
void *buf = &lib.template get<unsigned char>(ct.C3);
std::memcpy(&a, &buf, sizeof(void*));
}

return constructor<Signature>(s,a);
}

template<typename Class, typename Lib>
destructor<Class> load_dtor(Lib & lib, const mangled_storage_impl::dtor_sym & dt) {
typedef typename destructor<Class>::standard_t stand;
typedef typename destructor<Class>::deleting_t delet;

stand s = nullptr;
delet d = nullptr;

if (!dt.D1.empty()) {
s = &lib.template get< typename boost::remove_pointer<stand>::type >(dt.D1);
}

if (!dt.D0.empty()) {
d = &lib.template get< typename boost::remove_pointer<delet>::type >(dt.D0);
}

return destructor<Class>(s,d);

}

#endif

}}} 

#endif 
