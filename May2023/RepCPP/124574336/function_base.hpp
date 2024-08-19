


#ifndef BOOST_FUNCTION_BASE_HEADER
#define BOOST_FUNCTION_BASE_HEADER

#include <stdexcept>
#include <string>
#include <memory>
#include <new>
#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/integer.hpp>
#include <boost/type_index.hpp>
#include <boost/type_traits/has_trivial_copy.hpp>
#include <boost/type_traits/has_trivial_destructor.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_volatile.hpp>
#include <boost/type_traits/composite_traits.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/config/workaround.hpp>
#include <boost/type_traits/alignment_of.hpp>
#ifndef BOOST_NO_SFINAE
#include <boost/type_traits/enable_if.hpp>
#else
#include <boost/type_traits/integral_constant.hpp>
#endif
#include <boost/function_equal.hpp>
#include <boost/function/function_fwd.hpp>

#if defined(BOOST_MSVC)
#   pragma warning( push )
#   pragma warning( disable : 4793 ) 
#   pragma warning( disable : 4127 ) 
#endif

#if defined(__ICL) && __ICL <= 600 || defined(__MWERKS__) && __MWERKS__ < 0x2406 && !defined(BOOST_STRICT_CONFIG)
#  define BOOST_FUNCTION_TARGET_FIX(x) x
#else
#  define BOOST_FUNCTION_TARGET_FIX(x)
#endif 

#  define BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor,Type)              \
typename ::boost::enable_if_<          \
!(::boost::is_integral<Functor>::value), \
Type>::type

namespace boost {
namespace detail {
namespace function {
class X;


union function_buffer_members
{
typedef void* obj_ptr_t;
mutable obj_ptr_t obj_ptr;

struct type_t {
const boost::typeindex::type_info* type;

bool const_qualified;
bool volatile_qualified;
} type;

typedef void (*func_ptr_t)();
mutable func_ptr_t func_ptr;

struct bound_memfunc_ptr_t {
void (X::*memfunc_ptr)(int);
void* obj_ptr;
} bound_memfunc_ptr;

struct obj_ref_t {
mutable void* obj_ptr;
bool is_const_qualified;
bool is_volatile_qualified;
} obj_ref;
};

union BOOST_SYMBOL_VISIBLE function_buffer
{
mutable function_buffer_members members;

mutable char data[sizeof(function_buffer_members)];
};


struct unusable
{
unusable() {}
template<typename T> unusable(const T&) {}
};


template<typename T> struct function_return_type { typedef T type; };

template<>
struct function_return_type<void>
{
typedef unusable type;
};

enum functor_manager_operation_type {
clone_functor_tag,
move_functor_tag,
destroy_functor_tag,
check_functor_type_tag,
get_functor_type_tag
};

struct function_ptr_tag {};
struct function_obj_tag {};
struct member_ptr_tag {};
struct function_obj_ref_tag {};

template<typename F>
class get_function_tag
{
typedef typename conditional<(is_pointer<F>::value),
function_ptr_tag,
function_obj_tag>::type ptr_or_obj_tag;

typedef typename conditional<(is_member_pointer<F>::value),
member_ptr_tag,
ptr_or_obj_tag>::type ptr_or_obj_or_mem_tag;

typedef typename conditional<(is_reference_wrapper<F>::value),
function_obj_ref_tag,
ptr_or_obj_or_mem_tag>::type or_ref_tag;

public:
typedef or_ref_tag type;
};

template<typename F>
struct reference_manager
{
static inline void
manage(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op)
{
switch (op) {
case clone_functor_tag:
out_buffer.members.obj_ref = in_buffer.members.obj_ref;
return;

case move_functor_tag:
out_buffer.members.obj_ref = in_buffer.members.obj_ref;
in_buffer.members.obj_ref.obj_ptr = 0;
return;

case destroy_functor_tag:
out_buffer.members.obj_ref.obj_ptr = 0;
return;

case check_functor_type_tag:
{
if (*out_buffer.members.type.type == boost::typeindex::type_id<F>()
&& (!in_buffer.members.obj_ref.is_const_qualified
|| out_buffer.members.type.const_qualified)
&& (!in_buffer.members.obj_ref.is_volatile_qualified
|| out_buffer.members.type.volatile_qualified))
out_buffer.members.obj_ptr = in_buffer.members.obj_ref.obj_ptr;
else
out_buffer.members.obj_ptr = 0;
}
return;

case get_functor_type_tag:
out_buffer.members.type.type = &boost::typeindex::type_id<F>().type_info();
out_buffer.members.type.const_qualified = in_buffer.members.obj_ref.is_const_qualified;
out_buffer.members.type.volatile_qualified = in_buffer.members.obj_ref.is_volatile_qualified;
return;
}
}
};


template<typename F>
struct function_allows_small_object_optimization
{
BOOST_STATIC_CONSTANT
(bool,
value = ((sizeof(F) <= sizeof(function_buffer) &&
(alignment_of<function_buffer>::value
% alignment_of<F>::value == 0))));
};

template <typename F,typename A>
struct functor_wrapper: public F, public A
{
functor_wrapper( F f, A a ):
F(f),
A(a)
{
}

functor_wrapper(const functor_wrapper& f) :
F(static_cast<const F&>(f)),
A(static_cast<const A&>(f))
{
}
};


template<typename Functor>
struct functor_manager_common
{
typedef Functor functor_type;

static inline void
manage_ptr(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op)
{
if (op == clone_functor_tag)
out_buffer.members.func_ptr = in_buffer.members.func_ptr;
else if (op == move_functor_tag) {
out_buffer.members.func_ptr = in_buffer.members.func_ptr;
in_buffer.members.func_ptr = 0;
} else if (op == destroy_functor_tag)
out_buffer.members.func_ptr = 0;
else if (op == check_functor_type_tag) {
if (*out_buffer.members.type.type == boost::typeindex::type_id<Functor>())
out_buffer.members.obj_ptr = &in_buffer.members.func_ptr;
else
out_buffer.members.obj_ptr = 0;
} else  {
out_buffer.members.type.type = &boost::typeindex::type_id<Functor>().type_info();
out_buffer.members.type.const_qualified = false;
out_buffer.members.type.volatile_qualified = false;
}
}

static inline void
manage_small(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op)
{
if (op == clone_functor_tag || op == move_functor_tag) {
const functor_type* in_functor =
reinterpret_cast<const functor_type*>(in_buffer.data);
new (reinterpret_cast<void*>(out_buffer.data)) functor_type(*in_functor);

if (op == move_functor_tag) {
functor_type* f = reinterpret_cast<functor_type*>(in_buffer.data);
(void)f; 
f->~Functor();
}
} else if (op == destroy_functor_tag) {
functor_type* f = reinterpret_cast<functor_type*>(out_buffer.data);
(void)f; 
f->~Functor();
} else if (op == check_functor_type_tag) {
if (*out_buffer.members.type.type == boost::typeindex::type_id<Functor>())
out_buffer.members.obj_ptr = in_buffer.data;
else
out_buffer.members.obj_ptr = 0;
} else  {
out_buffer.members.type.type = &boost::typeindex::type_id<Functor>().type_info();
out_buffer.members.type.const_qualified = false;
out_buffer.members.type.volatile_qualified = false;
}
}
};

template<typename Functor>
struct functor_manager
{
private:
typedef Functor functor_type;

static inline void
manager(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op, function_ptr_tag)
{
functor_manager_common<Functor>::manage_ptr(in_buffer,out_buffer,op);
}

static inline void
manager(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op, true_type)
{
functor_manager_common<Functor>::manage_small(in_buffer,out_buffer,op);
}

static inline void
manager(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op, false_type)
{
if (op == clone_functor_tag) {
const functor_type* f =
static_cast<const functor_type*>(in_buffer.members.obj_ptr);
functor_type* new_f = new functor_type(*f);
out_buffer.members.obj_ptr = new_f;
} else if (op == move_functor_tag) {
out_buffer.members.obj_ptr = in_buffer.members.obj_ptr;
in_buffer.members.obj_ptr = 0;
} else if (op == destroy_functor_tag) {

functor_type* f =
static_cast<functor_type*>(out_buffer.members.obj_ptr);
delete f;
out_buffer.members.obj_ptr = 0;
} else if (op == check_functor_type_tag) {
if (*out_buffer.members.type.type == boost::typeindex::type_id<Functor>())
out_buffer.members.obj_ptr = in_buffer.members.obj_ptr;
else
out_buffer.members.obj_ptr = 0;
} else  {
out_buffer.members.type.type = &boost::typeindex::type_id<Functor>().type_info();
out_buffer.members.type.const_qualified = false;
out_buffer.members.type.volatile_qualified = false;
}
}

static inline void
manager(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op, function_obj_tag)
{
manager(in_buffer, out_buffer, op,
integral_constant<bool, (function_allows_small_object_optimization<functor_type>::value)>());
}

static inline void
manager(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op, member_ptr_tag)
{
manager(in_buffer, out_buffer, op, true_type());
}

public:

static inline void
manage(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op)
{
typedef typename get_function_tag<functor_type>::type tag_type;
if (op == get_functor_type_tag) {
out_buffer.members.type.type = &boost::typeindex::type_id<functor_type>().type_info();
out_buffer.members.type.const_qualified = false;
out_buffer.members.type.volatile_qualified = false;
} else {
manager(in_buffer, out_buffer, op, tag_type());
}
}
};

template<typename Functor, typename Allocator>
struct functor_manager_a
{
private:
typedef Functor functor_type;

static inline void
manager(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op, function_ptr_tag)
{
functor_manager_common<Functor>::manage_ptr(in_buffer,out_buffer,op);
}

static inline void
manager(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op, true_type)
{
functor_manager_common<Functor>::manage_small(in_buffer,out_buffer,op);
}

static inline void
manager(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op, false_type)
{
typedef functor_wrapper<Functor,Allocator> functor_wrapper_type;
#if defined(BOOST_NO_CXX11_ALLOCATOR)
typedef typename Allocator::template rebind<functor_wrapper_type>::other
wrapper_allocator_type;
typedef typename wrapper_allocator_type::pointer wrapper_allocator_pointer_type;
#else
using wrapper_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<functor_wrapper_type>;
using wrapper_allocator_pointer_type = typename std::allocator_traits<wrapper_allocator_type>::pointer;
#endif

if (op == clone_functor_tag) {
const functor_wrapper_type* f =
static_cast<const functor_wrapper_type*>(in_buffer.members.obj_ptr);
wrapper_allocator_type wrapper_allocator(static_cast<Allocator const &>(*f));
wrapper_allocator_pointer_type copy = wrapper_allocator.allocate(1);
#if defined(BOOST_NO_CXX11_ALLOCATOR)
wrapper_allocator.construct(copy, *f);
#else
std::allocator_traits<wrapper_allocator_type>::construct(wrapper_allocator, copy, *f);
#endif

functor_wrapper_type* new_f = static_cast<functor_wrapper_type*>(copy);
out_buffer.members.obj_ptr = new_f;
} else if (op == move_functor_tag) {
out_buffer.members.obj_ptr = in_buffer.members.obj_ptr;
in_buffer.members.obj_ptr = 0;
} else if (op == destroy_functor_tag) {

functor_wrapper_type* victim =
static_cast<functor_wrapper_type*>(in_buffer.members.obj_ptr);
wrapper_allocator_type wrapper_allocator(static_cast<Allocator const &>(*victim));
#if defined(BOOST_NO_CXX11_ALLOCATOR)
wrapper_allocator.destroy(victim);
#else
std::allocator_traits<wrapper_allocator_type>::destroy(wrapper_allocator, victim);
#endif
wrapper_allocator.deallocate(victim,1);
out_buffer.members.obj_ptr = 0;
} else if (op == check_functor_type_tag) {
if (*out_buffer.members.type.type == boost::typeindex::type_id<Functor>())
out_buffer.members.obj_ptr = in_buffer.members.obj_ptr;
else
out_buffer.members.obj_ptr = 0;
} else  {
out_buffer.members.type.type = &boost::typeindex::type_id<Functor>().type_info();
out_buffer.members.type.const_qualified = false;
out_buffer.members.type.volatile_qualified = false;
}
}

static inline void
manager(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op, function_obj_tag)
{
manager(in_buffer, out_buffer, op,
integral_constant<bool, (function_allows_small_object_optimization<functor_type>::value)>());
}

public:

static inline void
manage(const function_buffer& in_buffer, function_buffer& out_buffer,
functor_manager_operation_type op)
{
typedef typename get_function_tag<functor_type>::type tag_type;
if (op == get_functor_type_tag) {
out_buffer.members.type.type = &boost::typeindex::type_id<functor_type>().type_info();
out_buffer.members.type.const_qualified = false;
out_buffer.members.type.volatile_qualified = false;
} else {
manager(in_buffer, out_buffer, op, tag_type());
}
}
};

struct useless_clear_type {};

#ifdef BOOST_NO_SFINAE
template<typename Function, typename Functor>
bool
compare_equal(const Function& f, const Functor&, int, true_type)
{ return f.empty(); }

template<typename Function, typename Functor>
bool
compare_not_equal(const Function& f, const Functor&, int,
true_type)
{ return !f.empty(); }

template<typename Function, typename Functor>
bool
compare_equal(const Function& f, const Functor& g, long,
false_type)
{
if (const Functor* fp = f.template target<Functor>())
return function_equal(*fp, g);
else return false;
}

template<typename Function, typename Functor>
bool
compare_equal(const Function& f, const reference_wrapper<Functor>& g,
int, false_type)
{
if (const Functor* fp = f.template target<Functor>())
return fp == g.get_pointer();
else return false;
}

template<typename Function, typename Functor>
bool
compare_not_equal(const Function& f, const Functor& g, long,
false_type)
{
if (const Functor* fp = f.template target<Functor>())
return !function_equal(*fp, g);
else return true;
}

template<typename Function, typename Functor>
bool
compare_not_equal(const Function& f,
const reference_wrapper<Functor>& g, int,
false_type)
{
if (const Functor* fp = f.template target<Functor>())
return fp != g.get_pointer();
else return true;
}
#endif 


struct vtable_base
{
void (*manager)(const function_buffer& in_buffer,
function_buffer& out_buffer,
functor_manager_operation_type op);
};
} 
} 


class function_base
{
public:
function_base() : vtable(0) { }


bool empty() const { return !vtable; }


const boost::typeindex::type_info& target_type() const
{
if (!vtable) return boost::typeindex::type_id<void>().type_info();

detail::function::function_buffer type;
get_vtable()->manager(functor, type, detail::function::get_functor_type_tag);
return *type.members.type.type;
}

template<typename Functor>
Functor* target()
{
if (!vtable) return 0;

detail::function::function_buffer type_result;
type_result.members.type.type = &boost::typeindex::type_id<Functor>().type_info();
type_result.members.type.const_qualified = is_const<Functor>::value;
type_result.members.type.volatile_qualified = is_volatile<Functor>::value;
get_vtable()->manager(functor, type_result,
detail::function::check_functor_type_tag);
return static_cast<Functor*>(type_result.members.obj_ptr);
}

template<typename Functor>
const Functor* target() const
{
if (!vtable) return 0;

detail::function::function_buffer type_result;
type_result.members.type.type = &boost::typeindex::type_id<Functor>().type_info();
type_result.members.type.const_qualified = true;
type_result.members.type.volatile_qualified = is_volatile<Functor>::value;
get_vtable()->manager(functor, type_result,
detail::function::check_functor_type_tag);
return static_cast<const Functor*>(type_result.members.obj_ptr);
}

template<typename F>
bool contains(const F& f) const
{
if (const F* fp = this->template target<F>())
{
return function_equal(*fp, f);
} else {
return false;
}
}

#if defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3
template<typename Functor>
BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor, bool)
operator==(Functor g) const
{
if (const Functor* fp = target<Functor>())
return function_equal(*fp, g);
else return false;
}

template<typename Functor>
BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor, bool)
operator!=(Functor g) const
{
if (const Functor* fp = target<Functor>())
return !function_equal(*fp, g);
else return true;
}
#endif

public: 
detail::function::vtable_base* get_vtable() const {
return reinterpret_cast<detail::function::vtable_base*>(
reinterpret_cast<std::size_t>(vtable) & ~static_cast<std::size_t>(0x01));
}

bool has_trivial_copy_and_destroy() const {
return reinterpret_cast<std::size_t>(vtable) & 0x01;
}

detail::function::vtable_base* vtable;
mutable detail::function::function_buffer functor;
};

#if defined(BOOST_CLANG)
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wweak-vtables"
#endif

class BOOST_SYMBOL_VISIBLE bad_function_call : public std::runtime_error
{
public:
bad_function_call() : std::runtime_error("call to empty boost::function") {}
};
#if defined(BOOST_CLANG)
#   pragma clang diagnostic pop
#endif

#ifndef BOOST_NO_SFINAE
inline bool operator==(const function_base& f,
detail::function::useless_clear_type*)
{
return f.empty();
}

inline bool operator!=(const function_base& f,
detail::function::useless_clear_type*)
{
return !f.empty();
}

inline bool operator==(detail::function::useless_clear_type*,
const function_base& f)
{
return f.empty();
}

inline bool operator!=(detail::function::useless_clear_type*,
const function_base& f)
{
return !f.empty();
}
#endif

#ifdef BOOST_NO_SFINAE
template<typename Functor>
inline bool operator==(const function_base& f, Functor g)
{
typedef integral_constant<bool, (is_integral<Functor>::value)> integral;
return detail::function::compare_equal(f, g, 0, integral());
}

template<typename Functor>
inline bool operator==(Functor g, const function_base& f)
{
typedef integral_constant<bool, (is_integral<Functor>::value)> integral;
return detail::function::compare_equal(f, g, 0, integral());
}

template<typename Functor>
inline bool operator!=(const function_base& f, Functor g)
{
typedef integral_constant<bool, (is_integral<Functor>::value)> integral;
return detail::function::compare_not_equal(f, g, 0, integral());
}

template<typename Functor>
inline bool operator!=(Functor g, const function_base& f)
{
typedef integral_constant<bool, (is_integral<Functor>::value)> integral;
return detail::function::compare_not_equal(f, g, 0, integral());
}
#else

#  if !(defined(__GNUC__) && __GNUC__ == 3 && __GNUC_MINOR__ <= 3)
template<typename Functor>
BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor, bool)
operator==(const function_base& f, Functor g)
{
if (const Functor* fp = f.template target<Functor>())
return function_equal(*fp, g);
else return false;
}

template<typename Functor>
BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor, bool)
operator==(Functor g, const function_base& f)
{
if (const Functor* fp = f.template target<Functor>())
return function_equal(g, *fp);
else return false;
}

template<typename Functor>
BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor, bool)
operator!=(const function_base& f, Functor g)
{
if (const Functor* fp = f.template target<Functor>())
return !function_equal(*fp, g);
else return true;
}

template<typename Functor>
BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor, bool)
operator!=(Functor g, const function_base& f)
{
if (const Functor* fp = f.template target<Functor>())
return !function_equal(g, *fp);
else return true;
}
#  endif

template<typename Functor>
BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor, bool)
operator==(const function_base& f, reference_wrapper<Functor> g)
{
if (const Functor* fp = f.template target<Functor>())
return fp == g.get_pointer();
else return false;
}

template<typename Functor>
BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor, bool)
operator==(reference_wrapper<Functor> g, const function_base& f)
{
if (const Functor* fp = f.template target<Functor>())
return g.get_pointer() == fp;
else return false;
}

template<typename Functor>
BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor, bool)
operator!=(const function_base& f, reference_wrapper<Functor> g)
{
if (const Functor* fp = f.template target<Functor>())
return fp != g.get_pointer();
else return true;
}

template<typename Functor>
BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL(Functor, bool)
operator!=(reference_wrapper<Functor> g, const function_base& f)
{
if (const Functor* fp = f.template target<Functor>())
return g.get_pointer() != fp;
else return true;
}

#endif 

namespace detail {
namespace function {
inline bool has_empty_target(const function_base* f)
{
return f->empty();
}

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1310)
inline bool has_empty_target(const void*)
{
return false;
}
#else
inline bool has_empty_target(...)
{
return false;
}
#endif
} 
} 
} 

#undef BOOST_FUNCTION_ENABLE_IF_NOT_INTEGRAL

#if defined(BOOST_MSVC)
#   pragma warning( pop )
#endif

#endif 
