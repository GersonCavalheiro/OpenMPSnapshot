



#ifndef OMEGA_H_ANY_HPP
#define OMEGA_H_ANY_HPP

#include <Omega_h_fail.hpp>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>

namespace Omega_h {

class bad_any_cast : public std::bad_cast {
public:
const char* what() const noexcept override;
};

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

class any final {
public:
any() : vtable(nullptr) {}

any(const any& rhs) : vtable(rhs.vtable) {
if (!rhs.empty()) {
rhs.vtable->copy(rhs.storage, this->storage);
}
}

any(any&& rhs) noexcept : vtable(rhs.vtable) {
if (!rhs.empty()) {
rhs.vtable->move(rhs.storage, this->storage);
rhs.vtable = nullptr;
}
}

~any() { this->clear(); }

template <typename ValueType,
typename = typename std::enable_if<!std::is_same<
typename std::decay<ValueType>::type, any>::value>::type>
any(ValueType&& value) {
static_assert(
std::is_copy_constructible<typename std::decay<ValueType>::type>::value,
"T shall satisfy the CopyConstructible requirements.");
this->construct(std::forward<ValueType>(value));
}

any& operator=(const any& rhs) {
any(rhs).swap(*this);
return *this;
}

any& operator=(any&& rhs) noexcept {
any(std::move(rhs)).swap(*this);
return *this;
}

template <typename ValueType,
typename = typename std::enable_if<!std::is_same<
typename std::decay<ValueType>::type, any>::value>::type>
any& operator=(ValueType&& value) {
static_assert(
std::is_copy_constructible<typename std::decay<ValueType>::type>::value,
"T shall satisfy the CopyConstructible requirements.");
any(std::forward<ValueType>(value)).swap(*this);
return *this;
}

void clear() noexcept {
if (!empty()) {
this->vtable->destroy(storage);
this->vtable = nullptr;
}
}

bool empty() const noexcept { return this->vtable == nullptr; }

const std::type_info& type() const noexcept {
return empty() ? typeid(void) : this->vtable->type();
}

void swap(any& rhs) noexcept {
if (this->vtable != rhs.vtable) {
any tmp(std::move(rhs));

rhs.vtable = this->vtable;
if (this->vtable != nullptr) {
this->vtable->move(this->storage, rhs.storage);
}

this->vtable = tmp.vtable;
if (tmp.vtable != nullptr) {
tmp.vtable->move(tmp.storage, this->storage);
tmp.vtable = nullptr;
}
} else  
{
if (this->vtable != nullptr)
this->vtable->swap(this->storage, rhs.storage);
}
}

private:  
union storage_union {
using stack_storage_t = typename std::aligned_storage<2 * sizeof(void*),
std::alignment_of<void*>::value>::type;

void* dynamic;
stack_storage_t stack;  
};

struct vtable_type {

const std::type_info& (*type)() noexcept;

void (*destroy)(storage_union&) noexcept;

void (*copy)(const storage_union& src, storage_union& dest);

void (*move)(storage_union& src, storage_union& dest) noexcept;

void (*swap)(storage_union& lhs, storage_union& rhs) noexcept;
};

template <typename T>
struct vtable_dynamic {
static const std::type_info& type() noexcept { return typeid(T); }

static void destroy(storage_union& storage) noexcept {
delete reinterpret_cast<T*>(storage.dynamic);
}

static void copy(const storage_union& src, storage_union& dest) {
dest.dynamic = new T(*reinterpret_cast<const T*>(src.dynamic));
}

static void move(storage_union& src, storage_union& dest) noexcept {
dest.dynamic = src.dynamic;
src.dynamic = nullptr;
}

static void swap(storage_union& lhs, storage_union& rhs) noexcept {
std::swap(lhs.dynamic, rhs.dynamic);
}
};

template <typename T>
struct vtable_stack {
static const std::type_info& type() noexcept { return typeid(T); }

static void destroy(storage_union& storage) noexcept {
reinterpret_cast<T*>(&storage.stack)->~T();
}

static void copy(const storage_union& src, storage_union& dest) {
new (&dest.stack) T(reinterpret_cast<const T&>(src.stack));
}

static void move(storage_union& src, storage_union& dest) noexcept {
new (&dest.stack) T(std::move(reinterpret_cast<T&>(src.stack)));
destroy(src);
}

static void swap(storage_union& lhs, storage_union& rhs) noexcept {
std::swap(
reinterpret_cast<T&>(lhs.stack), reinterpret_cast<T&>(rhs.stack));
}
};

template <typename T>
struct requires_allocation
: std::integral_constant<bool,
!(std::is_nothrow_move_constructible<T>::value  
&& sizeof(T) <= sizeof(storage_union::stack) &&
std::alignment_of<T>::value <=
std::alignment_of<storage_union::stack_storage_t>::value)> {
};

template <typename T>
static vtable_type* vtable_for_type() {
using VTableType = typename std::conditional<requires_allocation<T>::value,
vtable_dynamic<T>, vtable_stack<T>>::type;
static vtable_type table = {
VTableType::type,
VTableType::destroy,
VTableType::copy,
VTableType::move,
VTableType::swap,
};
return &table;
}

protected:
template <typename T>
friend const T* any_cast(const any* operand) noexcept;
template <typename T>
friend T* any_cast(any* operand) noexcept;

bool is_typed(const std::type_info& t) const {
return is_same(this->type(), t);
}

static bool is_same(const std::type_info& a, const std::type_info& b) {
#ifdef ANY_IMPL_FAST_TYPE_INFO_COMPARE
return &a == &b;
#else
return a == b;
#endif
}

template <typename T>
const T* cast() const noexcept {
return requires_allocation<typename std::decay<T>::type>::value
? reinterpret_cast<const T*>(storage.dynamic)
: reinterpret_cast<const T*>(&storage.stack);
}

template <typename T>
T* cast() noexcept {
return requires_allocation<typename std::decay<T>::type>::value
? reinterpret_cast<T*>(storage.dynamic)
: reinterpret_cast<T*>(&storage.stack);
}

private:
storage_union storage;  
vtable_type* vtable;

template <typename ValueType, typename T>
typename std::enable_if<requires_allocation<T>::value>::type do_construct(
ValueType&& value) {
storage.dynamic = new T(std::forward<ValueType>(value));
}

template <typename ValueType, typename T>
typename std::enable_if<!requires_allocation<T>::value>::type do_construct(
ValueType&& value) {
new (&storage.stack) T(std::forward<ValueType>(value));
}

template <typename ValueType>
void construct(ValueType&& value) {
using T = typename std::decay<ValueType>::type;

this->vtable = vtable_for_type<T>();

do_construct<ValueType, T>(std::forward<ValueType>(value));
}
};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace detail {
template <typename ValueType>
inline ValueType any_cast_move_if_true(
typename std::remove_reference<ValueType>::type* p, std::true_type) {
return std::move(*p);
}

template <typename ValueType>
inline ValueType any_cast_move_if_true(
typename std::remove_reference<ValueType>::type* p, std::false_type) {
return *p;
}
}  

template <typename ValueType>
inline ValueType any_cast(const any& operand) {
auto p = any_cast<typename std::add_const<
typename std::remove_reference<ValueType>::type>::type>(&operand);
if (p == nullptr) throw bad_any_cast();
return *p;
}

template <typename ValueType>
inline ValueType any_cast(any& operand) {
auto p = any_cast<typename std::remove_reference<ValueType>::type>(&operand);
if (p == nullptr) throw bad_any_cast();
return *p;
}

template <typename ValueType>
inline ValueType any_cast(any&& operand) {
using can_move = std::integral_constant<bool,
std::is_move_constructible<ValueType>::value &&
!std::is_lvalue_reference<ValueType>::value>;

auto p = any_cast<typename std::remove_reference<ValueType>::type>(&operand);
if (p == nullptr) throw bad_any_cast();
return detail::any_cast_move_if_true<ValueType>(p, can_move());
}

template <typename T>
inline const T* any_cast(const any* operand) noexcept {
if (operand == nullptr || !operand->is_typed(typeid(T)))
return nullptr;
else
return operand->cast<T>();
}

template <typename T>
inline T* any_cast(any* operand) noexcept {
if (operand == nullptr || !operand->is_typed(typeid(T)))
return nullptr;
else
return operand->cast<T>();
}

template <typename T>
T&& move_value(any& a) {
auto any_ptr = &(a);
auto value_ptr = any_cast<T>(any_ptr);
OMEGA_H_CHECK(value_ptr != nullptr);
return std::move(*value_ptr);
}

inline void swap(Omega_h::any& lhs, Omega_h::any& rhs) noexcept {
lhs.swap(rhs);
}

}  

#endif
