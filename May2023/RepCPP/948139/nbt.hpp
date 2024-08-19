#pragma once
#ifndef NBT_HPP_
#define NBT_HPP_

#include <logger.hpp>
#include <map>
#include <nbt/iterators.hpp>
#include <nbt/tag_types.hpp>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace nbt {

class NBT {
private:
template <typename NBTType> friend class nbt::iter;

public:
using tag_end_t = int8_t;
using tag_byte_t = int8_t;
using tag_short_t = int16_t;
using tag_int_t = int32_t;
using tag_long_t = int64_t;
using tag_float_t = float;
using tag_double_t = double;
using tag_byte_array_t = std::vector<int8_t>;
using tag_string_t = std::string;
using tag_list_t = std::vector<NBT>;
using tag_compound_t = std::map<std::string, NBT>;
using tag_int_array_t = std::vector<int32_t>;
using tag_long_array_t = std::vector<int64_t>;

using key_type = std::string;
using value_type = NBT;
using reference = value_type &;
using const_reference = const value_type &;
using difference_type = std::ptrdiff_t;
using size_type = std::size_t;
using allocator_type = std::allocator<NBT>;
using pointer = typename std::allocator_traits<allocator_type>::pointer;
using const_pointer =
typename std::allocator_traits<allocator_type>::const_pointer;
using iterator = iter<NBT>;
using const_iterator = iter<const NBT>;

NBT(const tag_type type, key_type name_ = "")
: type(type), content(type), name(name_){};

NBT(std::nullptr_t = nullptr) : NBT(tag_type::tag_end){};

template <
typename Integer,
typename std::enable_if<std::is_integral<Integer>::value, int>::type = 0>
NBT(const Integer value, key_type name_ = "") : name(name_) {
switch (std::alignment_of<Integer>()) {
case 1:
type = tag_type::tag_byte;
content = tag_content(tag_byte_t(value));
break;

case 2:
type = tag_type::tag_short;
content = tag_content(tag_short_t(value));
break;

case 3:
case 4:
type = tag_type::tag_int;
content = tag_content(tag_int_t(value));
break;

default:
type = tag_type::tag_long;
content = tag_content(tag_long_t(value));
break;
}
}

template <typename Float,
typename std::enable_if<std::is_floating_point<Float>::value,
int>::type = 0>
NBT(const Float value, key_type name_ = "") : name(name_) {
switch (std::alignment_of<Float>()) {
case 4:
type = tag_type::tag_float;
content = tag_content(tag_float_t(value));
break;

default:
type = tag_type::tag_double;
content = tag_content(tag_double_t(value));
break;
}
}

NBT(const tag_string_t str, key_type name_ = "")
: type(tag_type::tag_string), content(str), name(name_){};

template <
typename Integer,
typename std::enable_if<std::is_integral<Integer>::value, int>::type = 0>
NBT(const std::vector<Integer> &data, key_type name_ = "") : name(name_) {
switch (std::alignment_of<Integer>()) {
case 1:
type = tag_type::tag_byte_array;
content = tag_content(tag_byte_array_t(data.begin(), data.end()));
break;

case 2:
case 3:
case 4:
type = tag_type::tag_int_array;
content = tag_content(tag_int_array_t(data.begin(), data.end()));
break;

default:
type = tag_type::tag_long_array;
content = tag_content(tag_long_array_t(data.begin(), data.end()));
break;
}
}

NBT(const tag_list_t &data, key_type name_ = "")
: type(tag_type::tag_list), content(data), name(name_){};

NBT(const tag_compound_t &data, key_type name_ = "")
: type(tag_type::tag_compound), content(data), name(name_){};

NBT(const NBT &other) { *this = other; }

NBT(NBT &&other) noexcept { *this = std::move(other); }

~NBT() { content.destroy(type); }

void set_name(const std::string &name_) { name = name_; };
std::string get_name() const { return name; };

nbt::tag_type get_type() const { return type; };

constexpr bool is_end() const noexcept { return type == tag_type::tag_end; }
constexpr bool is_byte() const noexcept { return type == tag_type::tag_byte; }
constexpr bool is_short() const noexcept {
return type == tag_type::tag_short;
}
constexpr bool is_int() const noexcept { return type == tag_type::tag_int; }
constexpr bool is_long() const noexcept { return type == tag_type::tag_long; }
constexpr bool is_float() const noexcept {
return type == tag_type::tag_float;
}
constexpr bool is_double() const noexcept {
return type == tag_type::tag_double;
}
constexpr bool is_byte_array() const noexcept {
return type == tag_type::tag_byte_array;
}
constexpr bool is_string() const noexcept {
return type == tag_type::tag_string;
}
constexpr bool is_list() const noexcept { return type == tag_type::tag_list; }
constexpr bool is_compound() const noexcept {
return type == tag_type::tag_compound;
}
constexpr bool is_int_array() const noexcept {
return type == tag_type::tag_int_array;
}
constexpr bool is_long_array() const noexcept {
return type == tag_type::tag_long_array;
}

reference at(size_type index) {
if (is_list()) {
return content.list->at(index);
} else {
throw(std::domain_error("Cannot use at() with " +
std::string(type_name())));
}
}

const_reference at(size_type index) const {
if (is_list()) {
return content.list->at(index);
} else {
throw(std::domain_error("Cannot use at() with " +
std::string(type_name())));
}
}

reference at(const std::string &key) {
if (is_compound()) {
return content.compound->at(key);
}
throw(std::domain_error("Invalid type"));
}

const_reference at(const std::string &key) const {
if (is_compound()) {
return content.compound->at(key);
}
throw(std::domain_error("Invalid type"));
}

reference operator[](size_type index) {
if (is_list()) {
return content.list->operator[](index);
}
throw(std::domain_error(
"Cannot use operator[] with a numeric argument on tag of type " +
std::string(type_name())));
}

const_reference operator[](size_type index) const {
if (is_list()) {
return content.list->operator[](index);
}
throw(std::domain_error(
"Cannot use operator[] with a numeric argument on tag of type " +
std::string(type_name())));
}

reference operator[](const std::string &key) {
if (is_compound()) {
return content.compound->operator[](key);
}
throw(std::domain_error(fmt::format(
"Cannot use operator[] with a string argument on tag of type {}",
type_name())));
}

const_reference operator[](const std::string &key) const {
if (is_compound()) {
auto query = content.compound->find(key);
if (query != content.compound->end()) {
return query->second;
} else
throw(std::out_of_range("Key " + key + " not found"));
}
throw(std::domain_error(
"Cannot use operator[] with a string argument on tag of type " +
std::string(type_name())));
}

std::pair<tag_compound_t::iterator, bool>
insert(const tag_compound_t::value_type &value) {
if (is_compound())
return content.compound->insert(value);
throw(std::domain_error("Cannot use insert on tag of type " +
std::string(type_name())));
}

std::pair<tag_compound_t::iterator, bool>
insert(tag_compound_t::value_type &&value) {
if (is_compound())
return content.compound->insert(
std::forward<tag_compound_t::value_type>(value));
throw(std::domain_error("Cannot use insert on tag of type " +
std::string(type_name())));
}

size_type erase(const key_type &key) {
if (is_compound())
return content.compound->erase(key);
throw(std::domain_error(
"Cannot use erase with a string argument on tag of type " +
std::string(type_name())));
}

template <typename T> reference operator[](T *key) {
if (is_end()) {
type = tag_type::tag_compound;
content = tag_type::tag_compound;
}

if (is_compound()) {
return content.compound->operator[](key);
}

throw(std::domain_error(
fmt::format("Cannot use operator[] with type `{}` (key `{}` on `{}`)",
type_name(), key, get_name())));
}

NBT &operator=(const NBT &other) noexcept {
type = other.type;
name = other.name;
content = other.content.clone(type);

return *this;
}

NBT &operator=(NBT &&other) noexcept {
std::swap(type, other.type);
std::swap(name, other.name);
std::swap(content, other.content);

return *this;
}

const char *type_name() const noexcept {
switch (type) {
case tag_type::tag_byte:
return "byte";
case tag_type::tag_short:
return "short";
case tag_type::tag_int:
return "int";
case tag_type::tag_long:
return "long";
case tag_type::tag_float:
return "float";
case tag_type::tag_double:
return "double";
case tag_type::tag_byte_array:
return "byte_array";
case tag_type::tag_string:
return "string";
case tag_type::tag_list:
return "list";
case tag_type::tag_compound:
return "compound";
case tag_type::tag_int_array:
return "int_array";
case tag_type::tag_long_array:
return "long_array";
default:
return "end";
}
}

reference front() { return *begin(); }

const_reference front() const { return *cbegin(); }

reference back() {
auto tmp = end();
--tmp;
return *tmp;
}

const_reference back() const {
auto tmp = cend();
--tmp;
return *tmp;
}

iterator begin() noexcept {
iterator result(this);
result.set_begin();
return result;
}

const_iterator begin() const noexcept { return cbegin(); }

const_iterator cbegin() const noexcept {
const_iterator result(this);
result.set_begin();
return result;
}

iterator end() noexcept {
iterator result(this);
result.set_end();
return result;
}

const_iterator end() const noexcept { return cend(); }

const_iterator cend() const noexcept {
const_iterator result(this);
result.set_end();
return result;
}

iterator find(std::string &&key) {
auto result = end();

if (is_compound()) {
result.it.compound_iterator =
content.compound->find(std::forward<std::string>(key));
}

return result;
}

const_iterator find(std::string &&key) const {
auto result = cend();

if (is_compound()) {
result.it.compound_iterator =
content.compound->find(std::forward<std::string>(key));
}

return result;
}

size_type count(std::string &&key) const {
return is_compound()
? content.compound->count(std::forward<std::string>(key))
: 0;
}

bool contains(std::string &&key) const {
return is_compound() &&
content.compound->find(std::forward<std::string>(key)) !=
content.compound->end();
}

bool empty() const noexcept {
switch (type) {
case tag_type::tag_end:
return true;
case tag_type::tag_byte_array:
return content.byte_array->empty();
case tag_type::tag_list:
return content.list->empty();
case tag_type::tag_int_array:
return content.int_array->empty();
case tag_type::tag_long_array:
return content.long_array->empty();
default:
return false;
}
}

size_type size() const noexcept {
switch (type) {
case tag_type::tag_end:
return 0;
case tag_type::tag_byte_array:
return content.byte_array->size();
case tag_type::tag_list:
return content.list->size();
case tag_type::tag_compound:
return content.compound->size();
case tag_type::tag_int_array:
return content.int_array->size();
case tag_type::tag_long_array:
return content.long_array->size();
default:
return 1;
}
}

private:
union tag_content {
tag_end_t end;
tag_byte_t byte;
tag_short_t short_n;
tag_int_t int_n;
tag_long_t long_n;
tag_float_t float_n;
tag_double_t double_n;
tag_byte_array_t *byte_array;
tag_string_t *string;
tag_list_t *list;
tag_compound_t *compound;
tag_int_array_t *int_array;
tag_long_array_t *long_array;

tag_content() = default;
tag_content(tag_byte_t v) : byte(v){};
tag_content(tag_short_t v) : short_n(v){};
tag_content(tag_int_t v) : int_n(v){};
tag_content(tag_long_t v) : long_n(v){};
tag_content(tag_float_t v) : float_n(v){};
tag_content(tag_double_t v) : double_n(v){};
tag_content(tag_type t) {
switch (t) {
case tag_type::tag_end: {
end = tag_end_t(0);
break;
}
case tag_type::tag_byte: {
byte = tag_byte_t(0);
break;
}
case tag_type::tag_short: {
short_n = tag_short_t(0);
break;
}
case tag_type::tag_int: {
int_n = tag_int_t(0);
break;
}
case tag_type::tag_long: {
long_n = tag_long_t(0);
break;
}
case tag_type::tag_float: {
float_n = tag_float_t(0);
break;
}
case tag_type::tag_double: {
double_n = tag_double_t(0);
break;
}
case tag_type::tag_byte_array: {
byte_array = new tag_byte_array_t{};
break;
}
case tag_type::tag_string: {
string = new tag_string_t("");
break;
}
case tag_type::tag_list: {
list = new tag_list_t();
break;
}
case tag_type::tag_compound: {
compound = new tag_compound_t();
break;
}
case tag_type::tag_int_array: {
int_array = new tag_int_array_t();
break;
}
case tag_type::tag_long_array: {
long_array = new tag_long_array_t();
break;
}
}
}

tag_content(const tag_byte_array_t &value) {
byte_array = new tag_byte_array_t(value);
}

tag_content(tag_byte_array_t &&value) {
byte_array = new tag_byte_array_t(std::move(value));
}

tag_content(const tag_int_array_t &value) {
int_array = new tag_int_array_t(value);
}

tag_content(tag_int_array_t &&value) {
int_array = new tag_int_array_t(std::move(value));
}

tag_content(const tag_long_array_t &value) {
long_array = new tag_long_array_t(value);
}

tag_content(tag_long_array_t &&value) {
long_array = new tag_long_array_t(std::move(value));
}

tag_content(const tag_string_t &value) : string(new tag_string_t(value)) {}

tag_content(tag_string_t &&value)
: string(new tag_string_t(std::move(value))) {}

tag_content(const tag_list_t &value) : list(new tag_list_t(value)) {}

tag_content(tag_list_t &&value) : list(new tag_list_t(std::move(value))) {}

tag_content(const tag_compound_t &value)
: compound(new tag_compound_t(value)) {}

tag_content(tag_compound_t &&value)
: compound(new tag_compound_t(std::move(value))) {}

tag_content clone(tag_type t) const {
switch (t) {
case tag_type::tag_byte:
return tag_content(byte);
case tag_type::tag_short:
return tag_content(short_n);
case tag_type::tag_int:
return tag_content(int_n);
case tag_type::tag_long:
return tag_content(long_n);
case tag_type::tag_float:
return tag_content(float_n);
case tag_type::tag_double:
return tag_content(double_n);
case tag_type::tag_string:
return tag_content(*string);
case tag_type::tag_byte_array:
return tag_content(*byte_array);
case tag_type::tag_int_array:
return tag_content(*int_array);
case tag_type::tag_long_array:
return tag_content(*long_array);
case tag_type::tag_compound:
return tag_content(*compound);
case tag_type::tag_list:
return tag_content(*list);
default:
return tag_content(tag_type::tag_end);
}
}

void destroy(tag_type &t) {
std::vector<NBT> stack;

if (t == tag_type::tag_compound) {
stack.reserve(compound->size());
for (auto &&it : *compound)
stack.push_back(std::move(it.second));
}

if (t == tag_type::tag_list) {
stack.reserve(list->size());
for (auto &&it : *list)
stack.push_back(std::move(it));
}

while (!stack.empty()) {
NBT current(std::move(stack.back()));
stack.pop_back();

if (current.type == tag_type::tag_compound) {
for (auto &&it : *current.content.compound)
stack.push_back(std::move(it.second));
current.content.compound->clear();
}
}

switch (t) {
case tag_type::tag_byte_array: {
delete byte_array;
break;
}
case tag_type::tag_string: {
delete string;
break;
}
case tag_type::tag_list: {
delete list;
break;
}
case tag_type::tag_compound: {
delete compound;
break;
}
case tag_type::tag_int_array: {
delete int_array;
break;
}
case tag_type::tag_long_array: {
delete long_array;
break;
}
default:
break;
}

t = tag_type::tag_end;
}
};

tag_type type = tag_type::tag_end;
tag_content content = {};
std::string name = "";


tag_byte_t *get_impl_ptr(tag_byte_t *) noexcept {
return is_byte() ? &content.byte : nullptr;
}

constexpr const tag_byte_t *get_impl_ptr(const tag_byte_t *) const noexcept {
return is_byte() ? &content.byte : nullptr;
}

tag_short_t *get_impl_ptr(tag_short_t *) noexcept {
return is_short() ? &content.short_n : nullptr;
}

constexpr const tag_short_t *
get_impl_ptr(const tag_short_t *) const noexcept {
return is_short() ? &content.short_n : nullptr;
}

tag_int_t *get_impl_ptr(tag_int_t *) noexcept {
return is_int() ? &content.int_n : nullptr;
}

constexpr const tag_int_t *get_impl_ptr(const tag_int_t *) const noexcept {
return is_int() ? &content.int_n : nullptr;
}

tag_long_t *get_impl_ptr(tag_long_t *) noexcept {
return is_long() ? &content.long_n : nullptr;
}

constexpr const tag_long_t *get_impl_ptr(const tag_long_t *) const noexcept {
return is_long() ? &content.long_n : nullptr;
}

tag_float_t *get_impl_ptr(tag_float_t *) noexcept {
return is_float() ? &content.float_n : nullptr;
}

constexpr const tag_float_t *
get_impl_ptr(const tag_float_t *) const noexcept {
return is_float() ? &content.float_n : nullptr;
}

tag_double_t *get_impl_ptr(tag_double_t *) noexcept {
return is_double() ? &content.double_n : nullptr;
}

constexpr const tag_double_t *
get_impl_ptr(const tag_double_t *) const noexcept {
return is_double() ? &content.double_n : nullptr;
}

tag_byte_array_t *get_impl_ptr(tag_byte_array_t *) noexcept {
return is_byte_array() ? content.byte_array : nullptr;
}

constexpr const tag_byte_array_t *
get_impl_ptr(const tag_byte_array_t *) const noexcept {
return is_byte_array() ? content.byte_array : nullptr;
}

tag_string_t *get_impl_ptr(tag_string_t *) noexcept {
return is_string() ? content.string : nullptr;
}

constexpr const tag_string_t *
get_impl_ptr(const tag_string_t *) const noexcept {
return is_string() ? content.string : nullptr;
}

tag_list_t *get_impl_ptr(tag_list_t *) noexcept {
return is_list() ? content.list : nullptr;
}

constexpr const tag_list_t *get_impl_ptr(const tag_list_t *) const noexcept {
return is_list() ? content.list : nullptr;
}

tag_compound_t *get_impl_ptr(tag_compound_t *) noexcept {
return is_compound() ? content.compound : nullptr;
}

constexpr const tag_compound_t *
get_impl_ptr(const tag_compound_t *) const noexcept {
return is_compound() ? content.compound : nullptr;
}

tag_int_array_t *get_impl_ptr(tag_int_array_t *) noexcept {
return is_int_array() ? content.int_array : nullptr;
}

constexpr const tag_int_array_t *
get_impl_ptr(const tag_int_array_t *) const noexcept {
return is_int_array() ? content.int_array : nullptr;
}

tag_long_array_t *get_impl_ptr(tag_long_array_t *) noexcept {
return is_long_array() ? content.long_array : nullptr;
}

constexpr const tag_long_array_t *
get_impl_ptr(const tag_long_array_t *) const noexcept {
return is_long_array() ? content.long_array : nullptr;
}

public:
template <typename PointerType,
typename std::enable_if<std::is_pointer<PointerType>::value,
int>::type = 0>
PointerType get_ptr() noexcept {
return get_impl_ptr(static_cast<PointerType>(nullptr));
}

template <
typename PointerType,
typename std::enable_if<std::is_pointer<PointerType>::value &&
std::is_const<typename std::remove_pointer<
PointerType>::type>::value,
int>::type = 0>
constexpr PointerType get_ptr() const noexcept {
return get_impl_ptr(static_cast<PointerType>(nullptr));
}

template <typename PointerType,
typename std::enable_if<std::is_pointer<PointerType>::value,
int>::type = 0>
auto get() noexcept -> decltype(get_ptr<PointerType>()) {
return get_ptr<PointerType>();
}

template <typename PointerType,
typename std::enable_if<std::is_pointer<PointerType>::value,
int>::type = 0>
constexpr auto get() const noexcept -> decltype(get_ptr<PointerType>()) {
return get_ptr<PointerType>();
}

template <typename ArithmeticType,
typename std::enable_if<std::is_arithmetic<ArithmeticType>::value,
int>::type = 0>
ArithmeticType get() const {
switch (get_type()) {
case tag_type::tag_byte:
return static_cast<ArithmeticType>(*get_ptr<const tag_byte_t *>());
case tag_type::tag_short:
return static_cast<ArithmeticType>(*get_ptr<const tag_short_t *>());
case tag_type::tag_int:
return static_cast<ArithmeticType>(*get_ptr<const tag_int_t *>());
case tag_type::tag_long:
return static_cast<ArithmeticType>(*get_ptr<const tag_long_t *>());
case tag_type::tag_float:
return static_cast<ArithmeticType>(*get_ptr<const tag_float_t *>());
case tag_type::tag_double:
return static_cast<ArithmeticType>(*get_ptr<const tag_double_t *>());
default:
throw(std::invalid_argument(fmt::format(
"Operation not available for tags of type `{}`", type_name())));
}
}

template <typename StringType,
typename std::enable_if<
std::is_same<StringType, tag_string_t>::value, int>::type = 0>
StringType get() const {
if (get_type() == tag_type::tag_string)
return static_cast<StringType>(*get_ptr<const tag_string_t *>());

throw(std::invalid_argument(fmt::format(
"Operation not available for tags of type `{}`", type_name())));
}
};

} 

#endif
