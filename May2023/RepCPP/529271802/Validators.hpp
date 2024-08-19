
#pragma once

#include "Macros.hpp"
#include "StringTools.hpp"
#include "TypeTools.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>


#if defined CLI11_CPP17 && defined __has_include && !defined CLI11_HAS_FILESYSTEM
#if __has_include(<filesystem>)
#if defined __MAC_OS_X_VERSION_MIN_REQUIRED && __MAC_OS_X_VERSION_MIN_REQUIRED < 101500
#define CLI11_HAS_FILESYSTEM 0
#else
#include <filesystem>
#if defined __cpp_lib_filesystem && __cpp_lib_filesystem >= 201703
#if defined _GLIBCXX_RELEASE && _GLIBCXX_RELEASE >= 9
#define CLI11_HAS_FILESYSTEM 1
#elif defined(__GLIBCXX__)
#define CLI11_HAS_FILESYSTEM 0
#else
#define CLI11_HAS_FILESYSTEM 1
#endif
#else
#define CLI11_HAS_FILESYSTEM 0
#endif
#endif
#endif
#endif

#if defined CLI11_HAS_FILESYSTEM && CLI11_HAS_FILESYSTEM > 0
#include <filesystem>  
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif


namespace CLI {

class Option;



class Validator {
protected:
std::function<std::string()> desc_function_{[]() { return std::string{}; }};

std::function<std::string(std::string &)> func_{[](std::string &) { return std::string{}; }};
std::string name_{};
int application_index_ = -1;
bool active_{true};
bool non_modifying_{false};

public:
Validator() = default;
explicit Validator(std::string validator_desc) : desc_function_([validator_desc]() { return validator_desc; }) {}
Validator(std::function<std::string(std::string &)> op, std::string validator_desc, std::string validator_name = "")
: desc_function_([validator_desc]() { return validator_desc; }), func_(std::move(op)),
name_(std::move(validator_name)) {}
Validator &operation(std::function<std::string(std::string &)> op) {
func_ = std::move(op);
return *this;
}
std::string operator()(std::string &str) const {
std::string retstring;
if(active_) {
if(non_modifying_) {
std::string value = str;
retstring = func_(value);
} else {
retstring = func_(str);
}
}
return retstring;
}

std::string operator()(const std::string &str) const {
std::string value = str;
return (active_) ? func_(value) : std::string{};
}

Validator &description(std::string validator_desc) {
desc_function_ = [validator_desc]() { return validator_desc; };
return *this;
}
Validator description(std::string validator_desc) const {
Validator newval(*this);
newval.desc_function_ = [validator_desc]() { return validator_desc; };
return newval;
}
std::string get_description() const {
if(active_) {
return desc_function_();
}
return std::string{};
}
Validator &name(std::string validator_name) {
name_ = std::move(validator_name);
return *this;
}
Validator name(std::string validator_name) const {
Validator newval(*this);
newval.name_ = std::move(validator_name);
return newval;
}
const std::string &get_name() const { return name_; }
Validator &active(bool active_val = true) {
active_ = active_val;
return *this;
}
Validator active(bool active_val = true) const {
Validator newval(*this);
newval.active_ = active_val;
return newval;
}

Validator &non_modifying(bool no_modify = true) {
non_modifying_ = no_modify;
return *this;
}
Validator &application_index(int app_index) {
application_index_ = app_index;
return *this;
}
Validator application_index(int app_index) const {
Validator newval(*this);
newval.application_index_ = app_index;
return newval;
}
int get_application_index() const { return application_index_; }
bool get_active() const { return active_; }

bool get_modifying() const { return !non_modifying_; }

Validator operator&(const Validator &other) const {
Validator newval;

newval._merge_description(*this, other, " AND ");

const std::function<std::string(std::string & filename)> &f1 = func_;
const std::function<std::string(std::string & filename)> &f2 = other.func_;

newval.func_ = [f1, f2](std::string &input) {
std::string s1 = f1(input);
std::string s2 = f2(input);
if(!s1.empty() && !s2.empty())
return std::string("(") + s1 + ") AND (" + s2 + ")";
else
return s1 + s2;
};

newval.active_ = (active_ & other.active_);
newval.application_index_ = application_index_;
return newval;
}

Validator operator|(const Validator &other) const {
Validator newval;

newval._merge_description(*this, other, " OR ");

const std::function<std::string(std::string &)> &f1 = func_;
const std::function<std::string(std::string &)> &f2 = other.func_;

newval.func_ = [f1, f2](std::string &input) {
std::string s1 = f1(input);
std::string s2 = f2(input);
if(s1.empty() || s2.empty())
return std::string();

return std::string("(") + s1 + ") OR (" + s2 + ")";
};
newval.active_ = (active_ & other.active_);
newval.application_index_ = application_index_;
return newval;
}

Validator operator!() const {
Validator newval;
const std::function<std::string()> &dfunc1 = desc_function_;
newval.desc_function_ = [dfunc1]() {
auto str = dfunc1();
return (!str.empty()) ? std::string("NOT ") + str : std::string{};
};
const std::function<std::string(std::string & res)> &f1 = func_;

newval.func_ = [f1, dfunc1](std::string &test) -> std::string {
std::string s1 = f1(test);
if(s1.empty()) {
return std::string("check ") + dfunc1() + " succeeded improperly";
}
return std::string{};
};
newval.active_ = active_;
newval.application_index_ = application_index_;
return newval;
}

private:
void _merge_description(const Validator &val1, const Validator &val2, const std::string &merger) {

const std::function<std::string()> &dfunc1 = val1.desc_function_;
const std::function<std::string()> &dfunc2 = val2.desc_function_;

desc_function_ = [=]() {
std::string f1 = dfunc1();
std::string f2 = dfunc2();
if((f1.empty()) || (f2.empty())) {
return f1 + f2;
}
return std::string(1, '(') + f1 + ')' + merger + '(' + f2 + ')';
};
}
};  

class CustomValidator : public Validator {
public:
};
namespace detail {

enum class path_type { nonexistent, file, directory };

#if defined CLI11_HAS_FILESYSTEM && CLI11_HAS_FILESYSTEM > 0
inline path_type check_path(const char *file) noexcept {
std::error_code ec;
auto stat = std::filesystem::status(file, ec);
if(ec) {
return path_type::nonexistent;
}
switch(stat.type()) {
case std::filesystem::file_type::none:
case std::filesystem::file_type::not_found:
return path_type::nonexistent;
case std::filesystem::file_type::directory:
return path_type::directory;
case std::filesystem::file_type::symlink:
case std::filesystem::file_type::block:
case std::filesystem::file_type::character:
case std::filesystem::file_type::fifo:
case std::filesystem::file_type::socket:
case std::filesystem::file_type::regular:
case std::filesystem::file_type::unknown:
default:
return path_type::file;
}
}
#else
inline path_type check_path(const char *file) noexcept {
#if defined(_MSC_VER)
struct __stat64 buffer;
if(_stat64(file, &buffer) == 0) {
return ((buffer.st_mode & S_IFDIR) != 0) ? path_type::directory : path_type::file;
}
#else
struct stat buffer;
if(stat(file, &buffer) == 0) {
return ((buffer.st_mode & S_IFDIR) != 0) ? path_type::directory : path_type::file;
}
#endif
return path_type::nonexistent;
}
#endif
class ExistingFileValidator : public Validator {
public:
ExistingFileValidator() : Validator("FILE") {
func_ = [](std::string &filename) {
auto path_result = check_path(filename.c_str());
if(path_result == path_type::nonexistent) {
return "File does not exist: " + filename;
}
if(path_result == path_type::directory) {
return "File is actually a directory: " + filename;
}
return std::string();
};
}
};

class ExistingDirectoryValidator : public Validator {
public:
ExistingDirectoryValidator() : Validator("DIR") {
func_ = [](std::string &filename) {
auto path_result = check_path(filename.c_str());
if(path_result == path_type::nonexistent) {
return "Directory does not exist: " + filename;
}
if(path_result == path_type::file) {
return "Directory is actually a file: " + filename;
}
return std::string();
};
}
};

class ExistingPathValidator : public Validator {
public:
ExistingPathValidator() : Validator("PATH(existing)") {
func_ = [](std::string &filename) {
auto path_result = check_path(filename.c_str());
if(path_result == path_type::nonexistent) {
return "Path does not exist: " + filename;
}
return std::string();
};
}
};

class NonexistentPathValidator : public Validator {
public:
NonexistentPathValidator() : Validator("PATH(non-existing)") {
func_ = [](std::string &filename) {
auto path_result = check_path(filename.c_str());
if(path_result != path_type::nonexistent) {
return "Path already exists: " + filename;
}
return std::string();
};
}
};

class IPV4Validator : public Validator {
public:
IPV4Validator() : Validator("IPV4") {
func_ = [](std::string &ip_addr) {
auto result = CLI::detail::split(ip_addr, '.');
if(result.size() != 4) {
return std::string("Invalid IPV4 address must have four parts (") + ip_addr + ')';
}
int num;
for(const auto &var : result) {
bool retval = detail::lexical_cast(var, num);
if(!retval) {
return std::string("Failed parsing number (") + var + ')';
}
if(num < 0 || num > 255) {
return std::string("Each IP number must be between 0 and 255 ") + var;
}
}
return std::string();
};
}
};

}  


const detail::ExistingFileValidator ExistingFile;

const detail::ExistingDirectoryValidator ExistingDirectory;

const detail::ExistingPathValidator ExistingPath;

const detail::NonexistentPathValidator NonexistentPath;

const detail::IPV4Validator ValidIPV4;

template <typename DesiredType> class TypeValidator : public Validator {
public:
explicit TypeValidator(const std::string &validator_name) : Validator(validator_name) {
func_ = [](std::string &input_string) {
auto val = DesiredType();
if(!detail::lexical_cast(input_string, val)) {
return std::string("Failed parsing ") + input_string + " as a " + detail::type_name<DesiredType>();
}
return std::string();
};
}
TypeValidator() : TypeValidator(detail::type_name<DesiredType>()) {}
};

const TypeValidator<double> Number("NUMBER");

class Range : public Validator {
public:
template <typename T>
Range(T min_val, T max_val, const std::string &validator_name = std::string{}) : Validator(validator_name) {
if(validator_name.empty()) {
std::stringstream out;
out << detail::type_name<T>() << " in [" << min_val << " - " << max_val << "]";
description(out.str());
}

func_ = [min_val, max_val](std::string &input) {
T val;
bool converted = detail::lexical_cast(input, val);
if((!converted) || (val < min_val || val > max_val))
return std::string("Value ") + input + " not in range " + std::to_string(min_val) + " to " +
std::to_string(max_val);

return std::string{};
};
}

template <typename T>
explicit Range(T max_val, const std::string &validator_name = std::string{})
: Range(static_cast<T>(0), max_val, validator_name) {}
};

const Range NonNegativeNumber((std::numeric_limits<double>::max)(), "NONNEGATIVE");

const Range PositiveNumber((std::numeric_limits<double>::min)(), (std::numeric_limits<double>::max)(), "POSITIVE");

class Bound : public Validator {
public:
template <typename T> Bound(T min_val, T max_val) {
std::stringstream out;
out << detail::type_name<T>() << " bounded to [" << min_val << " - " << max_val << "]";
description(out.str());

func_ = [min_val, max_val](std::string &input) {
T val;
bool converted = detail::lexical_cast(input, val);
if(!converted) {
return std::string("Value ") + input + " could not be converted";
}
if(val < min_val)
input = detail::to_string(min_val);
else if(val > max_val)
input = detail::to_string(max_val);

return std::string{};
};
}

template <typename T> explicit Bound(T max_val) : Bound(static_cast<T>(0), max_val) {}
};

namespace detail {
template <typename T,
enable_if_t<is_copyable_ptr<typename std::remove_reference<T>::type>::value, detail::enabler> = detail::dummy>
auto smart_deref(T value) -> decltype(*value) {
return *value;
}

template <
typename T,
enable_if_t<!is_copyable_ptr<typename std::remove_reference<T>::type>::value, detail::enabler> = detail::dummy>
typename std::remove_reference<T>::type &smart_deref(T &value) {
return value;
}
template <typename T> std::string generate_set(const T &set) {
using element_t = typename detail::element_type<T>::type;
using iteration_type_t = typename detail::pair_adaptor<element_t>::value_type;  
std::string out(1, '{');
out.append(detail::join(
detail::smart_deref(set),
[](const iteration_type_t &v) { return detail::pair_adaptor<element_t>::first(v); },
","));
out.push_back('}');
return out;
}

template <typename T> std::string generate_map(const T &map, bool key_only = false) {
using element_t = typename detail::element_type<T>::type;
using iteration_type_t = typename detail::pair_adaptor<element_t>::value_type;  
std::string out(1, '{');
out.append(detail::join(
detail::smart_deref(map),
[key_only](const iteration_type_t &v) {
std::string res{detail::to_string(detail::pair_adaptor<element_t>::first(v))};

if(!key_only) {
res.append("->");
res += detail::to_string(detail::pair_adaptor<element_t>::second(v));
}
return res;
},
","));
out.push_back('}');
return out;
}

template <typename C, typename V> struct has_find {
template <typename CC, typename VV>
static auto test(int) -> decltype(std::declval<CC>().find(std::declval<VV>()), std::true_type());
template <typename, typename> static auto test(...) -> decltype(std::false_type());

static const auto value = decltype(test<C, V>(0))::value;
using type = std::integral_constant<bool, value>;
};

template <typename T, typename V, enable_if_t<!has_find<T, V>::value, detail::enabler> = detail::dummy>
auto search(const T &set, const V &val) -> std::pair<bool, decltype(std::begin(detail::smart_deref(set)))> {
using element_t = typename detail::element_type<T>::type;
auto &setref = detail::smart_deref(set);
auto it = std::find_if(std::begin(setref), std::end(setref), [&val](decltype(*std::begin(setref)) v) {
return (detail::pair_adaptor<element_t>::first(v) == val);
});
return {(it != std::end(setref)), it};
}

template <typename T, typename V, enable_if_t<has_find<T, V>::value, detail::enabler> = detail::dummy>
auto search(const T &set, const V &val) -> std::pair<bool, decltype(std::begin(detail::smart_deref(set)))> {
auto &setref = detail::smart_deref(set);
auto it = setref.find(val);
return {(it != std::end(setref)), it};
}

template <typename T, typename V>
auto search(const T &set, const V &val, const std::function<V(V)> &filter_function)
-> std::pair<bool, decltype(std::begin(detail::smart_deref(set)))> {
using element_t = typename detail::element_type<T>::type;
auto res = search(set, val);
if((res.first) || (!(filter_function))) {
return res;
}
auto &setref = detail::smart_deref(set);
auto it = std::find_if(std::begin(setref), std::end(setref), [&](decltype(*std::begin(setref)) v) {
V a{detail::pair_adaptor<element_t>::first(v)};
a = filter_function(a);
return (a == val);
});
return {(it != std::end(setref)), it};
}


template <typename T>
inline typename std::enable_if<std::is_signed<T>::value, T>::type overflowCheck(const T &a, const T &b) {
if((a > 0) == (b > 0)) {
return ((std::numeric_limits<T>::max)() / (std::abs)(a) < (std::abs)(b));
} else {
return ((std::numeric_limits<T>::min)() / (std::abs)(a) > -(std::abs)(b));
}
}
template <typename T>
inline typename std::enable_if<!std::is_signed<T>::value, T>::type overflowCheck(const T &a, const T &b) {
return ((std::numeric_limits<T>::max)() / a < b);
}

template <typename T> typename std::enable_if<std::is_integral<T>::value, bool>::type checked_multiply(T &a, T b) {
if(a == 0 || b == 0 || a == 1 || b == 1) {
a *= b;
return true;
}
if(a == (std::numeric_limits<T>::min)() || b == (std::numeric_limits<T>::min)()) {
return false;
}
if(overflowCheck(a, b)) {
return false;
}
a *= b;
return true;
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type checked_multiply(T &a, T b) {
T c = a * b;
if(std::isinf(c) && !std::isinf(a) && !std::isinf(b)) {
return false;
}
a = c;
return true;
}

}  
class IsMember : public Validator {
public:
using filter_fn_t = std::function<std::string(std::string)>;

template <typename T, typename... Args>
IsMember(std::initializer_list<T> values, Args &&...args)
: IsMember(std::vector<T>(values), std::forward<Args>(args)...) {}

template <typename T> explicit IsMember(T &&set) : IsMember(std::forward<T>(set), nullptr) {}

template <typename T, typename F> explicit IsMember(T set, F filter_function) {

using element_t = typename detail::element_type<T>::type;             
using item_t = typename detail::pair_adaptor<element_t>::first_type;  

using local_item_t = typename IsMemberType<item_t>::type;  

std::function<local_item_t(local_item_t)> filter_fn = filter_function;

desc_function_ = [set]() { return detail::generate_set(detail::smart_deref(set)); };

func_ = [set, filter_fn](std::string &input) {
local_item_t b;
if(!detail::lexical_cast(input, b)) {
throw ValidationError(input);  
}
if(filter_fn) {
b = filter_fn(b);
}
auto res = detail::search(set, b, filter_fn);
if(res.first) {
if(filter_fn) {
input = detail::value_string(detail::pair_adaptor<element_t>::first(*(res.second)));
}

return std::string{};
}

return input + " not in " + detail::generate_set(detail::smart_deref(set));
};
}

template <typename T, typename... Args>
IsMember(T &&set, filter_fn_t filter_fn_1, filter_fn_t filter_fn_2, Args &&...other)
: IsMember(
std::forward<T>(set),
[filter_fn_1, filter_fn_2](std::string a) { return filter_fn_2(filter_fn_1(a)); },
other...) {}
};

template <typename T> using TransformPairs = std::vector<std::pair<std::string, T>>;

class Transformer : public Validator {
public:
using filter_fn_t = std::function<std::string(std::string)>;

template <typename... Args>
Transformer(std::initializer_list<std::pair<std::string, std::string>> values, Args &&...args)
: Transformer(TransformPairs<std::string>(values), std::forward<Args>(args)...) {}

template <typename T> explicit Transformer(T &&mapping) : Transformer(std::forward<T>(mapping), nullptr) {}

template <typename T, typename F> explicit Transformer(T mapping, F filter_function) {

static_assert(detail::pair_adaptor<typename detail::element_type<T>::type>::value,
"mapping must produce value pairs");
using element_t = typename detail::element_type<T>::type;             
using item_t = typename detail::pair_adaptor<element_t>::first_type;  
using local_item_t = typename IsMemberType<item_t>::type;             

std::function<local_item_t(local_item_t)> filter_fn = filter_function;

desc_function_ = [mapping]() { return detail::generate_map(detail::smart_deref(mapping)); };

func_ = [mapping, filter_fn](std::string &input) {
local_item_t b;
if(!detail::lexical_cast(input, b)) {
return std::string();
}
if(filter_fn) {
b = filter_fn(b);
}
auto res = detail::search(mapping, b, filter_fn);
if(res.first) {
input = detail::value_string(detail::pair_adaptor<element_t>::second(*res.second));
}
return std::string{};
};
}

template <typename T, typename... Args>
Transformer(T &&mapping, filter_fn_t filter_fn_1, filter_fn_t filter_fn_2, Args &&...other)
: Transformer(
std::forward<T>(mapping),
[filter_fn_1, filter_fn_2](std::string a) { return filter_fn_2(filter_fn_1(a)); },
other...) {}
};

class CheckedTransformer : public Validator {
public:
using filter_fn_t = std::function<std::string(std::string)>;

template <typename... Args>
CheckedTransformer(std::initializer_list<std::pair<std::string, std::string>> values, Args &&...args)
: CheckedTransformer(TransformPairs<std::string>(values), std::forward<Args>(args)...) {}

template <typename T> explicit CheckedTransformer(T mapping) : CheckedTransformer(std::move(mapping), nullptr) {}

template <typename T, typename F> explicit CheckedTransformer(T mapping, F filter_function) {

static_assert(detail::pair_adaptor<typename detail::element_type<T>::type>::value,
"mapping must produce value pairs");
using element_t = typename detail::element_type<T>::type;             
using item_t = typename detail::pair_adaptor<element_t>::first_type;  
using local_item_t = typename IsMemberType<item_t>::type;             
using iteration_type_t = typename detail::pair_adaptor<element_t>::value_type;  

std::function<local_item_t(local_item_t)> filter_fn = filter_function;

auto tfunc = [mapping]() {
std::string out("value in ");
out += detail::generate_map(detail::smart_deref(mapping)) + " OR {";
out += detail::join(
detail::smart_deref(mapping),
[](const iteration_type_t &v) { return detail::to_string(detail::pair_adaptor<element_t>::second(v)); },
",");
out.push_back('}');
return out;
};

desc_function_ = tfunc;

func_ = [mapping, tfunc, filter_fn](std::string &input) {
local_item_t b;
bool converted = detail::lexical_cast(input, b);
if(converted) {
if(filter_fn) {
b = filter_fn(b);
}
auto res = detail::search(mapping, b, filter_fn);
if(res.first) {
input = detail::value_string(detail::pair_adaptor<element_t>::second(*res.second));
return std::string{};
}
}
for(const auto &v : detail::smart_deref(mapping)) {
auto output_string = detail::value_string(detail::pair_adaptor<element_t>::second(v));
if(output_string == input) {
return std::string();
}
}

return "Check " + input + " " + tfunc() + " FAILED";
};
}

template <typename T, typename... Args>
CheckedTransformer(T &&mapping, filter_fn_t filter_fn_1, filter_fn_t filter_fn_2, Args &&...other)
: CheckedTransformer(
std::forward<T>(mapping),
[filter_fn_1, filter_fn_2](std::string a) { return filter_fn_2(filter_fn_1(a)); },
other...) {}
};

inline std::string ignore_case(std::string item) { return detail::to_lower(item); }

inline std::string ignore_underscore(std::string item) { return detail::remove_underscore(item); }

inline std::string ignore_space(std::string item) {
item.erase(std::remove(std::begin(item), std::end(item), ' '), std::end(item));
item.erase(std::remove(std::begin(item), std::end(item), '\t'), std::end(item));
return item;
}

class AsNumberWithUnit : public Validator {
public:
enum Options {
CASE_SENSITIVE = 0,
CASE_INSENSITIVE = 1,
UNIT_OPTIONAL = 0,
UNIT_REQUIRED = 2,
DEFAULT = CASE_INSENSITIVE | UNIT_OPTIONAL
};

template <typename Number>
explicit AsNumberWithUnit(std::map<std::string, Number> mapping,
Options opts = DEFAULT,
const std::string &unit_name = "UNIT") {
description(generate_description<Number>(unit_name, opts));
validate_mapping(mapping, opts);

func_ = [mapping, opts](std::string &input) -> std::string {
Number num;

detail::rtrim(input);
if(input.empty()) {
throw ValidationError("Input is empty");
}

auto unit_begin = input.end();
while(unit_begin > input.begin() && std::isalpha(*(unit_begin - 1), std::locale())) {
--unit_begin;
}

std::string unit{unit_begin, input.end()};
input.resize(static_cast<std::size_t>(std::distance(input.begin(), unit_begin)));
detail::trim(input);

if(opts & UNIT_REQUIRED && unit.empty()) {
throw ValidationError("Missing mandatory unit");
}
if(opts & CASE_INSENSITIVE) {
unit = detail::to_lower(unit);
}
if(unit.empty()) {
if(!detail::lexical_cast(input, num)) {
throw ValidationError(std::string("Value ") + input + " could not be converted to " +
detail::type_name<Number>());
}
return {};
}

auto it = mapping.find(unit);
if(it == mapping.end()) {
throw ValidationError(unit +
" unit not recognized. "
"Allowed values: " +
detail::generate_map(mapping, true));
}

if(!input.empty()) {
bool converted = detail::lexical_cast(input, num);
if(!converted) {
throw ValidationError(std::string("Value ") + input + " could not be converted to " +
detail::type_name<Number>());
}
bool ok = detail::checked_multiply(num, it->second);
if(!ok) {
throw ValidationError(detail::to_string(num) + " multiplied by " + unit +
" factor would cause number overflow. Use smaller value.");
}
} else {
num = static_cast<Number>(it->second);
}

input = detail::to_string(num);

return {};
};
}

private:
template <typename Number> static void validate_mapping(std::map<std::string, Number> &mapping, Options opts) {
for(auto &kv : mapping) {
if(kv.first.empty()) {
throw ValidationError("Unit must not be empty.");
}
if(!detail::isalpha(kv.first)) {
throw ValidationError("Unit must contain only letters.");
}
}

if(opts & CASE_INSENSITIVE) {
std::map<std::string, Number> lower_mapping;
for(auto &kv : mapping) {
auto s = detail::to_lower(kv.first);
if(lower_mapping.count(s)) {
throw ValidationError(std::string("Several matching lowercase unit representations are found: ") +
s);
}
lower_mapping[detail::to_lower(kv.first)] = kv.second;
}
mapping = std::move(lower_mapping);
}
}

template <typename Number> static std::string generate_description(const std::string &name, Options opts) {
std::stringstream out;
out << detail::type_name<Number>() << ' ';
if(opts & UNIT_REQUIRED) {
out << name;
} else {
out << '[' << name << ']';
}
return out.str();
}
};

class AsSizeValue : public AsNumberWithUnit {
public:
using result_t = std::uint64_t;

explicit AsSizeValue(bool kb_is_1000) : AsNumberWithUnit(get_mapping(kb_is_1000)) {
if(kb_is_1000) {
description("SIZE [b, kb(=1000b), kib(=1024b), ...]");
} else {
description("SIZE [b, kb(=1024b), ...]");
}
}

private:
static std::map<std::string, result_t> init_mapping(bool kb_is_1000) {
std::map<std::string, result_t> m;
result_t k_factor = kb_is_1000 ? 1000 : 1024;
result_t ki_factor = 1024;
result_t k = 1;
result_t ki = 1;
m["b"] = 1;
for(std::string p : {"k", "m", "g", "t", "p", "e"}) {
k *= k_factor;
ki *= ki_factor;
m[p] = k;
m[p + "b"] = k;
m[p + "i"] = ki;
m[p + "ib"] = ki;
}
return m;
}

static std::map<std::string, result_t> get_mapping(bool kb_is_1000) {
if(kb_is_1000) {
static auto m = init_mapping(true);
return m;
} else {
static auto m = init_mapping(false);
return m;
}
}
};

namespace detail {
inline std::pair<std::string, std::string> split_program_name(std::string commandline) {
std::pair<std::string, std::string> vals;
trim(commandline);
auto esp = commandline.find_first_of(' ', 1);
while(detail::check_path(commandline.substr(0, esp).c_str()) != path_type::file) {
esp = commandline.find_first_of(' ', esp + 1);
if(esp == std::string::npos) {
if(commandline[0] == '"' || commandline[0] == '\'' || commandline[0] == '`') {
bool embeddedQuote = false;
auto keyChar = commandline[0];
auto end = commandline.find_first_of(keyChar, 1);
while((end != std::string::npos) && (commandline[end - 1] == '\\')) {  
end = commandline.find_first_of(keyChar, end + 1);
embeddedQuote = true;
}
if(end != std::string::npos) {
vals.first = commandline.substr(1, end - 1);
esp = end + 1;
if(embeddedQuote) {
vals.first = find_and_replace(vals.first, std::string("\\") + keyChar, std::string(1, keyChar));
}
} else {
esp = commandline.find_first_of(' ', 1);
}
} else {
esp = commandline.find_first_of(' ', 1);
}

break;
}
}
if(vals.first.empty()) {
vals.first = commandline.substr(0, esp);
rtrim(vals.first);
}

vals.second = (esp != std::string::npos) ? commandline.substr(esp + 1) : std::string{};
ltrim(vals.second);
return vals;
}

}  

}  
