
#pragma once

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "StringTools.hpp"

namespace CLI {


namespace detail {
enum class enabler {};

constexpr enabler dummy = {};
}  

template <bool B, class T = void> using enable_if_t = typename std::enable_if<B, T>::type;

template <typename... Ts> struct make_void { using type = void; };

template <typename... Ts> using void_t = typename make_void<Ts...>::type;

template <bool B, class T, class F> using conditional_t = typename std::conditional<B, T, F>::type;

template <typename T> struct is_bool : std::false_type {};

template <> struct is_bool<bool> : std::true_type {};

template <typename T> struct is_shared_ptr : std::false_type {};

template <typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template <typename T> struct is_shared_ptr<const std::shared_ptr<T>> : std::true_type {};

template <typename T> struct is_copyable_ptr {
static bool const value = is_shared_ptr<T>::value || std::is_pointer<T>::value;
};

template <typename T> struct IsMemberType { using type = T; };

template <> struct IsMemberType<const char *> { using type = std::string; };

namespace detail {



template <typename T, typename Enable = void> struct element_type { using type = T; };

template <typename T> struct element_type<T, typename std::enable_if<is_copyable_ptr<T>::value>::type> {
using type = typename std::pointer_traits<T>::element_type;
};

template <typename T> struct element_value_type { using type = typename element_type<T>::type::value_type; };

template <typename T, typename _ = void> struct pair_adaptor : std::false_type {
using value_type = typename T::value_type;
using first_type = typename std::remove_const<value_type>::type;
using second_type = typename std::remove_const<value_type>::type;

template <typename Q> static auto first(Q &&pair_value) -> decltype(std::forward<Q>(pair_value)) {
return std::forward<Q>(pair_value);
}
template <typename Q> static auto second(Q &&pair_value) -> decltype(std::forward<Q>(pair_value)) {
return std::forward<Q>(pair_value);
}
};

template <typename T>
struct pair_adaptor<
T,
conditional_t<false, void_t<typename T::value_type::first_type, typename T::value_type::second_type>, void>>
: std::true_type {
using value_type = typename T::value_type;
using first_type = typename std::remove_const<typename value_type::first_type>::type;
using second_type = typename std::remove_const<typename value_type::second_type>::type;

template <typename Q> static auto first(Q &&pair_value) -> decltype(std::get<0>(std::forward<Q>(pair_value))) {
return std::get<0>(std::forward<Q>(pair_value));
}
template <typename Q> static auto second(Q &&pair_value) -> decltype(std::get<1>(std::forward<Q>(pair_value))) {
return std::get<1>(std::forward<Q>(pair_value));
}
};

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif
template <typename T, typename C> class is_direct_constructible {
template <typename TT, typename CC>
static auto test(int, std::true_type) -> decltype(
#ifdef __CUDACC__
#pragma diag_suppress 2361
#endif
TT { std::declval<CC>() }
#ifdef __CUDACC__
#pragma diag_default 2361
#endif
,
std::is_move_assignable<TT>());

template <typename TT, typename CC> static auto test(int, std::false_type) -> std::false_type;

template <typename, typename> static auto test(...) -> std::false_type;

public:
static constexpr bool value = decltype(test<T, C>(0, typename std::is_constructible<T, C>::type()))::value;
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif


template <typename T, typename S = std::ostringstream> class is_ostreamable {
template <typename TT, typename SS>
static auto test(int) -> decltype(std::declval<SS &>() << std::declval<TT>(), std::true_type());

template <typename, typename> static auto test(...) -> std::false_type;

public:
static constexpr bool value = decltype(test<T, S>(0))::value;
};

template <typename T, typename S = std::istringstream> class is_istreamable {
template <typename TT, typename SS>
static auto test(int) -> decltype(std::declval<SS &>() >> std::declval<TT &>(), std::true_type());

template <typename, typename> static auto test(...) -> std::false_type;

public:
static constexpr bool value = decltype(test<T, S>(0))::value;
};

template <typename T> class is_complex {
template <typename TT>
static auto test(int) -> decltype(std::declval<TT>().real(), std::declval<TT>().imag(), std::true_type());

template <typename> static auto test(...) -> std::false_type;

public:
static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T, enable_if_t<is_istreamable<T>::value, detail::enabler> = detail::dummy>
bool from_stream(const std::string &istring, T &obj) {
std::istringstream is;
is.str(istring);
is >> obj;
return !is.fail() && !is.rdbuf()->in_avail();
}

template <typename T, enable_if_t<!is_istreamable<T>::value, detail::enabler> = detail::dummy>
bool from_stream(const std::string & , T & ) {
return false;
}

template <typename T, typename _ = void> struct is_mutable_container : std::false_type {};

template <typename T>
struct is_mutable_container<
T,
conditional_t<false,
void_t<typename T::value_type,
decltype(std::declval<T>().end()),
decltype(std::declval<T>().clear()),
decltype(std::declval<T>().insert(std::declval<decltype(std::declval<T>().end())>(),
std::declval<const typename T::value_type &>()))>,
void>>
: public conditional_t<std::is_constructible<T, std::string>::value, std::false_type, std::true_type> {};

template <typename T, typename _ = void> struct is_readable_container : std::false_type {};

template <typename T>
struct is_readable_container<
T,
conditional_t<false, void_t<decltype(std::declval<T>().end()), decltype(std::declval<T>().begin())>, void>>
: public std::true_type {};

template <typename T, typename _ = void> struct is_wrapper : std::false_type {};

template <typename T>
struct is_wrapper<T, conditional_t<false, void_t<typename T::value_type>, void>> : public std::true_type {};

template <typename S> class is_tuple_like {
template <typename SS>
static auto test(int) -> decltype(std::tuple_size<typename std::decay<SS>::type>::value, std::true_type{});
template <typename> static auto test(...) -> std::false_type;

public:
static constexpr bool value = decltype(test<S>(0))::value;
};

template <typename T, enable_if_t<std::is_convertible<T, std::string>::value, detail::enabler> = detail::dummy>
auto to_string(T &&value) -> decltype(std::forward<T>(value)) {
return std::forward<T>(value);
}

template <typename T,
enable_if_t<std::is_constructible<std::string, T>::value && !std::is_convertible<T, std::string>::value,
detail::enabler> = detail::dummy>
std::string to_string(const T &value) {
return std::string(value);
}

template <typename T,
enable_if_t<!std::is_convertible<std::string, T>::value && !std::is_constructible<std::string, T>::value &&
is_ostreamable<T>::value,
detail::enabler> = detail::dummy>
std::string to_string(T &&value) {
std::stringstream stream;
stream << value;
return stream.str();
}

template <typename T,
enable_if_t<!std::is_constructible<std::string, T>::value && !is_ostreamable<T>::value &&
!is_readable_container<typename std::remove_const<T>::type>::value,
detail::enabler> = detail::dummy>
std::string to_string(T &&) {
return std::string{};
}

template <typename T,
enable_if_t<!std::is_constructible<std::string, T>::value && !is_ostreamable<T>::value &&
is_readable_container<T>::value,
detail::enabler> = detail::dummy>
std::string to_string(T &&variable) {
std::vector<std::string> defaults;
auto cval = variable.begin();
auto end = variable.end();
while(cval != end) {
defaults.emplace_back(CLI::detail::to_string(*cval));
++cval;
}
return std::string("[" + detail::join(defaults) + "]");
}

template <typename T1,
typename T2,
typename T,
enable_if_t<std::is_same<T1, T2>::value, detail::enabler> = detail::dummy>
auto checked_to_string(T &&value) -> decltype(to_string(std::forward<T>(value))) {
return to_string(std::forward<T>(value));
}

template <typename T1,
typename T2,
typename T,
enable_if_t<!std::is_same<T1, T2>::value, detail::enabler> = detail::dummy>
std::string checked_to_string(T &&) {
return std::string{};
}
template <typename T, enable_if_t<std::is_arithmetic<T>::value, detail::enabler> = detail::dummy>
std::string value_string(const T &value) {
return std::to_string(value);
}
template <typename T, enable_if_t<std::is_enum<T>::value, detail::enabler> = detail::dummy>
std::string value_string(const T &value) {
return std::to_string(static_cast<typename std::underlying_type<T>::type>(value));
}
template <typename T,
enable_if_t<!std::is_enum<T>::value && !std::is_arithmetic<T>::value, detail::enabler> = detail::dummy>
auto value_string(const T &value) -> decltype(to_string(value)) {
return to_string(value);
}

template <typename T, typename def, typename Enable = void> struct wrapped_type { using type = def; };

template <typename T, typename def> struct wrapped_type<T, def, typename std::enable_if<is_wrapper<T>::value>::type> {
using type = typename T::value_type;
};

template <typename T, typename Enable = void> struct type_count_base { static const int value{0}; };

template <typename T>
struct type_count_base<T,
typename std::enable_if<!is_tuple_like<T>::value && !is_mutable_container<T>::value &&
!std::is_void<T>::value>::type> {
static constexpr int value{1};
};

template <typename T>
struct type_count_base<T, typename std::enable_if<is_tuple_like<T>::value && !is_mutable_container<T>::value>::type> {
static constexpr int value{std::tuple_size<T>::value};
};

template <typename T> struct type_count_base<T, typename std::enable_if<is_mutable_container<T>::value>::type> {
static constexpr int value{type_count_base<typename T::value_type>::value};
};


template <typename T> struct subtype_count;

template <typename T> struct subtype_count_min;

template <typename T, typename Enable = void> struct type_count { static const int value{0}; };

template <typename T>
struct type_count<T,
typename std::enable_if<!is_wrapper<T>::value && !is_tuple_like<T>::value && !is_complex<T>::value &&
!std::is_void<T>::value>::type> {
static constexpr int value{1};
};

template <typename T> struct type_count<T, typename std::enable_if<is_complex<T>::value>::type> {
static constexpr int value{2};
};

template <typename T> struct type_count<T, typename std::enable_if<is_mutable_container<T>::value>::type> {
static constexpr int value{subtype_count<typename T::value_type>::value};
};

template <typename T>
struct type_count<T,
typename std::enable_if<is_wrapper<T>::value && !is_complex<T>::value && !is_tuple_like<T>::value &&
!is_mutable_container<T>::value>::type> {
static constexpr int value{type_count<typename T::value_type>::value};
};

template <typename T, std::size_t I>
constexpr typename std::enable_if<I == type_count_base<T>::value, int>::type tuple_type_size() {
return 0;
}

template <typename T, std::size_t I>
constexpr typename std::enable_if < I<type_count_base<T>::value, int>::type tuple_type_size() {
return subtype_count<typename std::tuple_element<I, T>::type>::value + tuple_type_size<T, I + 1>();
}

template <typename T> struct type_count<T, typename std::enable_if<is_tuple_like<T>::value>::type> {
static constexpr int value{tuple_type_size<T, 0>()};
};

template <typename T> struct subtype_count {
static constexpr int value{is_mutable_container<T>::value ? expected_max_vector_size : type_count<T>::value};
};

template <typename T, typename Enable = void> struct type_count_min { static const int value{0}; };

template <typename T>
struct type_count_min<
T,
typename std::enable_if<!is_mutable_container<T>::value && !is_tuple_like<T>::value && !is_wrapper<T>::value &&
!is_complex<T>::value && !std::is_void<T>::value>::type> {
static constexpr int value{type_count<T>::value};
};

template <typename T> struct type_count_min<T, typename std::enable_if<is_complex<T>::value>::type> {
static constexpr int value{1};
};

template <typename T>
struct type_count_min<
T,
typename std::enable_if<is_wrapper<T>::value && !is_complex<T>::value && !is_tuple_like<T>::value>::type> {
static constexpr int value{subtype_count_min<typename T::value_type>::value};
};

template <typename T, std::size_t I>
constexpr typename std::enable_if<I == type_count_base<T>::value, int>::type tuple_type_size_min() {
return 0;
}

template <typename T, std::size_t I>
constexpr typename std::enable_if < I<type_count_base<T>::value, int>::type tuple_type_size_min() {
return subtype_count_min<typename std::tuple_element<I, T>::type>::value + tuple_type_size_min<T, I + 1>();
}

template <typename T> struct type_count_min<T, typename std::enable_if<is_tuple_like<T>::value>::type> {
static constexpr int value{tuple_type_size_min<T, 0>()};
};

template <typename T> struct subtype_count_min {
static constexpr int value{is_mutable_container<T>::value
? ((type_count<T>::value < expected_max_vector_size) ? type_count<T>::value : 0)
: type_count_min<T>::value};
};

template <typename T, typename Enable = void> struct expected_count { static const int value{0}; };

template <typename T>
struct expected_count<T,
typename std::enable_if<!is_mutable_container<T>::value && !is_wrapper<T>::value &&
!std::is_void<T>::value>::type> {
static constexpr int value{1};
};
template <typename T> struct expected_count<T, typename std::enable_if<is_mutable_container<T>::value>::type> {
static constexpr int value{expected_max_vector_size};
};

template <typename T>
struct expected_count<T, typename std::enable_if<!is_mutable_container<T>::value && is_wrapper<T>::value>::type> {
static constexpr int value{expected_count<typename T::value_type>::value};
};

enum class object_category : int {
char_value = 1,
integral_value = 2,
unsigned_integral = 4,
enumeration = 6,
boolean_value = 8,
floating_point = 10,
number_constructible = 12,
double_constructible = 14,
integer_constructible = 16,
string_assignable = 23,
string_constructible = 24,
other = 45,
wrapper_value = 50,
complex_number = 60,
tuple_value = 70,
container_value = 80,

};


template <typename T, typename Enable = void> struct classify_object {
static constexpr object_category value{object_category::other};
};

template <typename T>
struct classify_object<
T,
typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, char>::value && std::is_signed<T>::value &&
!is_bool<T>::value && !std::is_enum<T>::value>::type> {
static constexpr object_category value{object_category::integral_value};
};

template <typename T>
struct classify_object<T,
typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value &&
!std::is_same<T, char>::value && !is_bool<T>::value>::type> {
static constexpr object_category value{object_category::unsigned_integral};
};

template <typename T>
struct classify_object<T, typename std::enable_if<std::is_same<T, char>::value && !std::is_enum<T>::value>::type> {
static constexpr object_category value{object_category::char_value};
};

template <typename T> struct classify_object<T, typename std::enable_if<is_bool<T>::value>::type> {
static constexpr object_category value{object_category::boolean_value};
};

template <typename T> struct classify_object<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
static constexpr object_category value{object_category::floating_point};
};

template <typename T>
struct classify_object<T,
typename std::enable_if<!std::is_floating_point<T>::value && !std::is_integral<T>::value &&
std::is_assignable<T &, std::string>::value>::type> {
static constexpr object_category value{object_category::string_assignable};
};

template <typename T>
struct classify_object<
T,
typename std::enable_if<!std::is_floating_point<T>::value && !std::is_integral<T>::value &&
!std::is_assignable<T &, std::string>::value && (type_count<T>::value == 1) &&
std::is_constructible<T, std::string>::value>::type> {
static constexpr object_category value{object_category::string_constructible};
};

template <typename T> struct classify_object<T, typename std::enable_if<std::is_enum<T>::value>::type> {
static constexpr object_category value{object_category::enumeration};
};

template <typename T> struct classify_object<T, typename std::enable_if<is_complex<T>::value>::type> {
static constexpr object_category value{object_category::complex_number};
};

template <typename T> struct uncommon_type {
using type = typename std::conditional<!std::is_floating_point<T>::value && !std::is_integral<T>::value &&
!std::is_assignable<T &, std::string>::value &&
!std::is_constructible<T, std::string>::value && !is_complex<T>::value &&
!is_mutable_container<T>::value && !std::is_enum<T>::value,
std::true_type,
std::false_type>::type;
static constexpr bool value = type::value;
};

template <typename T>
struct classify_object<T,
typename std::enable_if<(!is_mutable_container<T>::value && is_wrapper<T>::value &&
!is_tuple_like<T>::value && uncommon_type<T>::value)>::type> {
static constexpr object_category value{object_category::wrapper_value};
};

template <typename T>
struct classify_object<T,
typename std::enable_if<uncommon_type<T>::value && type_count<T>::value == 1 &&
!is_wrapper<T>::value && is_direct_constructible<T, double>::value &&
is_direct_constructible<T, int>::value>::type> {
static constexpr object_category value{object_category::number_constructible};
};

template <typename T>
struct classify_object<T,
typename std::enable_if<uncommon_type<T>::value && type_count<T>::value == 1 &&
!is_wrapper<T>::value && !is_direct_constructible<T, double>::value &&
is_direct_constructible<T, int>::value>::type> {
static constexpr object_category value{object_category::integer_constructible};
};

template <typename T>
struct classify_object<T,
typename std::enable_if<uncommon_type<T>::value && type_count<T>::value == 1 &&
!is_wrapper<T>::value && is_direct_constructible<T, double>::value &&
!is_direct_constructible<T, int>::value>::type> {
static constexpr object_category value{object_category::double_constructible};
};

template <typename T>
struct classify_object<
T,
typename std::enable_if<is_tuple_like<T>::value &&
((type_count<T>::value >= 2 && !is_wrapper<T>::value) ||
(uncommon_type<T>::value && !is_direct_constructible<T, double>::value &&
!is_direct_constructible<T, int>::value))>::type> {
static constexpr object_category value{object_category::tuple_value};
};

template <typename T> struct classify_object<T, typename std::enable_if<is_mutable_container<T>::value>::type> {
static constexpr object_category value{object_category::container_value};
};



template <typename T,
enable_if_t<classify_object<T>::value == object_category::char_value, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
return "CHAR";
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::integral_value ||
classify_object<T>::value == object_category::integer_constructible,
detail::enabler> = detail::dummy>
constexpr const char *type_name() {
return "INT";
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::unsigned_integral, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
return "UINT";
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::floating_point ||
classify_object<T>::value == object_category::number_constructible ||
classify_object<T>::value == object_category::double_constructible,
detail::enabler> = detail::dummy>
constexpr const char *type_name() {
return "FLOAT";
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::enumeration, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
return "ENUM";
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::boolean_value, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
return "BOOLEAN";
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::complex_number, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
return "COMPLEX";
}

template <typename T,
enable_if_t<classify_object<T>::value >= object_category::string_assignable &&
classify_object<T>::value <= object_category::other,
detail::enabler> = detail::dummy>
constexpr const char *type_name() {
return "TEXT";
}
template <typename T,
enable_if_t<classify_object<T>::value == object_category::tuple_value && type_count_base<T>::value >= 2,
detail::enabler> = detail::dummy>
std::string type_name();  

template <typename T,
enable_if_t<classify_object<T>::value == object_category::container_value ||
classify_object<T>::value == object_category::wrapper_value,
detail::enabler> = detail::dummy>
std::string type_name();  

template <typename T,
enable_if_t<classify_object<T>::value == object_category::tuple_value && type_count_base<T>::value == 1,
detail::enabler> = detail::dummy>
inline std::string type_name() {
return type_name<typename std::decay<typename std::tuple_element<0, T>::type>::type>();
}

template <typename T, std::size_t I>
inline typename std::enable_if<I == type_count_base<T>::value, std::string>::type tuple_name() {
return std::string{};
}

template <typename T, std::size_t I>
inline typename std::enable_if<(I < type_count_base<T>::value), std::string>::type tuple_name() {
std::string str = std::string(type_name<typename std::decay<typename std::tuple_element<I, T>::type>::type>()) +
',' + tuple_name<T, I + 1>();
if(str.back() == ',')
str.pop_back();
return str;
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::tuple_value && type_count_base<T>::value >= 2,
detail::enabler>>
inline std::string type_name() {
auto tname = std::string(1, '[') + tuple_name<T, 0>();
tname.push_back(']');
return tname;
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::container_value ||
classify_object<T>::value == object_category::wrapper_value,
detail::enabler>>
inline std::string type_name() {
return type_name<typename T::value_type>();
}


template <typename T, enable_if_t<std::is_unsigned<T>::value, detail::enabler> = detail::dummy>
bool integral_conversion(const std::string &input, T &output) noexcept {
if(input.empty()) {
return false;
}
char *val = nullptr;
std::uint64_t output_ll = std::strtoull(input.c_str(), &val, 0);
output = static_cast<T>(output_ll);
return val == (input.c_str() + input.size()) && static_cast<std::uint64_t>(output) == output_ll;
}

template <typename T, enable_if_t<std::is_signed<T>::value, detail::enabler> = detail::dummy>
bool integral_conversion(const std::string &input, T &output) noexcept {
if(input.empty()) {
return false;
}
char *val = nullptr;
std::int64_t output_ll = std::strtoll(input.c_str(), &val, 0);
output = static_cast<T>(output_ll);
return val == (input.c_str() + input.size()) && static_cast<std::int64_t>(output) == output_ll;
}

inline std::int64_t to_flag_value(std::string val) {
static const std::string trueString("true");
static const std::string falseString("false");
if(val == trueString) {
return 1;
}
if(val == falseString) {
return -1;
}
val = detail::to_lower(val);
std::int64_t ret;
if(val.size() == 1) {
if(val[0] >= '1' && val[0] <= '9') {
return (static_cast<std::int64_t>(val[0]) - '0');
}
switch(val[0]) {
case '0':
case 'f':
case 'n':
case '-':
ret = -1;
break;
case 't':
case 'y':
case '+':
ret = 1;
break;
default:
throw std::invalid_argument("unrecognized character");
}
return ret;
}
if(val == trueString || val == "on" || val == "yes" || val == "enable") {
ret = 1;
} else if(val == falseString || val == "off" || val == "no" || val == "disable") {
ret = -1;
} else {
ret = std::stoll(val);
}
return ret;
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::integral_value ||
classify_object<T>::value == object_category::unsigned_integral,
detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
return integral_conversion(input, output);
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::char_value, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
if(input.size() == 1) {
output = static_cast<T>(input[0]);
return true;
}
return integral_conversion(input, output);
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::boolean_value, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
try {
auto out = to_flag_value(input);
output = (out > 0);
return true;
} catch(const std::invalid_argument &) {
return false;
} catch(const std::out_of_range &) {
output = (input[0] != '-');
return true;
}
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::floating_point, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
if(input.empty()) {
return false;
}
char *val = nullptr;
auto output_ld = std::strtold(input.c_str(), &val);
output = static_cast<T>(output_ld);
return val == (input.c_str() + input.size());
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::complex_number, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
using XC = typename wrapped_type<T, double>::type;
XC x{0.0}, y{0.0};
auto str1 = input;
bool worked = false;
auto nloc = str1.find_last_of("+-");
if(nloc != std::string::npos && nloc > 0) {
worked = detail::lexical_cast(str1.substr(0, nloc), x);
str1 = str1.substr(nloc);
if(str1.back() == 'i' || str1.back() == 'j')
str1.pop_back();
worked = worked && detail::lexical_cast(str1, y);
} else {
if(str1.back() == 'i' || str1.back() == 'j') {
str1.pop_back();
worked = detail::lexical_cast(str1, y);
x = XC{0};
} else {
worked = detail::lexical_cast(str1, x);
y = XC{0};
}
}
if(worked) {
output = T{x, y};
return worked;
}
return from_stream(input, output);
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::string_assignable, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
output = input;
return true;
}

template <
typename T,
enable_if_t<classify_object<T>::value == object_category::string_constructible, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
output = T(input);
return true;
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::enumeration, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
typename std::underlying_type<T>::type val;
if(!integral_conversion(input, val)) {
return false;
}
output = static_cast<T>(val);
return true;
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::wrapper_value &&
std::is_assignable<T &, typename T::value_type>::value,
detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
typename T::value_type val;
if(lexical_cast(input, val)) {
output = val;
return true;
}
return from_stream(input, output);
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::wrapper_value &&
!std::is_assignable<T &, typename T::value_type>::value && std::is_assignable<T &, T>::value,
detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
typename T::value_type val;
if(lexical_cast(input, val)) {
output = T{val};
return true;
}
return from_stream(input, output);
}

template <
typename T,
enable_if_t<classify_object<T>::value == object_category::number_constructible, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
int val;
if(integral_conversion(input, val)) {
output = T(val);
return true;
} else {
double dval;
if(lexical_cast(input, dval)) {
output = T{dval};
return true;
}
}
return from_stream(input, output);
}

template <
typename T,
enable_if_t<classify_object<T>::value == object_category::integer_constructible, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
int val;
if(integral_conversion(input, val)) {
output = T(val);
return true;
}
return from_stream(input, output);
}

template <
typename T,
enable_if_t<classify_object<T>::value == object_category::double_constructible, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
double val;
if(lexical_cast(input, val)) {
output = T{val};
return true;
}
return from_stream(input, output);
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::other && std::is_assignable<T &, int>::value,
detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
int val;
if(integral_conversion(input, val)) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800)
#endif
output = val;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
return true;
}
return from_stream(input, output);
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::other && !std::is_assignable<T &, int>::value,
detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
static_assert(is_istreamable<T>::value,
"option object type must have a lexical cast overload or streaming input operator(>>) defined, if it "
"is convertible from another type use the add_option<T, XC>(...) with XC being the known type");
return from_stream(input, output);
}

template <typename AssignTo,
typename ConvertTo,
enable_if_t<std::is_same<AssignTo, ConvertTo>::value &&
(classify_object<AssignTo>::value == object_category::string_assignable ||
classify_object<AssignTo>::value == object_category::string_constructible),
detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
return lexical_cast(input, output);
}

template <typename AssignTo,
typename ConvertTo,
enable_if_t<std::is_same<AssignTo, ConvertTo>::value && std::is_assignable<AssignTo &, AssignTo>::value &&
classify_object<AssignTo>::value != object_category::string_assignable &&
classify_object<AssignTo>::value != object_category::string_constructible,
detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
if(input.empty()) {
output = AssignTo{};
return true;
}

return lexical_cast(input, output);
}

template <typename AssignTo,
typename ConvertTo,
enable_if_t<std::is_same<AssignTo, ConvertTo>::value && !std::is_assignable<AssignTo &, AssignTo>::value &&
classify_object<AssignTo>::value == object_category::wrapper_value,
detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
if(input.empty()) {
typename AssignTo::value_type emptyVal{};
output = emptyVal;
return true;
}
return lexical_cast(input, output);
}

template <typename AssignTo,
typename ConvertTo,
enable_if_t<std::is_same<AssignTo, ConvertTo>::value && !std::is_assignable<AssignTo &, AssignTo>::value &&
classify_object<AssignTo>::value != object_category::wrapper_value &&
std::is_assignable<AssignTo &, int>::value,
detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
if(input.empty()) {
output = 0;
return true;
}
int val;
if(lexical_cast(input, val)) {
output = val;
return true;
}
return false;
}

template <typename AssignTo,
typename ConvertTo,
enable_if_t<!std::is_same<AssignTo, ConvertTo>::value && std::is_assignable<AssignTo &, ConvertTo &>::value,
detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
ConvertTo val{};
bool parse_result = (!input.empty()) ? lexical_cast<ConvertTo>(input, val) : true;
if(parse_result) {
output = val;
}
return parse_result;
}

template <
typename AssignTo,
typename ConvertTo,
enable_if_t<!std::is_same<AssignTo, ConvertTo>::value && !std::is_assignable<AssignTo &, ConvertTo &>::value &&
std::is_move_assignable<AssignTo>::value,
detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, AssignTo &output) {
ConvertTo val{};
bool parse_result = input.empty() ? true : lexical_cast<ConvertTo>(input, val);
if(parse_result) {
output = AssignTo(val);  
}
return parse_result;
}

template <typename AssignTo,
typename ConvertTo,
enable_if_t<classify_object<ConvertTo>::value <= object_category::other &&
classify_object<AssignTo>::value <= object_category::wrapper_value,
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
return lexical_assign<AssignTo, ConvertTo>(strings[0], output);
}

template <typename AssignTo,
typename ConvertTo,
enable_if_t<(type_count<AssignTo>::value <= 2) && expected_count<AssignTo>::value == 1 &&
is_tuple_like<ConvertTo>::value && type_count_base<ConvertTo>::value == 2,
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
typename std::remove_const<typename std::tuple_element<0, ConvertTo>::type>::type v1;
typename std::tuple_element<1, ConvertTo>::type v2;
bool retval = lexical_assign<decltype(v1), decltype(v1)>(strings[0], v1);
if(strings.size() > 1) {
retval = retval && lexical_assign<decltype(v2), decltype(v2)>(strings[1], v2);
}
if(retval) {
output = AssignTo{v1, v2};
}
return retval;
}

template <class AssignTo,
class ConvertTo,
enable_if_t<is_mutable_container<AssignTo>::value && is_mutable_container<ConvertTo>::value &&
type_count<ConvertTo>::value == 1,
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
output.erase(output.begin(), output.end());
for(const auto &elem : strings) {
typename AssignTo::value_type out;
bool retval = lexical_assign<typename AssignTo::value_type, typename ConvertTo::value_type>(elem, out);
if(!retval) {
return false;
}
output.insert(output.end(), std::move(out));
}
return (!output.empty());
}

template <class AssignTo, class ConvertTo, enable_if_t<is_complex<ConvertTo>::value, detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std::string> &strings, AssignTo &output) {

if(strings.size() >= 2 && !strings[1].empty()) {
using XC2 = typename wrapped_type<ConvertTo, double>::type;
XC2 x{0.0}, y{0.0};
auto str1 = strings[1];
if(str1.back() == 'i' || str1.back() == 'j') {
str1.pop_back();
}
auto worked = detail::lexical_cast(strings[0], x) && detail::lexical_cast(str1, y);
if(worked) {
output = ConvertTo{x, y};
}
return worked;
} else {
return lexical_assign<AssignTo, ConvertTo>(strings[0], output);
}
}

template <class AssignTo,
class ConvertTo,
enable_if_t<is_mutable_container<AssignTo>::value && (expected_count<ConvertTo>::value == 1) &&
(type_count<ConvertTo>::value == 1),
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
bool retval = true;
output.clear();
output.reserve(strings.size());
for(const auto &elem : strings) {

output.emplace_back();
retval = retval && lexical_assign<typename AssignTo::value_type, ConvertTo>(elem, output.back());
}
return (!output.empty()) && retval;
}


template <class AssignTo,
class ConvertTo,
enable_if_t<is_mutable_container<AssignTo>::value && is_mutable_container<ConvertTo>::value &&
type_count_base<ConvertTo>::value == 2,
detail::enabler> = detail::dummy>
bool lexical_conversion(std::vector<std::string> strings, AssignTo &output);

template <class AssignTo,
class ConvertTo,
enable_if_t<is_mutable_container<AssignTo>::value && is_mutable_container<ConvertTo>::value &&
type_count_base<ConvertTo>::value != 2 &&
((type_count<ConvertTo>::value > 2) ||
(type_count<ConvertTo>::value > type_count_base<ConvertTo>::value)),
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std::string> &strings, AssignTo &output);

template <class AssignTo,
class ConvertTo,
enable_if_t<is_tuple_like<AssignTo>::value && is_tuple_like<ConvertTo>::value &&
(type_count_base<ConvertTo>::value != type_count<ConvertTo>::value ||
type_count<ConvertTo>::value > 2),
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std::string> &strings, AssignTo &output);  

template <typename AssignTo,
typename ConvertTo,
enable_if_t<!is_tuple_like<AssignTo>::value && !is_mutable_container<AssignTo>::value &&
classify_object<ConvertTo>::value != object_category::wrapper_value &&
(is_mutable_container<ConvertTo>::value || type_count<ConvertTo>::value > 2),
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {

if(strings.size() > 1 || (!strings.empty() && !(strings.front().empty()))) {
ConvertTo val;
auto retval = lexical_conversion<ConvertTo, ConvertTo>(strings, val);
output = AssignTo{val};
return retval;
}
output = AssignTo{};
return true;
}

template <class AssignTo, class ConvertTo, std::size_t I>
inline typename std::enable_if<(I >= type_count_base<AssignTo>::value), bool>::type
tuple_conversion(const std::vector<std::string> &, AssignTo &) {
return true;
}

template <class AssignTo, class ConvertTo>
inline typename std::enable_if<!is_mutable_container<ConvertTo>::value && type_count<ConvertTo>::value == 1, bool>::type
tuple_type_conversion(std::vector<std::string> &strings, AssignTo &output) {
auto retval = lexical_assign<AssignTo, ConvertTo>(strings[0], output);
strings.erase(strings.begin());
return retval;
}

template <class AssignTo, class ConvertTo>
inline typename std::enable_if<!is_mutable_container<ConvertTo>::value && (type_count<ConvertTo>::value > 1) &&
type_count<ConvertTo>::value == type_count_min<ConvertTo>::value,
bool>::type
tuple_type_conversion(std::vector<std::string> &strings, AssignTo &output) {
auto retval = lexical_conversion<AssignTo, ConvertTo>(strings, output);
strings.erase(strings.begin(), strings.begin() + type_count<ConvertTo>::value);
return retval;
}

template <class AssignTo, class ConvertTo>
inline typename std::enable_if<is_mutable_container<ConvertTo>::value ||
type_count<ConvertTo>::value != type_count_min<ConvertTo>::value,
bool>::type
tuple_type_conversion(std::vector<std::string> &strings, AssignTo &output) {

std::size_t index{subtype_count_min<ConvertTo>::value};
const std::size_t mx_count{subtype_count<ConvertTo>::value};
const std::size_t mx{(std::max)(mx_count, strings.size())};

while(index < mx) {
if(is_separator(strings[index])) {
break;
}
++index;
}
bool retval = lexical_conversion<AssignTo, ConvertTo>(
std::vector<std::string>(strings.begin(), strings.begin() + static_cast<std::ptrdiff_t>(index)), output);
strings.erase(strings.begin(), strings.begin() + static_cast<std::ptrdiff_t>(index) + 1);
return retval;
}

template <class AssignTo, class ConvertTo, std::size_t I>
inline typename std::enable_if<(I < type_count_base<AssignTo>::value), bool>::type
tuple_conversion(std::vector<std::string> strings, AssignTo &output) {
bool retval = true;
using ConvertToElement = typename std::
conditional<is_tuple_like<ConvertTo>::value, typename std::tuple_element<I, ConvertTo>::type, ConvertTo>::type;
if(!strings.empty()) {
retval = retval && tuple_type_conversion<typename std::tuple_element<I, AssignTo>::type, ConvertToElement>(
strings, std::get<I>(output));
}
retval = retval && tuple_conversion<AssignTo, ConvertTo, I + 1>(std::move(strings), output);
return retval;
}

template <class AssignTo,
class ConvertTo,
enable_if_t<is_mutable_container<AssignTo>::value && is_mutable_container<ConvertTo>::value &&
type_count_base<ConvertTo>::value == 2,
detail::enabler>>
bool lexical_conversion(std::vector<std::string> strings, AssignTo &output) {
output.clear();
while(!strings.empty()) {

typename std::remove_const<typename std::tuple_element<0, typename ConvertTo::value_type>::type>::type v1;
typename std::tuple_element<1, typename ConvertTo::value_type>::type v2;
bool retval = tuple_type_conversion<decltype(v1), decltype(v1)>(strings, v1);
if(!strings.empty()) {
retval = retval && tuple_type_conversion<decltype(v2), decltype(v2)>(strings, v2);
}
if(retval) {
output.insert(output.end(), typename AssignTo::value_type{v1, v2});
} else {
return false;
}
}
return (!output.empty());
}

template <class AssignTo,
class ConvertTo,
enable_if_t<is_tuple_like<AssignTo>::value && is_tuple_like<ConvertTo>::value &&
(type_count_base<ConvertTo>::value != type_count<ConvertTo>::value ||
type_count<ConvertTo>::value > 2),
detail::enabler>>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
static_assert(
!is_tuple_like<ConvertTo>::value || type_count_base<AssignTo>::value == type_count_base<ConvertTo>::value,
"if the conversion type is defined as a tuple it must be the same size as the type you are converting to");
return tuple_conversion<AssignTo, ConvertTo, 0>(strings, output);
}

template <class AssignTo,
class ConvertTo,
enable_if_t<is_mutable_container<AssignTo>::value && is_mutable_container<ConvertTo>::value &&
type_count_base<ConvertTo>::value != 2 &&
((type_count<ConvertTo>::value > 2) ||
(type_count<ConvertTo>::value > type_count_base<ConvertTo>::value)),
detail::enabler>>
bool lexical_conversion(const std::vector<std ::string> &strings, AssignTo &output) {
bool retval = true;
output.clear();
std::vector<std::string> temp;
std::size_t ii{0};
std::size_t icount{0};
std::size_t xcm{type_count<ConvertTo>::value};
auto ii_max = strings.size();
while(ii < ii_max) {
temp.push_back(strings[ii]);
++ii;
++icount;
if(icount == xcm || is_separator(temp.back()) || ii == ii_max) {
if(static_cast<int>(xcm) > type_count_min<ConvertTo>::value && is_separator(temp.back())) {
temp.pop_back();
}
typename AssignTo::value_type temp_out;
retval = retval &&
lexical_conversion<typename AssignTo::value_type, typename ConvertTo::value_type>(temp, temp_out);
temp.clear();
if(!retval) {
return false;
}
output.insert(output.end(), std::move(temp_out));
icount = 0;
}
}
return retval;
}

template <typename AssignTo,
class ConvertTo,
enable_if_t<classify_object<ConvertTo>::value == object_category::wrapper_value &&
std::is_assignable<ConvertTo &, ConvertTo>::value,
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std::string> &strings, AssignTo &output) {
if(strings.empty() || strings.front().empty()) {
output = ConvertTo{};
return true;
}
typename ConvertTo::value_type val;
if(lexical_conversion<typename ConvertTo::value_type, typename ConvertTo::value_type>(strings, val)) {
output = ConvertTo{val};
return true;
}
return false;
}

template <typename AssignTo,
class ConvertTo,
enable_if_t<classify_object<ConvertTo>::value == object_category::wrapper_value &&
!std::is_assignable<AssignTo &, ConvertTo>::value,
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std::string> &strings, AssignTo &output) {
using ConvertType = typename ConvertTo::value_type;
if(strings.empty() || strings.front().empty()) {
output = ConvertType{};
return true;
}
ConvertType val;
if(lexical_conversion<typename ConvertTo::value_type, typename ConvertTo::value_type>(strings, val)) {
output = val;
return true;
}
return false;
}

template <typename T, enable_if_t<std::is_unsigned<T>::value, detail::enabler> = detail::dummy>
void sum_flag_vector(const std::vector<std::string> &flags, T &output) {
std::int64_t count{0};
for(auto &flag : flags) {
count += detail::to_flag_value(flag);
}
output = (count > 0) ? static_cast<T>(count) : T{0};
}

template <typename T, enable_if_t<std::is_signed<T>::value, detail::enabler> = detail::dummy>
void sum_flag_vector(const std::vector<std::string> &flags, T &output) {
std::int64_t count{0};
for(auto &flag : flags) {
count += detail::to_flag_value(flag);
}
output = static_cast<T>(count);
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800)
#endif

template <typename T,
enable_if_t<!std::is_signed<T>::value && !std::is_unsigned<T>::value, detail::enabler> = detail::dummy>
void sum_flag_vector(const std::vector<std::string> &flags, T &output) {
std::int64_t count{0};
for(auto &flag : flags) {
count += detail::to_flag_value(flag);
}
std::string out = detail::to_string(count);
lexical_cast(out, output);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

}  
}  
