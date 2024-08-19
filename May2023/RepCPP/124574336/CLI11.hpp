#pragma once




#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>




#define CLI11_VERSION_MAJOR 1
#define CLI11_VERSION_MINOR 9
#define CLI11_VERSION_PATCH 1
#define CLI11_VERSION "1.9.1"






#if !(defined(_MSC_VER) && __cplusplus == 199711L) && !defined(__INTEL_COMPILER)
#if __cplusplus >= 201402L
#define CLI11_CPP14
#if __cplusplus >= 201703L
#define CLI11_CPP17
#if __cplusplus > 201703L
#define CLI11_CPP20
#endif
#endif
#endif
#elif defined(_MSC_VER) && __cplusplus == 199711L
#if _MSVC_LANG >= 201402L
#define CLI11_CPP14
#if _MSVC_LANG > 201402L && _MSC_VER >= 1910
#define CLI11_CPP17
#if __MSVC_LANG > 201703L && _MSC_VER >= 1910
#define CLI11_CPP20
#endif
#endif
#endif
#endif

#if defined(CLI11_CPP14)
#define CLI11_DEPRECATED(reason) [[deprecated(reason)]]
#elif defined(_MSC_VER)
#define CLI11_DEPRECATED(reason) __declspec(deprecated(reason))
#else
#define CLI11_DEPRECATED(reason) __attribute__((deprecated(reason)))
#endif






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

namespace enums {

template <typename T, typename = typename std::enable_if<std::is_enum<T>::value>::type>
std::ostream &operator<<(std::ostream &in, const T &item) {
return in << static_cast<typename std::underlying_type<T>::type>(item);
}

}  

using enums::operator<<;

namespace detail {
constexpr int expected_max_vector_size{1 << 29};
inline std::vector<std::string> split(const std::string &s, char delim) {
std::vector<std::string> elems;
if(s.empty()) {
elems.emplace_back();
} else {
std::stringstream ss;
ss.str(s);
std::string item;
while(std::getline(ss, item, delim)) {
elems.push_back(item);
}
}
return elems;
}

template <typename T> std::string join(const T &v, std::string delim = ",") {
std::ostringstream s;
auto beg = std::begin(v);
auto end = std::end(v);
if(beg != end)
s << *beg++;
while(beg != end) {
s << delim << *beg++;
}
return s.str();
}

template <typename T,
typename Callable,
typename = typename std::enable_if<!std::is_constructible<std::string, Callable>::value>::type>
std::string join(const T &v, Callable func, std::string delim = ",") {
std::ostringstream s;
auto beg = std::begin(v);
auto end = std::end(v);
if(beg != end)
s << func(*beg++);
while(beg != end) {
s << delim << func(*beg++);
}
return s.str();
}

template <typename T> std::string rjoin(const T &v, std::string delim = ",") {
std::ostringstream s;
for(std::size_t start = 0; start < v.size(); start++) {
if(start > 0)
s << delim;
s << v[v.size() - start - 1];
}
return s.str();
}


inline std::string &ltrim(std::string &str) {
auto it = std::find_if(str.begin(), str.end(), [](char ch) { return !std::isspace<char>(ch, std::locale()); });
str.erase(str.begin(), it);
return str;
}

inline std::string &ltrim(std::string &str, const std::string &filter) {
auto it = std::find_if(str.begin(), str.end(), [&filter](char ch) { return filter.find(ch) == std::string::npos; });
str.erase(str.begin(), it);
return str;
}

inline std::string &rtrim(std::string &str) {
auto it = std::find_if(str.rbegin(), str.rend(), [](char ch) { return !std::isspace<char>(ch, std::locale()); });
str.erase(it.base(), str.end());
return str;
}

inline std::string &rtrim(std::string &str, const std::string &filter) {
auto it =
std::find_if(str.rbegin(), str.rend(), [&filter](char ch) { return filter.find(ch) == std::string::npos; });
str.erase(it.base(), str.end());
return str;
}

inline std::string &trim(std::string &str) { return ltrim(rtrim(str)); }

inline std::string &trim(std::string &str, const std::string filter) { return ltrim(rtrim(str, filter), filter); }

inline std::string trim_copy(const std::string &str) {
std::string s = str;
return trim(s);
}

inline std::string &remove_quotes(std::string &str) {
if(str.length() > 1 && (str.front() == '"' || str.front() == '\'')) {
if(str.front() == str.back()) {
str.pop_back();
str.erase(str.begin(), str.begin() + 1);
}
}
return str;
}

inline std::string trim_copy(const std::string &str, const std::string &filter) {
std::string s = str;
return trim(s, filter);
}
inline std::ostream &format_help(std::ostream &out, std::string name, std::string description, std::size_t wid) {
name = "  " + name;
out << std::setw(static_cast<int>(wid)) << std::left << name;
if(!description.empty()) {
if(name.length() >= wid)
out << "\n" << std::setw(static_cast<int>(wid)) << "";
for(const char c : description) {
out.put(c);
if(c == '\n') {
out << std::setw(static_cast<int>(wid)) << "";
}
}
}
out << "\n";
return out;
}

template <typename T> bool valid_first_char(T c) {
return std::isalnum(c, std::locale()) || c == '_' || c == '?' || c == '@';
}

template <typename T> bool valid_later_char(T c) { return valid_first_char(c) || c == '.' || c == '-'; }

inline bool valid_name_string(const std::string &str) {
if(str.empty() || !valid_first_char(str[0]))
return false;
for(auto c : str.substr(1))
if(!valid_later_char(c))
return false;
return true;
}

inline bool isalpha(const std::string &str) {
return std::all_of(str.begin(), str.end(), [](char c) { return std::isalpha(c, std::locale()); });
}

inline std::string to_lower(std::string str) {
std::transform(std::begin(str), std::end(str), std::begin(str), [](const std::string::value_type &x) {
return std::tolower(x, std::locale());
});
return str;
}

inline std::string remove_underscore(std::string str) {
str.erase(std::remove(std::begin(str), std::end(str), '_'), std::end(str));
return str;
}

inline std::string find_and_replace(std::string str, std::string from, std::string to) {

std::size_t start_pos = 0;

while((start_pos = str.find(from, start_pos)) != std::string::npos) {
str.replace(start_pos, from.length(), to);
start_pos += to.length();
}

return str;
}

inline bool has_default_flag_values(const std::string &flags) {
return (flags.find_first_of("{!") != std::string::npos);
}

inline void remove_default_flag_values(std::string &flags) {
auto loc = flags.find_first_of('{');
while(loc != std::string::npos) {
auto finish = flags.find_first_of("},", loc + 1);
if((finish != std::string::npos) && (flags[finish] == '}')) {
flags.erase(flags.begin() + static_cast<std::ptrdiff_t>(loc),
flags.begin() + static_cast<std::ptrdiff_t>(finish) + 1);
}
loc = flags.find_first_of('{', loc + 1);
}
flags.erase(std::remove(flags.begin(), flags.end(), '!'), flags.end());
}

inline std::ptrdiff_t find_member(std::string name,
const std::vector<std::string> names,
bool ignore_case = false,
bool ignore_underscore = false) {
auto it = std::end(names);
if(ignore_case) {
if(ignore_underscore) {
name = detail::to_lower(detail::remove_underscore(name));
it = std::find_if(std::begin(names), std::end(names), [&name](std::string local_name) {
return detail::to_lower(detail::remove_underscore(local_name)) == name;
});
} else {
name = detail::to_lower(name);
it = std::find_if(std::begin(names), std::end(names), [&name](std::string local_name) {
return detail::to_lower(local_name) == name;
});
}

} else if(ignore_underscore) {
name = detail::remove_underscore(name);
it = std::find_if(std::begin(names), std::end(names), [&name](std::string local_name) {
return detail::remove_underscore(local_name) == name;
});
} else {
it = std::find(std::begin(names), std::end(names), name);
}

return (it != std::end(names)) ? (it - std::begin(names)) : (-1);
}

template <typename Callable> inline std::string find_and_modify(std::string str, std::string trigger, Callable modify) {
std::size_t start_pos = 0;
while((start_pos = str.find(trigger, start_pos)) != std::string::npos) {
start_pos = modify(str, start_pos);
}
return str;
}

inline std::vector<std::string> split_up(std::string str, char delimiter = '\0') {

const std::string delims("\'\"`");
auto find_ws = [delimiter](char ch) {
return (delimiter == '\0') ? (std::isspace<char>(ch, std::locale()) != 0) : (ch == delimiter);
};
trim(str);

std::vector<std::string> output;
bool embeddedQuote = false;
char keyChar = ' ';
while(!str.empty()) {
if(delims.find_first_of(str[0]) != std::string::npos) {
keyChar = str[0];
auto end = str.find_first_of(keyChar, 1);
while((end != std::string::npos) && (str[end - 1] == '\\')) {  
end = str.find_first_of(keyChar, end + 1);
embeddedQuote = true;
}
if(end != std::string::npos) {
output.push_back(str.substr(1, end - 1));
str = str.substr(end + 1);
} else {
output.push_back(str.substr(1));
str = "";
}
} else {
auto it = std::find_if(std::begin(str), std::end(str), find_ws);
if(it != std::end(str)) {
std::string value = std::string(str.begin(), it);
output.push_back(value);
str = std::string(it + 1, str.end());
} else {
output.push_back(str);
str = "";
}
}
if(embeddedQuote) {
output.back() = find_and_replace(output.back(), std::string("\\") + keyChar, std::string(1, keyChar));
embeddedQuote = false;
}
trim(str);
}
return output;
}

inline std::string fix_newlines(const std::string &leader, std::string input) {
std::string::size_type n = 0;
while(n != std::string::npos && n < input.size()) {
n = input.find('\n', n);
if(n != std::string::npos) {
input = input.substr(0, n + 1) + leader + input.substr(n + 1);
n += leader.size();
}
}
return input;
}

inline std::size_t escape_detect(std::string &str, std::size_t offset) {
auto next = str[offset + 1];
if((next == '\"') || (next == '\'') || (next == '`')) {
auto astart = str.find_last_of("-/ \"\'`", offset - 1);
if(astart != std::string::npos) {
if(str[astart] == ((str[offset] == '=') ? '-' : '/'))
str[offset] = ' ';  
}
}
return offset + 1;
}

inline std::string &add_quotes_if_needed(std::string &str) {
if((str.front() != '"' && str.front() != '\'') || str.front() != str.back()) {
char quote = str.find('"') < str.find('\'') ? '\'' : '"';
if(str.find(' ') != std::string::npos) {
str.insert(0, 1, quote);
str.append(1, quote);
}
}
return str;
}

}  

}  


namespace CLI {

#define CLI11_ERROR_DEF(parent, name)                                                                                  \
protected:                                                                                                           \
name(std::string ename, std::string msg, int exit_code) : parent(std::move(ename), std::move(msg), exit_code) {}   \
name(std::string ename, std::string msg, ExitCodes exit_code)                                                      \
: parent(std::move(ename), std::move(msg), exit_code) {}                                                       \
\
public:                                                                                                              \
name(std::string msg, ExitCodes exit_code) : parent(#name, std::move(msg), exit_code) {}                           \
name(std::string msg, int exit_code) : parent(#name, std::move(msg), exit_code) {}

#define CLI11_ERROR_SIMPLE(name)                                                                                       \
explicit name(std::string msg) : name(#name, msg, ExitCodes::name) {}

enum class ExitCodes {
Success = 0,
IncorrectConstruction = 100,
BadNameString,
OptionAlreadyAdded,
FileError,
ConversionError,
ValidationError,
RequiredError,
RequiresError,
ExcludesError,
ExtrasError,
ConfigError,
InvalidError,
HorribleError,
OptionNotFound,
ArgumentMismatch,
BaseClass = 127
};



class Error : public std::runtime_error {
int actual_exit_code;
std::string error_name{"Error"};

public:
int get_exit_code() const { return actual_exit_code; }

std::string get_name() const { return error_name; }

Error(std::string name, std::string msg, int exit_code = static_cast<int>(ExitCodes::BaseClass))
: runtime_error(msg), actual_exit_code(exit_code), error_name(std::move(name)) {}

Error(std::string name, std::string msg, ExitCodes exit_code) : Error(name, msg, static_cast<int>(exit_code)) {}
};


class ConstructionError : public Error {
CLI11_ERROR_DEF(Error, ConstructionError)
};

class IncorrectConstruction : public ConstructionError {
CLI11_ERROR_DEF(ConstructionError, IncorrectConstruction)
CLI11_ERROR_SIMPLE(IncorrectConstruction)
static IncorrectConstruction PositionalFlag(std::string name) {
return IncorrectConstruction(name + ": Flags cannot be positional");
}
static IncorrectConstruction Set0Opt(std::string name) {
return IncorrectConstruction(name + ": Cannot set 0 expected, use a flag instead");
}
static IncorrectConstruction SetFlag(std::string name) {
return IncorrectConstruction(name + ": Cannot set an expected number for flags");
}
static IncorrectConstruction ChangeNotVector(std::string name) {
return IncorrectConstruction(name + ": You can only change the expected arguments for vectors");
}
static IncorrectConstruction AfterMultiOpt(std::string name) {
return IncorrectConstruction(
name + ": You can't change expected arguments after you've changed the multi option policy!");
}
static IncorrectConstruction MissingOption(std::string name) {
return IncorrectConstruction("Option " + name + " is not defined");
}
static IncorrectConstruction MultiOptionPolicy(std::string name) {
return IncorrectConstruction(name + ": multi_option_policy only works for flags and exact value options");
}
};

class BadNameString : public ConstructionError {
CLI11_ERROR_DEF(ConstructionError, BadNameString)
CLI11_ERROR_SIMPLE(BadNameString)
static BadNameString OneCharName(std::string name) { return BadNameString("Invalid one char name: " + name); }
static BadNameString BadLongName(std::string name) { return BadNameString("Bad long name: " + name); }
static BadNameString DashesOnly(std::string name) {
return BadNameString("Must have a name, not just dashes: " + name);
}
static BadNameString MultiPositionalNames(std::string name) {
return BadNameString("Only one positional name allowed, remove: " + name);
}
};

class OptionAlreadyAdded : public ConstructionError {
CLI11_ERROR_DEF(ConstructionError, OptionAlreadyAdded)
explicit OptionAlreadyAdded(std::string name)
: OptionAlreadyAdded(name + " is already added", ExitCodes::OptionAlreadyAdded) {}
static OptionAlreadyAdded Requires(std::string name, std::string other) {
return OptionAlreadyAdded(name + " requires " + other, ExitCodes::OptionAlreadyAdded);
}
static OptionAlreadyAdded Excludes(std::string name, std::string other) {
return OptionAlreadyAdded(name + " excludes " + other, ExitCodes::OptionAlreadyAdded);
}
};


class ParseError : public Error {
CLI11_ERROR_DEF(Error, ParseError)
};


class Success : public ParseError {
CLI11_ERROR_DEF(ParseError, Success)
Success() : Success("Successfully completed, should be caught and quit", ExitCodes::Success) {}
};

class CallForHelp : public ParseError {
CLI11_ERROR_DEF(ParseError, CallForHelp)
CallForHelp() : CallForHelp("This should be caught in your main function, see examples", ExitCodes::Success) {}
};

class CallForAllHelp : public ParseError {
CLI11_ERROR_DEF(ParseError, CallForAllHelp)
CallForAllHelp()
: CallForAllHelp("This should be caught in your main function, see examples", ExitCodes::Success) {}
};

class RuntimeError : public ParseError {
CLI11_ERROR_DEF(ParseError, RuntimeError)
explicit RuntimeError(int exit_code = 1) : RuntimeError("Runtime error", exit_code) {}
};

class FileError : public ParseError {
CLI11_ERROR_DEF(ParseError, FileError)
CLI11_ERROR_SIMPLE(FileError)
static FileError Missing(std::string name) { return FileError(name + " was not readable (missing?)"); }
};

class ConversionError : public ParseError {
CLI11_ERROR_DEF(ParseError, ConversionError)
CLI11_ERROR_SIMPLE(ConversionError)
ConversionError(std::string member, std::string name)
: ConversionError("The value " + member + " is not an allowed value for " + name) {}
ConversionError(std::string name, std::vector<std::string> results)
: ConversionError("Could not convert: " + name + " = " + detail::join(results)) {}
static ConversionError TooManyInputsFlag(std::string name) {
return ConversionError(name + ": too many inputs for a flag");
}
static ConversionError TrueFalse(std::string name) {
return ConversionError(name + ": Should be true/false or a number");
}
};

class ValidationError : public ParseError {
CLI11_ERROR_DEF(ParseError, ValidationError)
CLI11_ERROR_SIMPLE(ValidationError)
explicit ValidationError(std::string name, std::string msg) : ValidationError(name + ": " + msg) {}
};

class RequiredError : public ParseError {
CLI11_ERROR_DEF(ParseError, RequiredError)
explicit RequiredError(std::string name) : RequiredError(name + " is required", ExitCodes::RequiredError) {}
static RequiredError Subcommand(std::size_t min_subcom) {
if(min_subcom == 1) {
return RequiredError("A subcommand");
}
return RequiredError("Requires at least " + std::to_string(min_subcom) + " subcommands",
ExitCodes::RequiredError);
}
static RequiredError
Option(std::size_t min_option, std::size_t max_option, std::size_t used, const std::string &option_list) {
if((min_option == 1) && (max_option == 1) && (used == 0))
return RequiredError("Exactly 1 option from [" + option_list + "]");
if((min_option == 1) && (max_option == 1) && (used > 1)) {
return RequiredError("Exactly 1 option from [" + option_list + "] is required and " + std::to_string(used) +
" were given",
ExitCodes::RequiredError);
}
if((min_option == 1) && (used == 0))
return RequiredError("At least 1 option from [" + option_list + "]");
if(used < min_option) {
return RequiredError("Requires at least " + std::to_string(min_option) + " options used and only " +
std::to_string(used) + "were given from [" + option_list + "]",
ExitCodes::RequiredError);
}
if(max_option == 1)
return RequiredError("Requires at most 1 options be given from [" + option_list + "]",
ExitCodes::RequiredError);

return RequiredError("Requires at most " + std::to_string(max_option) + " options be used and " +
std::to_string(used) + "were given from [" + option_list + "]",
ExitCodes::RequiredError);
}
};

class ArgumentMismatch : public ParseError {
CLI11_ERROR_DEF(ParseError, ArgumentMismatch)
CLI11_ERROR_SIMPLE(ArgumentMismatch)
ArgumentMismatch(std::string name, int expected, std::size_t received)
: ArgumentMismatch(expected > 0 ? ("Expected exactly " + std::to_string(expected) + " arguments to " + name +
", got " + std::to_string(received))
: ("Expected at least " + std::to_string(-expected) + " arguments to " + name +
", got " + std::to_string(received)),
ExitCodes::ArgumentMismatch) {}

static ArgumentMismatch AtLeast(std::string name, int num, std::size_t received) {
return ArgumentMismatch(name + ": At least " + std::to_string(num) + " required but received " +
std::to_string(received));
}
static ArgumentMismatch AtMost(std::string name, int num, std::size_t received) {
return ArgumentMismatch(name + ": At Most " + std::to_string(num) + " required but received " +
std::to_string(received));
}
static ArgumentMismatch TypedAtLeast(std::string name, int num, std::string type) {
return ArgumentMismatch(name + ": " + std::to_string(num) + " required " + type + " missing");
}
static ArgumentMismatch FlagOverride(std::string name) {
return ArgumentMismatch(name + " was given a disallowed flag override");
}
};

class RequiresError : public ParseError {
CLI11_ERROR_DEF(ParseError, RequiresError)
RequiresError(std::string curname, std::string subname)
: RequiresError(curname + " requires " + subname, ExitCodes::RequiresError) {}
};

class ExcludesError : public ParseError {
CLI11_ERROR_DEF(ParseError, ExcludesError)
ExcludesError(std::string curname, std::string subname)
: ExcludesError(curname + " excludes " + subname, ExitCodes::ExcludesError) {}
};

class ExtrasError : public ParseError {
CLI11_ERROR_DEF(ParseError, ExtrasError)
explicit ExtrasError(std::vector<std::string> args)
: ExtrasError((args.size() > 1 ? "The following arguments were not expected: "
: "The following argument was not expected: ") +
detail::rjoin(args, " "),
ExitCodes::ExtrasError) {}
ExtrasError(const std::string &name, std::vector<std::string> args)
: ExtrasError(name,
(args.size() > 1 ? "The following arguments were not expected: "
: "The following argument was not expected: ") +
detail::rjoin(args, " "),
ExitCodes::ExtrasError) {}
};

class ConfigError : public ParseError {
CLI11_ERROR_DEF(ParseError, ConfigError)
CLI11_ERROR_SIMPLE(ConfigError)
static ConfigError Extras(std::string item) { return ConfigError("INI was not able to parse " + item); }
static ConfigError NotConfigurable(std::string item) {
return ConfigError(item + ": This option is not allowed in a configuration file");
}
};

class InvalidError : public ParseError {
CLI11_ERROR_DEF(ParseError, InvalidError)
explicit InvalidError(std::string name)
: InvalidError(name + ": Too many positional arguments with unlimited expected args", ExitCodes::InvalidError) {
}
};

class HorribleError : public ParseError {
CLI11_ERROR_DEF(ParseError, HorribleError)
CLI11_ERROR_SIMPLE(HorribleError)
};


class OptionNotFound : public Error {
CLI11_ERROR_DEF(Error, OptionNotFound)
explicit OptionNotFound(std::string name) : OptionNotFound(name + " not found", ExitCodes::OptionNotFound) {}
};

#undef CLI11_ERROR_DEF
#undef CLI11_ERROR_SIMPLE


}  


namespace CLI {


namespace detail {
enum class enabler {};

constexpr enabler dummy = {};
}  

template <bool B, class T = void> using enable_if_t = typename std::enable_if<B, T>::type;

template <typename... Ts> struct make_void { using type = void; };

template <typename... Ts> using void_t = typename make_void<Ts...>::type;

template <bool B, class T, class F> using conditional_t = typename std::conditional<B, T, F>::type;

template <typename T> struct is_vector : std::false_type {};

template <class T, class A> struct is_vector<std::vector<T, A>> : std::true_type {};

template <class T, class A> struct is_vector<const std::vector<T, A>> : std::true_type {};

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

template <typename S> class is_tuple_like {
template <typename SS>
static auto test(int) -> decltype(std::tuple_size<SS>::value, std::true_type{});
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
!is_vector<typename std::remove_reference<typename std::remove_const<T>::type>::type>::value,
detail::enabler> = detail::dummy>
std::string to_string(T &&) {
return std::string{};
}

template <typename T,
enable_if_t<!std::is_constructible<std::string, T>::value && !is_ostreamable<T>::value &&
is_vector<typename std::remove_reference<typename std::remove_const<T>::type>::type>::value,
detail::enabler> = detail::dummy>
std::string to_string(T &&variable) {
std::vector<std::string> defaults;
defaults.reserve(variable.size());
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

template <typename T, typename Enable = void> struct type_count { static const int value{0}; };

template <typename T> struct type_count<T, typename std::enable_if<is_tuple_like<T>::value>::type> {
static constexpr int value{std::tuple_size<T>::value};
};
template <typename T>
struct type_count<
T,
typename std::enable_if<!is_vector<T>::value && !is_tuple_like<T>::value && !std::is_void<T>::value>::type> {
static constexpr int value{1};
};

template <typename T> struct type_count<T, typename std::enable_if<is_vector<T>::value>::type> {
static constexpr int value{is_vector<typename T::value_type>::value ? expected_max_vector_size
: type_count<typename T::value_type>::value};
};

template <typename T, typename Enable = void> struct expected_count { static const int value{0}; };

template <typename T>
struct expected_count<T, typename std::enable_if<!is_vector<T>::value && !std::is_void<T>::value>::type> {
static constexpr int value{1};
};
template <typename T> struct expected_count<T, typename std::enable_if<is_vector<T>::value>::type> {
static constexpr int value{expected_max_vector_size};
};

enum class object_category : int {
integral_value = 2,
unsigned_integral = 4,
enumeration = 6,
boolean_value = 8,
floating_point = 10,
number_constructible = 12,
double_constructible = 14,
integer_constructible = 16,
vector_value = 30,
tuple_value = 35,
string_assignable = 50,
string_constructible = 60,
other = 200,

};

template <typename T, typename Enable = void> struct classify_object {
static constexpr object_category value{object_category::other};
};

template <typename T>
struct classify_object<T,
typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value &&
!is_bool<T>::value && !std::is_enum<T>::value>::type> {
static constexpr object_category value{object_category::integral_value};
};

template <typename T>
struct classify_object<
T,
typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value && !is_bool<T>::value>::type> {
static constexpr object_category value{object_category::unsigned_integral};
};

template <typename T> struct classify_object<T, typename std::enable_if<is_bool<T>::value>::type> {
static constexpr object_category value{object_category::boolean_value};
};

template <typename T> struct classify_object<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
static constexpr object_category value{object_category::floating_point};
};

template <typename T>
struct classify_object<
T,
typename std::enable_if<!std::is_floating_point<T>::value && !std::is_integral<T>::value &&
std::is_assignable<T &, std::string>::value && !is_vector<T>::value>::type> {
static constexpr object_category value{object_category::string_assignable};
};

template <typename T>
struct classify_object<
T,
typename std::enable_if<!std::is_floating_point<T>::value && !std::is_integral<T>::value &&
!std::is_assignable<T &, std::string>::value &&
std::is_constructible<T, std::string>::value && !is_vector<T>::value>::type> {
static constexpr object_category value{object_category::string_constructible};
};

template <typename T> struct classify_object<T, typename std::enable_if<std::is_enum<T>::value>::type> {
static constexpr object_category value{object_category::enumeration};
};

template <typename T> struct uncommon_type {
using type = typename std::conditional<!std::is_floating_point<T>::value && !std::is_integral<T>::value &&
!std::is_assignable<T &, std::string>::value &&
!std::is_constructible<T, std::string>::value && !is_vector<T>::value &&
!std::is_enum<T>::value,
std::true_type,
std::false_type>::type;
static constexpr bool value = type::value;
};

template <typename T>
struct classify_object<T,
typename std::enable_if<uncommon_type<T>::value && type_count<T>::value == 1 &&
is_direct_constructible<T, double>::value &&
is_direct_constructible<T, int>::value>::type> {
static constexpr object_category value{object_category::number_constructible};
};

template <typename T>
struct classify_object<T,
typename std::enable_if<uncommon_type<T>::value && type_count<T>::value == 1 &&
!is_direct_constructible<T, double>::value &&
is_direct_constructible<T, int>::value>::type> {
static constexpr object_category value{object_category::integer_constructible};
};

template <typename T>
struct classify_object<T,
typename std::enable_if<uncommon_type<T>::value && type_count<T>::value == 1 &&
is_direct_constructible<T, double>::value &&
!is_direct_constructible<T, int>::value>::type> {
static constexpr object_category value{object_category::double_constructible};
};

template <typename T>
struct classify_object<T,
typename std::enable_if<(type_count<T>::value >= 2 && !is_vector<T>::value) ||
(is_tuple_like<T>::value && uncommon_type<T>::value &&
!is_direct_constructible<T, double>::value &&
!is_direct_constructible<T, int>::value)>::type> {
static constexpr object_category value{object_category::tuple_value};
};

template <typename T> struct classify_object<T, typename std::enable_if<is_vector<T>::value>::type> {
static constexpr object_category value{object_category::vector_value};
};



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
enable_if_t<classify_object<T>::value >= object_category::string_assignable, detail::enabler> = detail::dummy>
constexpr const char *type_name() {
return "TEXT";
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::tuple_value && type_count<T>::value == 1,
detail::enabler> = detail::dummy>
inline std::string type_name() {
return type_name<typename std::tuple_element<0, T>::type>();
}

template <typename T, std::size_t I>
inline typename std::enable_if<I == type_count<T>::value, std::string>::type tuple_name() {
return std::string{};
}

template <typename T, std::size_t I>
inline typename std::enable_if < I<type_count<T>::value, std::string>::type tuple_name() {
std::string str = std::string(type_name<typename std::tuple_element<I, T>::type>()) + ',' + tuple_name<T, I + 1>();
if(str.back() == ',')
str.pop_back();
return str;
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::tuple_value && type_count<T>::value >= 2,
detail::enabler> = detail::dummy>
std::string type_name() {
auto tname = std::string(1, '[') + tuple_name<T, 0>();
tname.push_back(']');
return tname;
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::vector_value, detail::enabler> = detail::dummy>
inline std::string type_name() {
return type_name<typename T::value_type>();
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
enable_if_t<classify_object<T>::value == object_category::integral_value, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
try {
std::size_t n = 0;
std::int64_t output_ll = std::stoll(input, &n, 0);
output = static_cast<T>(output_ll);
return n == input.size() && static_cast<std::int64_t>(output) == output_ll;
} catch(const std::invalid_argument &) {
return false;
} catch(const std::out_of_range &) {
return false;
}
}

template <typename T,
enable_if_t<classify_object<T>::value == object_category::unsigned_integral, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
if(!input.empty() && input.front() == '-')
return false;  

try {
std::size_t n = 0;
std::uint64_t output_ll = std::stoull(input, &n, 0);
output = static_cast<T>(output_ll);
return n == input.size() && static_cast<std::uint64_t>(output) == output_ll;
} catch(const std::invalid_argument &) {
return false;
} catch(const std::out_of_range &) {
return false;
}
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
try {
std::size_t n = 0;
output = static_cast<T>(std::stold(input, &n));
return n == input.size();
} catch(const std::invalid_argument &) {
return false;
} catch(const std::out_of_range &) {
return false;
}
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
bool retval = detail::lexical_cast(input, val);
if(!retval) {
return false;
}
output = static_cast<T>(val);
return true;
}

template <
typename T,
enable_if_t<classify_object<T>::value == object_category::number_constructible, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
int val;
if(lexical_cast(input, val)) {
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
if(lexical_cast(input, val)) {
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

template <typename T, enable_if_t<classify_object<T>::value == object_category::other, detail::enabler> = detail::dummy>
bool lexical_cast(const std::string &input, T &output) {
static_assert(is_istreamable<T>::value,
"option object type must have a lexical cast overload or streaming input operator(>>) defined, if it "
"is convertible from another type use the add_option<T, XC>(...) with XC being the known type");
return from_stream(input, output);
}

template <
typename T,
typename XC,
enable_if_t<std::is_same<T, XC>::value && (classify_object<T>::value == object_category::string_assignable ||
classify_object<T>::value == object_category::string_constructible),
detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, T &output) {
return lexical_cast(input, output);
}

template <typename T,
typename XC,
enable_if_t<std::is_same<T, XC>::value && classify_object<T>::value != object_category::string_assignable &&
classify_object<T>::value != object_category::string_constructible,
detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, T &output) {
if(input.empty()) {
output = T{};
return true;
}
return lexical_cast(input, output);
}

template <
typename T,
typename XC,
enable_if_t<!std::is_same<T, XC>::value && std::is_assignable<T &, XC &>::value, detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, T &output) {
XC val{};
bool parse_result = (!input.empty()) ? lexical_cast<XC>(input, val) : true;
if(parse_result) {
output = val;
}
return parse_result;
}

template <typename T,
typename XC,
enable_if_t<!std::is_same<T, XC>::value && !std::is_assignable<T &, XC &>::value &&
std::is_move_assignable<T>::value,
detail::enabler> = detail::dummy>
bool lexical_assign(const std::string &input, T &output) {
XC val{};
bool parse_result = input.empty() ? true : lexical_cast<XC>(input, val);
if(parse_result) {
output = T(val);  
}
return parse_result;
}
template <
typename T,
typename XC,
enable_if_t<!is_tuple_like<T>::value && !is_tuple_like<XC>::value && !is_vector<T>::value && !is_vector<XC>::value,
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, T &output) {
return lexical_assign<T, XC>(strings[0], output);
}

template <typename T,
typename XC,
enable_if_t<type_count<T>::value == 1 && type_count<XC>::value == 2, detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, T &output) {
typename std::tuple_element<0, XC>::type v1;
typename std::tuple_element<1, XC>::type v2;
bool retval = lexical_assign<decltype(v1), decltype(v1)>(strings[0], v1);
if(strings.size() > 1) {
retval = retval && lexical_assign<decltype(v2), decltype(v2)>(strings[1], v2);
}
if(retval) {
output = T{v1, v2};
}
return retval;
}

template <class T,
class XC,
enable_if_t<expected_count<T>::value == expected_max_vector_size &&
expected_count<XC>::value == expected_max_vector_size && type_count<XC>::value == 1,
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, T &output) {
output.clear();
output.reserve(strings.size());
for(const auto &elem : strings) {

output.emplace_back();
bool retval = lexical_assign<typename T::value_type, typename XC::value_type>(elem, output.back());
if(!retval) {
return false;
}
}
return (!output.empty());
}

template <class T,
class XC,
enable_if_t<expected_count<T>::value == expected_max_vector_size &&
expected_count<XC>::value == expected_max_vector_size && type_count<XC>::value == 2,
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, T &output) {
output.clear();
for(std::size_t ii = 0; ii < strings.size(); ii += 2) {

typename std::tuple_element<0, typename XC::value_type>::type v1;
typename std::tuple_element<1, typename XC::value_type>::type v2;
bool retval = lexical_assign<decltype(v1), decltype(v1)>(strings[ii], v1);
if(strings.size() > ii + 1) {
retval = retval && lexical_assign<decltype(v2), decltype(v2)>(strings[ii + 1], v2);
}
if(retval) {
output.emplace_back(v1, v2);
} else {
return false;
}
}
return (!output.empty());
}

template <class T,
class XC,
enable_if_t<(expected_count<T>::value == expected_max_vector_size) && (expected_count<XC>::value == 1) &&
(type_count<XC>::value == 1),
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, T &output) {
bool retval = true;
output.clear();
output.reserve(strings.size());
for(const auto &elem : strings) {

output.emplace_back();
retval = retval && lexical_assign<typename T::value_type, XC>(elem, output.back());
}
return (!output.empty()) && retval;
}
template <typename T,
typename XC,
enable_if_t<!is_tuple_like<T>::value && !is_vector<T>::value && is_vector<XC>::value, detail::enabler> =
detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, T &output) {

if(strings.size() > 1 || (!strings.empty() && !(strings.front().empty()))) {
XC val;
auto retval = lexical_conversion<XC, XC>(strings, val);
output = T{val};
return retval;
}
output = T{};
return true;
}

template <class T, class XC, std::size_t I>
inline typename std::enable_if<I >= type_count<T>::value, bool>::type tuple_conversion(const std::vector<std::string> &,
T &) {
return true;
}
template <class T, class XC, std::size_t I>
inline typename std::enable_if <
I<type_count<T>::value, bool>::type tuple_conversion(const std::vector<std::string> &strings, T &output) {
bool retval = true;
if(strings.size() > I) {
retval = retval && lexical_assign<typename std::tuple_element<I, T>::type,
typename std::conditional<is_tuple_like<XC>::value,
typename std::tuple_element<I, XC>::type,
XC>::type>(strings[I], std::get<I>(output));
}
retval = retval && tuple_conversion<T, XC, I + 1>(strings, output);
return retval;
}

template <class T, class XC, enable_if_t<is_tuple_like<T>::value, detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, T &output) {
static_assert(
!is_tuple_like<XC>::value || type_count<T>::value == type_count<XC>::value,
"if the conversion type is defined as a tuple it must be the same size as the type you are converting to");
return tuple_conversion<T, XC, 0>(strings, output);
}

template <class T,
class XC,
enable_if_t<expected_count<T>::value == expected_max_vector_size &&
expected_count<XC>::value == expected_max_vector_size && (type_count<XC>::value > 2),
detail::enabler> = detail::dummy>
bool lexical_conversion(const std::vector<std ::string> &strings, T &output) {
bool retval = true;
output.clear();
std::vector<std::string> temp;
std::size_t ii = 0;
std::size_t icount = 0;
std::size_t xcm = type_count<XC>::value;
while(ii < strings.size()) {
temp.push_back(strings[ii]);
++ii;
++icount;
if(icount == xcm || temp.back().empty()) {
if(static_cast<int>(xcm) == expected_max_vector_size) {
temp.pop_back();
}
output.emplace_back();
retval = retval && lexical_conversion<typename T::value_type, typename XC::value_type>(temp, output.back());
temp.clear();
if(!retval) {
return false;
}
icount = 0;
}
}
return retval;
}
template <typename T,
enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, detail::enabler> = detail::dummy>
void sum_flag_vector(const std::vector<std::string> &flags, T &output) {
std::int64_t count{0};
for(auto &flag : flags) {
count += detail::to_flag_value(flag);
}
output = (count > 0) ? static_cast<T>(count) : T{0};
}

template <typename T,
enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, detail::enabler> = detail::dummy>
void sum_flag_vector(const std::vector<std::string> &flags, T &output) {
std::int64_t count{0};
for(auto &flag : flags) {
count += detail::to_flag_value(flag);
}
output = static_cast<T>(count);
}

}  
}  


namespace CLI {
namespace detail {

inline bool split_short(const std::string &current, std::string &name, std::string &rest) {
if(current.size() > 1 && current[0] == '-' && valid_first_char(current[1])) {
name = current.substr(1, 1);
rest = current.substr(2);
return true;
}
return false;
}

inline bool split_long(const std::string &current, std::string &name, std::string &value) {
if(current.size() > 2 && current.substr(0, 2) == "--" && valid_first_char(current[2])) {
auto loc = current.find_first_of('=');
if(loc != std::string::npos) {
name = current.substr(2, loc - 2);
value = current.substr(loc + 1);
} else {
name = current.substr(2);
value = "";
}
return true;
}
return false;
}

inline bool split_windows_style(const std::string &current, std::string &name, std::string &value) {
if(current.size() > 1 && current[0] == '/' && valid_first_char(current[1])) {
auto loc = current.find_first_of(':');
if(loc != std::string::npos) {
name = current.substr(1, loc - 1);
value = current.substr(loc + 1);
} else {
name = current.substr(1);
value = "";
}
return true;
}
return false;
}

inline std::vector<std::string> split_names(std::string current) {
std::vector<std::string> output;
std::size_t val;
while((val = current.find(",")) != std::string::npos) {
output.push_back(trim_copy(current.substr(0, val)));
current = current.substr(val + 1);
}
output.push_back(trim_copy(current));
return output;
}

inline std::vector<std::pair<std::string, std::string>> get_default_flag_values(const std::string &str) {
std::vector<std::string> flags = split_names(str);
flags.erase(std::remove_if(flags.begin(),
flags.end(),
[](const std::string &name) {
return ((name.empty()) || (!(((name.find_first_of('{') != std::string::npos) &&
(name.back() == '}')) ||
(name[0] == '!'))));
}),
flags.end());
std::vector<std::pair<std::string, std::string>> output;
output.reserve(flags.size());
for(auto &flag : flags) {
auto def_start = flag.find_first_of('{');
std::string defval = "false";
if((def_start != std::string::npos) && (flag.back() == '}')) {
defval = flag.substr(def_start + 1);
defval.pop_back();
flag.erase(def_start, std::string::npos);
}
flag.erase(0, flag.find_first_not_of("-!"));
output.emplace_back(flag, defval);
}
return output;
}

inline std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>
get_names(const std::vector<std::string> &input) {

std::vector<std::string> short_names;
std::vector<std::string> long_names;
std::string pos_name;

for(std::string name : input) {
if(name.length() == 0) {
continue;
}
if(name.length() > 1 && name[0] == '-' && name[1] != '-') {
if(name.length() == 2 && valid_first_char(name[1]))
short_names.emplace_back(1, name[1]);
else
throw BadNameString::OneCharName(name);
} else if(name.length() > 2 && name.substr(0, 2) == "--") {
name = name.substr(2);
if(valid_name_string(name))
long_names.push_back(name);
else
throw BadNameString::BadLongName(name);
} else if(name == "-" || name == "--") {
throw BadNameString::DashesOnly(name);
} else {
if(pos_name.length() > 0)
throw BadNameString::MultiPositionalNames(name);
pos_name = name;
}
}

return std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>(
short_names, long_names, pos_name);
}

}  
}  


namespace CLI {

class App;

struct ConfigItem {
std::vector<std::string> parents{};

std::string name{};

std::vector<std::string> inputs{};

std::string fullname() const {
std::vector<std::string> tmp = parents;
tmp.emplace_back(name);
return detail::join(tmp, ".");
}
};

class Config {
protected:
std::vector<ConfigItem> items{};

public:
virtual std::string to_config(const App *, bool, bool, std::string) const = 0;

virtual std::vector<ConfigItem> from_config(std::istream &) const = 0;

virtual std::string to_flag(const ConfigItem &item) const {
if(item.inputs.size() == 1) {
return item.inputs.at(0);
}
throw ConversionError::TooManyInputsFlag(item.fullname());
}

std::vector<ConfigItem> from_file(const std::string &name) {
std::ifstream input{name};
if(!input.good())
throw FileError::Missing(name);

return from_config(input);
}

virtual ~Config() = default;
};

class ConfigBase : public Config {
protected:
char commentChar = ';';
char arrayStart = '\0';
char arrayEnd = '\0';
char arraySeparator = ' ';
char valueDelimiter = '=';

public:
std::string
to_config(const App * , bool default_also, bool write_description, std::string prefix) const override;

std::vector<ConfigItem> from_config(std::istream &input) const override;
ConfigBase *comment(char cchar) {
commentChar = cchar;
return this;
}
ConfigBase *arrayBounds(char aStart, char aEnd) {
arrayStart = aStart;
arrayEnd = aEnd;
return this;
}
ConfigBase *arrayDelimiter(char aSep) {
arraySeparator = aSep;
return this;
}
ConfigBase *valueSeparator(char vSep) {
valueDelimiter = vSep;
return this;
}
};

using ConfigINI = ConfigBase;

class ConfigTOML : public ConfigINI {

public:
ConfigTOML() {
commentChar = '#';
arrayStart = '[';
arrayEnd = ']';
arraySeparator = ',';
valueDelimiter = '=';
}
};
}  


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

class PositiveNumber : public Validator {
public:
PositiveNumber() : Validator("POSITIVE") {
func_ = [](std::string &number_str) {
double number;
if(!detail::lexical_cast(number_str, number)) {
return std::string("Failed parsing number: (") + number_str + ')';
}
if(number <= 0) {
return std::string("Number less or equal to 0: (") + number_str + ')';
}
return std::string();
};
}
};
class NonNegativeNumber : public Validator {
public:
NonNegativeNumber() : Validator("NONNEGATIVE") {
func_ = [](std::string &number_str) {
double number;
if(!detail::lexical_cast(number_str, number)) {
return std::string("Failed parsing number: (") + number_str + ')';
}
if(number < 0) {
return std::string("Number less than 0: (") + number_str + ')';
}
return std::string();
};
}
};

class Number : public Validator {
public:
Number() : Validator("NUMBER") {
func_ = [](std::string &number_str) {
double number;
if(!detail::lexical_cast(number_str, number)) {
return std::string("Failed parsing as a number (") + number_str + ')';
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

const detail::PositiveNumber PositiveNumber;

const detail::NonNegativeNumber NonNegativeNumber;

const detail::Number Number;

class Range : public Validator {
public:
template <typename T> Range(T min, T max) {
std::stringstream out;
out << detail::type_name<T>() << " in [" << min << " - " << max << "]";
description(out.str());

func_ = [min, max](std::string &input) {
T val;
bool converted = detail::lexical_cast(input, val);
if((!converted) || (val < min || val > max))
return std::string("Value ") + input + " not in range " + std::to_string(min) + " to " +
std::to_string(max);

return std::string();
};
}

template <typename T> explicit Range(T max) : Range(static_cast<T>(0), max) {}
};

class Bound : public Validator {
public:
template <typename T> Bound(T min, T max) {
std::stringstream out;
out << detail::type_name<T>() << " bounded to [" << min << " - " << max << "]";
description(out.str());

func_ = [min, max](std::string &input) {
T val;
bool converted = detail::lexical_cast(input, val);
if(!converted) {
return std::string("Value ") + input + " could not be converted";
}
if(val < min)
input = detail::to_string(min);
else if(val > max)
input = detail::to_string(max);

return std::string{};
};
}

template <typename T> explicit Bound(T max) : Bound(static_cast<T>(0), max) {}
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
IsMember(std::initializer_list<T> values, Args &&... args)
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

std::string out(" not in ");
out += detail::generate_set(detail::smart_deref(set));
return out;
};
}

template <typename T, typename... Args>
IsMember(T &&set, filter_fn_t filter_fn_1, filter_fn_t filter_fn_2, Args &&... other)
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
Transformer(std::initializer_list<std::pair<std::string, std::string>> values, Args &&... args)
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
Transformer(T &&mapping, filter_fn_t filter_fn_1, filter_fn_t filter_fn_2, Args &&... other)
: Transformer(
std::forward<T>(mapping),
[filter_fn_1, filter_fn_2](std::string a) { return filter_fn_2(filter_fn_1(a)); },
other...) {}
};

class CheckedTransformer : public Validator {
public:
using filter_fn_t = std::function<std::string(std::string)>;

template <typename... Args>
CheckedTransformer(std::initializer_list<std::pair<std::string, std::string>> values, Args &&... args)
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
CheckedTransformer(T &&mapping, filter_fn_t filter_fn_1, filter_fn_t filter_fn_2, Args &&... other)
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

bool converted = detail::lexical_cast(input, num);
if(!converted) {
throw ValidationError(std::string("Value ") + input + " could not be converted to " +
detail::type_name<Number>());
}

if(unit.empty()) {
return {};
}

auto it = mapping.find(unit);
if(it == mapping.end()) {
throw ValidationError(unit +
" unit not recognized. "
"Allowed values: " +
detail::generate_map(mapping, true));
}

bool ok = detail::checked_multiply(num, it->second);
if(!ok) {
throw ValidationError(detail::to_string(num) + " multiplied by " + unit +
" factor would cause number overflow. Use smaller value.");
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
esp = commandline.find_first_of(' ', 1);
break;
}
}
vals.first = commandline.substr(0, esp);
rtrim(vals.first);
vals.second = (esp != std::string::npos) ? commandline.substr(esp + 1) : std::string{};
ltrim(vals.second);
return vals;
}

}  

}  


namespace CLI {

class Option;
class App;


enum class AppFormatMode {
Normal,  
All,     
Sub,     
};

class FormatterBase {
protected:

std::size_t column_width_{30};

std::map<std::string, std::string> labels_{};


public:
FormatterBase() = default;
FormatterBase(const FormatterBase &) = default;
FormatterBase(FormatterBase &&) = default;

virtual ~FormatterBase() noexcept {}  

virtual std::string make_help(const App *, std::string, AppFormatMode) const = 0;


void label(std::string key, std::string val) { labels_[key] = val; }

void column_width(std::size_t val) { column_width_ = val; }


std::string get_label(std::string key) const {
if(labels_.find(key) == labels_.end())
return key;
else
return labels_.at(key);
}

std::size_t get_column_width() const { return column_width_; }

};

class FormatterLambda final : public FormatterBase {
using funct_t = std::function<std::string(const App *, std::string, AppFormatMode)>;

funct_t lambda_;

public:
explicit FormatterLambda(funct_t funct) : lambda_(std::move(funct)) {}

~FormatterLambda() noexcept override {}  

std::string make_help(const App *app, std::string name, AppFormatMode mode) const override {
return lambda_(app, name, mode);
}
};

class Formatter : public FormatterBase {
public:
Formatter() = default;
Formatter(const Formatter &) = default;
Formatter(Formatter &&) = default;


virtual std::string make_group(std::string group, bool is_positional, std::vector<const Option *> opts) const;

virtual std::string make_positionals(const App *app) const;

std::string make_groups(const App *app, AppFormatMode mode) const;

virtual std::string make_subcommands(const App *app, AppFormatMode mode) const;

virtual std::string make_subcommand(const App *sub) const;

virtual std::string make_expanded(const App *sub) const;

virtual std::string make_footer(const App *app) const;

virtual std::string make_description(const App *app) const;

virtual std::string make_usage(const App *app, std::string name) const;

std::string make_help(const App * , std::string, AppFormatMode) const override;


virtual std::string make_option(const Option *opt, bool is_positional) const {
std::stringstream out;
detail::format_help(
out, make_option_name(opt, is_positional) + make_option_opts(opt), make_option_desc(opt), column_width_);
return out.str();
}

virtual std::string make_option_name(const Option *, bool) const;

virtual std::string make_option_opts(const Option *) const;

virtual std::string make_option_desc(const Option *) const;

virtual std::string make_option_usage(const Option *opt) const;

};

}  


namespace CLI {

using results_t = std::vector<std::string>;
using callback_t = std::function<bool(const results_t &)>;

class Option;
class App;

using Option_p = std::unique_ptr<Option>;
enum class MultiOptionPolicy : char {
Throw,      
TakeLast,   
TakeFirst,  
Join,       
TakeAll     
};

template <typename CRTP> class OptionBase {
friend App;

protected:
std::string group_ = std::string("Options");

bool required_{false};

bool ignore_case_{false};

bool ignore_underscore_{false};

bool configurable_{true};

bool disable_flag_override_{false};

char delimiter_{'\0'};

bool always_capture_default_{false};

MultiOptionPolicy multi_option_policy_{MultiOptionPolicy::Throw};

template <typename T> void copy_to(T *other) const {
other->group(group_);
other->required(required_);
other->ignore_case(ignore_case_);
other->ignore_underscore(ignore_underscore_);
other->configurable(configurable_);
other->disable_flag_override(disable_flag_override_);
other->delimiter(delimiter_);
other->always_capture_default(always_capture_default_);
other->multi_option_policy(multi_option_policy_);
}

public:

CRTP *group(const std::string &name) {
group_ = name;
return static_cast<CRTP *>(this);
}

CRTP *required(bool value = true) {
required_ = value;
return static_cast<CRTP *>(this);
}

CRTP *mandatory(bool value = true) { return required(value); }

CRTP *always_capture_default(bool value = true) {
always_capture_default_ = value;
return static_cast<CRTP *>(this);
}


const std::string &get_group() const { return group_; }

bool get_required() const { return required_; }

bool get_ignore_case() const { return ignore_case_; }

bool get_ignore_underscore() const { return ignore_underscore_; }

bool get_configurable() const { return configurable_; }

bool get_disable_flag_override() const { return disable_flag_override_; }

char get_delimiter() const { return delimiter_; }

bool get_always_capture_default() const { return always_capture_default_; }

MultiOptionPolicy get_multi_option_policy() const { return multi_option_policy_; }


CRTP *take_last() {
auto self = static_cast<CRTP *>(this);
self->multi_option_policy(MultiOptionPolicy::TakeLast);
return self;
}

CRTP *take_first() {
auto self = static_cast<CRTP *>(this);
self->multi_option_policy(MultiOptionPolicy::TakeFirst);
return self;
}

CRTP *take_all() {
auto self = static_cast<CRTP *>(this);
self->multi_option_policy(MultiOptionPolicy::TakeAll);
return self;
}

CRTP *join() {
auto self = static_cast<CRTP *>(this);
self->multi_option_policy(MultiOptionPolicy::Join);
return self;
}

CRTP *join(char delim) {
auto self = static_cast<CRTP *>(this);
self->delimiter_ = delim;
self->multi_option_policy(MultiOptionPolicy::Join);
return self;
}

CRTP *configurable(bool value = true) {
configurable_ = value;
return static_cast<CRTP *>(this);
}

CRTP *delimiter(char value = '\0') {
delimiter_ = value;
return static_cast<CRTP *>(this);
}
};

class OptionDefaults : public OptionBase<OptionDefaults> {
public:
OptionDefaults() = default;


OptionDefaults *multi_option_policy(MultiOptionPolicy value = MultiOptionPolicy::Throw) {
multi_option_policy_ = value;
return this;
}

OptionDefaults *ignore_case(bool value = true) {
ignore_case_ = value;
return this;
}

OptionDefaults *ignore_underscore(bool value = true) {
ignore_underscore_ = value;
return this;
}

OptionDefaults *disable_flag_override(bool value = true) {
disable_flag_override_ = value;
return this;
}

OptionDefaults *delimiter(char value = '\0') {
delimiter_ = value;
return this;
}
};

class Option : public OptionBase<Option> {
friend App;

protected:

std::vector<std::string> snames_{};

std::vector<std::string> lnames_{};

std::vector<std::pair<std::string, std::string>> default_flag_values_{};

std::vector<std::string> fnames_{};

std::string pname_{};

std::string envname_{};


std::string description_{};

std::string default_str_{};

std::function<std::string()> type_name_{[]() { return std::string(); }};

std::function<std::string()> default_function_{};


int type_size_max_{1};
int type_size_min_{1};

int expected_min_{1};
int expected_max_{1};

std::vector<Validator> validators_{};

std::set<Option *> needs_{};

std::set<Option *> excludes_{};


App *parent_{nullptr};

callback_t callback_{};


results_t results_{};
results_t proc_results_{};
enum class option_state {
parsing = 0,       
validated = 2,     
reduced = 4,       
callback_run = 6,  
};
option_state current_option_state_{option_state::parsing};
bool allow_extra_args_{false};
bool flag_like_{false};
bool run_callback_for_default_{false};

Option(std::string option_name, std::string option_description, callback_t callback, App *parent)
: description_(std::move(option_description)), parent_(parent), callback_(std::move(callback)) {
std::tie(snames_, lnames_, pname_) = detail::get_names(detail::split_names(option_name));
}

public:

Option(const Option &) = delete;
Option &operator=(const Option &) = delete;

std::size_t count() const { return results_.size(); }

bool empty() const { return results_.empty(); }

explicit operator bool() const { return !empty(); }

void clear() {
results_.clear();
current_option_state_ = option_state::parsing;
}


Option *expected(int value) {
if(value < 0) {
expected_min_ = -value;
if(expected_max_ < expected_min_) {
expected_max_ = expected_min_;
}
allow_extra_args_ = true;
flag_like_ = false;
} else if(value == detail::expected_max_vector_size) {
expected_min_ = 1;
expected_max_ = detail::expected_max_vector_size;
allow_extra_args_ = true;
flag_like_ = false;
} else {
expected_min_ = value;
expected_max_ = value;
flag_like_ = (expected_min_ == 0);
}
return this;
}

Option *expected(int value_min, int value_max) {
if(value_min < 0) {
value_min = -value_min;
}

if(value_max < 0) {
value_max = detail::expected_max_vector_size;
}
if(value_max < value_min) {
expected_min_ = value_max;
expected_max_ = value_min;
} else {
expected_max_ = value_max;
expected_min_ = value_min;
}

return this;
}
Option *allow_extra_args(bool value = true) {
allow_extra_args_ = value;
return this;
}
bool get_allow_extra_args() const { return allow_extra_args_; }

Option *run_callback_for_default(bool value = true) {
run_callback_for_default_ = value;
return this;
}
bool get_run_callback_for_default() const { return run_callback_for_default_; }

Option *check(Validator validator, const std::string &validator_name = "") {
validator.non_modifying();
validators_.push_back(std::move(validator));
if(!validator_name.empty())
validators_.back().name(validator_name);
return this;
}

Option *check(std::function<std::string(const std::string &)> Validator,
std::string Validator_description = "",
std::string Validator_name = "") {
validators_.emplace_back(Validator, std::move(Validator_description), std::move(Validator_name));
validators_.back().non_modifying();
return this;
}

Option *transform(Validator Validator, const std::string &Validator_name = "") {
validators_.insert(validators_.begin(), std::move(Validator));
if(!Validator_name.empty())
validators_.front().name(Validator_name);
return this;
}

Option *transform(const std::function<std::string(std::string)> &func,
std::string transform_description = "",
std::string transform_name = "") {
validators_.insert(validators_.begin(),
Validator(
[func](std::string &val) {
val = func(val);
return std::string{};
},
std::move(transform_description),
std::move(transform_name)));

return this;
}

Option *each(const std::function<void(std::string)> &func) {
validators_.emplace_back(
[func](std::string &inout) {
func(inout);
return std::string{};
},
std::string{});
return this;
}
Validator *get_validator(const std::string &Validator_name = "") {
for(auto &Validator : validators_) {
if(Validator_name == Validator.get_name()) {
return &Validator;
}
}
if((Validator_name.empty()) && (!validators_.empty())) {
return &(validators_.front());
}
throw OptionNotFound(std::string{"Validator "} + Validator_name + " Not Found");
}

Validator *get_validator(int index) {
if(index >= 0 && index < static_cast<int>(validators_.size())) {
return &(validators_[static_cast<decltype(validators_)::size_type>(index)]);
}
throw OptionNotFound("Validator index is not valid");
}

Option *needs(Option *opt) {
if(opt != this) {
needs_.insert(opt);
}
return this;
}

template <typename T = App> Option *needs(std::string opt_name) {
auto opt = static_cast<T *>(parent_)->get_option_no_throw(opt_name);
if(opt == nullptr) {
throw IncorrectConstruction::MissingOption(opt_name);
}
return needs(opt);
}

template <typename A, typename B, typename... ARG> Option *needs(A opt, B opt1, ARG... args) {
needs(opt);
return needs(opt1, args...);
}

bool remove_needs(Option *opt) {
auto iterator = std::find(std::begin(needs_), std::end(needs_), opt);

if(iterator == std::end(needs_)) {
return false;
}
needs_.erase(iterator);
return true;
}

Option *excludes(Option *opt) {
if(opt == this) {
throw(IncorrectConstruction("and option cannot exclude itself"));
}
excludes_.insert(opt);

opt->excludes_.insert(this);


return this;
}

template <typename T = App> Option *excludes(std::string opt_name) {
auto opt = static_cast<T *>(parent_)->get_option_no_throw(opt_name);
if(opt == nullptr) {
throw IncorrectConstruction::MissingOption(opt_name);
}
return excludes(opt);
}

template <typename A, typename B, typename... ARG> Option *excludes(A opt, B opt1, ARG... args) {
excludes(opt);
return excludes(opt1, args...);
}

bool remove_excludes(Option *opt) {
auto iterator = std::find(std::begin(excludes_), std::end(excludes_), opt);

if(iterator == std::end(excludes_)) {
return false;
}
excludes_.erase(iterator);
return true;
}

Option *envname(std::string name) {
envname_ = std::move(name);
return this;
}

template <typename T = App> Option *ignore_case(bool value = true) {
if(!ignore_case_ && value) {
ignore_case_ = value;
auto *parent = static_cast<T *>(parent_);
for(const Option_p &opt : parent->options_) {
if(opt.get() == this) {
continue;
}
auto &omatch = opt->matching_name(*this);
if(!omatch.empty()) {
ignore_case_ = false;
throw OptionAlreadyAdded("adding ignore case caused a name conflict with " + omatch);
}
}
} else {
ignore_case_ = value;
}
return this;
}

template <typename T = App> Option *ignore_underscore(bool value = true) {

if(!ignore_underscore_ && value) {
ignore_underscore_ = value;
auto *parent = static_cast<T *>(parent_);
for(const Option_p &opt : parent->options_) {
if(opt.get() == this) {
continue;
}
auto &omatch = opt->matching_name(*this);
if(!omatch.empty()) {
ignore_underscore_ = false;
throw OptionAlreadyAdded("adding ignore underscore caused a name conflict with " + omatch);
}
}
} else {
ignore_underscore_ = value;
}
return this;
}

Option *multi_option_policy(MultiOptionPolicy value = MultiOptionPolicy::Throw) {
if(value != multi_option_policy_) {
if(multi_option_policy_ == MultiOptionPolicy::Throw && expected_max_ == detail::expected_max_vector_size &&
expected_min_ > 1) {  
expected_max_ = expected_min_;
}
multi_option_policy_ = value;
current_option_state_ = option_state::parsing;
}
return this;
}

Option *disable_flag_override(bool value = true) {
disable_flag_override_ = value;
return this;
}

int get_type_size() const { return type_size_min_; }

int get_type_size_min() const { return type_size_min_; }
int get_type_size_max() const { return type_size_max_; }

std::string get_envname() const { return envname_; }

std::set<Option *> get_needs() const { return needs_; }

std::set<Option *> get_excludes() const { return excludes_; }

std::string get_default_str() const { return default_str_; }

callback_t get_callback() const { return callback_; }

const std::vector<std::string> &get_lnames() const { return lnames_; }

const std::vector<std::string> &get_snames() const { return snames_; }

const std::vector<std::string> &get_fnames() const { return fnames_; }

int get_expected() const { return expected_min_; }

int get_expected_min() const { return expected_min_; }
int get_expected_max() const { return expected_max_; }

int get_items_expected_min() const { return type_size_min_ * expected_min_; }

int get_items_expected_max() const {
int t = type_size_max_;
return detail::checked_multiply(t, expected_max_) ? t : detail::expected_max_vector_size;
}
int get_items_expected() const { return get_items_expected_min(); }

bool get_positional() const { return pname_.length() > 0; }

bool nonpositional() const { return (snames_.size() + lnames_.size()) > 0; }

bool has_description() const { return description_.length() > 0; }

const std::string &get_description() const { return description_; }

Option *description(std::string option_description) {
description_ = std::move(option_description);
return this;
}


std::string get_name(bool positional = false,  
bool all_options = false  
) const {
if(get_group().empty())
return {};  

if(all_options) {

std::vector<std::string> name_list;

if((positional && (!pname_.empty())) || (snames_.empty() && lnames_.empty())) {
name_list.push_back(pname_);
}
if((get_items_expected() == 0) && (!fnames_.empty())) {
for(const std::string &sname : snames_) {
name_list.push_back("-" + sname);
if(check_fname(sname)) {
name_list.back() += "{" + get_flag_value(sname, "") + "}";
}
}

for(const std::string &lname : lnames_) {
name_list.push_back("--" + lname);
if(check_fname(lname)) {
name_list.back() += "{" + get_flag_value(lname, "") + "}";
}
}
} else {
for(const std::string &sname : snames_)
name_list.push_back("-" + sname);

for(const std::string &lname : lnames_)
name_list.push_back("--" + lname);
}

return detail::join(name_list);
}

if(positional)
return pname_;

if(!lnames_.empty())
return std::string(2, '-') + lnames_[0];

if(!snames_.empty())
return std::string(1, '-') + snames_[0];

return pname_;
}


void run_callback() {

if(current_option_state_ == option_state::parsing) {
_validate_results(results_);
current_option_state_ = option_state::validated;
}

if(current_option_state_ < option_state::reduced) {
_reduce_results(proc_results_, results_);
current_option_state_ = option_state::reduced;
}
if(current_option_state_ >= option_state::reduced) {
current_option_state_ = option_state::callback_run;
if(!(callback_)) {
return;
}
const results_t &send_results = proc_results_.empty() ? results_ : proc_results_;
bool local_result = callback_(send_results);

if(!local_result)
throw ConversionError(get_name(), results_);
}
}

const std::string &matching_name(const Option &other) const {
static const std::string estring;
for(const std::string &sname : snames_)
if(other.check_sname(sname))
return sname;
for(const std::string &lname : lnames_)
if(other.check_lname(lname))
return lname;

if(ignore_case_ ||
ignore_underscore_) {  
for(const std::string &sname : other.snames_)
if(check_sname(sname))
return sname;
for(const std::string &lname : other.lnames_)
if(check_lname(lname))
return lname;
}
return estring;
}
bool operator==(const Option &other) const { return !matching_name(other).empty(); }

bool check_name(std::string name) const {

if(name.length() > 2 && name[0] == '-' && name[1] == '-')
return check_lname(name.substr(2));
if(name.length() > 1 && name.front() == '-')
return check_sname(name.substr(1));

std::string local_pname = pname_;
if(ignore_underscore_) {
local_pname = detail::remove_underscore(local_pname);
name = detail::remove_underscore(name);
}
if(ignore_case_) {
local_pname = detail::to_lower(local_pname);
name = detail::to_lower(name);
}
return name == local_pname;
}

bool check_sname(std::string name) const {
return (detail::find_member(std::move(name), snames_, ignore_case_) >= 0);
}

bool check_lname(std::string name) const {
return (detail::find_member(std::move(name), lnames_, ignore_case_, ignore_underscore_) >= 0);
}

bool check_fname(std::string name) const {
if(fnames_.empty()) {
return false;
}
return (detail::find_member(std::move(name), fnames_, ignore_case_, ignore_underscore_) >= 0);
}

std::string get_flag_value(const std::string &name, std::string input_value) const {
static const std::string trueString{"true"};
static const std::string falseString{"false"};
static const std::string emptyString{"{}"};
if(disable_flag_override_) {
if(!((input_value.empty()) || (input_value == emptyString))) {
auto default_ind = detail::find_member(name, fnames_, ignore_case_, ignore_underscore_);
if(default_ind >= 0) {
if(default_flag_values_[static_cast<std::size_t>(default_ind)].second != input_value) {
throw(ArgumentMismatch::FlagOverride(name));
}
} else {
if(input_value != trueString) {
throw(ArgumentMismatch::FlagOverride(name));
}
}
}
}
auto ind = detail::find_member(name, fnames_, ignore_case_, ignore_underscore_);
if((input_value.empty()) || (input_value == emptyString)) {
if(flag_like_) {
return (ind < 0) ? trueString : default_flag_values_[static_cast<std::size_t>(ind)].second;
} else {
return (ind < 0) ? default_str_ : default_flag_values_[static_cast<std::size_t>(ind)].second;
}
}
if(ind < 0) {
return input_value;
}
if(default_flag_values_[static_cast<std::size_t>(ind)].second == falseString) {
try {
auto val = detail::to_flag_value(input_value);
return (val == 1) ? falseString : (val == (-1) ? trueString : std::to_string(-val));
} catch(const std::invalid_argument &) {
return input_value;
}
} else {
return input_value;
}
}

Option *add_result(std::string s) {
_add_result(std::move(s), results_);
current_option_state_ = option_state::parsing;
return this;
}

Option *add_result(std::string s, int &results_added) {
results_added = _add_result(std::move(s), results_);
current_option_state_ = option_state::parsing;
return this;
}

Option *add_result(std::vector<std::string> s) {
for(auto &str : s) {
_add_result(std::move(str), results_);
}
current_option_state_ = option_state::parsing;
return this;
}

results_t results() const { return results_; }

results_t reduced_results() const {
results_t res = proc_results_.empty() ? results_ : proc_results_;
if(current_option_state_ < option_state::reduced) {
if(current_option_state_ == option_state::parsing) {
res = results_;
_validate_results(res);
}
if(!res.empty()) {
results_t extra;
_reduce_results(extra, res);
if(!extra.empty()) {
res = std::move(extra);
}
}
}
return res;
}

template <typename T, enable_if_t<!std::is_const<T>::value, detail::enabler> = detail::dummy>
void results(T &output) const {
bool retval;
if(current_option_state_ >= option_state::reduced || (results_.size() == 1 && validators_.empty())) {
const results_t &res = (proc_results_.empty()) ? results_ : proc_results_;
retval = detail::lexical_conversion<T, T>(res, output);
} else {
results_t res;
if(results_.empty()) {
if(!default_str_.empty()) {
_add_result(std::string(default_str_), res);
_validate_results(res);
results_t extra;
_reduce_results(extra, res);
if(!extra.empty()) {
res = std::move(extra);
}
} else {
res.emplace_back();
}
} else {
res = reduced_results();
}
retval = detail::lexical_conversion<T, T>(res, output);
}
if(!retval) {
throw ConversionError(get_name(), results_);
}
}

template <typename T> T as() const {
T output;
results(output);
return output;
}

bool get_callback_run() const { return (current_option_state_ == option_state::callback_run); }


Option *type_name_fn(std::function<std::string()> typefun) {
type_name_ = std::move(typefun);
return this;
}

Option *type_name(std::string typeval) {
type_name_fn([typeval]() { return typeval; });
return this;
}

Option *type_size(int option_type_size) {
if(option_type_size < 0) {
type_size_max_ = -option_type_size;
type_size_min_ = -option_type_size;
expected_max_ = detail::expected_max_vector_size;
} else {
type_size_max_ = option_type_size;
if(type_size_max_ < detail::expected_max_vector_size) {
type_size_min_ = option_type_size;
}
if(type_size_max_ == 0)
required_ = false;
}
return this;
}
Option *type_size(int option_type_size_min, int option_type_size_max) {
if(option_type_size_min < 0 || option_type_size_max < 0) {
expected_max_ = detail::expected_max_vector_size;
option_type_size_min = (std::abs)(option_type_size_min);
option_type_size_max = (std::abs)(option_type_size_max);
}

if(option_type_size_min > option_type_size_max) {
type_size_max_ = option_type_size_min;
type_size_min_ = option_type_size_max;
} else {
type_size_min_ = option_type_size_min;
type_size_max_ = option_type_size_max;
}
if(type_size_max_ == 0) {
required_ = false;
}
return this;
}

Option *default_function(const std::function<std::string()> &func) {
default_function_ = func;
return this;
}

Option *capture_default_str() {
if(default_function_) {
default_str_ = default_function_();
}
return this;
}

Option *default_str(std::string val) {
default_str_ = std::move(val);
return this;
}

template <typename X> Option *default_val(const X &val) {
std::string val_str = detail::to_string(val);
auto old_option_state = current_option_state_;
results_t old_results{std::move(results_)};
results_.clear();
try {
add_result(val_str);
if(run_callback_for_default_) {
run_callback();  
current_option_state_ = option_state::parsing;
} else {
_validate_results(results_);
current_option_state_ = old_option_state;
}
} catch(const CLI::Error &) {
results_ = std::move(old_results);
current_option_state_ = old_option_state;
throw;
}
results_ = std::move(old_results);
default_str_ = std::move(val_str);
return this;
}

std::string get_type_name() const {
std::string full_type_name = type_name_();
if(!validators_.empty()) {
for(auto &Validator : validators_) {
std::string vtype = Validator.get_description();
if(!vtype.empty()) {
full_type_name += ":" + vtype;
}
}
}
return full_type_name;
}

private:
void _validate_results(results_t &res) const {
if(!validators_.empty()) {
if(type_size_max_ > 1) {  
int index = 0;
if(get_items_expected_max() < static_cast<int>(res.size()) &&
multi_option_policy_ == CLI::MultiOptionPolicy::TakeLast) {
index = get_items_expected_max() - static_cast<int>(res.size());
}

for(std::string &result : res) {
if(result.empty() && type_size_max_ != type_size_min_ && index >= 0) {
index = 0;  
continue;
}
auto err_msg = _validate(result, (index >= 0) ? (index % type_size_max_) : index);
if(!err_msg.empty())
throw ValidationError(get_name(), err_msg);
++index;
}
} else {
int index = 0;
if(expected_max_ < static_cast<int>(res.size()) &&
multi_option_policy_ == CLI::MultiOptionPolicy::TakeLast) {
index = expected_max_ - static_cast<int>(res.size());
}
for(std::string &result : res) {
auto err_msg = _validate(result, index);
++index;
if(!err_msg.empty())
throw ValidationError(get_name(), err_msg);
}
}
}
}


void _reduce_results(results_t &res, const results_t &original) const {


res.clear();
switch(multi_option_policy_) {
case MultiOptionPolicy::TakeAll:
break;
case MultiOptionPolicy::TakeLast: {
std::size_t trim_size = std::min<std::size_t>(
static_cast<std::size_t>(std::max<int>(get_items_expected_max(), 1)), original.size());
if(original.size() != trim_size) {
res.assign(original.end() - static_cast<results_t::difference_type>(trim_size), original.end());
}
} break;
case MultiOptionPolicy::TakeFirst: {
std::size_t trim_size = std::min<std::size_t>(
static_cast<std::size_t>(std::max<int>(get_items_expected_max(), 1)), original.size());
if(original.size() != trim_size) {
res.assign(original.begin(), original.begin() + static_cast<results_t::difference_type>(trim_size));
}
} break;
case MultiOptionPolicy::Join:
if(results_.size() > 1) {
res.push_back(detail::join(original, std::string(1, (delimiter_ == '\0') ? '\n' : delimiter_)));
}
break;
case MultiOptionPolicy::Throw:
default: {
auto num_min = static_cast<std::size_t>(get_items_expected_min());
auto num_max = static_cast<std::size_t>(get_items_expected_max());
if(num_min == 0) {
num_min = 1;
}
if(num_max == 0) {
num_max = 1;
}
if(original.size() < num_min) {
throw ArgumentMismatch::AtLeast(get_name(), static_cast<int>(num_min), original.size());
}
if(original.size() > num_max) {
throw ArgumentMismatch::AtMost(get_name(), static_cast<int>(num_max), original.size());
}
break;
}
}
}

std::string _validate(std::string &result, int index) const {
std::string err_msg;
if(result.empty() && expected_min_ == 0) {
return err_msg;
}
for(const auto &vali : validators_) {
auto v = vali.get_application_index();
if(v == -1 || v == index) {
try {
err_msg = vali(result);
} catch(const ValidationError &err) {
err_msg = err.what();
}
if(!err_msg.empty())
break;
}
}

return err_msg;
}

int _add_result(std::string &&result, std::vector<std::string> &res) const {
int result_count = 0;
if(allow_extra_args_ && !result.empty() && result.front() == '[' &&
result.back() == ']') {  
result.pop_back();

for(auto &var : CLI::detail::split(result.substr(1), ',')) {
if(!var.empty()) {
result_count += _add_result(std::move(var), res);
}
}
return result_count;
}
if(delimiter_ == '\0') {
res.push_back(std::move(result));
++result_count;
} else {
if((result.find_first_of(delimiter_) != std::string::npos)) {
for(const auto &var : CLI::detail::split(result, delimiter_)) {
if(!var.empty()) {
res.push_back(var);
++result_count;
}
}
} else {
res.push_back(std::move(result));
++result_count;
}
}
return result_count;
}
};  

}  


namespace CLI {

#ifndef CLI11_PARSE
#define CLI11_PARSE(app, argc, argv)                                                                                   \
try {                                                                                                              \
(app).parse((argc), (argv));                                                                                   \
} catch(const CLI::ParseError &e) {                                                                                \
return (app).exit(e);                                                                                          \
}
#endif

namespace detail {
enum class Classifier { NONE, POSITIONAL_MARK, SHORT, LONG, WINDOWS, SUBCOMMAND, SUBCOMMAND_TERMINATOR };
struct AppFriend;
}  

namespace FailureMessage {
std::string simple(const App *app, const Error &e);
std::string help(const App *app, const Error &e);
}  


enum class config_extras_mode : char { error = 0, ignore, capture };

class App;

using App_p = std::shared_ptr<App>;

class Option_group;

class App {
friend Option;
friend detail::AppFriend;

protected:


std::string name_{};

std::string description_{};

bool allow_extras_{false};

config_extras_mode allow_config_extras_{config_extras_mode::ignore};

bool prefix_command_{false};

bool has_automatic_name_{false};

bool required_{false};

bool disabled_{false};

bool pre_parse_called_{false};

bool immediate_callback_{false};

std::function<void(std::size_t)> pre_parse_callback_{};

std::function<void()> parse_complete_callback_{};
std::function<void()> final_callback_{};


OptionDefaults option_defaults_{};

std::vector<Option_p> options_{};


std::string footer_{};

std::function<std::string()> footer_callback_{};

Option *help_ptr_{nullptr};

Option *help_all_ptr_{nullptr};

std::shared_ptr<FormatterBase> formatter_{new Formatter()};

std::function<std::string(const App *, const Error &e)> failure_message_{FailureMessage::simple};


using missing_t = std::vector<std::pair<detail::Classifier, std::string>>;

missing_t missing_{};

std::vector<Option *> parse_order_{};

std::vector<App *> parsed_subcommands_{};

std::set<App *> exclude_subcommands_{};

std::set<Option *> exclude_options_{};

std::set<App *> need_subcommands_{};

std::set<Option *> need_options_{};


std::vector<App_p> subcommands_{};

bool ignore_case_{false};

bool ignore_underscore_{false};

bool fallthrough_{false};

bool allow_windows_style_options_{
#ifdef _WIN32
true
#else
false
#endif
};
bool positionals_at_end_{false};

enum class startup_mode : char { stable, enabled, disabled };
startup_mode default_startup{startup_mode::stable};

bool configurable_{false};

bool validate_positionals_{false};

App *parent_{nullptr};

std::size_t parsed_{0};

std::size_t require_subcommand_min_{0};

std::size_t require_subcommand_max_{0};

std::size_t require_option_min_{0};

std::size_t require_option_max_{0};

std::string group_{"Subcommands"};

std::vector<std::string> aliases_{};


Option *config_ptr_{nullptr};

std::shared_ptr<Config> config_formatter_{new ConfigINI()};


App(std::string app_description, std::string app_name, App *parent)
: name_(std::move(app_name)), description_(std::move(app_description)), parent_(parent) {
if(parent_ != nullptr) {
if(parent_->help_ptr_ != nullptr)
set_help_flag(parent_->help_ptr_->get_name(false, true), parent_->help_ptr_->get_description());
if(parent_->help_all_ptr_ != nullptr)
set_help_all_flag(parent_->help_all_ptr_->get_name(false, true),
parent_->help_all_ptr_->get_description());

option_defaults_ = parent_->option_defaults_;

failure_message_ = parent_->failure_message_;
allow_extras_ = parent_->allow_extras_;
allow_config_extras_ = parent_->allow_config_extras_;
prefix_command_ = parent_->prefix_command_;
immediate_callback_ = parent_->immediate_callback_;
ignore_case_ = parent_->ignore_case_;
ignore_underscore_ = parent_->ignore_underscore_;
fallthrough_ = parent_->fallthrough_;
validate_positionals_ = parent_->validate_positionals_;
configurable_ = parent_->configurable_;
allow_windows_style_options_ = parent_->allow_windows_style_options_;
group_ = parent_->group_;
footer_ = parent_->footer_;
formatter_ = parent_->formatter_;
config_formatter_ = parent_->config_formatter_;
require_subcommand_max_ = parent_->require_subcommand_max_;
}
}

public:

explicit App(std::string app_description = "", std::string app_name = "")
: App(app_description, app_name, nullptr) {
set_help_flag("-h,--help", "Print this help message and exit");
}

App(const App &) = delete;
App &operator=(const App &) = delete;

virtual ~App() = default;

App *callback(std::function<void()> app_callback) {
if(immediate_callback_) {
parse_complete_callback_ = std::move(app_callback);
} else {
final_callback_ = std::move(app_callback);
}
return this;
}

App *final_callback(std::function<void()> app_callback) {
final_callback_ = std::move(app_callback);
return this;
}

App *parse_complete_callback(std::function<void()> pc_callback) {
parse_complete_callback_ = std::move(pc_callback);
return this;
}

App *preparse_callback(std::function<void(std::size_t)> pp_callback) {
pre_parse_callback_ = std::move(pp_callback);
return this;
}

App *name(std::string app_name = "") {

if(parent_ != nullptr) {
auto oname = name_;
name_ = app_name;
auto &res = _compare_subcommand_names(*this, *_get_fallthrough_parent());
if(!res.empty()) {
name_ = oname;
throw(OptionAlreadyAdded(app_name + " conflicts with existing subcommand names"));
}
} else {
name_ = app_name;
}
has_automatic_name_ = false;
return this;
}

App *alias(std::string app_name) {
if(!detail::valid_name_string(app_name)) {
throw(IncorrectConstruction("alias is not a valid name string"));
}

if(parent_ != nullptr) {
aliases_.push_back(app_name);
auto &res = _compare_subcommand_names(*this, *_get_fallthrough_parent());
if(!res.empty()) {
aliases_.pop_back();
throw(OptionAlreadyAdded("alias already matches an existing subcommand: " + app_name));
}
} else {
aliases_.push_back(app_name);
}

return this;
}

App *allow_extras(bool allow = true) {
allow_extras_ = allow;
return this;
}

App *required(bool require = true) {
required_ = require;
return this;
}

App *disabled(bool disable = true) {
disabled_ = disable;
return this;
}

App *disabled_by_default(bool disable = true) {
if(disable) {
default_startup = startup_mode::disabled;
} else {
default_startup = (default_startup == startup_mode::enabled) ? startup_mode::enabled : startup_mode::stable;
}
return this;
}

App *enabled_by_default(bool enable = true) {
if(enable) {
default_startup = startup_mode::enabled;
} else {
default_startup =
(default_startup == startup_mode::disabled) ? startup_mode::disabled : startup_mode::stable;
}
return this;
}

App *immediate_callback(bool immediate = true) {
immediate_callback_ = immediate;
if(immediate_callback_) {
if(final_callback_ && !(parse_complete_callback_)) {
std::swap(final_callback_, parse_complete_callback_);
}
} else if(!(final_callback_) && parse_complete_callback_) {
std::swap(final_callback_, parse_complete_callback_);
}
return this;
}

App *validate_positionals(bool validate = true) {
validate_positionals_ = validate;
return this;
}

App *allow_config_extras(bool allow = true) {
if(allow) {
allow_config_extras_ = config_extras_mode::capture;
allow_extras_ = true;
} else {
allow_config_extras_ = config_extras_mode::error;
}
return this;
}

App *allow_config_extras(config_extras_mode mode) {
allow_config_extras_ = mode;
return this;
}

App *prefix_command(bool allow = true) {
prefix_command_ = allow;
return this;
}

App *ignore_case(bool value = true) {
if(value && !ignore_case_) {
ignore_case_ = true;
auto *p = (parent_ != nullptr) ? _get_fallthrough_parent() : this;
auto &match = _compare_subcommand_names(*this, *p);
if(!match.empty()) {
ignore_case_ = false;  
throw OptionAlreadyAdded("ignore case would cause subcommand name conflicts: " + match);
}
}
ignore_case_ = value;
return this;
}

App *allow_windows_style_options(bool value = true) {
allow_windows_style_options_ = value;
return this;
}

App *positionals_at_end(bool value = true) {
positionals_at_end_ = value;
return this;
}

App *configurable(bool value = true) {
configurable_ = value;
return this;
}

App *ignore_underscore(bool value = true) {
if(value && !ignore_underscore_) {
ignore_underscore_ = true;
auto *p = (parent_ != nullptr) ? _get_fallthrough_parent() : this;
auto &match = _compare_subcommand_names(*this, *p);
if(!match.empty()) {
ignore_underscore_ = false;
throw OptionAlreadyAdded("ignore underscore would cause subcommand name conflicts: " + match);
}
}
ignore_underscore_ = value;
return this;
}

App *formatter(std::shared_ptr<FormatterBase> fmt) {
formatter_ = fmt;
return this;
}

App *formatter_fn(std::function<std::string(const App *, std::string, AppFormatMode)> fmt) {
formatter_ = std::make_shared<FormatterLambda>(fmt);
return this;
}

App *config_formatter(std::shared_ptr<Config> fmt) {
config_formatter_ = fmt;
return this;
}

bool parsed() const { return parsed_ > 0; }

OptionDefaults *option_defaults() { return &option_defaults_; }


Option *add_option(std::string option_name,
callback_t option_callback,
std::string option_description = "",
bool defaulted = false,
std::function<std::string()> func = {}) {
Option myopt{option_name, option_description, option_callback, this};

if(std::find_if(std::begin(options_), std::end(options_), [&myopt](const Option_p &v) {
return *v == myopt;
}) == std::end(options_)) {
options_.emplace_back();
Option_p &option = options_.back();
option.reset(new Option(option_name, option_description, option_callback, this));

option->default_function(func);

if(defaulted)
option->capture_default_str();

option_defaults_.copy_to(option.get());

if(!defaulted && option->get_always_capture_default())
option->capture_default_str();

return option.get();
}
for(auto &opt : options_) {
auto &matchname = opt->matching_name(myopt);
if(!matchname.empty()) {
throw(OptionAlreadyAdded("added option matched existing option name: " + matchname));
}
}
throw(OptionAlreadyAdded("added option matched existing option name"));  
}

template <typename AssignTo,
typename ConvertTo = AssignTo,
enable_if_t<!std::is_const<ConvertTo>::value, detail::enabler> = detail::dummy>
Option *add_option(std::string option_name,
AssignTo &variable,  
std::string option_description = "",
bool defaulted = false) {

auto fun = [&variable](const CLI::results_t &res) {  
return detail::lexical_conversion<AssignTo, ConvertTo>(res, variable);
};

Option *opt = add_option(option_name, fun, option_description, defaulted, [&variable]() {
return CLI::detail::checked_to_string<AssignTo, ConvertTo>(variable);
});
opt->type_name(detail::type_name<ConvertTo>());
auto Tcount = detail::type_count<AssignTo>::value;
auto XCcount = detail::type_count<ConvertTo>::value;
opt->type_size((std::max)(Tcount, XCcount));
opt->expected(detail::expected_count<ConvertTo>::value);
opt->run_callback_for_default();
return opt;
}

template <typename T>
Option *add_option_function(std::string option_name,
const std::function<void(const T &)> &func,  
std::string option_description = "") {

auto fun = [func](const CLI::results_t &res) {
T variable;
bool result = detail::lexical_conversion<T, T>(res, variable);
if(result) {
func(variable);
}
return result;
};

Option *opt = add_option(option_name, std::move(fun), option_description, false);
opt->type_name(detail::type_name<T>());
opt->type_size(detail::type_count<T>::value);
opt->expected(detail::expected_count<T>::value);
return opt;
}

Option *add_option(std::string option_name) {
return add_option(option_name, CLI::callback_t(), std::string{}, false);
}

template <typename T,
enable_if_t<std::is_const<T>::value && std::is_constructible<std::string, T>::value, detail::enabler> =
detail::dummy>
Option *add_option(std::string option_name, T &option_description) {
return add_option(option_name, CLI::callback_t(), option_description, false);
}

Option *set_help_flag(std::string flag_name = "", const std::string &help_description = "") {
if(help_ptr_ != nullptr) {
remove_option(help_ptr_);
help_ptr_ = nullptr;
}

if(!flag_name.empty()) {
help_ptr_ = add_flag(flag_name, help_description);
help_ptr_->configurable(false);
}

return help_ptr_;
}

Option *set_help_all_flag(std::string help_name = "", const std::string &help_description = "") {
if(help_all_ptr_ != nullptr) {
remove_option(help_all_ptr_);
help_all_ptr_ = nullptr;
}

if(!help_name.empty()) {
help_all_ptr_ = add_flag(help_name, help_description);
help_all_ptr_->configurable(false);
}

return help_all_ptr_;
}

private:
Option *_add_flag_internal(std::string flag_name, CLI::callback_t fun, std::string flag_description) {
Option *opt;
if(detail::has_default_flag_values(flag_name)) {
auto flag_defaults = detail::get_default_flag_values(flag_name);
detail::remove_default_flag_values(flag_name);
opt = add_option(std::move(flag_name), std::move(fun), std::move(flag_description), false);
for(const auto &fname : flag_defaults)
opt->fnames_.push_back(fname.first);
opt->default_flag_values_ = std::move(flag_defaults);
} else {
opt = add_option(std::move(flag_name), std::move(fun), std::move(flag_description), false);
}
if(opt->get_positional()) {
auto pos_name = opt->get_name(true);
remove_option(opt);
throw IncorrectConstruction::PositionalFlag(pos_name);
}
opt->multi_option_policy(MultiOptionPolicy::TakeLast);
opt->expected(0);
opt->required(false);
return opt;
}

public:
Option *add_flag(std::string flag_name) { return _add_flag_internal(flag_name, CLI::callback_t(), std::string{}); }

template <typename T,
enable_if_t<std::is_const<T>::value && std::is_constructible<std::string, T>::value, detail::enabler> =
detail::dummy>
Option *add_flag(std::string flag_name, T &flag_description) {
return _add_flag_internal(flag_name, CLI::callback_t(), flag_description);
}

template <typename T,
enable_if_t<std::is_integral<T>::value && !is_bool<T>::value, detail::enabler> = detail::dummy>
Option *add_flag(std::string flag_name,
T &flag_count,  
std::string flag_description = "") {
flag_count = 0;
CLI::callback_t fun = [&flag_count](const CLI::results_t &res) {
try {
detail::sum_flag_vector(res, flag_count);
} catch(const std::invalid_argument &) {
return false;
}
return true;
};
return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description))
->multi_option_policy(MultiOptionPolicy::TakeAll);
}

template <typename T,
enable_if_t<!is_vector<T>::value && !std::is_const<T>::value &&
(!std::is_integral<T>::value || is_bool<T>::value) &&
!std::is_constructible<std::function<void(int)>, T>::value,
detail::enabler> = detail::dummy>
Option *add_flag(std::string flag_name,
T &flag_result,  
std::string flag_description = "") {

CLI::callback_t fun = [&flag_result](const CLI::results_t &res) {
return CLI::detail::lexical_cast(res[0], flag_result);
};
return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description))->run_callback_for_default();
}

template <
typename T,
enable_if_t<!std::is_assignable<std::function<void(std::int64_t)>, T>::value, detail::enabler> = detail::dummy>
Option *add_flag(std::string flag_name,
std::vector<T> &flag_results,  
std::string flag_description = "") {
CLI::callback_t fun = [&flag_results](const CLI::results_t &res) {
bool retval = true;
for(const auto &elem : res) {
flag_results.emplace_back();
retval &= detail::lexical_cast(elem, flag_results.back());
}
return retval;
};
return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description))
->multi_option_policy(MultiOptionPolicy::TakeAll)
->run_callback_for_default();
}

Option *add_flag_callback(std::string flag_name,
std::function<void(void)> function,  
std::string flag_description = "") {

CLI::callback_t fun = [function](const CLI::results_t &res) {
bool trigger{false};
auto result = CLI::detail::lexical_cast(res[0], trigger);
if(result && trigger) {
function();
}
return result;
};
return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description));
}

Option *add_flag_function(std::string flag_name,
std::function<void(std::int64_t)> function,  
std::string flag_description = "") {

CLI::callback_t fun = [function](const CLI::results_t &res) {
std::int64_t flag_count = 0;
detail::sum_flag_vector(res, flag_count);
function(flag_count);
return true;
};
return _add_flag_internal(flag_name, std::move(fun), std::move(flag_description))
->multi_option_policy(MultiOptionPolicy::TakeAll);
}

#ifdef CLI11_CPP14
Option *add_flag(std::string flag_name,
std::function<void(std::int64_t)> function,  
std::string flag_description = "") {
return add_flag_function(std::move(flag_name), std::move(function), std::move(flag_description));
}
#endif

template <typename T>
Option *add_set(std::string option_name,
T &member,            
std::set<T> options,  
std::string option_description = "") {

Option *opt = add_option(option_name, member, std::move(option_description));
opt->check(IsMember{options});
return opt;
}

template <typename T>
Option *add_mutable_set(std::string option_name,
T &member,                   
const std::set<T> &options,  
std::string option_description = "") {

Option *opt = add_option(option_name, member, std::move(option_description));
opt->check(IsMember{&options});
return opt;
}

template <typename T>
Option *add_set(std::string option_name,
T &member,            
std::set<T> options,  
std::string option_description,
bool defaulted) {

Option *opt = add_option(option_name, member, std::move(option_description), defaulted);
opt->check(IsMember{options});
return opt;
}

template <typename T>
Option *add_mutable_set(std::string option_name,
T &member,                   
const std::set<T> &options,  
std::string option_description,
bool defaulted) {

Option *opt = add_option(option_name, member, std::move(option_description), defaulted);
opt->check(IsMember{&options});
return opt;
}

template <typename T, typename XC = double>
Option *add_complex(std::string option_name,
T &variable,
std::string option_description = "",
bool defaulted = false,
std::string label = "COMPLEX") {

CLI::callback_t fun = [&variable](const results_t &res) {
XC x, y;
bool worked;
if(res.size() >= 2 && !res[1].empty()) {
auto str1 = res[1];
if(str1.back() == 'i' || str1.back() == 'j')
str1.pop_back();
worked = detail::lexical_cast(res[0], x) && detail::lexical_cast(str1, y);
} else {
auto str1 = res.front();
auto nloc = str1.find_last_of('-');
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
}
if(worked)
variable = T{x, y};
return worked;
};

auto default_function = [&variable]() { return CLI::detail::checked_to_string<T, T>(variable); };

CLI::Option *opt =
add_option(option_name, std::move(fun), std::move(option_description), defaulted, default_function);

opt->type_name(label)->type_size(1, 2)->delimiter('+')->run_callback_for_default();
return opt;
}

Option *set_config(std::string option_name = "",
std::string default_filename = "",
const std::string &help_message = "Read an ini file",
bool config_required = false) {

if(config_ptr_ != nullptr) {
remove_option(config_ptr_);
config_ptr_ = nullptr;  
}

if(!option_name.empty()) {
config_ptr_ = add_option(option_name, help_message);
if(config_required) {
config_ptr_->required();
}
if(!default_filename.empty()) {
config_ptr_->default_str(std::move(default_filename));
}
config_ptr_->configurable(false);
}

return config_ptr_;
}

bool remove_option(Option *opt) {
for(Option_p &op : options_) {
op->remove_needs(opt);
op->remove_excludes(opt);
}

if(help_ptr_ == opt)
help_ptr_ = nullptr;
if(help_all_ptr_ == opt)
help_all_ptr_ = nullptr;

auto iterator =
std::find_if(std::begin(options_), std::end(options_), [opt](const Option_p &v) { return v.get() == opt; });
if(iterator != std::end(options_)) {
options_.erase(iterator);
return true;
}
return false;
}

template <typename T = Option_group>
T *add_option_group(std::string group_name, std::string group_description = "") {
auto option_group = std::make_shared<T>(std::move(group_description), group_name, this);
auto ptr = option_group.get();
App_p app_ptr = std::dynamic_pointer_cast<App>(option_group);
add_subcommand(std::move(app_ptr));
return ptr;
}


App *add_subcommand(std::string subcommand_name = "", std::string subcommand_description = "") {
if(!subcommand_name.empty() && !detail::valid_name_string(subcommand_name)) {
throw IncorrectConstruction("subcommand name is not valid");
}
CLI::App_p subcom = std::shared_ptr<App>(new App(std::move(subcommand_description), subcommand_name, this));
return add_subcommand(std::move(subcom));
}

App *add_subcommand(CLI::App_p subcom) {
if(!subcom)
throw IncorrectConstruction("passed App is not valid");
auto ckapp = (name_.empty() && parent_ != nullptr) ? _get_fallthrough_parent() : this;
auto &mstrg = _compare_subcommand_names(*subcom, *ckapp);
if(!mstrg.empty()) {
throw(OptionAlreadyAdded("subcommand name or alias matches existing subcommand: " + mstrg));
}
subcom->parent_ = this;
subcommands_.push_back(std::move(subcom));
return subcommands_.back().get();
}

bool remove_subcommand(App *subcom) {
for(App_p &sub : subcommands_) {
sub->remove_excludes(subcom);
sub->remove_needs(subcom);
}

auto iterator = std::find_if(
std::begin(subcommands_), std::end(subcommands_), [subcom](const App_p &v) { return v.get() == subcom; });
if(iterator != std::end(subcommands_)) {
subcommands_.erase(iterator);
return true;
}
return false;
}
App *get_subcommand(const App *subcom) const {
if(subcom == nullptr)
throw OptionNotFound("nullptr passed");
for(const App_p &subcomptr : subcommands_)
if(subcomptr.get() == subcom)
return subcomptr.get();
throw OptionNotFound(subcom->get_name());
}

App *get_subcommand(std::string subcom) const {
auto subc = _find_subcommand(subcom, false, false);
if(subc == nullptr)
throw OptionNotFound(subcom);
return subc;
}
App *get_subcommand(int index = 0) const {
if(index >= 0) {
auto uindex = static_cast<unsigned>(index);
if(uindex < subcommands_.size())
return subcommands_[uindex].get();
}
throw OptionNotFound(std::to_string(index));
}

CLI::App_p get_subcommand_ptr(App *subcom) const {
if(subcom == nullptr)
throw OptionNotFound("nullptr passed");
for(const App_p &subcomptr : subcommands_)
if(subcomptr.get() == subcom)
return subcomptr;
throw OptionNotFound(subcom->get_name());
}

CLI::App_p get_subcommand_ptr(std::string subcom) const {
for(const App_p &subcomptr : subcommands_)
if(subcomptr->check_name(subcom))
return subcomptr;
throw OptionNotFound(subcom);
}

CLI::App_p get_subcommand_ptr(int index = 0) const {
if(index >= 0) {
auto uindex = static_cast<unsigned>(index);
if(uindex < subcommands_.size())
return subcommands_[uindex];
}
throw OptionNotFound(std::to_string(index));
}

App *get_option_group(std::string group_name) const {
for(const App_p &app : subcommands_) {
if(app->name_.empty() && app->group_ == group_name) {
return app.get();
}
}
throw OptionNotFound(group_name);
}

std::size_t count() const { return parsed_; }

std::size_t count_all() const {
std::size_t cnt{0};
for(auto &opt : options_) {
cnt += opt->count();
}
for(auto &sub : subcommands_) {
cnt += sub->count_all();
}
if(!get_name().empty()) {  
cnt += parsed_;
}
return cnt;
}

App *group(std::string group_name) {
group_ = group_name;
return this;
}

App *require_subcommand() {
require_subcommand_min_ = 1;
require_subcommand_max_ = 0;
return this;
}

App *require_subcommand(int value) {
if(value < 0) {
require_subcommand_min_ = 0;
require_subcommand_max_ = static_cast<std::size_t>(-value);
} else {
require_subcommand_min_ = static_cast<std::size_t>(value);
require_subcommand_max_ = static_cast<std::size_t>(value);
}
return this;
}

App *require_subcommand(std::size_t min, std::size_t max) {
require_subcommand_min_ = min;
require_subcommand_max_ = max;
return this;
}

App *require_option() {
require_option_min_ = 1;
require_option_max_ = 0;
return this;
}

App *require_option(int value) {
if(value < 0) {
require_option_min_ = 0;
require_option_max_ = static_cast<std::size_t>(-value);
} else {
require_option_min_ = static_cast<std::size_t>(value);
require_option_max_ = static_cast<std::size_t>(value);
}
return this;
}

App *require_option(std::size_t min, std::size_t max) {
require_option_min_ = min;
require_option_max_ = max;
return this;
}

App *fallthrough(bool value = true) {
fallthrough_ = value;
return this;
}

explicit operator bool() const { return parsed_ > 0; }


virtual void pre_callback() {}

void clear() {

parsed_ = 0;
pre_parse_called_ = false;

missing_.clear();
parsed_subcommands_.clear();
for(const Option_p &opt : options_) {
opt->clear();
}
for(const App_p &subc : subcommands_) {
subc->clear();
}
}

void parse(int argc, const char *const *argv) {
if(name_.empty() || has_automatic_name_) {
has_automatic_name_ = true;
name_ = argv[0];
}

std::vector<std::string> args;
args.reserve(static_cast<std::size_t>(argc) - 1);
for(int i = argc - 1; i > 0; i--)
args.emplace_back(argv[i]);
parse(std::move(args));
}

void parse(std::string commandline, bool program_name_included = false) {

if(program_name_included) {
auto nstr = detail::split_program_name(commandline);
if((name_.empty()) || (has_automatic_name_)) {
has_automatic_name_ = true;
name_ = nstr.first;
}
commandline = std::move(nstr.second);
} else {
detail::trim(commandline);
}
if(!commandline.empty()) {
commandline = detail::find_and_modify(commandline, "=", detail::escape_detect);
if(allow_windows_style_options_)
commandline = detail::find_and_modify(commandline, ":", detail::escape_detect);
}

auto args = detail::split_up(std::move(commandline));
args.erase(std::remove(args.begin(), args.end(), std::string{}), args.end());
std::reverse(args.begin(), args.end());

parse(std::move(args));
}

void parse(std::vector<std::string> &args) {
if(parsed_ > 0)
clear();

parsed_ = 1;
_validate();
_configure();
parent_ = nullptr;
parsed_ = 0;

_parse(args);
run_callback();
}

void parse(std::vector<std::string> &&args) {
if(parsed_ > 0)
clear();

parsed_ = 1;
_validate();
_configure();
parent_ = nullptr;
parsed_ = 0;

_parse(std::move(args));
run_callback();
}

void failure_message(std::function<std::string(const App *, const Error &e)> function) {
failure_message_ = function;
}

int exit(const Error &e, std::ostream &out = std::cout, std::ostream &err = std::cerr) const {

if(e.get_name() == "RuntimeError")
return e.get_exit_code();

if(e.get_name() == "CallForHelp") {
out << help();
return e.get_exit_code();
}

if(e.get_name() == "CallForAllHelp") {
out << help("", AppFormatMode::All);
return e.get_exit_code();
}

if(e.get_exit_code() != static_cast<int>(ExitCodes::Success)) {
if(failure_message_)
err << failure_message_(this, e) << std::flush;
}

return e.get_exit_code();
}


std::size_t count(std::string option_name) const { return get_option(option_name)->count(); }

std::vector<App *> get_subcommands() const { return parsed_subcommands_; }

std::vector<const App *> get_subcommands(const std::function<bool(const App *)> &filter) const {
std::vector<const App *> subcomms(subcommands_.size());
std::transform(std::begin(subcommands_), std::end(subcommands_), std::begin(subcomms), [](const App_p &v) {
return v.get();
});

if(filter) {
subcomms.erase(std::remove_if(std::begin(subcomms),
std::end(subcomms),
[&filter](const App *app) { return !filter(app); }),
std::end(subcomms));
}

return subcomms;
}

std::vector<App *> get_subcommands(const std::function<bool(App *)> &filter) {
std::vector<App *> subcomms(subcommands_.size());
std::transform(std::begin(subcommands_), std::end(subcommands_), std::begin(subcomms), [](const App_p &v) {
return v.get();
});

if(filter) {
subcomms.erase(
std::remove_if(std::begin(subcomms), std::end(subcomms), [&filter](App *app) { return !filter(app); }),
std::end(subcomms));
}

return subcomms;
}

bool got_subcommand(const App *subcom) const {
return get_subcommand(subcom)->parsed_ > 0;
}

bool got_subcommand(std::string subcommand_name) const { return get_subcommand(subcommand_name)->parsed_ > 0; }

App *excludes(Option *opt) {
if(opt == nullptr) {
throw OptionNotFound("nullptr passed");
}
exclude_options_.insert(opt);
return this;
}

App *excludes(App *app) {
if(app == nullptr) {
throw OptionNotFound("nullptr passed");
}
if(app == this) {
throw OptionNotFound("cannot self reference in needs");
}
auto res = exclude_subcommands_.insert(app);
if(res.second) {
app->exclude_subcommands_.insert(this);
}
return this;
}

App *needs(Option *opt) {
if(opt == nullptr) {
throw OptionNotFound("nullptr passed");
}
need_options_.insert(opt);
return this;
}

App *needs(App *app) {
if(app == nullptr) {
throw OptionNotFound("nullptr passed");
}
if(app == this) {
throw OptionNotFound("cannot self reference in needs");
}
need_subcommands_.insert(app);
return this;
}

bool remove_excludes(Option *opt) {
auto iterator = std::find(std::begin(exclude_options_), std::end(exclude_options_), opt);
if(iterator == std::end(exclude_options_)) {
return false;
}
exclude_options_.erase(iterator);
return true;
}

bool remove_excludes(App *app) {
auto iterator = std::find(std::begin(exclude_subcommands_), std::end(exclude_subcommands_), app);
if(iterator == std::end(exclude_subcommands_)) {
return false;
}
auto other_app = *iterator;
exclude_subcommands_.erase(iterator);
other_app->remove_excludes(this);
return true;
}

bool remove_needs(Option *opt) {
auto iterator = std::find(std::begin(need_options_), std::end(need_options_), opt);
if(iterator == std::end(need_options_)) {
return false;
}
need_options_.erase(iterator);
return true;
}

bool remove_needs(App *app) {
auto iterator = std::find(std::begin(need_subcommands_), std::end(need_subcommands_), app);
if(iterator == std::end(need_subcommands_)) {
return false;
}
need_subcommands_.erase(iterator);
return true;
}


App *footer(std::string footer_string) {
footer_ = std::move(footer_string);
return this;
}
App *footer(std::function<std::string()> footer_function) {
footer_callback_ = std::move(footer_function);
return this;
}
std::string config_to_str(bool default_also = false, bool write_description = false) const {
return config_formatter_->to_config(this, default_also, write_description, "");
}

std::string help(std::string prev = "", AppFormatMode mode = AppFormatMode::Normal) const {
if(prev.empty())
prev = get_name();
else
prev += " " + get_name();

auto selected_subcommands = get_subcommands();
if(!selected_subcommands.empty()) {
return selected_subcommands.at(0)->help(prev, mode);
}
return formatter_->make_help(this, prev, mode);
}


std::shared_ptr<FormatterBase> get_formatter() const { return formatter_; }

std::shared_ptr<Config> get_config_formatter() const { return config_formatter_; }

std::shared_ptr<ConfigBase> get_config_formatter_base() const {
#if defined(__cpp_rtti) || (defined(__GXX_RTTI) && __GXX_RTTI) || (defined(_HAS_STATIC_RTTI) && (_HAS_STATIC_RTTI == 0))
return std::dynamic_pointer_cast<ConfigBase>(config_formatter_);
#else
return std::static_pointer_cast<ConfigBase>(config_formatter_);
#endif
}

std::string get_description() const { return description_; }

App *description(std::string app_description) {
description_ = std::move(app_description);
return this;
}

std::vector<const Option *> get_options(const std::function<bool(const Option *)> filter = {}) const {
std::vector<const Option *> options(options_.size());
std::transform(std::begin(options_), std::end(options_), std::begin(options), [](const Option_p &val) {
return val.get();
});

if(filter) {
options.erase(std::remove_if(std::begin(options),
std::end(options),
[&filter](const Option *opt) { return !filter(opt); }),
std::end(options));
}

return options;
}

std::vector<Option *> get_options(const std::function<bool(Option *)> filter = {}) {
std::vector<Option *> options(options_.size());
std::transform(std::begin(options_), std::end(options_), std::begin(options), [](const Option_p &val) {
return val.get();
});

if(filter) {
options.erase(
std::remove_if(std::begin(options), std::end(options), [&filter](Option *opt) { return !filter(opt); }),
std::end(options));
}

return options;
}

Option *get_option_no_throw(std::string option_name) noexcept {
for(Option_p &opt : options_) {
if(opt->check_name(option_name)) {
return opt.get();
}
}
for(auto &subc : subcommands_) {
if(subc->get_name().empty()) {
auto opt = subc->get_option_no_throw(option_name);
if(opt != nullptr) {
return opt;
}
}
}
return nullptr;
}

const Option *get_option_no_throw(std::string option_name) const noexcept {
for(const Option_p &opt : options_) {
if(opt->check_name(option_name)) {
return opt.get();
}
}
for(const auto &subc : subcommands_) {
if(subc->get_name().empty()) {
auto opt = subc->get_option_no_throw(option_name);
if(opt != nullptr) {
return opt;
}
}
}
return nullptr;
}

const Option *get_option(std::string option_name) const {
auto opt = get_option_no_throw(option_name);
if(opt == nullptr) {
throw OptionNotFound(option_name);
}
return opt;
}

Option *get_option(std::string option_name) {
auto opt = get_option_no_throw(option_name);
if(opt == nullptr) {
throw OptionNotFound(option_name);
}
return opt;
}

const Option *operator[](const std::string &option_name) const { return get_option(option_name); }

const Option *operator[](const char *option_name) const { return get_option(option_name); }

bool get_ignore_case() const { return ignore_case_; }

bool get_ignore_underscore() const { return ignore_underscore_; }

bool get_fallthrough() const { return fallthrough_; }

bool get_allow_windows_style_options() const { return allow_windows_style_options_; }

bool get_positionals_at_end() const { return positionals_at_end_; }

bool get_configurable() const { return configurable_; }

const std::string &get_group() const { return group_; }

std::string get_footer() const { return (footer_callback_) ? footer_callback_() + '\n' + footer_ : footer_; }

std::size_t get_require_subcommand_min() const { return require_subcommand_min_; }

std::size_t get_require_subcommand_max() const { return require_subcommand_max_; }

std::size_t get_require_option_min() const { return require_option_min_; }

std::size_t get_require_option_max() const { return require_option_max_; }

bool get_prefix_command() const { return prefix_command_; }

bool get_allow_extras() const { return allow_extras_; }

bool get_required() const { return required_; }

bool get_disabled() const { return disabled_; }

bool get_immediate_callback() const { return immediate_callback_; }

bool get_disabled_by_default() const { return (default_startup == startup_mode::disabled); }

bool get_enabled_by_default() const { return (default_startup == startup_mode::enabled); }
bool get_validate_positionals() const { return validate_positionals_; }

config_extras_mode get_allow_config_extras() const { return allow_config_extras_; }

Option *get_help_ptr() { return help_ptr_; }

const Option *get_help_ptr() const { return help_ptr_; }

const Option *get_help_all_ptr() const { return help_all_ptr_; }

Option *get_config_ptr() { return config_ptr_; }

const Option *get_config_ptr() const { return config_ptr_; }

App *get_parent() { return parent_; }

const App *get_parent() const { return parent_; }

const std::string &get_name() const { return name_; }

const std::vector<std::string> &get_aliases() const { return aliases_; }

App *clear_aliases() {
aliases_.clear();
return this;
}

std::string get_display_name() const { return (!name_.empty()) ? name_ : "[Option Group: " + get_group() + "]"; }

bool check_name(std::string name_to_check) const {
std::string local_name = name_;
if(ignore_underscore_) {
local_name = detail::remove_underscore(name_);
name_to_check = detail::remove_underscore(name_to_check);
}
if(ignore_case_) {
local_name = detail::to_lower(name_);
name_to_check = detail::to_lower(name_to_check);
}

if(local_name == name_to_check) {
return true;
}
for(auto les : aliases_) {
if(ignore_underscore_) {
les = detail::remove_underscore(les);
}
if(ignore_case_) {
les = detail::to_lower(les);
}
if(les == name_to_check) {
return true;
}
}
return false;
}

std::vector<std::string> get_groups() const {
std::vector<std::string> groups;

for(const Option_p &opt : options_) {
if(std::find(groups.begin(), groups.end(), opt->get_group()) == groups.end()) {
groups.push_back(opt->get_group());
}
}

return groups;
}

const std::vector<Option *> &parse_order() const { return parse_order_; }

std::vector<std::string> remaining(bool recurse = false) const {
std::vector<std::string> miss_list;
for(const std::pair<detail::Classifier, std::string> &miss : missing_) {
miss_list.push_back(std::get<1>(miss));
}
if(recurse) {
if(!allow_extras_) {
for(const auto &sub : subcommands_) {
if(sub->name_.empty() && !sub->missing_.empty()) {
for(const std::pair<detail::Classifier, std::string> &miss : sub->missing_) {
miss_list.push_back(std::get<1>(miss));
}
}
}
}

for(const App *sub : parsed_subcommands_) {
std::vector<std::string> output = sub->remaining(recurse);
std::copy(std::begin(output), std::end(output), std::back_inserter(miss_list));
}
}
return miss_list;
}

std::vector<std::string> remaining_for_passthrough(bool recurse = false) const {
std::vector<std::string> miss_list = remaining(recurse);
std::reverse(std::begin(miss_list), std::end(miss_list));
return miss_list;
}

std::size_t remaining_size(bool recurse = false) const {
auto remaining_options = static_cast<std::size_t>(std::count_if(
std::begin(missing_), std::end(missing_), [](const std::pair<detail::Classifier, std::string> &val) {
return val.first != detail::Classifier::POSITIONAL_MARK;
}));

if(recurse) {
for(const App_p &sub : subcommands_) {
remaining_options += sub->remaining_size(recurse);
}
}
return remaining_options;
}


protected:
void _validate() const {
auto pcount = std::count_if(std::begin(options_), std::end(options_), [](const Option_p &opt) {
return opt->get_items_expected_max() >= detail::expected_max_vector_size && !opt->nonpositional();
});
if(pcount > 1) {
auto pcount_req = std::count_if(std::begin(options_), std::end(options_), [](const Option_p &opt) {
return opt->get_items_expected_max() >= detail::expected_max_vector_size && !opt->nonpositional() &&
opt->get_required();
});
if(pcount - pcount_req > 1) {
throw InvalidError(name_);
}
}

std::size_t nameless_subs{0};
for(const App_p &app : subcommands_) {
app->_validate();
if(app->get_name().empty())
++nameless_subs;
}

if(require_option_min_ > 0) {
if(require_option_max_ > 0) {
if(require_option_max_ < require_option_min_) {
throw(InvalidError("Required min options greater than required max options",
ExitCodes::InvalidError));
}
}
if(require_option_min_ > (options_.size() + nameless_subs)) {
throw(InvalidError("Required min options greater than number of available options",
ExitCodes::InvalidError));
}
}
}

void _configure() {
if(default_startup == startup_mode::enabled) {
disabled_ = false;
} else if(default_startup == startup_mode::disabled) {
disabled_ = true;
}
for(const App_p &app : subcommands_) {
if(app->has_automatic_name_) {
app->name_.clear();
}
if(app->name_.empty()) {
app->fallthrough_ = false;  
app->prefix_command_ = false;
}
app->parent_ = this;
app->_configure();
}
}
void run_callback(bool final_mode = false) {
pre_callback();
if(!final_mode && parse_complete_callback_) {
parse_complete_callback_();
}
for(App *subc : get_subcommands()) {
subc->run_callback(true);
}
for(auto &subc : subcommands_) {
if(subc->name_.empty() && subc->count_all() > 0) {
subc->run_callback(true);
}
}

if(final_callback_ && (parsed_ > 0)) {
if(!name_.empty() || count_all() > 0) {
final_callback_();
}
}
}

bool _valid_subcommand(const std::string &current, bool ignore_used = true) const {
if(require_subcommand_max_ != 0 && parsed_subcommands_.size() >= require_subcommand_max_) {
return parent_ != nullptr && parent_->_valid_subcommand(current, ignore_used);
}
auto com = _find_subcommand(current, true, ignore_used);
if(com != nullptr) {
return true;
}
return parent_ != nullptr && parent_->_valid_subcommand(current, ignore_used);
}

detail::Classifier _recognize(const std::string &current, bool ignore_used_subcommands = true) const {
std::string dummy1, dummy2;

if(current == "--")
return detail::Classifier::POSITIONAL_MARK;
if(_valid_subcommand(current, ignore_used_subcommands))
return detail::Classifier::SUBCOMMAND;
if(detail::split_long(current, dummy1, dummy2))
return detail::Classifier::LONG;
if(detail::split_short(current, dummy1, dummy2)) {
if(dummy1[0] >= '0' && dummy1[0] <= '9') {
if(get_option_no_throw(std::string{'-', dummy1[0]}) == nullptr) {
return detail::Classifier::NONE;
}
}
return detail::Classifier::SHORT;
}
if((allow_windows_style_options_) && (detail::split_windows_style(current, dummy1, dummy2)))
return detail::Classifier::WINDOWS;
if((current == "++") && !name_.empty() && parent_ != nullptr)
return detail::Classifier::SUBCOMMAND_TERMINATOR;
return detail::Classifier::NONE;
}


void _process_config_file() {
if(config_ptr_ != nullptr) {
bool config_required = config_ptr_->get_required();
bool file_given = config_ptr_->count() > 0;
auto config_file = config_ptr_->as<std::string>();
if(config_file.empty()) {
if(config_required) {
throw FileError::Missing("no specified config file");
}
return;
}

auto path_result = detail::check_path(config_file.c_str());
if(path_result == detail::path_type::file) {
try {
std::vector<ConfigItem> values = config_formatter_->from_file(config_file);
_parse_config(values);
if(!file_given) {
config_ptr_->add_result(config_file);
}
} catch(const FileError &) {
if(config_required || file_given)
throw;
}
} else if(config_required || file_given) {
throw FileError::Missing(config_file);
}
}
}

void _process_env() {
for(const Option_p &opt : options_) {
if(opt->count() == 0 && !opt->envname_.empty()) {
char *buffer = nullptr;
std::string ename_string;

#ifdef _MSC_VER
std::size_t sz = 0;
if(_dupenv_s(&buffer, &sz, opt->envname_.c_str()) == 0 && buffer != nullptr) {
ename_string = std::string(buffer);
free(buffer);
}
#else
buffer = std::getenv(opt->envname_.c_str());
if(buffer != nullptr)
ename_string = std::string(buffer);
#endif

if(!ename_string.empty()) {
opt->add_result(ename_string);
}
}
}

for(App_p &sub : subcommands_) {
if(sub->get_name().empty() || !sub->parse_complete_callback_)
sub->_process_env();
}
}

void _process_callbacks() {

for(App_p &sub : subcommands_) {
if(sub->get_name().empty() && sub->parse_complete_callback_) {
if(sub->count_all() > 0) {
sub->_process_callbacks();
sub->run_callback();
}
}
}

for(const Option_p &opt : options_) {
if(opt->count() > 0 && !opt->get_callback_run()) {
opt->run_callback();
}
}
for(App_p &sub : subcommands_) {
if(!sub->parse_complete_callback_) {
sub->_process_callbacks();
}
}
}

void _process_help_flags(bool trigger_help = false, bool trigger_all_help = false) const {
const Option *help_ptr = get_help_ptr();
const Option *help_all_ptr = get_help_all_ptr();

if(help_ptr != nullptr && help_ptr->count() > 0)
trigger_help = true;
if(help_all_ptr != nullptr && help_all_ptr->count() > 0)
trigger_all_help = true;

if(!parsed_subcommands_.empty()) {
for(const App *sub : parsed_subcommands_)
sub->_process_help_flags(trigger_help, trigger_all_help);

} else if(trigger_all_help) {
throw CallForAllHelp();
} else if(trigger_help) {
throw CallForHelp();
}
}

void _process_requirements() {
bool excluded{false};
std::string excluder;
for(auto &opt : exclude_options_) {
if(opt->count() > 0) {
excluded = true;
excluder = opt->get_name();
}
}
for(auto &subc : exclude_subcommands_) {
if(subc->count_all() > 0) {
excluded = true;
excluder = subc->get_display_name();
}
}
if(excluded) {
if(count_all() > 0) {
throw ExcludesError(get_display_name(), excluder);
}
return;
}

bool missing_needed{false};
std::string missing_need;
for(auto &opt : need_options_) {
if(opt->count() == 0) {
missing_needed = true;
missing_need = opt->get_name();
}
}
for(auto &subc : need_subcommands_) {
if(subc->count_all() == 0) {
missing_needed = true;
missing_need = subc->get_display_name();
}
}
if(missing_needed) {
if(count_all() > 0) {
throw RequiresError(get_display_name(), missing_need);
}
return;
}

std::size_t used_options = 0;
for(const Option_p &opt : options_) {

if(opt->count() != 0) {
++used_options;
}
if(opt->get_required() && opt->count() == 0) {
throw RequiredError(opt->get_name());
}
for(const Option *opt_req : opt->needs_)
if(opt->count() > 0 && opt_req->count() == 0)
throw RequiresError(opt->get_name(), opt_req->get_name());
for(const Option *opt_ex : opt->excludes_)
if(opt->count() > 0 && opt_ex->count() != 0)
throw ExcludesError(opt->get_name(), opt_ex->get_name());
}
if(require_subcommand_min_ > 0) {
auto selected_subcommands = get_subcommands();
if(require_subcommand_min_ > selected_subcommands.size())
throw RequiredError::Subcommand(require_subcommand_min_);
}


for(App_p &sub : subcommands_) {
if(sub->disabled_)
continue;
if(sub->name_.empty() && sub->count_all() > 0) {
++used_options;
}
}

if(require_option_min_ > used_options || (require_option_max_ > 0 && require_option_max_ < used_options)) {
auto option_list = detail::join(options_, [](const Option_p &ptr) { return ptr->get_name(false, true); });
if(option_list.compare(0, 10, "-h,--help,") == 0) {
option_list.erase(0, 10);
}
auto subc_list = get_subcommands([](App *app) { return ((app->get_name().empty()) && (!app->disabled_)); });
if(!subc_list.empty()) {
option_list += "," + detail::join(subc_list, [](const App *app) { return app->get_display_name(); });
}
throw RequiredError::Option(require_option_min_, require_option_max_, used_options, option_list);
}

for(App_p &sub : subcommands_) {
if(sub->disabled_)
continue;
if(sub->name_.empty() && sub->required_ == false) {
if(sub->count_all() == 0) {
if(require_option_min_ > 0 && require_option_min_ <= used_options) {
continue;
}
if(require_option_max_ > 0 && used_options >= require_option_min_) {
continue;
}
}
}
if(sub->count() > 0 || sub->name_.empty()) {
sub->_process_requirements();
}

if(sub->required_ && sub->count_all() == 0) {
throw(CLI::RequiredError(sub->get_display_name()));
}
}
}

void _process() {
_process_config_file();
_process_env();
_process_callbacks();
_process_help_flags();
_process_requirements();
}

void _process_extras() {
if(!(allow_extras_ || prefix_command_)) {
std::size_t num_left_over = remaining_size();
if(num_left_over > 0) {
throw ExtrasError(name_, remaining(false));
}
}

for(App_p &sub : subcommands_) {
if(sub->count() > 0)
sub->_process_extras();
}
}

void _process_extras(std::vector<std::string> &args) {
if(!(allow_extras_ || prefix_command_)) {
std::size_t num_left_over = remaining_size();
if(num_left_over > 0) {
args = remaining(false);
throw ExtrasError(name_, args);
}
}

for(App_p &sub : subcommands_) {
if(sub->count() > 0)
sub->_process_extras(args);
}
}

void increment_parsed() {
++parsed_;
for(App_p &sub : subcommands_) {
if(sub->get_name().empty())
sub->increment_parsed();
}
}
void _parse(std::vector<std::string> &args) {
increment_parsed();
_trigger_pre_parse(args.size());
bool positional_only = false;

while(!args.empty()) {
if(!_parse_single(args, positional_only)) {
break;
}
}

if(parent_ == nullptr) {
_process();

_process_extras(args);

args = remaining_for_passthrough(false);
} else if(parse_complete_callback_) {
_process_env();
_process_callbacks();
_process_help_flags();
_process_requirements();
run_callback();
}
}

void _parse(std::vector<std::string> &&args) {
increment_parsed();
_trigger_pre_parse(args.size());
bool positional_only = false;

while(!args.empty()) {
_parse_single(args, positional_only);
}
_process();

_process_extras();
}

void _parse_config(std::vector<ConfigItem> &args) {
for(ConfigItem item : args) {
if(!_parse_single_config(item) && allow_config_extras_ == config_extras_mode::error)
throw ConfigError::Extras(item.fullname());
}
}

bool _parse_single_config(const ConfigItem &item, std::size_t level = 0) {
if(level < item.parents.size()) {
try {
auto subcom = get_subcommand(item.parents.at(level));
auto result = subcom->_parse_single_config(item, level + 1);

return result;
} catch(const OptionNotFound &) {
return false;
}
}
if(item.name == "++") {
if(configurable_) {
increment_parsed();
_trigger_pre_parse(2);
if(parent_ != nullptr) {
parent_->parsed_subcommands_.push_back(this);
}
}
return true;
}
if(item.name == "--") {
if(configurable_) {
_process_callbacks();
_process_requirements();
run_callback();
}
return true;
}
Option *op = get_option_no_throw("--" + item.name);
if(op == nullptr) {
if(get_allow_config_extras() == config_extras_mode::capture)
missing_.emplace_back(detail::Classifier::NONE, item.fullname());
return false;
}

if(!op->get_configurable())
throw ConfigError::NotConfigurable(item.fullname());

if(op->empty()) {
if(op->get_expected_min() == 0) {
auto res = config_formatter_->to_flag(item);
res = op->get_flag_value(item.name, res);

op->add_result(res);

} else {
op->add_result(item.inputs);
op->run_callback();
}
}

return true;
}

bool _parse_single(std::vector<std::string> &args, bool &positional_only) {
bool retval = true;
detail::Classifier classifier = positional_only ? detail::Classifier::NONE : _recognize(args.back());
switch(classifier) {
case detail::Classifier::POSITIONAL_MARK:
args.pop_back();
positional_only = true;
if((!_has_remaining_positionals()) && (parent_ != nullptr)) {
retval = false;
} else {
_move_to_missing(classifier, "--");
}
break;
case detail::Classifier::SUBCOMMAND_TERMINATOR:
args.pop_back();
retval = false;
break;
case detail::Classifier::SUBCOMMAND:
retval = _parse_subcommand(args);
break;
case detail::Classifier::LONG:
case detail::Classifier::SHORT:
case detail::Classifier::WINDOWS:
_parse_arg(args, classifier);
break;
case detail::Classifier::NONE:
retval = _parse_positional(args, false);
if(retval && positionals_at_end_) {
positional_only = true;
}
break;
default:
throw HorribleError("unrecognized classifier (you should not see this!)");
}
return retval;
}

std::size_t _count_remaining_positionals(bool required_only = false) const {
std::size_t retval = 0;
for(const Option_p &opt : options_) {
if(opt->get_positional() && (!required_only || opt->get_required())) {
if(opt->get_items_expected_min() > 0 &&
static_cast<int>(opt->count()) < opt->get_items_expected_min()) {
retval += static_cast<std::size_t>(opt->get_items_expected_min()) - opt->count();
}
}
}
return retval;
}

bool _has_remaining_positionals() const {
for(const Option_p &opt : options_) {
if(opt->get_positional() && ((static_cast<int>(opt->count()) < opt->get_items_expected_min()))) {
return true;
}
}

return false;
}

bool _parse_positional(std::vector<std::string> &args, bool haltOnSubcommand) {

const std::string &positional = args.back();

if(positionals_at_end_) {
auto arg_rem = args.size();
auto remreq = _count_remaining_positionals(true);
if(arg_rem <= remreq) {
for(const Option_p &opt : options_) {
if(opt->get_positional() && opt->required_) {
if(static_cast<int>(opt->count()) < opt->get_items_expected_min()) {
if(validate_positionals_) {
std::string pos = positional;
pos = opt->_validate(pos, 0);
if(!pos.empty()) {
continue;
}
}
opt->add_result(positional);
parse_order_.push_back(opt.get());
args.pop_back();
return true;
}
}
}
}
}
for(const Option_p &opt : options_) {
if(opt->get_positional() &&
(static_cast<int>(opt->count()) < opt->get_items_expected_min() || opt->get_allow_extra_args())) {
if(validate_positionals_) {
std::string pos = positional;
pos = opt->_validate(pos, 0);
if(!pos.empty()) {
continue;
}
}
opt->add_result(positional);
parse_order_.push_back(opt.get());
args.pop_back();
return true;
}
}

for(auto &subc : subcommands_) {
if((subc->name_.empty()) && (!subc->disabled_)) {
if(subc->_parse_positional(args, false)) {
if(!subc->pre_parse_called_) {
subc->_trigger_pre_parse(args.size());
}
return true;
}
}
}
if(parent_ != nullptr && fallthrough_)
return _get_fallthrough_parent()->_parse_positional(args, static_cast<bool>(parse_complete_callback_));

auto com = _find_subcommand(args.back(), true, false);
if(com != nullptr && (require_subcommand_max_ == 0 || require_subcommand_max_ > parsed_subcommands_.size())) {
if(haltOnSubcommand) {
return false;
}
args.pop_back();
com->_parse(args);
return true;
}
auto parent_app = (parent_ != nullptr) ? _get_fallthrough_parent() : this;
com = parent_app->_find_subcommand(args.back(), true, false);
if(com != nullptr && (com->parent_->require_subcommand_max_ == 0 ||
com->parent_->require_subcommand_max_ > com->parent_->parsed_subcommands_.size())) {
return false;
}

if(positionals_at_end_) {
throw CLI::ExtrasError(name_, args);
}
if(parent_ != nullptr && name_.empty()) {
return false;
}
_move_to_missing(detail::Classifier::NONE, positional);
args.pop_back();
if(prefix_command_) {
while(!args.empty()) {
_move_to_missing(detail::Classifier::NONE, args.back());
args.pop_back();
}
}

return true;
}

App *_find_subcommand(const std::string &subc_name, bool ignore_disabled, bool ignore_used) const noexcept {
for(const App_p &com : subcommands_) {
if(com->disabled_ && ignore_disabled)
continue;
if(com->get_name().empty()) {
auto subc = com->_find_subcommand(subc_name, ignore_disabled, ignore_used);
if(subc != nullptr) {
return subc;
}
}
if(com->check_name(subc_name)) {
if((!*com) || !ignore_used)
return com.get();
}
}
return nullptr;
}

bool _parse_subcommand(std::vector<std::string> &args) {
if(_count_remaining_positionals( true) > 0) {
_parse_positional(args, false);
return true;
}
auto com = _find_subcommand(args.back(), true, true);
if(com != nullptr) {
args.pop_back();
parsed_subcommands_.push_back(com);
com->_parse(args);
auto parent_app = com->parent_;
while(parent_app != this) {
parent_app->_trigger_pre_parse(args.size());
parent_app->parsed_subcommands_.push_back(com);
parent_app = parent_app->parent_;
}
return true;
}

if(parent_ == nullptr)
throw HorribleError("Subcommand " + args.back() + " missing");
return false;
}

bool _parse_arg(std::vector<std::string> &args, detail::Classifier current_type) {

std::string current = args.back();

std::string arg_name;
std::string value;
std::string rest;

switch(current_type) {
case detail::Classifier::LONG:
if(!detail::split_long(current, arg_name, value))
throw HorribleError("Long parsed but missing (you should not see this):" + args.back());
break;
case detail::Classifier::SHORT:
if(!detail::split_short(current, arg_name, rest))
throw HorribleError("Short parsed but missing! You should not see this");
break;
case detail::Classifier::WINDOWS:
if(!detail::split_windows_style(current, arg_name, value))
throw HorribleError("windows option parsed but missing! You should not see this");
break;
case detail::Classifier::SUBCOMMAND:
case detail::Classifier::SUBCOMMAND_TERMINATOR:
case detail::Classifier::POSITIONAL_MARK:
case detail::Classifier::NONE:
default:
throw HorribleError("parsing got called with invalid option! You should not see this");
}

auto op_ptr =
std::find_if(std::begin(options_), std::end(options_), [arg_name, current_type](const Option_p &opt) {
if(current_type == detail::Classifier::LONG)
return opt->check_lname(arg_name);
if(current_type == detail::Classifier::SHORT)
return opt->check_sname(arg_name);
return opt->check_lname(arg_name) || opt->check_sname(arg_name);
});

if(op_ptr == std::end(options_)) {
for(auto &subc : subcommands_) {
if(subc->name_.empty() && !subc->disabled_) {
if(subc->_parse_arg(args, current_type)) {
if(!subc->pre_parse_called_) {
subc->_trigger_pre_parse(args.size());
}
return true;
}
}
}
if(parent_ != nullptr && fallthrough_)
return _get_fallthrough_parent()->_parse_arg(args, current_type);
if(parent_ != nullptr && name_.empty()) {
return false;
}
args.pop_back();
_move_to_missing(current_type, current);
return true;
}

args.pop_back();

Option_p &op = *op_ptr;

int min_num = (std::min)(op->get_type_size_min(), op->get_items_expected_min());
int max_num = op->get_items_expected_max();

int collected = 0;     
int result_count = 0;  
if(max_num == 0) {
auto res = op->get_flag_value(arg_name, value);
op->add_result(res);
parse_order_.push_back(op.get());
} else if(!value.empty()) {  
op->add_result(value, result_count);
parse_order_.push_back(op.get());
collected += result_count;
} else if(!rest.empty()) {
op->add_result(rest, result_count);
parse_order_.push_back(op.get());
rest = "";
collected += result_count;
}

while(min_num > collected && !args.empty()) {
std::string current_ = args.back();
args.pop_back();
op->add_result(current_, result_count);
parse_order_.push_back(op.get());
collected += result_count;
}

if(min_num > collected) {  
throw ArgumentMismatch::TypedAtLeast(op->get_name(), min_num, op->get_type_name());
}

if(max_num > collected || op->get_allow_extra_args()) {  
auto remreqpos = _count_remaining_positionals(true);
while((collected < max_num || op->get_allow_extra_args()) && !args.empty() &&
_recognize(args.back(), false) == detail::Classifier::NONE) {
if(remreqpos >= args.size()) {
break;
}

op->add_result(args.back(), result_count);
parse_order_.push_back(op.get());
args.pop_back();
collected += result_count;
}

if(!args.empty() && _recognize(args.back()) == detail::Classifier::POSITIONAL_MARK)
args.pop_back();
if(min_num == 0 && max_num > 0 && collected == 0) {
auto res = op->get_flag_value(arg_name, std::string{});
op->add_result(res);
parse_order_.push_back(op.get());
}
}

if(min_num > 0 && op->get_type_size_max() != min_num && collected % op->get_type_size_max() != 0) {
op->add_result(std::string{});
}

if(!rest.empty()) {
rest = "-" + rest;
args.push_back(rest);
}
return true;
}

void _trigger_pre_parse(std::size_t remaining_args) {
if(!pre_parse_called_) {
pre_parse_called_ = true;
if(pre_parse_callback_) {
pre_parse_callback_(remaining_args);
}
} else if(immediate_callback_) {
if(!name_.empty()) {
auto pcnt = parsed_;
auto extras = std::move(missing_);
clear();
parsed_ = pcnt;
pre_parse_called_ = true;
missing_ = std::move(extras);
}
}
}

App *_get_fallthrough_parent() {
if(parent_ == nullptr) {
throw(HorribleError("No Valid parent"));
}
auto fallthrough_parent = parent_;
while((fallthrough_parent->parent_ != nullptr) && (fallthrough_parent->get_name().empty())) {
fallthrough_parent = fallthrough_parent->parent_;
}
return fallthrough_parent;
}

const std::string &_compare_subcommand_names(const App &subcom, const App &base) const {
static const std::string estring;
if(subcom.disabled_) {
return estring;
}
for(auto &subc : base.subcommands_) {
if(subc.get() != &subcom) {
if(subc->disabled_) {
continue;
}
if(!subcom.get_name().empty()) {
if(subc->check_name(subcom.get_name())) {
return subcom.get_name();
}
}
if(!subc->get_name().empty()) {
if(subcom.check_name(subc->get_name())) {
return subc->get_name();
}
}
for(const auto &les : subcom.aliases_) {
if(subc->check_name(les)) {
return les;
}
}
for(const auto &les : subc->aliases_) {
if(subcom.check_name(les)) {
return les;
}
}
if(subc->get_name().empty()) {
auto &cmpres = _compare_subcommand_names(subcom, *subc);
if(!cmpres.empty()) {
return cmpres;
}
}
if(subcom.get_name().empty()) {
auto &cmpres = _compare_subcommand_names(*subc, subcom);
if(!cmpres.empty()) {
return cmpres;
}
}
}
}
return estring;
}
void _move_to_missing(detail::Classifier val_type, const std::string &val) {
if(allow_extras_ || subcommands_.empty()) {
missing_.emplace_back(val_type, val);
return;
}
for(auto &subc : subcommands_) {
if(subc->name_.empty() && subc->allow_extras_) {
subc->missing_.emplace_back(val_type, val);
return;
}
}
missing_.emplace_back(val_type, val);
}

public:
void _move_option(Option *opt, App *app) {
if(opt == nullptr) {
throw OptionNotFound("the option is NULL");
}
bool found = false;
for(auto &subc : subcommands_) {
if(app == subc.get()) {
found = true;
}
}
if(!found) {
throw OptionNotFound("The Given app is not a subcommand");
}

if((help_ptr_ == opt) || (help_all_ptr_ == opt))
throw OptionAlreadyAdded("cannot move help options");

if(config_ptr_ == opt)
throw OptionAlreadyAdded("cannot move config file options");

auto iterator =
std::find_if(std::begin(options_), std::end(options_), [opt](const Option_p &v) { return v.get() == opt; });
if(iterator != std::end(options_)) {
const auto &opt_p = *iterator;
if(std::find_if(std::begin(app->options_), std::end(app->options_), [&opt_p](const Option_p &v) {
return (*v == *opt_p);
}) == std::end(app->options_)) {
app->options_.push_back(std::move(*iterator));
options_.erase(iterator);
} else {
throw OptionAlreadyAdded("option was not located: " + opt->get_name());
}
} else {
throw OptionNotFound("could not locate the given Option");
}
}
};  

class Option_group : public App {
public:
Option_group(std::string group_description, std::string group_name, App *parent)
: App(std::move(group_description), "", parent) {
group(group_name);
}
using App::add_option;
Option *add_option(Option *opt) {
if(get_parent() == nullptr) {
throw OptionNotFound("Unable to locate the specified option");
}
get_parent()->_move_option(opt, this);
return opt;
}
void add_options(Option *opt) { add_option(opt); }
template <typename... Args> void add_options(Option *opt, Args... args) {
add_option(opt);
add_options(args...);
}
using App::add_subcommand;
App *add_subcommand(App *subcom) {
App_p subc = subcom->get_parent()->get_subcommand_ptr(subcom);
subc->get_parent()->remove_subcommand(subcom);
add_subcommand(std::move(subc));
return subcom;
}
};
inline void TriggerOn(App *trigger_app, App *app_to_enable) {
app_to_enable->enabled_by_default(false);
app_to_enable->disabled_by_default();
trigger_app->preparse_callback([app_to_enable](std::size_t) { app_to_enable->disabled(false); });
}

inline void TriggerOn(App *trigger_app, std::vector<App *> apps_to_enable) {
for(auto &app : apps_to_enable) {
app->enabled_by_default(false);
app->disabled_by_default();
}

trigger_app->preparse_callback([apps_to_enable](std::size_t) {
for(auto &app : apps_to_enable) {
app->disabled(false);
}
});
}

inline void TriggerOff(App *trigger_app, App *app_to_enable) {
app_to_enable->disabled_by_default(false);
app_to_enable->enabled_by_default();
trigger_app->preparse_callback([app_to_enable](std::size_t) { app_to_enable->disabled(); });
}

inline void TriggerOff(App *trigger_app, std::vector<App *> apps_to_enable) {
for(auto &app : apps_to_enable) {
app->disabled_by_default(false);
app->enabled_by_default();
}

trigger_app->preparse_callback([apps_to_enable](std::size_t) {
for(auto &app : apps_to_enable) {
app->disabled();
}
});
}

inline void deprecate_option(Option *opt, const std::string &replacement = "") {
Validator deprecate_warning{[opt, replacement](std::string &) {
std::cout << opt->get_name() << " is deprecated please use '" << replacement
<< "' instead\n";
return std::string();
},
"DEPRECATED"};
deprecate_warning.application_index(0);
opt->check(deprecate_warning);
if(!replacement.empty()) {
opt->description(opt->get_description() + " DEPRECATED: please use '" + replacement + "' instead");
}
}

inline void deprecate_option(App *app, const std::string &option_name, const std::string &replacement = "") {
auto opt = app->get_option(option_name);
deprecate_option(opt, replacement);
}

inline void deprecate_option(App &app, const std::string &option_name, const std::string &replacement = "") {
auto opt = app.get_option(option_name);
deprecate_option(opt, replacement);
}

inline void retire_option(App *app, Option *opt) {
App temp;
auto option_copy = temp.add_option(opt->get_name(false, true))
->type_size(opt->get_type_size_min(), opt->get_type_size_max())
->expected(opt->get_expected_min(), opt->get_expected_max())
->allow_extra_args(opt->get_allow_extra_args());

app->remove_option(opt);
auto opt2 = app->add_option(option_copy->get_name(false, true), "option has been retired and has no effect")
->type_name("RETIRED")
->default_str("RETIRED")
->type_size(option_copy->get_type_size_min(), option_copy->get_type_size_max())
->expected(option_copy->get_expected_min(), option_copy->get_expected_max())
->allow_extra_args(option_copy->get_allow_extra_args());

Validator retired_warning{[opt2](std::string &) {
std::cout << "WARNING " << opt2->get_name() << " is retired and has no effect\n";
return std::string();
},
""};
retired_warning.application_index(0);
opt2->check(retired_warning);
}

inline void retire_option(App &app, Option *opt) { retire_option(&app, opt); }

inline void retire_option(App *app, const std::string &option_name) {

auto opt = app->get_option_no_throw(option_name);
if(opt != nullptr) {
retire_option(app, opt);
return;
}
auto opt2 = app->add_option(option_name, "option has been retired and has no effect")
->type_name("RETIRED")
->expected(0, 1)
->default_str("RETIRED");
Validator retired_warning{[opt2](std::string &) {
std::cout << "WARNING " << opt2->get_name() << " is retired and has no effect\n";
return std::string();
},
""};
retired_warning.application_index(0);
opt2->check(retired_warning);
}

inline void retire_option(App &app, const std::string &option_name) { retire_option(&app, option_name); }

namespace FailureMessage {

inline std::string simple(const App *app, const Error &e) {
std::string header = std::string(e.what()) + "\n";
std::vector<std::string> names;

if(app->get_help_ptr() != nullptr)
names.push_back(app->get_help_ptr()->get_name());

if(app->get_help_all_ptr() != nullptr)
names.push_back(app->get_help_all_ptr()->get_name());

if(!names.empty())
header += "Run with " + detail::join(names, " or ") + " for more information.\n";

return header;
}

inline std::string help(const App *app, const Error &e) {
std::string header = std::string("ERROR: ") + e.get_name() + ": " + e.what() + "\n";
header += app->help();
return header;
}

}  

namespace detail {
struct AppFriend {
#ifdef CLI11_CPP14

template <typename... Args> static decltype(auto) parse_arg(App *app, Args &&... args) {
return app->_parse_arg(std::forward<Args>(args)...);
}

template <typename... Args> static decltype(auto) parse_subcommand(App *app, Args &&... args) {
return app->_parse_subcommand(std::forward<Args>(args)...);
}
#else
template <typename... Args>
static auto parse_arg(App *app, Args &&... args) ->
typename std::result_of<decltype (&App::_parse_arg)(App, Args...)>::type {
return app->_parse_arg(std::forward<Args>(args)...);
}

template <typename... Args>
static auto parse_subcommand(App *app, Args &&... args) ->
typename std::result_of<decltype (&App::_parse_subcommand)(App, Args...)>::type {
return app->_parse_subcommand(std::forward<Args>(args)...);
}
#endif
static App *get_fallthrough_parent(App *app) { return app->_get_fallthrough_parent(); }
};
}  

}  


namespace CLI {

namespace detail {

inline std::string convert_arg_for_ini(const std::string &arg) {
if(arg.empty()) {
return std::string(2, '"');
}
if(arg == "true" || arg == "false" || arg == "nan" || arg == "inf") {
return arg;
}
if(arg.compare(0, 2, "0x") != 0 && arg.compare(0, 2, "0X") != 0) {
double val;
if(detail::lexical_cast(arg, val)) {
return arg;
}
}
if(arg.size() == 1) {
return std::string("'") + arg + '\'';
}
if(arg.front() == '0') {
if(arg[1] == 'x') {
if(std::all_of(arg.begin() + 2, arg.end(), [](char x) {
return (x >= '0' && x <= '9') || (x >= 'A' && x <= 'F') || (x >= 'a' && x <= 'f');
})) {
return arg;
}
} else if(arg[1] == 'o') {
if(std::all_of(arg.begin() + 2, arg.end(), [](char x) { return (x >= '0' && x <= '7'); })) {
return arg;
}
} else if(arg[1] == 'b') {
if(std::all_of(arg.begin() + 2, arg.end(), [](char x) { return (x == '0' || x == '1'); })) {
return arg;
}
}
}
if(arg.find_first_of('"') == std::string::npos) {
return std::string("\"") + arg + '"';
} else {
return std::string("'") + arg + '\'';
}
}

inline std::string
ini_join(const std::vector<std::string> &args, char sepChar = ',', char arrayStart = '[', char arrayEnd = ']') {
std::string joined;
if(args.size() > 1 && arrayStart != '\0') {
joined.push_back(arrayStart);
}
std::size_t start = 0;
for(const auto &arg : args) {
if(start++ > 0) {
joined.push_back(sepChar);
if(isspace(sepChar) == 0) {
joined.push_back(' ');
}
}
joined.append(convert_arg_for_ini(arg));
}
if(args.size() > 1 && arrayEnd != '\0') {
joined.push_back(arrayEnd);
}
return joined;
}

inline std::vector<std::string> generate_parents(const std::string &section, std::string &name) {
std::vector<std::string> parents;
if(detail::to_lower(section) != "default") {
if(section.find('.') != std::string::npos) {
parents = detail::split(section, '.');
} else {
parents = {section};
}
}
if(name.find('.') != std::string::npos) {
std::vector<std::string> plist = detail::split(name, '.');
name = plist.back();
detail::remove_quotes(name);
plist.pop_back();
parents.insert(parents.end(), plist.begin(), plist.end());
}

for(auto &parent : parents) {
detail::remove_quotes(parent);
}
return parents;
}

inline void checkParentSegments(std::vector<ConfigItem> &output, const std::string &currentSection) {

std::string estring;
auto parents = detail::generate_parents(currentSection, estring);
if(!output.empty() && output.back().name == "--") {
std::size_t msize = (parents.size() > 1U) ? parents.size() : 2;
while(output.back().parents.size() >= msize) {
output.push_back(output.back());
output.back().parents.pop_back();
}

if(parents.size() > 1) {
std::size_t common = 0;
std::size_t mpair = (std::min)(output.back().parents.size(), parents.size() - 1);
for(std::size_t ii = 0; ii < mpair; ++ii) {
if(output.back().parents[ii] != parents[ii]) {
break;
}
++common;
}
if(common == mpair) {
output.pop_back();
} else {
while(output.back().parents.size() > common + 1) {
output.push_back(output.back());
output.back().parents.pop_back();
}
}
for(std::size_t ii = common; ii < parents.size() - 1; ++ii) {
output.emplace_back();
output.back().parents.assign(parents.begin(), parents.begin() + static_cast<std::ptrdiff_t>(ii) + 1);
output.back().name = "++";
}
}
} else if(parents.size() > 1) {
for(std::size_t ii = 0; ii < parents.size() - 1; ++ii) {
output.emplace_back();
output.back().parents.assign(parents.begin(), parents.begin() + static_cast<std::ptrdiff_t>(ii) + 1);
output.back().name = "++";
}
}

output.emplace_back();
output.back().parents = std::move(parents);
output.back().name = "++";
}
}  

inline std::vector<ConfigItem> ConfigBase::from_config(std::istream &input) const {
std::string line;
std::string section = "default";

std::vector<ConfigItem> output;
bool defaultArray = (arrayStart == '\0' || arrayStart == ' ') && arrayStart == arrayEnd;
char aStart = (defaultArray) ? '[' : arrayStart;
char aEnd = (defaultArray) ? ']' : arrayEnd;
char aSep = (defaultArray && arraySeparator == ' ') ? ',' : arraySeparator;

while(getline(input, line)) {
std::vector<std::string> items_buffer;
std::string name;

detail::trim(line);
std::size_t len = line.length();
if(len > 1 && line.front() == '[' && line.back() == ']') {
if(section != "default") {
output.emplace_back();
output.back().parents = detail::generate_parents(section, name);
output.back().name = "--";
}
section = line.substr(1, len - 2);
if(section.size() > 1 && section.front() == '[' && section.back() == ']') {
section = section.substr(1, section.size() - 2);
}
if(detail::to_lower(section) == "default") {
section = "default";
} else {
detail::checkParentSegments(output, section);
}
continue;
}
if(len == 0) {
continue;
}
if(line.front() == ';' || line.front() == '#' || line.front() == commentChar) {
continue;
}

auto pos = line.find(valueDelimiter);
if(pos != std::string::npos) {
name = detail::trim_copy(line.substr(0, pos));
std::string item = detail::trim_copy(line.substr(pos + 1));
if(item.size() > 1 && item.front() == aStart && item.back() == aEnd) {
items_buffer = detail::split_up(item.substr(1, item.length() - 2), aSep);
} else if(defaultArray && item.find_first_of(aSep) != std::string::npos) {
items_buffer = detail::split_up(item, aSep);
} else if(defaultArray && item.find_first_of(' ') != std::string::npos) {
items_buffer = detail::split_up(item);
} else {
items_buffer = {item};
}
} else {
name = detail::trim_copy(line);
items_buffer = {"true"};
}
if(name.find('.') == std::string::npos) {
detail::remove_quotes(name);
}
for(auto &it : items_buffer) {
detail::remove_quotes(it);
}

std::vector<std::string> parents = detail::generate_parents(section, name);

if(!output.empty() && name == output.back().name && parents == output.back().parents) {
output.back().inputs.insert(output.back().inputs.end(), items_buffer.begin(), items_buffer.end());
} else {
output.emplace_back();
output.back().parents = std::move(parents);
output.back().name = std::move(name);
output.back().inputs = std::move(items_buffer);
}
}
if(section != "default") {
std::string ename;
output.emplace_back();
output.back().parents = detail::generate_parents(section, ename);
output.back().name = "--";
while(output.back().parents.size() > 1) {
output.push_back(output.back());
output.back().parents.pop_back();
}
}
return output;
}

inline std::string
ConfigBase::to_config(const App *app, bool default_also, bool write_description, std::string prefix) const {
std::stringstream out;
std::string commentLead;
commentLead.push_back(commentChar);
commentLead.push_back(' ');

std::vector<std::string> groups = app->get_groups();
bool defaultUsed = false;
groups.insert(groups.begin(), std::string("Options"));
if(write_description) {
out << commentLead << app->get_description() << '\n';
}
for(auto &group : groups) {
if(group == "Options" || group.empty()) {
if(defaultUsed) {
continue;
}
defaultUsed = true;
}
if(write_description && group != "Options" && !group.empty()) {
out << '\n' << commentLead << group << " Options\n";
}
for(const Option *opt : app->get_options({})) {

if(!opt->get_lnames().empty() && opt->get_configurable()) {
if(opt->get_group() != group) {
if(!(group == "Options" && opt->get_group().empty())) {
continue;
}
}
std::string name = prefix + opt->get_lnames()[0];
std::string value = detail::ini_join(opt->reduced_results(), arraySeparator, arrayStart, arrayEnd);

if(value.empty() && default_also) {
if(!opt->get_default_str().empty()) {
value = detail::convert_arg_for_ini(opt->get_default_str());
} else if(opt->get_expected_min() == 0) {
value = "false";
}
}

if(!value.empty()) {
if(write_description && opt->has_description()) {
out << '\n';
out << commentLead << detail::fix_newlines(commentLead, opt->get_description()) << '\n';
}
out << name << valueDelimiter << value << '\n';
}
}
}
}
auto subcommands = app->get_subcommands({});
for(const App *subcom : subcommands) {
if(subcom->get_name().empty()) {
if(write_description && !subcom->get_group().empty()) {
out << '\n' << commentLead << subcom->get_group() << " Options\n";
}
out << to_config(subcom, default_also, write_description, prefix);
}
}

for(const App *subcom : subcommands) {
if(!subcom->get_name().empty()) {
if(subcom->get_configurable() && app->got_subcommand(subcom)) {
if(!prefix.empty() || app->get_parent() == nullptr) {
out << '[' << prefix << subcom->get_name() << "]\n";
} else {
std::string subname = app->get_name() + "." + subcom->get_name();
auto p = app->get_parent();
while(p->get_parent() != nullptr) {
subname = p->get_name() + "." + subname;
p = p->get_parent();
}
out << '[' << subname << "]\n";
}
out << to_config(subcom, default_also, write_description, "");
} else {
out << to_config(subcom, default_also, write_description, prefix + subcom->get_name() + ".");
}
}
}

return out.str();
}

}  


namespace CLI {

inline std::string
Formatter::make_group(std::string group, bool is_positional, std::vector<const Option *> opts) const {
std::stringstream out;

out << "\n" << group << ":\n";
for(const Option *opt : opts) {
out << make_option(opt, is_positional);
}

return out.str();
}

inline std::string Formatter::make_positionals(const App *app) const {
std::vector<const Option *> opts =
app->get_options([](const Option *opt) { return !opt->get_group().empty() && opt->get_positional(); });

if(opts.empty())
return std::string();

return make_group(get_label("Positionals"), true, opts);
}

inline std::string Formatter::make_groups(const App *app, AppFormatMode mode) const {
std::stringstream out;
std::vector<std::string> groups = app->get_groups();

for(const std::string &group : groups) {
std::vector<const Option *> opts = app->get_options([app, mode, &group](const Option *opt) {
return opt->get_group() == group                     
&& opt->nonpositional()                       
&& (mode != AppFormatMode::Sub                
|| (app->get_help_ptr() != opt            
&& app->get_help_all_ptr() != opt));  
});
if(!group.empty() && !opts.empty()) {
out << make_group(group, false, opts);

if(group != groups.back())
out << "\n";
}
}

return out.str();
}

inline std::string Formatter::make_description(const App *app) const {
std::string desc = app->get_description();
auto min_options = app->get_require_option_min();
auto max_options = app->get_require_option_max();
if(app->get_required()) {
desc += " REQUIRED ";
}
if((max_options == min_options) && (min_options > 0)) {
if(min_options == 1) {
desc += " \n[Exactly 1 of the following options is required]";
} else {
desc += " \n[Exactly " + std::to_string(min_options) + "options from the following list are required]";
}
} else if(max_options > 0) {
if(min_options > 0) {
desc += " \n[Between " + std::to_string(min_options) + " and " + std::to_string(max_options) +
" of the follow options are required]";
} else {
desc += " \n[At most " + std::to_string(max_options) + " of the following options are allowed]";
}
} else if(min_options > 0) {
desc += " \n[At least " + std::to_string(min_options) + " of the following options are required]";
}
return (!desc.empty()) ? desc + "\n" : std::string{};
}

inline std::string Formatter::make_usage(const App *app, std::string name) const {
std::stringstream out;

out << get_label("Usage") << ":" << (name.empty() ? "" : " ") << name;

std::vector<std::string> groups = app->get_groups();

std::vector<const Option *> non_pos_options =
app->get_options([](const Option *opt) { return opt->nonpositional(); });
if(!non_pos_options.empty())
out << " [" << get_label("OPTIONS") << "]";

std::vector<const Option *> positionals = app->get_options([](const Option *opt) { return opt->get_positional(); });

if(!positionals.empty()) {
std::vector<std::string> positional_names(positionals.size());
std::transform(positionals.begin(), positionals.end(), positional_names.begin(), [this](const Option *opt) {
return make_option_usage(opt);
});

out << " " << detail::join(positional_names, " ");
}

if(!app->get_subcommands(
[](const CLI::App *subc) { return ((!subc->get_disabled()) && (!subc->get_name().empty())); })
.empty()) {
out << " " << (app->get_require_subcommand_min() == 0 ? "[" : "")
<< get_label(app->get_require_subcommand_max() < 2 || app->get_require_subcommand_min() > 1 ? "SUBCOMMAND"
: "SUBCOMMANDS")
<< (app->get_require_subcommand_min() == 0 ? "]" : "");
}

out << std::endl;

return out.str();
}

inline std::string Formatter::make_footer(const App *app) const {
std::string footer = app->get_footer();
if(footer.empty()) {
return std::string{};
}
return footer + "\n";
}

inline std::string Formatter::make_help(const App *app, std::string name, AppFormatMode mode) const {

if(mode == AppFormatMode::Sub)
return make_expanded(app);

std::stringstream out;
if((app->get_name().empty()) && (app->get_parent() != nullptr)) {
if(app->get_group() != "Subcommands") {
out << app->get_group() << ':';
}
}

out << make_description(app);
out << make_usage(app, name);
out << make_positionals(app);
out << make_groups(app, mode);
out << make_subcommands(app, mode);
out << '\n' << make_footer(app);

return out.str();
}

inline std::string Formatter::make_subcommands(const App *app, AppFormatMode mode) const {
std::stringstream out;

std::vector<const App *> subcommands = app->get_subcommands({});

std::vector<std::string> subcmd_groups_seen;
for(const App *com : subcommands) {
if(com->get_name().empty()) {
if(!com->get_group().empty()) {
out << make_expanded(com);
}
continue;
}
std::string group_key = com->get_group();
if(!group_key.empty() &&
std::find_if(subcmd_groups_seen.begin(), subcmd_groups_seen.end(), [&group_key](std::string a) {
return detail::to_lower(a) == detail::to_lower(group_key);
}) == subcmd_groups_seen.end())
subcmd_groups_seen.push_back(group_key);
}

for(const std::string &group : subcmd_groups_seen) {
out << "\n" << group << ":\n";
std::vector<const App *> subcommands_group = app->get_subcommands(
[&group](const App *sub_app) { return detail::to_lower(sub_app->get_group()) == detail::to_lower(group); });
for(const App *new_com : subcommands_group) {
if(new_com->get_name().empty())
continue;
if(mode != AppFormatMode::All) {
out << make_subcommand(new_com);
} else {
out << new_com->help(new_com->get_name(), AppFormatMode::Sub);
out << "\n";
}
}
}

return out.str();
}

inline std::string Formatter::make_subcommand(const App *sub) const {
std::stringstream out;
detail::format_help(out, sub->get_name(), sub->get_description(), column_width_);
return out.str();
}

inline std::string Formatter::make_expanded(const App *sub) const {
std::stringstream out;
out << sub->get_display_name() << "\n";

out << make_description(sub);
out << make_positionals(sub);
out << make_groups(sub, AppFormatMode::Sub);
out << make_subcommands(sub, AppFormatMode::Sub);

std::string tmp = detail::find_and_replace(out.str(), "\n\n", "\n");
tmp = tmp.substr(0, tmp.size() - 1);  

return detail::find_and_replace(tmp, "\n", "\n  ") + "\n";
}

inline std::string Formatter::make_option_name(const Option *opt, bool is_positional) const {
if(is_positional)
return opt->get_name(true, false);

return opt->get_name(false, true);
}

inline std::string Formatter::make_option_opts(const Option *opt) const {
std::stringstream out;

if(opt->get_type_size() != 0) {
if(!opt->get_type_name().empty())
out << " " << get_label(opt->get_type_name());
if(!opt->get_default_str().empty())
out << "=" << opt->get_default_str();
if(opt->get_expected_max() == detail::expected_max_vector_size)
out << " ...";
else if(opt->get_expected_min() > 1)
out << " x " << opt->get_expected();

if(opt->get_required())
out << " " << get_label("REQUIRED");
}
if(!opt->get_envname().empty())
out << " (" << get_label("Env") << ":" << opt->get_envname() << ")";
if(!opt->get_needs().empty()) {
out << " " << get_label("Needs") << ":";
for(const Option *op : opt->get_needs())
out << " " << op->get_name();
}
if(!opt->get_excludes().empty()) {
out << " " << get_label("Excludes") << ":";
for(const Option *op : opt->get_excludes())
out << " " << op->get_name();
}
return out.str();
}

inline std::string Formatter::make_option_desc(const Option *opt) const { return opt->get_description(); }

inline std::string Formatter::make_option_usage(const Option *opt) const {
std::stringstream out;
out << make_option_name(opt, true);
if(opt->get_expected_max() >= detail::expected_max_vector_size)
out << "...";
else if(opt->get_expected_max() > 1)
out << "(" << opt->get_expected() << "x)";

return opt->get_required() ? out.str() : "[" + out.str() + "]";
}

}  

