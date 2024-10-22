
#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "StringTools.hpp"

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
