

#pragma once

#include <cinttypes>
#include <cstdio>
#include <iomanip>
#include <ostream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static const std::regex program_options_regex{"[, \\f\\n\\r\\t\\v]+",
std::regex_constants::optimize};

using variables_map = std::set<std::string>;

class value_base
{
protected:
bool m_has_default = false;

public:
bool has_default() const
{
return m_has_default;
}

virtual ~value_base() = default;
};

template <typename T>
class value : public value_base
{
T* m_var; 

public:
explicit value(T* var)
: m_var(var)
{
}

T* get_ptr() const
{
return m_var;
}

value* operator->()
{
return this;
}

value& default_value(T val)
{
*m_var        = std::move(val);
m_has_default = true;
return *this;
}
};

using bool_switch = value<bool>;

class options_description
{
class desc_option
{
std::string m_opts;
value_base* m_val;
std::string m_desc;

public:
template <typename T>
desc_option(std::string opts, value<T> val, std::string desc)
: m_opts(std::move(opts))
, m_val(new auto(std::move(val)))
, m_desc(std::move(desc))
{
}

desc_option(std::string opts, std::string desc)
: m_opts(std::move(opts))
, m_val(nullptr)
, m_desc(std::move(desc))
{
}

desc_option(const desc_option&) = delete;

desc_option(desc_option&& other)
: m_opts(std::move(other.m_opts))
, m_val(other.m_val)
, m_desc(std::move(other.m_desc))
{
other.m_val = nullptr;
}

~desc_option()
{
delete m_val;
}

const std::string& get_opts() const
{
return m_opts;
}

const value_base* get_val() const
{
return m_val;
}

const std::string& get_desc() const
{
return m_desc;
}

void set_val(int& argc, char**& argv) const
{
bool match = false;
if(dynamic_cast<value<int32_t>*>(m_val))
{
auto* val = dynamic_cast<value<int32_t>*>(m_val)->get_ptr();
match     = argc && sscanf(*argv, "%" SCNd32, val) == 1;
}
else if(dynamic_cast<value<uint32_t>*>(m_val))
{
auto* val = dynamic_cast<value<uint32_t>*>(m_val)->get_ptr();
match     = argc && sscanf(*argv, "%" SCNu32, val) == 1;
}
else if(dynamic_cast<value<int64_t>*>(m_val))
{
auto* val = dynamic_cast<value<int64_t>*>(m_val)->get_ptr();
match     = argc && sscanf(*argv, "%" SCNd64, val) == 1;
}
else if(dynamic_cast<value<uint64_t>*>(m_val))
{
auto* val = dynamic_cast<value<uint64_t>*>(m_val)->get_ptr();
match     = argc && sscanf(*argv, "%" SCNu64, val) == 1;
}
else if(dynamic_cast<value<float>*>(m_val))
{
auto* val = dynamic_cast<value<float>*>(m_val)->get_ptr();
match     = argc && sscanf(*argv, "%f", val) == 1;
}
else if(dynamic_cast<value<double>*>(m_val))
{
auto* val = dynamic_cast<value<double>*>(m_val)->get_ptr();
match     = argc && sscanf(*argv, "%lf", val) == 1;
}
else if(dynamic_cast<value<char>*>(m_val))
{
auto* val = dynamic_cast<value<char>*>(m_val)->get_ptr();
match     = argc && sscanf(*argv, " %c", val) == 1;
}
else if(dynamic_cast<value<bool>*>(m_val))
{
auto* val = dynamic_cast<value<bool>*>(m_val)->get_ptr();
*val      = true;
return;
}
else if(dynamic_cast<value<std::string>*>(m_val))
{
if(argc)
{
*dynamic_cast<value<std::string>*>(m_val)->get_ptr() = *argv;
match                                                = true;
}
}
else
{
throw std::logic_error("Internal error: Unsupported data type");
}

if(!match)
throw std::invalid_argument(argc ? *argv : "Missing required argument");

++argv;
--argc;
}
};

std::string              m_desc;
std::vector<desc_option> m_optlist;

class desc_optionlist
{
std::vector<desc_option>& m_list;

public:
explicit desc_optionlist(std::vector<desc_option>& list)
: m_list(list)
{
}

template <typename... Ts>
desc_optionlist operator()(Ts&&... arg)
{
m_list.push_back(desc_option(std::forward<Ts>(arg)...));
return *this;
}
};

public:
explicit options_description(std::string desc)
: m_desc(std::move(desc))
{
}

desc_optionlist add_options() &
{
return desc_optionlist(m_optlist);
}

void parse_option(int& argc, char**& argv, variables_map& vm, bool ignoreUnknown = false) const
{
for(const auto& opt : m_optlist)
{
std::string canonical_name;

for(std::sregex_token_iterator tok{
opt.get_opts().begin(), opt.get_opts().end(), program_options_regex, -1};
tok != std::sregex_token_iterator();
++tok)
{
if(!canonical_name.length())
canonical_name = tok->str();

const char* prefix = tok->length() == 1 ? "-" : "--";

if(*argv == prefix + tok->str())
{
++argv;
--argc;

if(opt.get_val())
opt.set_val(argc, argv);
else
vm.insert(canonical_name);
return; 
}
}
}

if(ignoreUnknown)
{
++argv;
--argc;
}
else
{
throw std::invalid_argument(*argv);
}
}

friend std::ostream& operator<<(std::ostream& os, const options_description& d)
{
for(const auto& opt : d.m_optlist)
{
bool               first = true;
const char*        delim = "";
std::ostringstream left;

for(std::sregex_token_iterator tok{opt.get_opts().begin(),
opt.get_opts().end(),
program_options_regex,
-1};
tok != std::sregex_token_iterator();
++tok, first = false, delim = " ")
{
const char* prefix = tok->length() == 1 ? "-" : "--";
left << delim << (first ? "" : "[ ") << prefix << tok->str() << (first ? "" : " ]");
}

const value_base* val = opt.get_val();
if(val && !dynamic_cast<const value<bool>*>(val))
{
left << " arg";
if(val->has_default())
{
left << " (=";
if(dynamic_cast<const value<int32_t>*>(val))
left << *dynamic_cast<const value<int32_t>*>(val)->get_ptr();
else if(dynamic_cast<const value<uint32_t>*>(val))
left << *dynamic_cast<const value<uint32_t>*>(val)->get_ptr();
else if(dynamic_cast<const value<int64_t>*>(val))
left << *dynamic_cast<const value<int64_t>*>(val)->get_ptr();
else if(dynamic_cast<const value<uint64_t>*>(val))
left << *dynamic_cast<const value<uint64_t>*>(val)->get_ptr();
else if(dynamic_cast<const value<float>*>(val))
left << *dynamic_cast<const value<float>*>(val)->get_ptr();
else if(dynamic_cast<const value<double>*>(val))
left << *dynamic_cast<const value<double>*>(val)->get_ptr();
else if(dynamic_cast<const value<char>*>(val))
left << *dynamic_cast<const value<char>*>(val)->get_ptr();
else if(dynamic_cast<const value<std::string>*>(val))
left << *dynamic_cast<const value<std::string>*>(val)->get_ptr();
else
throw std::logic_error("Internal error: Unsupported data type");
left << ")";
}
}
os << std::setw(36) << std::left << left.str() << " " << opt.get_desc() << "\n\n";
}
return os << std::flush;
}
};

class parse_command_line
{
variables_map m_vm;

public:
parse_command_line(int                        argc,
char**                     argv,
const options_description& desc,
bool                       ignoreUnknown = false)
{
++argv; 
--argc;
while(argc)
desc.parse_option(argc, argv, m_vm, ignoreUnknown);
}

friend void store(const parse_command_line& p, variables_map& vm)
{
vm = p.m_vm;
}

friend void store(parse_command_line&& p, variables_map& vm)
{
vm = std::move(p.m_vm);
}
};

inline void notify(const variables_map&) {}
