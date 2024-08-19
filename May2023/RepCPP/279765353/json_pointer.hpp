#pragma once

#include <algorithm> 
#include <cassert> 
#include <cctype> 
#include <numeric> 
#include <string> 
#include <utility> 
#include <vector> 

#include <nlohmann/detail/exceptions.hpp>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/detail/value_t.hpp>

namespace nlohmann
{
template<typename BasicJsonType>
class json_pointer
{
NLOHMANN_BASIC_JSON_TPL_DECLARATION
friend class basic_json;

public:

explicit json_pointer(const std::string& s = "")
: reference_tokens(split(s))
{}


std::string to_string() const
{
return std::accumulate(reference_tokens.begin(), reference_tokens.end(),
std::string{},
[](const std::string & a, const std::string & b)
{
return a + "/" + escape(b);
});
}

operator std::string() const
{
return to_string();
}


json_pointer& operator/=(const json_pointer& ptr)
{
reference_tokens.insert(reference_tokens.end(),
ptr.reference_tokens.begin(),
ptr.reference_tokens.end());
return *this;
}


json_pointer& operator/=(std::string token)
{
push_back(std::move(token));
return *this;
}


json_pointer& operator/=(std::size_t array_idx)
{
return *this /= std::to_string(array_idx);
}


friend json_pointer operator/(const json_pointer& lhs,
const json_pointer& rhs)
{
return json_pointer(lhs) /= rhs;
}


friend json_pointer operator/(const json_pointer& ptr, std::string token)
{
return json_pointer(ptr) /= std::move(token);
}


friend json_pointer operator/(const json_pointer& ptr, std::size_t array_idx)
{
return json_pointer(ptr) /= array_idx;
}


json_pointer parent_pointer() const
{
if (empty())
{
return *this;
}

json_pointer res = *this;
res.pop_back();
return res;
}


void pop_back()
{
if (JSON_HEDLEY_UNLIKELY(empty()))
{
JSON_THROW(detail::out_of_range::create(405, "JSON pointer has no parent"));
}

reference_tokens.pop_back();
}


const std::string& back() const
{
if (JSON_HEDLEY_UNLIKELY(empty()))
{
JSON_THROW(detail::out_of_range::create(405, "JSON pointer has no parent"));
}

return reference_tokens.back();
}


void push_back(const std::string& token)
{
reference_tokens.push_back(token);
}

void push_back(std::string&& token)
{
reference_tokens.push_back(std::move(token));
}


bool empty() const noexcept
{
return reference_tokens.empty();
}

private:

static int array_index(const std::string& s)
{
if (JSON_HEDLEY_UNLIKELY(s.size() > 1 and s[0] == '0'))
{
JSON_THROW(detail::parse_error::create(106, 0,
"array index '" + s +
"' must not begin with '0'"));
}

if (JSON_HEDLEY_UNLIKELY(s.size() > 1 and not (s[0] >= '1' and s[0] <= '9')))
{
JSON_THROW(detail::parse_error::create(109, 0, "array index '" + s + "' is not a number"));
}

std::size_t processed_chars = 0;
int res = 0;
JSON_TRY
{
res = std::stoi(s, &processed_chars);
}
JSON_CATCH(std::out_of_range&)
{
JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + s + "'"));
}

if (JSON_HEDLEY_UNLIKELY(processed_chars != s.size()))
{
JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + s + "'"));
}

return res;
}

json_pointer top() const
{
if (JSON_HEDLEY_UNLIKELY(empty()))
{
JSON_THROW(detail::out_of_range::create(405, "JSON pointer has no parent"));
}

json_pointer result = *this;
result.reference_tokens = {reference_tokens[0]};
return result;
}


BasicJsonType& get_and_create(BasicJsonType& j) const
{
using size_type = typename BasicJsonType::size_type;
auto result = &j;

for (const auto& reference_token : reference_tokens)
{
switch (result->type())
{
case detail::value_t::null:
{
if (reference_token == "0")
{
result = &result->operator[](0);
}
else
{
result = &result->operator[](reference_token);
}
break;
}

case detail::value_t::object:
{
result = &result->operator[](reference_token);
break;
}

case detail::value_t::array:
{
result = &result->operator[](static_cast<size_type>(array_index(reference_token)));
break;
}


default:
JSON_THROW(detail::type_error::create(313, "invalid value to unflatten"));
}
}

return *result;
}


BasicJsonType& get_unchecked(BasicJsonType* ptr) const
{
using size_type = typename BasicJsonType::size_type;
for (const auto& reference_token : reference_tokens)
{
if (ptr->is_null())
{
const bool nums =
std::all_of(reference_token.begin(), reference_token.end(),
[](const unsigned char x)
{
return std::isdigit(x);
});

*ptr = (nums or reference_token == "-")
? detail::value_t::array
: detail::value_t::object;
}

switch (ptr->type())
{
case detail::value_t::object:
{
ptr = &ptr->operator[](reference_token);
break;
}

case detail::value_t::array:
{
if (reference_token == "-")
{
ptr = &ptr->operator[](ptr->m_value.array->size());
}
else
{
ptr = &ptr->operator[](
static_cast<size_type>(array_index(reference_token)));
}
break;
}

default:
JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + reference_token + "'"));
}
}

return *ptr;
}


BasicJsonType& get_checked(BasicJsonType* ptr) const
{
using size_type = typename BasicJsonType::size_type;
for (const auto& reference_token : reference_tokens)
{
switch (ptr->type())
{
case detail::value_t::object:
{
ptr = &ptr->at(reference_token);
break;
}

case detail::value_t::array:
{
if (JSON_HEDLEY_UNLIKELY(reference_token == "-"))
{
JSON_THROW(detail::out_of_range::create(402,
"array index '-' (" + std::to_string(ptr->m_value.array->size()) +
") is out of range"));
}

ptr = &ptr->at(static_cast<size_type>(array_index(reference_token)));
break;
}

default:
JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + reference_token + "'"));
}
}

return *ptr;
}


const BasicJsonType& get_unchecked(const BasicJsonType* ptr) const
{
using size_type = typename BasicJsonType::size_type;
for (const auto& reference_token : reference_tokens)
{
switch (ptr->type())
{
case detail::value_t::object:
{
ptr = &ptr->operator[](reference_token);
break;
}

case detail::value_t::array:
{
if (JSON_HEDLEY_UNLIKELY(reference_token == "-"))
{
JSON_THROW(detail::out_of_range::create(402,
"array index '-' (" + std::to_string(ptr->m_value.array->size()) +
") is out of range"));
}

ptr = &ptr->operator[](
static_cast<size_type>(array_index(reference_token)));
break;
}

default:
JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + reference_token + "'"));
}
}

return *ptr;
}


const BasicJsonType& get_checked(const BasicJsonType* ptr) const
{
using size_type = typename BasicJsonType::size_type;
for (const auto& reference_token : reference_tokens)
{
switch (ptr->type())
{
case detail::value_t::object:
{
ptr = &ptr->at(reference_token);
break;
}

case detail::value_t::array:
{
if (JSON_HEDLEY_UNLIKELY(reference_token == "-"))
{
JSON_THROW(detail::out_of_range::create(402,
"array index '-' (" + std::to_string(ptr->m_value.array->size()) +
") is out of range"));
}

ptr = &ptr->at(static_cast<size_type>(array_index(reference_token)));
break;
}

default:
JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + reference_token + "'"));
}
}

return *ptr;
}


bool contains(const BasicJsonType* ptr) const
{
using size_type = typename BasicJsonType::size_type;
for (const auto& reference_token : reference_tokens)
{
switch (ptr->type())
{
case detail::value_t::object:
{
if (not ptr->contains(reference_token))
{
return false;
}

ptr = &ptr->operator[](reference_token);
break;
}

case detail::value_t::array:
{
if (JSON_HEDLEY_UNLIKELY(reference_token == "-"))
{
return false;
}
if (JSON_HEDLEY_UNLIKELY(reference_token.size() == 1 and not ("0" <= reference_token and reference_token <= "9")))
{
return false;
}
if (JSON_HEDLEY_UNLIKELY(reference_token.size() > 1))
{
if (JSON_HEDLEY_UNLIKELY(not ('1' <= reference_token[0] and reference_token[0] <= '9')))
{
return false;
}
for (std::size_t i = 1; i < reference_token.size(); i++)
{
if (JSON_HEDLEY_UNLIKELY(not ('0' <= reference_token[i] and reference_token[i] <= '9')))
{
return false;
}
}
}

const auto idx = static_cast<size_type>(array_index(reference_token));
if (idx >= ptr->size())
{
return false;
}

ptr = &ptr->operator[](idx);
break;
}

default:
{
return false;
}
}
}

return true;
}


static std::vector<std::string> split(const std::string& reference_string)
{
std::vector<std::string> result;

if (reference_string.empty())
{
return result;
}

if (JSON_HEDLEY_UNLIKELY(reference_string[0] != '/'))
{
JSON_THROW(detail::parse_error::create(107, 1,
"JSON pointer must be empty or begin with '/' - was: '" +
reference_string + "'"));
}

for (
std::size_t slash = reference_string.find_first_of('/', 1),
start = 1;
start != 0;
start = (slash == std::string::npos) ? 0 : slash + 1,
slash = reference_string.find_first_of('/', start))
{
auto reference_token = reference_string.substr(start, slash - start);

for (std::size_t pos = reference_token.find_first_of('~');
pos != std::string::npos;
pos = reference_token.find_first_of('~', pos + 1))
{
assert(reference_token[pos] == '~');

if (JSON_HEDLEY_UNLIKELY(pos == reference_token.size() - 1 or
(reference_token[pos + 1] != '0' and
reference_token[pos + 1] != '1')))
{
JSON_THROW(detail::parse_error::create(108, 0, "escape character '~' must be followed with '0' or '1'"));
}
}

unescape(reference_token);
result.push_back(reference_token);
}

return result;
}


static void replace_substring(std::string& s, const std::string& f,
const std::string& t)
{
assert(not f.empty());
for (auto pos = s.find(f);                
pos != std::string::npos;         
s.replace(pos, f.size(), t),      
pos = s.find(f, pos + t.size()))  
{}
}

static std::string escape(std::string s)
{
replace_substring(s, "~", "~0");
replace_substring(s, "/", "~1");
return s;
}

static void unescape(std::string& s)
{
replace_substring(s, "~1", "/");
replace_substring(s, "~0", "~");
}


static void flatten(const std::string& reference_string,
const BasicJsonType& value,
BasicJsonType& result)
{
switch (value.type())
{
case detail::value_t::array:
{
if (value.m_value.array->empty())
{
result[reference_string] = nullptr;
}
else
{
for (std::size_t i = 0; i < value.m_value.array->size(); ++i)
{
flatten(reference_string + "/" + std::to_string(i),
value.m_value.array->operator[](i), result);
}
}
break;
}

case detail::value_t::object:
{
if (value.m_value.object->empty())
{
result[reference_string] = nullptr;
}
else
{
for (const auto& element : *value.m_value.object)
{
flatten(reference_string + "/" + escape(element.first), element.second, result);
}
}
break;
}

default:
{
result[reference_string] = value;
break;
}
}
}


static BasicJsonType
unflatten(const BasicJsonType& value)
{
if (JSON_HEDLEY_UNLIKELY(not value.is_object()))
{
JSON_THROW(detail::type_error::create(314, "only objects can be unflattened"));
}

BasicJsonType result;

for (const auto& element : *value.m_value.object)
{
if (JSON_HEDLEY_UNLIKELY(not element.second.is_primitive()))
{
JSON_THROW(detail::type_error::create(315, "values in object must be primitive"));
}

json_pointer(element.first).get_and_create(result) = element.second;
}

return result;
}


friend bool operator==(json_pointer const& lhs,
json_pointer const& rhs) noexcept
{
return lhs.reference_tokens == rhs.reference_tokens;
}


friend bool operator!=(json_pointer const& lhs,
json_pointer const& rhs) noexcept
{
return not (lhs == rhs);
}

std::vector<std::string> reference_tokens;
};
}  
