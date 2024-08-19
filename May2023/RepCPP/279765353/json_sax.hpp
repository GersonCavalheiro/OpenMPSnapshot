#pragma once

#include <cassert> 
#include <cstddef>
#include <string> 
#include <utility> 
#include <vector> 

#include <nlohmann/detail/exceptions.hpp>
#include <nlohmann/detail/macro_scope.hpp>

namespace nlohmann
{


template<typename BasicJsonType>
struct json_sax
{
using number_integer_t = typename BasicJsonType::number_integer_t;
using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
using number_float_t = typename BasicJsonType::number_float_t;
using string_t = typename BasicJsonType::string_t;
using binary_t = typename BasicJsonType::binary_t;


virtual bool null() = 0;


virtual bool boolean(bool val) = 0;


virtual bool number_integer(number_integer_t val) = 0;


virtual bool number_unsigned(number_unsigned_t val) = 0;


virtual bool number_float(number_float_t val, const string_t& s) = 0;


virtual bool string(string_t& val) = 0;


virtual bool binary(binary_t& val) = 0;


virtual bool start_object(std::size_t elements) = 0;


virtual bool key(string_t& val) = 0;


virtual bool end_object() = 0;


virtual bool start_array(std::size_t elements) = 0;


virtual bool end_array() = 0;


virtual bool parse_error(std::size_t position,
const std::string& last_token,
const detail::exception& ex) = 0;

virtual ~json_sax() = default;
};


namespace detail
{

template<typename BasicJsonType>
class json_sax_dom_parser
{
public:
using number_integer_t = typename BasicJsonType::number_integer_t;
using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
using number_float_t = typename BasicJsonType::number_float_t;
using string_t = typename BasicJsonType::string_t;
using binary_t = typename BasicJsonType::binary_t;


explicit json_sax_dom_parser(BasicJsonType& r, const bool allow_exceptions_ = true)
: root(r), allow_exceptions(allow_exceptions_)
{}

json_sax_dom_parser(const json_sax_dom_parser&) = delete;
json_sax_dom_parser(json_sax_dom_parser&&) = default;
json_sax_dom_parser& operator=(const json_sax_dom_parser&) = delete;
json_sax_dom_parser& operator=(json_sax_dom_parser&&) = default;
~json_sax_dom_parser() = default;

bool null()
{
handle_value(nullptr);
return true;
}

bool boolean(bool val)
{
handle_value(val);
return true;
}

bool number_integer(number_integer_t val)
{
handle_value(val);
return true;
}

bool number_unsigned(number_unsigned_t val)
{
handle_value(val);
return true;
}

bool number_float(number_float_t val, const string_t& )
{
handle_value(val);
return true;
}

bool string(string_t& val)
{
handle_value(val);
return true;
}

bool binary(binary_t& val)
{
handle_value(std::move(val));
return true;
}

bool start_object(std::size_t len)
{
ref_stack.push_back(handle_value(BasicJsonType::value_t::object));

if (JSON_HEDLEY_UNLIKELY(len != std::size_t(-1) and len > ref_stack.back()->max_size()))
{
JSON_THROW(out_of_range::create(408,
"excessive object size: " + std::to_string(len)));
}

return true;
}

bool key(string_t& val)
{
object_element = &(ref_stack.back()->m_value.object->operator[](val));
return true;
}

bool end_object()
{
ref_stack.pop_back();
return true;
}

bool start_array(std::size_t len)
{
ref_stack.push_back(handle_value(BasicJsonType::value_t::array));

if (JSON_HEDLEY_UNLIKELY(len != std::size_t(-1) and len > ref_stack.back()->max_size()))
{
JSON_THROW(out_of_range::create(408,
"excessive array size: " + std::to_string(len)));
}

return true;
}

bool end_array()
{
ref_stack.pop_back();
return true;
}

bool parse_error(std::size_t , const std::string& ,
const detail::exception& ex)
{
errored = true;
if (allow_exceptions)
{
switch ((ex.id / 100) % 100)
{
case 1:
JSON_THROW(*static_cast<const detail::parse_error*>(&ex));
case 4:
JSON_THROW(*static_cast<const detail::out_of_range*>(&ex));
case 2:
JSON_THROW(*static_cast<const detail::invalid_iterator*>(&ex));
case 3:
JSON_THROW(*static_cast<const detail::type_error*>(&ex));
case 5:
JSON_THROW(*static_cast<const detail::other_error*>(&ex));
default:
assert(false);
}
}
return false;
}

constexpr bool is_errored() const
{
return errored;
}

private:

template<typename Value>
JSON_HEDLEY_RETURNS_NON_NULL
BasicJsonType* handle_value(Value&& v)
{
if (ref_stack.empty())
{
root = BasicJsonType(std::forward<Value>(v));
return &root;
}

assert(ref_stack.back()->is_array() or ref_stack.back()->is_object());

if (ref_stack.back()->is_array())
{
ref_stack.back()->m_value.array->emplace_back(std::forward<Value>(v));
return &(ref_stack.back()->m_value.array->back());
}

assert(ref_stack.back()->is_object());
assert(object_element);
*object_element = BasicJsonType(std::forward<Value>(v));
return object_element;
}

BasicJsonType& root;
std::vector<BasicJsonType*> ref_stack {};
BasicJsonType* object_element = nullptr;
bool errored = false;
const bool allow_exceptions = true;
};

template<typename BasicJsonType>
class json_sax_dom_callback_parser
{
public:
using number_integer_t = typename BasicJsonType::number_integer_t;
using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
using number_float_t = typename BasicJsonType::number_float_t;
using string_t = typename BasicJsonType::string_t;
using binary_t = typename BasicJsonType::binary_t;
using parser_callback_t = typename BasicJsonType::parser_callback_t;
using parse_event_t = typename BasicJsonType::parse_event_t;

json_sax_dom_callback_parser(BasicJsonType& r,
const parser_callback_t cb,
const bool allow_exceptions_ = true)
: root(r), callback(cb), allow_exceptions(allow_exceptions_)
{
keep_stack.push_back(true);
}

json_sax_dom_callback_parser(const json_sax_dom_callback_parser&) = delete;
json_sax_dom_callback_parser(json_sax_dom_callback_parser&&) = default;
json_sax_dom_callback_parser& operator=(const json_sax_dom_callback_parser&) = delete;
json_sax_dom_callback_parser& operator=(json_sax_dom_callback_parser&&) = default;
~json_sax_dom_callback_parser() = default;

bool null()
{
handle_value(nullptr);
return true;
}

bool boolean(bool val)
{
handle_value(val);
return true;
}

bool number_integer(number_integer_t val)
{
handle_value(val);
return true;
}

bool number_unsigned(number_unsigned_t val)
{
handle_value(val);
return true;
}

bool number_float(number_float_t val, const string_t& )
{
handle_value(val);
return true;
}

bool string(string_t& val)
{
handle_value(val);
return true;
}

bool binary(binary_t& val)
{
handle_value(std::move(val));
return true;
}

bool start_object(std::size_t len)
{
const bool keep = callback(static_cast<int>(ref_stack.size()), parse_event_t::object_start, discarded);
keep_stack.push_back(keep);

auto val = handle_value(BasicJsonType::value_t::object, true);
ref_stack.push_back(val.second);

if (ref_stack.back() and JSON_HEDLEY_UNLIKELY(len != std::size_t(-1) and len > ref_stack.back()->max_size()))
{
JSON_THROW(out_of_range::create(408, "excessive object size: " + std::to_string(len)));
}

return true;
}

bool key(string_t& val)
{
BasicJsonType k = BasicJsonType(val);

const bool keep = callback(static_cast<int>(ref_stack.size()), parse_event_t::key, k);
key_keep_stack.push_back(keep);

if (keep and ref_stack.back())
{
object_element = &(ref_stack.back()->m_value.object->operator[](val) = discarded);
}

return true;
}

bool end_object()
{
if (ref_stack.back() and not callback(static_cast<int>(ref_stack.size()) - 1, parse_event_t::object_end, *ref_stack.back()))
{
*ref_stack.back() = discarded;
}

assert(not ref_stack.empty());
assert(not keep_stack.empty());
ref_stack.pop_back();
keep_stack.pop_back();

if (not ref_stack.empty() and ref_stack.back() and ref_stack.back()->is_structured())
{
for (auto it = ref_stack.back()->begin(); it != ref_stack.back()->end(); ++it)
{
if (it->is_discarded())
{
ref_stack.back()->erase(it);
break;
}
}
}

return true;
}

bool start_array(std::size_t len)
{
const bool keep = callback(static_cast<int>(ref_stack.size()), parse_event_t::array_start, discarded);
keep_stack.push_back(keep);

auto val = handle_value(BasicJsonType::value_t::array, true);
ref_stack.push_back(val.second);

if (ref_stack.back() and JSON_HEDLEY_UNLIKELY(len != std::size_t(-1) and len > ref_stack.back()->max_size()))
{
JSON_THROW(out_of_range::create(408, "excessive array size: " + std::to_string(len)));
}

return true;
}

bool end_array()
{
bool keep = true;

if (ref_stack.back())
{
keep = callback(static_cast<int>(ref_stack.size()) - 1, parse_event_t::array_end, *ref_stack.back());
if (not keep)
{
*ref_stack.back() = discarded;
}
}

assert(not ref_stack.empty());
assert(not keep_stack.empty());
ref_stack.pop_back();
keep_stack.pop_back();

if (not keep and not ref_stack.empty() and ref_stack.back()->is_array())
{
ref_stack.back()->m_value.array->pop_back();
}

return true;
}

bool parse_error(std::size_t , const std::string& ,
const detail::exception& ex)
{
errored = true;
if (allow_exceptions)
{
switch ((ex.id / 100) % 100)
{
case 1:
JSON_THROW(*static_cast<const detail::parse_error*>(&ex));
case 4:
JSON_THROW(*static_cast<const detail::out_of_range*>(&ex));
case 2:
JSON_THROW(*static_cast<const detail::invalid_iterator*>(&ex));
case 3:
JSON_THROW(*static_cast<const detail::type_error*>(&ex));
case 5:
JSON_THROW(*static_cast<const detail::other_error*>(&ex));
default:
assert(false);
}
}
return false;
}

constexpr bool is_errored() const
{
return errored;
}

private:

template<typename Value>
std::pair<bool, BasicJsonType*> handle_value(Value&& v, const bool skip_callback = false)
{
assert(not keep_stack.empty());

if (not keep_stack.back())
{
return {false, nullptr};
}

auto value = BasicJsonType(std::forward<Value>(v));

const bool keep = skip_callback or callback(static_cast<int>(ref_stack.size()), parse_event_t::value, value);

if (not keep)
{
return {false, nullptr};
}

if (ref_stack.empty())
{
root = std::move(value);
return {true, &root};
}

if (not ref_stack.back())
{
return {false, nullptr};
}

assert(ref_stack.back()->is_array() or ref_stack.back()->is_object());

if (ref_stack.back()->is_array())
{
ref_stack.back()->m_value.array->push_back(std::move(value));
return {true, &(ref_stack.back()->m_value.array->back())};
}

assert(ref_stack.back()->is_object());
assert(not key_keep_stack.empty());
const bool store_element = key_keep_stack.back();
key_keep_stack.pop_back();

if (not store_element)
{
return {false, nullptr};
}

assert(object_element);
*object_element = std::move(value);
return {true, object_element};
}

BasicJsonType& root;
std::vector<BasicJsonType*> ref_stack {};
std::vector<bool> keep_stack {};
std::vector<bool> key_keep_stack {};
BasicJsonType* object_element = nullptr;
bool errored = false;
const parser_callback_t callback = nullptr;
const bool allow_exceptions = true;
BasicJsonType discarded = BasicJsonType::value_t::discarded;
};

template<typename BasicJsonType>
class json_sax_acceptor
{
public:
using number_integer_t = typename BasicJsonType::number_integer_t;
using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
using number_float_t = typename BasicJsonType::number_float_t;
using string_t = typename BasicJsonType::string_t;
using binary_t = typename BasicJsonType::binary_t;

bool null()
{
return true;
}

bool boolean(bool )
{
return true;
}

bool number_integer(number_integer_t )
{
return true;
}

bool number_unsigned(number_unsigned_t )
{
return true;
}

bool number_float(number_float_t , const string_t& )
{
return true;
}

bool string(string_t& )
{
return true;
}

bool binary(binary_t& )
{
return true;
}

bool start_object(std::size_t  = std::size_t(-1))
{
return true;
}

bool key(string_t& )
{
return true;
}

bool end_object()
{
return true;
}

bool start_array(std::size_t  = std::size_t(-1))
{
return true;
}

bool end_array()
{
return true;
}

bool parse_error(std::size_t , const std::string& , const detail::exception& )
{
return false;
}
};
}  

}  
