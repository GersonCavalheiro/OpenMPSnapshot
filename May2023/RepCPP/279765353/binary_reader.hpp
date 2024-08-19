#pragma once

#include <algorithm> 
#include <array> 
#include <cassert> 
#include <cmath> 
#include <cstddef> 
#include <cstdint> 
#include <cstdio> 
#include <cstring> 
#include <iterator> 
#include <limits> 
#include <string> 
#include <utility> 

#include <nlohmann/detail/exceptions.hpp>
#include <nlohmann/detail/input/input_adapters.hpp>
#include <nlohmann/detail/input/json_sax.hpp>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/detail/meta/is_sax.hpp>
#include <nlohmann/detail/value_t.hpp>

namespace nlohmann
{
namespace detail
{


static inline bool little_endianess(int num = 1) noexcept
{
return *reinterpret_cast<char*>(&num) == 1;
}




template<typename BasicJsonType, typename InputAdapterType, typename SAX = json_sax_dom_parser<BasicJsonType>>
class binary_reader
{
using number_integer_t = typename BasicJsonType::number_integer_t;
using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
using number_float_t = typename BasicJsonType::number_float_t;
using string_t = typename BasicJsonType::string_t;
using binary_t = typename BasicJsonType::binary_t;
using json_sax_t = SAX;
using char_type = typename InputAdapterType::char_type;
using char_int_type = typename std::char_traits<char_type>::int_type;

public:

explicit binary_reader(InputAdapterType&& adapter) : ia(std::move(adapter))
{
(void)detail::is_sax_static_asserts<SAX, BasicJsonType> {};
}

binary_reader(const binary_reader&) = delete;
binary_reader(binary_reader&&) = default;
binary_reader& operator=(const binary_reader&) = delete;
binary_reader& operator=(binary_reader&&) = default;
~binary_reader() = default;


JSON_HEDLEY_NON_NULL(3)
bool sax_parse(const input_format_t format,
json_sax_t* sax_,
const bool strict = true)
{
sax = sax_;
bool result = false;

switch (format)
{
case input_format_t::bson:
result = parse_bson_internal();
break;

case input_format_t::cbor:
result = parse_cbor_internal();
break;

case input_format_t::msgpack:
result = parse_msgpack_internal();
break;

case input_format_t::ubjson:
result = parse_ubjson_internal();
break;

default:            
assert(false);  
}

if (result and strict)
{
if (format == input_format_t::ubjson)
{
get_ignore_noop();
}
else
{
get();
}

if (JSON_HEDLEY_UNLIKELY(current != std::char_traits<char_type>::eof()))
{
return sax->parse_error(chars_read, get_token_string(),
parse_error::create(110, chars_read, exception_message(format, "expected end of input; last byte: 0x" + get_token_string(), "value")));
}
}

return result;
}

private:


bool parse_bson_internal()
{
std::int32_t document_size;
get_number<std::int32_t, true>(input_format_t::bson, document_size);

if (JSON_HEDLEY_UNLIKELY(not sax->start_object(std::size_t(-1))))
{
return false;
}

if (JSON_HEDLEY_UNLIKELY(not parse_bson_element_list(false)))
{
return false;
}

return sax->end_object();
}


bool get_bson_cstr(string_t& result)
{
auto out = std::back_inserter(result);
while (true)
{
get();
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::bson, "cstring")))
{
return false;
}
if (current == 0x00)
{
return true;
}
*out++ = static_cast<typename string_t::value_type>(current);
}

return true;
}


template<typename NumberType>
bool get_bson_string(const NumberType len, string_t& result)
{
if (JSON_HEDLEY_UNLIKELY(len < 1))
{
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, exception_message(input_format_t::bson, "string length must be at least 1, is " + std::to_string(len), "string")));
}

return get_string(input_format_t::bson, len - static_cast<NumberType>(1), result) and get() != std::char_traits<char_type>::eof();
}


template<typename NumberType>
bool get_bson_binary(const NumberType len, binary_t& result)
{
if (JSON_HEDLEY_UNLIKELY(len < 0))
{
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, exception_message(input_format_t::bson, "byte array length cannot be negative, is " + std::to_string(len), "binary")));
}

std::uint8_t subtype;
get_number<std::uint8_t>(input_format_t::bson, subtype);
result.set_subtype(subtype);

return get_binary(input_format_t::bson, len, result);
}


bool parse_bson_element_internal(const char_int_type element_type,
const std::size_t element_type_parse_position)
{
switch (element_type)
{
case 0x01: 
{
double number;
return get_number<double, true>(input_format_t::bson, number) and sax->number_float(static_cast<number_float_t>(number), "");
}

case 0x02: 
{
std::int32_t len;
string_t value;
return get_number<std::int32_t, true>(input_format_t::bson, len) and get_bson_string(len, value) and sax->string(value);
}

case 0x03: 
{
return parse_bson_internal();
}

case 0x04: 
{
return parse_bson_array();
}

case 0x05: 
{
std::int32_t len;
binary_t value;
return get_number<std::int32_t, true>(input_format_t::bson, len) and get_bson_binary(len, value) and sax->binary(value);
}

case 0x08: 
{
return sax->boolean(get() != 0);
}

case 0x0A: 
{
return sax->null();
}

case 0x10: 
{
std::int32_t value;
return get_number<std::int32_t, true>(input_format_t::bson, value) and sax->number_integer(value);
}

case 0x12: 
{
std::int64_t value;
return get_number<std::int64_t, true>(input_format_t::bson, value) and sax->number_integer(value);
}

default: 
{
std::array<char, 3> cr{{}};
(std::snprintf)(cr.data(), cr.size(), "%.2hhX", static_cast<unsigned char>(element_type));
return sax->parse_error(element_type_parse_position, std::string(cr.data()), parse_error::create(114, element_type_parse_position, "Unsupported BSON record type 0x" + std::string(cr.data())));
}
}
}


bool parse_bson_element_list(const bool is_array)
{
string_t key;

while (auto element_type = get())
{
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::bson, "element list")))
{
return false;
}

const std::size_t element_type_parse_position = chars_read;
if (JSON_HEDLEY_UNLIKELY(not get_bson_cstr(key)))
{
return false;
}

if (not is_array and not sax->key(key))
{
return false;
}

if (JSON_HEDLEY_UNLIKELY(not parse_bson_element_internal(element_type, element_type_parse_position)))
{
return false;
}

key.clear();
}

return true;
}


bool parse_bson_array()
{
std::int32_t document_size;
get_number<std::int32_t, true>(input_format_t::bson, document_size);

if (JSON_HEDLEY_UNLIKELY(not sax->start_array(std::size_t(-1))))
{
return false;
}

if (JSON_HEDLEY_UNLIKELY(not parse_bson_element_list(true)))
{
return false;
}

return sax->end_array();
}



bool parse_cbor_internal(const bool get_char = true)
{
switch (get_char ? get() : current)
{
case std::char_traits<char_type>::eof():
return unexpect_eof(input_format_t::cbor, "value");

case 0x00:
case 0x01:
case 0x02:
case 0x03:
case 0x04:
case 0x05:
case 0x06:
case 0x07:
case 0x08:
case 0x09:
case 0x0A:
case 0x0B:
case 0x0C:
case 0x0D:
case 0x0E:
case 0x0F:
case 0x10:
case 0x11:
case 0x12:
case 0x13:
case 0x14:
case 0x15:
case 0x16:
case 0x17:
return sax->number_unsigned(static_cast<number_unsigned_t>(current));

case 0x18: 
{
std::uint8_t number;
return get_number(input_format_t::cbor, number) and sax->number_unsigned(number);
}

case 0x19: 
{
std::uint16_t number;
return get_number(input_format_t::cbor, number) and sax->number_unsigned(number);
}

case 0x1A: 
{
std::uint32_t number;
return get_number(input_format_t::cbor, number) and sax->number_unsigned(number);
}

case 0x1B: 
{
std::uint64_t number;
return get_number(input_format_t::cbor, number) and sax->number_unsigned(number);
}

case 0x20:
case 0x21:
case 0x22:
case 0x23:
case 0x24:
case 0x25:
case 0x26:
case 0x27:
case 0x28:
case 0x29:
case 0x2A:
case 0x2B:
case 0x2C:
case 0x2D:
case 0x2E:
case 0x2F:
case 0x30:
case 0x31:
case 0x32:
case 0x33:
case 0x34:
case 0x35:
case 0x36:
case 0x37:
return sax->number_integer(static_cast<std::int8_t>(0x20 - 1 - current));

case 0x38: 
{
std::uint8_t number;
return get_number(input_format_t::cbor, number) and sax->number_integer(static_cast<number_integer_t>(-1) - number);
}

case 0x39: 
{
std::uint16_t number;
return get_number(input_format_t::cbor, number) and sax->number_integer(static_cast<number_integer_t>(-1) - number);
}

case 0x3A: 
{
std::uint32_t number;
return get_number(input_format_t::cbor, number) and sax->number_integer(static_cast<number_integer_t>(-1) - number);
}

case 0x3B: 
{
std::uint64_t number;
return get_number(input_format_t::cbor, number) and sax->number_integer(static_cast<number_integer_t>(-1)
- static_cast<number_integer_t>(number));
}

case 0x40:
case 0x41:
case 0x42:
case 0x43:
case 0x44:
case 0x45:
case 0x46:
case 0x47:
case 0x48:
case 0x49:
case 0x4A:
case 0x4B:
case 0x4C:
case 0x4D:
case 0x4E:
case 0x4F:
case 0x50:
case 0x51:
case 0x52:
case 0x53:
case 0x54:
case 0x55:
case 0x56:
case 0x57:
case 0x58: 
case 0x59: 
case 0x5A: 
case 0x5B: 
case 0x5F: 
{
binary_t b;
return get_cbor_binary(b) and sax->binary(b);
}

case 0x60:
case 0x61:
case 0x62:
case 0x63:
case 0x64:
case 0x65:
case 0x66:
case 0x67:
case 0x68:
case 0x69:
case 0x6A:
case 0x6B:
case 0x6C:
case 0x6D:
case 0x6E:
case 0x6F:
case 0x70:
case 0x71:
case 0x72:
case 0x73:
case 0x74:
case 0x75:
case 0x76:
case 0x77:
case 0x78: 
case 0x79: 
case 0x7A: 
case 0x7B: 
case 0x7F: 
{
string_t s;
return get_cbor_string(s) and sax->string(s);
}

case 0x80:
case 0x81:
case 0x82:
case 0x83:
case 0x84:
case 0x85:
case 0x86:
case 0x87:
case 0x88:
case 0x89:
case 0x8A:
case 0x8B:
case 0x8C:
case 0x8D:
case 0x8E:
case 0x8F:
case 0x90:
case 0x91:
case 0x92:
case 0x93:
case 0x94:
case 0x95:
case 0x96:
case 0x97:
return get_cbor_array(static_cast<std::size_t>(static_cast<unsigned int>(current) & 0x1Fu));

case 0x98: 
{
std::uint8_t len;
return get_number(input_format_t::cbor, len) and get_cbor_array(static_cast<std::size_t>(len));
}

case 0x99: 
{
std::uint16_t len;
return get_number(input_format_t::cbor, len) and get_cbor_array(static_cast<std::size_t>(len));
}

case 0x9A: 
{
std::uint32_t len;
return get_number(input_format_t::cbor, len) and get_cbor_array(static_cast<std::size_t>(len));
}

case 0x9B: 
{
std::uint64_t len;
return get_number(input_format_t::cbor, len) and get_cbor_array(static_cast<std::size_t>(len));
}

case 0x9F: 
return get_cbor_array(std::size_t(-1));

case 0xA0:
case 0xA1:
case 0xA2:
case 0xA3:
case 0xA4:
case 0xA5:
case 0xA6:
case 0xA7:
case 0xA8:
case 0xA9:
case 0xAA:
case 0xAB:
case 0xAC:
case 0xAD:
case 0xAE:
case 0xAF:
case 0xB0:
case 0xB1:
case 0xB2:
case 0xB3:
case 0xB4:
case 0xB5:
case 0xB6:
case 0xB7:
return get_cbor_object(static_cast<std::size_t>(static_cast<unsigned int>(current) & 0x1Fu));

case 0xB8: 
{
std::uint8_t len;
return get_number(input_format_t::cbor, len) and get_cbor_object(static_cast<std::size_t>(len));
}

case 0xB9: 
{
std::uint16_t len;
return get_number(input_format_t::cbor, len) and get_cbor_object(static_cast<std::size_t>(len));
}

case 0xBA: 
{
std::uint32_t len;
return get_number(input_format_t::cbor, len) and get_cbor_object(static_cast<std::size_t>(len));
}

case 0xBB: 
{
std::uint64_t len;
return get_number(input_format_t::cbor, len) and get_cbor_object(static_cast<std::size_t>(len));
}

case 0xBF: 
return get_cbor_object(std::size_t(-1));

case 0xF4: 
return sax->boolean(false);

case 0xF5: 
return sax->boolean(true);

case 0xF6: 
return sax->null();

case 0xF9: 
{
const auto byte1_raw = get();
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::cbor, "number")))
{
return false;
}
const auto byte2_raw = get();
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::cbor, "number")))
{
return false;
}

const auto byte1 = static_cast<unsigned char>(byte1_raw);
const auto byte2 = static_cast<unsigned char>(byte2_raw);

const auto half = static_cast<unsigned int>((byte1 << 8u) + byte2);
const double val = [&half]
{
const int exp = (half >> 10u) & 0x1Fu;
const unsigned int mant = half & 0x3FFu;
assert(0 <= exp and exp <= 32);
assert(mant <= 1024);
switch (exp)
{
case 0:
return std::ldexp(mant, -24);
case 31:
return (mant == 0)
? std::numeric_limits<double>::infinity()
: std::numeric_limits<double>::quiet_NaN();
default:
return std::ldexp(mant + 1024, exp - 25);
}
}();
return sax->number_float((half & 0x8000u) != 0
? static_cast<number_float_t>(-val)
: static_cast<number_float_t>(val), "");
}

case 0xFA: 
{
float number;
return get_number(input_format_t::cbor, number) and sax->number_float(static_cast<number_float_t>(number), "");
}

case 0xFB: 
{
double number;
return get_number(input_format_t::cbor, number) and sax->number_float(static_cast<number_float_t>(number), "");
}

default: 
{
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, exception_message(input_format_t::cbor, "invalid byte: 0x" + last_token, "value")));
}
}
}


bool get_cbor_string(string_t& result)
{
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::cbor, "string")))
{
return false;
}

switch (current)
{
case 0x60:
case 0x61:
case 0x62:
case 0x63:
case 0x64:
case 0x65:
case 0x66:
case 0x67:
case 0x68:
case 0x69:
case 0x6A:
case 0x6B:
case 0x6C:
case 0x6D:
case 0x6E:
case 0x6F:
case 0x70:
case 0x71:
case 0x72:
case 0x73:
case 0x74:
case 0x75:
case 0x76:
case 0x77:
{
return get_string(input_format_t::cbor, static_cast<unsigned int>(current) & 0x1Fu, result);
}

case 0x78: 
{
std::uint8_t len;
return get_number(input_format_t::cbor, len) and get_string(input_format_t::cbor, len, result);
}

case 0x79: 
{
std::uint16_t len;
return get_number(input_format_t::cbor, len) and get_string(input_format_t::cbor, len, result);
}

case 0x7A: 
{
std::uint32_t len;
return get_number(input_format_t::cbor, len) and get_string(input_format_t::cbor, len, result);
}

case 0x7B: 
{
std::uint64_t len;
return get_number(input_format_t::cbor, len) and get_string(input_format_t::cbor, len, result);
}

case 0x7F: 
{
while (get() != 0xFF)
{
string_t chunk;
if (not get_cbor_string(chunk))
{
return false;
}
result.append(chunk);
}
return true;
}

default:
{
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, exception_message(input_format_t::cbor, "expected length specification (0x60-0x7B) or indefinite string type (0x7F); last byte: 0x" + last_token, "string")));
}
}
}


bool get_cbor_binary(binary_t& result)
{
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::cbor, "binary")))
{
return false;
}

switch (current)
{
case 0x40:
case 0x41:
case 0x42:
case 0x43:
case 0x44:
case 0x45:
case 0x46:
case 0x47:
case 0x48:
case 0x49:
case 0x4A:
case 0x4B:
case 0x4C:
case 0x4D:
case 0x4E:
case 0x4F:
case 0x50:
case 0x51:
case 0x52:
case 0x53:
case 0x54:
case 0x55:
case 0x56:
case 0x57:
{
return get_binary(input_format_t::cbor, static_cast<unsigned int>(current) & 0x1Fu, result);
}

case 0x58: 
{
std::uint8_t len;
return get_number(input_format_t::cbor, len) and
get_binary(input_format_t::cbor, len, result);
}

case 0x59: 
{
std::uint16_t len;
return get_number(input_format_t::cbor, len) and
get_binary(input_format_t::cbor, len, result);
}

case 0x5A: 
{
std::uint32_t len;
return get_number(input_format_t::cbor, len) and
get_binary(input_format_t::cbor, len, result);
}

case 0x5B: 
{
std::uint64_t len;
return get_number(input_format_t::cbor, len) and
get_binary(input_format_t::cbor, len, result);
}

case 0x5F: 
{
while (get() != 0xFF)
{
binary_t chunk;
if (not get_cbor_binary(chunk))
{
return false;
}
result.insert(result.end(), chunk.begin(), chunk.end());
}
return true;
}

default:
{
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, exception_message(input_format_t::cbor, "expected length specification (0x40-0x5B) or indefinite binary array type (0x5F); last byte: 0x" + last_token, "binary")));
}
}
}


bool get_cbor_array(const std::size_t len)
{
if (JSON_HEDLEY_UNLIKELY(not sax->start_array(len)))
{
return false;
}

if (len != std::size_t(-1))
{
for (std::size_t i = 0; i < len; ++i)
{
if (JSON_HEDLEY_UNLIKELY(not parse_cbor_internal()))
{
return false;
}
}
}
else
{
while (get() != 0xFF)
{
if (JSON_HEDLEY_UNLIKELY(not parse_cbor_internal(false)))
{
return false;
}
}
}

return sax->end_array();
}


bool get_cbor_object(const std::size_t len)
{
if (JSON_HEDLEY_UNLIKELY(not sax->start_object(len)))
{
return false;
}

string_t key;
if (len != std::size_t(-1))
{
for (std::size_t i = 0; i < len; ++i)
{
get();
if (JSON_HEDLEY_UNLIKELY(not get_cbor_string(key) or not sax->key(key)))
{
return false;
}

if (JSON_HEDLEY_UNLIKELY(not parse_cbor_internal()))
{
return false;
}
key.clear();
}
}
else
{
while (get() != 0xFF)
{
if (JSON_HEDLEY_UNLIKELY(not get_cbor_string(key) or not sax->key(key)))
{
return false;
}

if (JSON_HEDLEY_UNLIKELY(not parse_cbor_internal()))
{
return false;
}
key.clear();
}
}

return sax->end_object();
}



bool parse_msgpack_internal()
{
switch (get())
{
case std::char_traits<char_type>::eof():
return unexpect_eof(input_format_t::msgpack, "value");

case 0x00:
case 0x01:
case 0x02:
case 0x03:
case 0x04:
case 0x05:
case 0x06:
case 0x07:
case 0x08:
case 0x09:
case 0x0A:
case 0x0B:
case 0x0C:
case 0x0D:
case 0x0E:
case 0x0F:
case 0x10:
case 0x11:
case 0x12:
case 0x13:
case 0x14:
case 0x15:
case 0x16:
case 0x17:
case 0x18:
case 0x19:
case 0x1A:
case 0x1B:
case 0x1C:
case 0x1D:
case 0x1E:
case 0x1F:
case 0x20:
case 0x21:
case 0x22:
case 0x23:
case 0x24:
case 0x25:
case 0x26:
case 0x27:
case 0x28:
case 0x29:
case 0x2A:
case 0x2B:
case 0x2C:
case 0x2D:
case 0x2E:
case 0x2F:
case 0x30:
case 0x31:
case 0x32:
case 0x33:
case 0x34:
case 0x35:
case 0x36:
case 0x37:
case 0x38:
case 0x39:
case 0x3A:
case 0x3B:
case 0x3C:
case 0x3D:
case 0x3E:
case 0x3F:
case 0x40:
case 0x41:
case 0x42:
case 0x43:
case 0x44:
case 0x45:
case 0x46:
case 0x47:
case 0x48:
case 0x49:
case 0x4A:
case 0x4B:
case 0x4C:
case 0x4D:
case 0x4E:
case 0x4F:
case 0x50:
case 0x51:
case 0x52:
case 0x53:
case 0x54:
case 0x55:
case 0x56:
case 0x57:
case 0x58:
case 0x59:
case 0x5A:
case 0x5B:
case 0x5C:
case 0x5D:
case 0x5E:
case 0x5F:
case 0x60:
case 0x61:
case 0x62:
case 0x63:
case 0x64:
case 0x65:
case 0x66:
case 0x67:
case 0x68:
case 0x69:
case 0x6A:
case 0x6B:
case 0x6C:
case 0x6D:
case 0x6E:
case 0x6F:
case 0x70:
case 0x71:
case 0x72:
case 0x73:
case 0x74:
case 0x75:
case 0x76:
case 0x77:
case 0x78:
case 0x79:
case 0x7A:
case 0x7B:
case 0x7C:
case 0x7D:
case 0x7E:
case 0x7F:
return sax->number_unsigned(static_cast<number_unsigned_t>(current));

case 0x80:
case 0x81:
case 0x82:
case 0x83:
case 0x84:
case 0x85:
case 0x86:
case 0x87:
case 0x88:
case 0x89:
case 0x8A:
case 0x8B:
case 0x8C:
case 0x8D:
case 0x8E:
case 0x8F:
return get_msgpack_object(static_cast<std::size_t>(static_cast<unsigned int>(current) & 0x0Fu));

case 0x90:
case 0x91:
case 0x92:
case 0x93:
case 0x94:
case 0x95:
case 0x96:
case 0x97:
case 0x98:
case 0x99:
case 0x9A:
case 0x9B:
case 0x9C:
case 0x9D:
case 0x9E:
case 0x9F:
return get_msgpack_array(static_cast<std::size_t>(static_cast<unsigned int>(current) & 0x0Fu));

case 0xA0:
case 0xA1:
case 0xA2:
case 0xA3:
case 0xA4:
case 0xA5:
case 0xA6:
case 0xA7:
case 0xA8:
case 0xA9:
case 0xAA:
case 0xAB:
case 0xAC:
case 0xAD:
case 0xAE:
case 0xAF:
case 0xB0:
case 0xB1:
case 0xB2:
case 0xB3:
case 0xB4:
case 0xB5:
case 0xB6:
case 0xB7:
case 0xB8:
case 0xB9:
case 0xBA:
case 0xBB:
case 0xBC:
case 0xBD:
case 0xBE:
case 0xBF:
case 0xD9: 
case 0xDA: 
case 0xDB: 
{
string_t s;
return get_msgpack_string(s) and sax->string(s);
}

case 0xC0: 
return sax->null();

case 0xC2: 
return sax->boolean(false);

case 0xC3: 
return sax->boolean(true);

case 0xC4: 
case 0xC5: 
case 0xC6: 
case 0xC7: 
case 0xC8: 
case 0xC9: 
case 0xD4: 
case 0xD5: 
case 0xD6: 
case 0xD7: 
case 0xD8: 
{
binary_t b;
return get_msgpack_binary(b) and sax->binary(b);
}

case 0xCA: 
{
float number;
return get_number(input_format_t::msgpack, number) and sax->number_float(static_cast<number_float_t>(number), "");
}

case 0xCB: 
{
double number;
return get_number(input_format_t::msgpack, number) and sax->number_float(static_cast<number_float_t>(number), "");
}

case 0xCC: 
{
std::uint8_t number;
return get_number(input_format_t::msgpack, number) and sax->number_unsigned(number);
}

case 0xCD: 
{
std::uint16_t number;
return get_number(input_format_t::msgpack, number) and sax->number_unsigned(number);
}

case 0xCE: 
{
std::uint32_t number;
return get_number(input_format_t::msgpack, number) and sax->number_unsigned(number);
}

case 0xCF: 
{
std::uint64_t number;
return get_number(input_format_t::msgpack, number) and sax->number_unsigned(number);
}

case 0xD0: 
{
std::int8_t number;
return get_number(input_format_t::msgpack, number) and sax->number_integer(number);
}

case 0xD1: 
{
std::int16_t number;
return get_number(input_format_t::msgpack, number) and sax->number_integer(number);
}

case 0xD2: 
{
std::int32_t number;
return get_number(input_format_t::msgpack, number) and sax->number_integer(number);
}

case 0xD3: 
{
std::int64_t number;
return get_number(input_format_t::msgpack, number) and sax->number_integer(number);
}

case 0xDC: 
{
std::uint16_t len;
return get_number(input_format_t::msgpack, len) and get_msgpack_array(static_cast<std::size_t>(len));
}

case 0xDD: 
{
std::uint32_t len;
return get_number(input_format_t::msgpack, len) and get_msgpack_array(static_cast<std::size_t>(len));
}

case 0xDE: 
{
std::uint16_t len;
return get_number(input_format_t::msgpack, len) and get_msgpack_object(static_cast<std::size_t>(len));
}

case 0xDF: 
{
std::uint32_t len;
return get_number(input_format_t::msgpack, len) and get_msgpack_object(static_cast<std::size_t>(len));
}

case 0xE0:
case 0xE1:
case 0xE2:
case 0xE3:
case 0xE4:
case 0xE5:
case 0xE6:
case 0xE7:
case 0xE8:
case 0xE9:
case 0xEA:
case 0xEB:
case 0xEC:
case 0xED:
case 0xEE:
case 0xEF:
case 0xF0:
case 0xF1:
case 0xF2:
case 0xF3:
case 0xF4:
case 0xF5:
case 0xF6:
case 0xF7:
case 0xF8:
case 0xF9:
case 0xFA:
case 0xFB:
case 0xFC:
case 0xFD:
case 0xFE:
case 0xFF:
return sax->number_integer(static_cast<std::int8_t>(current));

default: 
{
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, exception_message(input_format_t::msgpack, "invalid byte: 0x" + last_token, "value")));
}
}
}


bool get_msgpack_string(string_t& result)
{
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::msgpack, "string")))
{
return false;
}

switch (current)
{
case 0xA0:
case 0xA1:
case 0xA2:
case 0xA3:
case 0xA4:
case 0xA5:
case 0xA6:
case 0xA7:
case 0xA8:
case 0xA9:
case 0xAA:
case 0xAB:
case 0xAC:
case 0xAD:
case 0xAE:
case 0xAF:
case 0xB0:
case 0xB1:
case 0xB2:
case 0xB3:
case 0xB4:
case 0xB5:
case 0xB6:
case 0xB7:
case 0xB8:
case 0xB9:
case 0xBA:
case 0xBB:
case 0xBC:
case 0xBD:
case 0xBE:
case 0xBF:
{
return get_string(input_format_t::msgpack, static_cast<unsigned int>(current) & 0x1Fu, result);
}

case 0xD9: 
{
std::uint8_t len;
return get_number(input_format_t::msgpack, len) and get_string(input_format_t::msgpack, len, result);
}

case 0xDA: 
{
std::uint16_t len;
return get_number(input_format_t::msgpack, len) and get_string(input_format_t::msgpack, len, result);
}

case 0xDB: 
{
std::uint32_t len;
return get_number(input_format_t::msgpack, len) and get_string(input_format_t::msgpack, len, result);
}

default:
{
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, exception_message(input_format_t::msgpack, "expected length specification (0xA0-0xBF, 0xD9-0xDB); last byte: 0x" + last_token, "string")));
}
}
}


bool get_msgpack_binary(binary_t& result)
{
auto assign_and_return_true = [&result](std::int8_t subtype)
{
result.set_subtype(static_cast<std::uint8_t>(subtype));
return true;
};

switch (current)
{
case 0xC4: 
{
std::uint8_t len;
return get_number(input_format_t::msgpack, len) and
get_binary(input_format_t::msgpack, len, result);
}

case 0xC5: 
{
std::uint16_t len;
return get_number(input_format_t::msgpack, len) and
get_binary(input_format_t::msgpack, len, result);
}

case 0xC6: 
{
std::uint32_t len;
return get_number(input_format_t::msgpack, len) and
get_binary(input_format_t::msgpack, len, result);
}

case 0xC7: 
{
std::uint8_t len;
std::int8_t subtype;
return get_number(input_format_t::msgpack, len) and
get_number(input_format_t::msgpack, subtype) and
get_binary(input_format_t::msgpack, len, result) and
assign_and_return_true(subtype);
}

case 0xC8: 
{
std::uint16_t len;
std::int8_t subtype;
return get_number(input_format_t::msgpack, len) and
get_number(input_format_t::msgpack, subtype) and
get_binary(input_format_t::msgpack, len, result) and
assign_and_return_true(subtype);
}

case 0xC9: 
{
std::uint32_t len;
std::int8_t subtype;
return get_number(input_format_t::msgpack, len) and
get_number(input_format_t::msgpack, subtype) and
get_binary(input_format_t::msgpack, len, result) and
assign_and_return_true(subtype);
}

case 0xD4: 
{
std::int8_t subtype;
return get_number(input_format_t::msgpack, subtype) and
get_binary(input_format_t::msgpack, 1, result) and
assign_and_return_true(subtype);
}

case 0xD5: 
{
std::int8_t subtype;
return get_number(input_format_t::msgpack, subtype) and
get_binary(input_format_t::msgpack, 2, result) and
assign_and_return_true(subtype);
}

case 0xD6: 
{
std::int8_t subtype;
return get_number(input_format_t::msgpack, subtype) and
get_binary(input_format_t::msgpack, 4, result) and
assign_and_return_true(subtype);
}

case 0xD7: 
{
std::int8_t subtype;
return get_number(input_format_t::msgpack, subtype) and
get_binary(input_format_t::msgpack, 8, result) and
assign_and_return_true(subtype);
}

case 0xD8: 
{
std::int8_t subtype;
return get_number(input_format_t::msgpack, subtype) and
get_binary(input_format_t::msgpack, 16, result) and
assign_and_return_true(subtype);
}

default:           
return false;  
}
}


bool get_msgpack_array(const std::size_t len)
{
if (JSON_HEDLEY_UNLIKELY(not sax->start_array(len)))
{
return false;
}

for (std::size_t i = 0; i < len; ++i)
{
if (JSON_HEDLEY_UNLIKELY(not parse_msgpack_internal()))
{
return false;
}
}

return sax->end_array();
}


bool get_msgpack_object(const std::size_t len)
{
if (JSON_HEDLEY_UNLIKELY(not sax->start_object(len)))
{
return false;
}

string_t key;
for (std::size_t i = 0; i < len; ++i)
{
get();
if (JSON_HEDLEY_UNLIKELY(not get_msgpack_string(key) or not sax->key(key)))
{
return false;
}

if (JSON_HEDLEY_UNLIKELY(not parse_msgpack_internal()))
{
return false;
}
key.clear();
}

return sax->end_object();
}



bool parse_ubjson_internal(const bool get_char = true)
{
return get_ubjson_value(get_char ? get_ignore_noop() : current);
}


bool get_ubjson_string(string_t& result, const bool get_char = true)
{
if (get_char)
{
get();  
}

if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::ubjson, "value")))
{
return false;
}

switch (current)
{
case 'U':
{
std::uint8_t len;
return get_number(input_format_t::ubjson, len) and get_string(input_format_t::ubjson, len, result);
}

case 'i':
{
std::int8_t len;
return get_number(input_format_t::ubjson, len) and get_string(input_format_t::ubjson, len, result);
}

case 'I':
{
std::int16_t len;
return get_number(input_format_t::ubjson, len) and get_string(input_format_t::ubjson, len, result);
}

case 'l':
{
std::int32_t len;
return get_number(input_format_t::ubjson, len) and get_string(input_format_t::ubjson, len, result);
}

case 'L':
{
std::int64_t len;
return get_number(input_format_t::ubjson, len) and get_string(input_format_t::ubjson, len, result);
}

default:
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, exception_message(input_format_t::ubjson, "expected length type specification (U, i, I, l, L); last byte: 0x" + last_token, "string")));
}
}


bool get_ubjson_size_value(std::size_t& result)
{
switch (get_ignore_noop())
{
case 'U':
{
std::uint8_t number;
if (JSON_HEDLEY_UNLIKELY(not get_number(input_format_t::ubjson, number)))
{
return false;
}
result = static_cast<std::size_t>(number);
return true;
}

case 'i':
{
std::int8_t number;
if (JSON_HEDLEY_UNLIKELY(not get_number(input_format_t::ubjson, number)))
{
return false;
}
result = static_cast<std::size_t>(number);
return true;
}

case 'I':
{
std::int16_t number;
if (JSON_HEDLEY_UNLIKELY(not get_number(input_format_t::ubjson, number)))
{
return false;
}
result = static_cast<std::size_t>(number);
return true;
}

case 'l':
{
std::int32_t number;
if (JSON_HEDLEY_UNLIKELY(not get_number(input_format_t::ubjson, number)))
{
return false;
}
result = static_cast<std::size_t>(number);
return true;
}

case 'L':
{
std::int64_t number;
if (JSON_HEDLEY_UNLIKELY(not get_number(input_format_t::ubjson, number)))
{
return false;
}
result = static_cast<std::size_t>(number);
return true;
}

default:
{
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, exception_message(input_format_t::ubjson, "expected length type specification (U, i, I, l, L) after '#'; last byte: 0x" + last_token, "size")));
}
}
}


bool get_ubjson_size_type(std::pair<std::size_t, char_int_type>& result)
{
result.first = string_t::npos; 
result.second = 0; 

get_ignore_noop();

if (current == '$')
{
result.second = get();  
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::ubjson, "type")))
{
return false;
}

get_ignore_noop();
if (JSON_HEDLEY_UNLIKELY(current != '#'))
{
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::ubjson, "value")))
{
return false;
}
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, exception_message(input_format_t::ubjson, "expected '#' after type information; last byte: 0x" + last_token, "size")));
}

return get_ubjson_size_value(result.first);
}

if (current == '#')
{
return get_ubjson_size_value(result.first);
}

return true;
}


bool get_ubjson_value(const char_int_type prefix)
{
switch (prefix)
{
case std::char_traits<char_type>::eof():  
return unexpect_eof(input_format_t::ubjson, "value");

case 'T':  
return sax->boolean(true);
case 'F':  
return sax->boolean(false);

case 'Z':  
return sax->null();

case 'U':
{
std::uint8_t number;
return get_number(input_format_t::ubjson, number) and sax->number_unsigned(number);
}

case 'i':
{
std::int8_t number;
return get_number(input_format_t::ubjson, number) and sax->number_integer(number);
}

case 'I':
{
std::int16_t number;
return get_number(input_format_t::ubjson, number) and sax->number_integer(number);
}

case 'l':
{
std::int32_t number;
return get_number(input_format_t::ubjson, number) and sax->number_integer(number);
}

case 'L':
{
std::int64_t number;
return get_number(input_format_t::ubjson, number) and sax->number_integer(number);
}

case 'd':
{
float number;
return get_number(input_format_t::ubjson, number) and sax->number_float(static_cast<number_float_t>(number), "");
}

case 'D':
{
double number;
return get_number(input_format_t::ubjson, number) and sax->number_float(static_cast<number_float_t>(number), "");
}

case 'C':  
{
get();
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(input_format_t::ubjson, "char")))
{
return false;
}
if (JSON_HEDLEY_UNLIKELY(current > 127))
{
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, exception_message(input_format_t::ubjson, "byte after 'C' must be in range 0x00..0x7F; last byte: 0x" + last_token, "char")));
}
string_t s(1, static_cast<typename string_t::value_type>(current));
return sax->string(s);
}

case 'S':  
{
string_t s;
return get_ubjson_string(s) and sax->string(s);
}

case '[':  
return get_ubjson_array();

case '{':  
return get_ubjson_object();

default: 
{
auto last_token = get_token_string();
return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, exception_message(input_format_t::ubjson, "invalid byte: 0x" + last_token, "value")));
}
}
}


bool get_ubjson_array()
{
std::pair<std::size_t, char_int_type> size_and_type;
if (JSON_HEDLEY_UNLIKELY(not get_ubjson_size_type(size_and_type)))
{
return false;
}

if (size_and_type.first != string_t::npos)
{
if (JSON_HEDLEY_UNLIKELY(not sax->start_array(size_and_type.first)))
{
return false;
}

if (size_and_type.second != 0)
{
if (size_and_type.second != 'N')
{
for (std::size_t i = 0; i < size_and_type.first; ++i)
{
if (JSON_HEDLEY_UNLIKELY(not get_ubjson_value(size_and_type.second)))
{
return false;
}
}
}
}
else
{
for (std::size_t i = 0; i < size_and_type.first; ++i)
{
if (JSON_HEDLEY_UNLIKELY(not parse_ubjson_internal()))
{
return false;
}
}
}
}
else
{
if (JSON_HEDLEY_UNLIKELY(not sax->start_array(std::size_t(-1))))
{
return false;
}

while (current != ']')
{
if (JSON_HEDLEY_UNLIKELY(not parse_ubjson_internal(false)))
{
return false;
}
get_ignore_noop();
}
}

return sax->end_array();
}


bool get_ubjson_object()
{
std::pair<std::size_t, char_int_type> size_and_type;
if (JSON_HEDLEY_UNLIKELY(not get_ubjson_size_type(size_and_type)))
{
return false;
}

string_t key;
if (size_and_type.first != string_t::npos)
{
if (JSON_HEDLEY_UNLIKELY(not sax->start_object(size_and_type.first)))
{
return false;
}

if (size_and_type.second != 0)
{
for (std::size_t i = 0; i < size_and_type.first; ++i)
{
if (JSON_HEDLEY_UNLIKELY(not get_ubjson_string(key) or not sax->key(key)))
{
return false;
}
if (JSON_HEDLEY_UNLIKELY(not get_ubjson_value(size_and_type.second)))
{
return false;
}
key.clear();
}
}
else
{
for (std::size_t i = 0; i < size_and_type.first; ++i)
{
if (JSON_HEDLEY_UNLIKELY(not get_ubjson_string(key) or not sax->key(key)))
{
return false;
}
if (JSON_HEDLEY_UNLIKELY(not parse_ubjson_internal()))
{
return false;
}
key.clear();
}
}
}
else
{
if (JSON_HEDLEY_UNLIKELY(not sax->start_object(std::size_t(-1))))
{
return false;
}

while (current != '}')
{
if (JSON_HEDLEY_UNLIKELY(not get_ubjson_string(key, false) or not sax->key(key)))
{
return false;
}
if (JSON_HEDLEY_UNLIKELY(not parse_ubjson_internal()))
{
return false;
}
get_ignore_noop();
key.clear();
}
}

return sax->end_object();
}




char_int_type get()
{
++chars_read;
return current = ia.get_character();
}


char_int_type get_ignore_noop()
{
do
{
get();
}
while (current == 'N');

return current;
}


template<typename NumberType, bool InputIsLittleEndian = false>
bool get_number(const input_format_t format, NumberType& result)
{
std::array<std::uint8_t, sizeof(NumberType)> vec;
for (std::size_t i = 0; i < sizeof(NumberType); ++i)
{
get();
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(format, "number")))
{
return false;
}

if (is_little_endian != InputIsLittleEndian)
{
vec[sizeof(NumberType) - i - 1] = static_cast<std::uint8_t>(current);
}
else
{
vec[i] = static_cast<std::uint8_t>(current); 
}
}

std::memcpy(&result, vec.data(), sizeof(NumberType));
return true;
}


template<typename NumberType>
bool get_string(const input_format_t format,
const NumberType len,
string_t& result)
{
bool success = true;
std::generate_n(std::back_inserter(result), len, [this, &success, &format]()
{
get();
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(format, "string")))
{
success = false;
}
return std::char_traits<char_type>::to_char_type(current);
});
return success;
}


template<typename NumberType>
bool get_binary(const input_format_t format,
const NumberType len,
binary_t& result)
{
bool success = true;
std::generate_n(std::back_inserter(result), len, [this, &success, &format]()
{
get();
if (JSON_HEDLEY_UNLIKELY(not unexpect_eof(format, "binary")))
{
success = false;
}
return static_cast<std::uint8_t>(current);
});
return success;
}


JSON_HEDLEY_NON_NULL(3)
bool unexpect_eof(const input_format_t format, const char* context) const
{
if (JSON_HEDLEY_UNLIKELY(current == std::char_traits<char_type>::eof()))
{
return sax->parse_error(chars_read, "<end of file>",
parse_error::create(110, chars_read, exception_message(format, "unexpected end of input", context)));
}
return true;
}


std::string get_token_string() const
{
std::array<char, 3> cr{{}};
(std::snprintf)(cr.data(), cr.size(), "%.2hhX", static_cast<unsigned char>(current));
return std::string{cr.data()};
}


std::string exception_message(const input_format_t format,
const std::string& detail,
const std::string& context) const
{
std::string error_msg = "syntax error while parsing ";

switch (format)
{
case input_format_t::cbor:
error_msg += "CBOR";
break;

case input_format_t::msgpack:
error_msg += "MessagePack";
break;

case input_format_t::ubjson:
error_msg += "UBJSON";
break;

case input_format_t::bson:
error_msg += "BSON";
break;

default:            
assert(false);  
}

return error_msg + " " + context + ": " + detail;
}

private:
InputAdapterType ia;

char_int_type current = std::char_traits<char_type>::eof();

std::size_t chars_read = 0;

const bool is_little_endian = little_endianess();

json_sax_t* sax = nullptr;
};
}  
}  
