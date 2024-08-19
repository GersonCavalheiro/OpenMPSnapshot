#pragma once

#include <array> 
#include <cassert> 
#include <cstddef> 
#include <cstdio> 
#include <cstring> 
#include <istream> 
#include <iterator> 
#include <memory> 
#include <numeric> 
#include <string> 
#include <type_traits> 
#include <utility> 

#include <nlohmann/detail/iterators/iterator_traits.hpp>
#include <nlohmann/detail/macro_scope.hpp>

namespace nlohmann
{
namespace detail
{
enum class input_format_t { json, cbor, msgpack, ubjson, bson };



class file_input_adapter
{
public:
using char_type = char;

JSON_HEDLEY_NON_NULL(2)
explicit file_input_adapter(std::FILE* f)  noexcept
: m_file(f)
{}

file_input_adapter(const file_input_adapter&) = delete;
file_input_adapter(file_input_adapter&&) = default;
file_input_adapter& operator=(const file_input_adapter&) = delete;
file_input_adapter& operator=(file_input_adapter&&) = delete;

std::char_traits<char>::int_type get_character() noexcept
{
return std::fgetc(m_file);
}

private:
std::FILE* m_file;
};



class input_stream_adapter
{
public:
using char_type = char;

~input_stream_adapter()
{
if (is)
{
is->clear(is->rdstate() & std::ios::eofbit);
}
}

explicit input_stream_adapter(std::istream& i)
: is(&i), sb(i.rdbuf())
{}

input_stream_adapter(const input_stream_adapter&) = delete;
input_stream_adapter& operator=(input_stream_adapter&) = delete;
input_stream_adapter& operator=(input_stream_adapter&& rhs) = delete;

input_stream_adapter(input_stream_adapter&& rhs) : is(rhs.is), sb(rhs.sb)
{
rhs.is = nullptr;
rhs.sb = nullptr;
}

std::char_traits<char>::int_type get_character()
{
auto res = sb->sbumpc();
if (JSON_HEDLEY_UNLIKELY(res == EOF))
{
is->clear(is->rdstate() | std::ios::eofbit);
}
return res;
}

private:
std::istream* is = nullptr;
std::streambuf* sb = nullptr;
};

template<typename IteratorType>
class iterator_input_adapter
{
public:
using char_type = typename std::iterator_traits<IteratorType>::value_type;

iterator_input_adapter(IteratorType first, IteratorType last)
: current(std::move(first)), end(std::move(last)) {}

typename std::char_traits<char_type>::int_type get_character()
{
if (JSON_HEDLEY_LIKELY(current != end))
{
auto result = std::char_traits<char_type>::to_int_type(*current);
std::advance(current, 1);
return result;
}
else
{
return std::char_traits<char_type>::eof();
}
}

private:
IteratorType current;
IteratorType end;

template<typename BaseInputAdapter, size_t T>
friend struct wide_string_input_helper;

bool empty() const
{
return current == end;
}

};


template<typename BaseInputAdapter, size_t T>
struct wide_string_input_helper;

template<typename BaseInputAdapter>
struct wide_string_input_helper<BaseInputAdapter, 4>
{
static void fill_buffer(BaseInputAdapter& input,
std::array<std::char_traits<char>::int_type, 4>& utf8_bytes,
size_t& utf8_bytes_index,
size_t& utf8_bytes_filled)
{
utf8_bytes_index = 0;

if (JSON_HEDLEY_UNLIKELY(input.empty()))
{
utf8_bytes[0] = std::char_traits<char>::eof();
utf8_bytes_filled = 1;
}
else
{
const auto wc = input.get_character();

if (wc < 0x80)
{
utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(wc);
utf8_bytes_filled = 1;
}
else if (wc <= 0x7FF)
{
utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xC0u | ((static_cast<unsigned int>(wc) >> 6u) & 0x1Fu));
utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | (static_cast<unsigned int>(wc) & 0x3Fu));
utf8_bytes_filled = 2;
}
else if (wc <= 0xFFFF)
{
utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xE0u | ((static_cast<unsigned int>(wc) >> 12u) & 0x0Fu));
utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | ((static_cast<unsigned int>(wc) >> 6u) & 0x3Fu));
utf8_bytes[2] = static_cast<std::char_traits<char>::int_type>(0x80u | (static_cast<unsigned int>(wc) & 0x3Fu));
utf8_bytes_filled = 3;
}
else if (wc <= 0x10FFFF)
{
utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xF0u | ((static_cast<unsigned int>(wc) >> 18u) & 0x07u));
utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | ((static_cast<unsigned int>(wc) >> 12u) & 0x3Fu));
utf8_bytes[2] = static_cast<std::char_traits<char>::int_type>(0x80u | ((static_cast<unsigned int>(wc) >> 6u) & 0x3Fu));
utf8_bytes[3] = static_cast<std::char_traits<char>::int_type>(0x80u | (static_cast<unsigned int>(wc) & 0x3Fu));
utf8_bytes_filled = 4;
}
else
{
utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(wc);
utf8_bytes_filled = 1;
}
}
}
};

template<typename BaseInputAdapter>
struct wide_string_input_helper<BaseInputAdapter, 2>
{
static void fill_buffer(BaseInputAdapter& input,
std::array<std::char_traits<char>::int_type, 4>& utf8_bytes,
size_t& utf8_bytes_index,
size_t& utf8_bytes_filled)
{
utf8_bytes_index = 0;

if (JSON_HEDLEY_UNLIKELY(input.empty()))
{
utf8_bytes[0] = std::char_traits<char>::eof();
utf8_bytes_filled = 1;
}
else
{
const auto wc = input.get_character();

if (wc < 0x80)
{
utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(wc);
utf8_bytes_filled = 1;
}
else if (wc <= 0x7FF)
{
utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xC0u | ((static_cast<unsigned int>(wc) >> 6u)));
utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | (static_cast<unsigned int>(wc) & 0x3Fu));
utf8_bytes_filled = 2;
}
else if (0xD800 > wc or wc >= 0xE000)
{
utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xE0u | ((static_cast<unsigned int>(wc) >> 12u)));
utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | ((static_cast<unsigned int>(wc) >> 6u) & 0x3Fu));
utf8_bytes[2] = static_cast<std::char_traits<char>::int_type>(0x80u | (static_cast<unsigned int>(wc) & 0x3Fu));
utf8_bytes_filled = 3;
}
else
{
if (JSON_HEDLEY_UNLIKELY(not input.empty()))
{
const auto wc2 = static_cast<unsigned int>(input.get_character());
const auto charcode = 0x10000u + (((static_cast<unsigned int>(wc) & 0x3FFu) << 10u) | (wc2 & 0x3FFu));
utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(0xF0u | (charcode >> 18u));
utf8_bytes[1] = static_cast<std::char_traits<char>::int_type>(0x80u | ((charcode >> 12u) & 0x3Fu));
utf8_bytes[2] = static_cast<std::char_traits<char>::int_type>(0x80u | ((charcode >> 6u) & 0x3Fu));
utf8_bytes[3] = static_cast<std::char_traits<char>::int_type>(0x80u | (charcode & 0x3Fu));
utf8_bytes_filled = 4;
}
else
{
utf8_bytes[0] = static_cast<std::char_traits<char>::int_type>(wc);
utf8_bytes_filled = 1;
}
}
}
}
};

template<typename BaseInputAdapter, typename WideCharType>
class wide_string_input_adapter
{
public:
using char_type = char;

wide_string_input_adapter(BaseInputAdapter base)
: base_adapter(base) {}

typename std::char_traits<char>::int_type get_character() noexcept
{
if (utf8_bytes_index == utf8_bytes_filled)
{
fill_buffer<sizeof(WideCharType)>();

assert(utf8_bytes_filled > 0);
assert(utf8_bytes_index == 0);
}

assert(utf8_bytes_filled > 0);
assert(utf8_bytes_index < utf8_bytes_filled);
return utf8_bytes[utf8_bytes_index++];
}

private:
BaseInputAdapter base_adapter;

template<size_t T>
void fill_buffer()
{
wide_string_input_helper<BaseInputAdapter, T>::fill_buffer(base_adapter, utf8_bytes, utf8_bytes_index, utf8_bytes_filled);
}

std::array<std::char_traits<char>::int_type, 4> utf8_bytes = {{0, 0, 0, 0}};

std::size_t utf8_bytes_index = 0;
std::size_t utf8_bytes_filled = 0;
};


template<typename IteratorType, typename Enable = void>
struct iterator_input_adapter_factory
{
using iterator_type = IteratorType;
using char_type = typename std::iterator_traits<iterator_type>::value_type;
using adapter_type = iterator_input_adapter<iterator_type>;

static adapter_type create(IteratorType first, IteratorType last)
{
return adapter_type(std::move(first), std::move(last));
}
};

template<typename T>
struct is_iterator_of_multibyte
{
using value_type = typename std::iterator_traits<T>::value_type;
enum
{
value = sizeof(value_type) > 1
};
};

template<typename IteratorType>
struct iterator_input_adapter_factory<IteratorType, enable_if_t<is_iterator_of_multibyte<IteratorType>::value>>
{
using iterator_type = IteratorType;
using char_type = typename std::iterator_traits<iterator_type>::value_type;
using base_adapter_type = iterator_input_adapter<iterator_type>;
using adapter_type = wide_string_input_adapter<base_adapter_type, char_type>;

static adapter_type create(IteratorType first, IteratorType last)
{
return adapter_type(base_adapter_type(std::move(first), std::move(last)));
}
};

template<typename IteratorType>
typename iterator_input_adapter_factory<IteratorType>::adapter_type input_adapter(IteratorType first, IteratorType last)
{
using factory_type = iterator_input_adapter_factory<IteratorType>;
return factory_type::create(first, last);
}

template<typename ContainerType>
auto input_adapter(const ContainerType& container) -> decltype(input_adapter(begin(container), end(container)))
{
using std::begin;
using std::end;

return input_adapter(begin(container), end(container));
}

inline file_input_adapter input_adapter(std::FILE* file)
{
return file_input_adapter(file);
}

inline input_stream_adapter input_adapter(std::istream& stream)
{
return input_stream_adapter(stream);
}

inline input_stream_adapter input_adapter(std::istream&& stream)
{
return input_stream_adapter(stream);
}

using contiguous_bytes_input_adapter = decltype(input_adapter(std::declval<const char*>(), std::declval<const char*>()));

template < typename CharT,
typename std::enable_if <
std::is_pointer<CharT>::value and
not std::is_array<CharT>::value and
std::is_integral<typename std::remove_pointer<CharT>::type>::value and
sizeof(typename std::remove_pointer<CharT>::type) == 1,
int >::type = 0 >
contiguous_bytes_input_adapter input_adapter(CharT b)
{
auto length = std::strlen(reinterpret_cast<const char*>(b));
auto ptr = reinterpret_cast<const char*>(b);
return input_adapter(ptr, ptr + length);
}

template<typename T, std::size_t N>
auto input_adapter(T (&array)[N]) -> decltype(input_adapter(array, array + N))
{
return input_adapter(array, array + N);
}

class span_input_adapter
{
public:
template<typename CharT,
typename std::enable_if<
std::is_pointer<CharT>::value and
std::is_integral<typename std::remove_pointer<CharT>::type>::value and
sizeof(typename std::remove_pointer<CharT>::type) == 1,
int>::type = 0>
span_input_adapter(CharT b, std::size_t l)
: ia(reinterpret_cast<const char*>(b), reinterpret_cast<const char*>(b) + l) {}

template<class IteratorType,
typename std::enable_if<
std::is_same<typename iterator_traits<IteratorType>::iterator_category, std::random_access_iterator_tag>::value,
int>::type = 0>
span_input_adapter(IteratorType first, IteratorType last)
: ia(input_adapter(first, last)) {}

contiguous_bytes_input_adapter&& get()
{
return std::move(ia);
}

private:
contiguous_bytes_input_adapter ia;
};
}  
}  
