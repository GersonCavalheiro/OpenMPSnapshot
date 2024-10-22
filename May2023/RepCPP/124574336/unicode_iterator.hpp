





#ifndef BOOST_REGEX_UNICODE_ITERATOR_HPP
#define BOOST_REGEX_UNICODE_ITERATOR_HPP
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/static_assert.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#ifndef BOOST_NO_STD_LOCALE
#include <sstream>
#include <ios>
#endif
#include <limits.h> 

namespace boost{

namespace detail{

static const ::boost::uint16_t high_surrogate_base = 0xD7C0u;
static const ::boost::uint16_t low_surrogate_base = 0xDC00u;
static const ::boost::uint32_t ten_bit_mask = 0x3FFu;

inline bool is_high_surrogate(::boost::uint16_t v)
{
return (v & 0xFFFFFC00u) == 0xd800u;
}
inline bool is_low_surrogate(::boost::uint16_t v)
{
return (v & 0xFFFFFC00u) == 0xdc00u;
}
template <class T>
inline bool is_surrogate(T v)
{
return (v & 0xFFFFF800u) == 0xd800;
}

inline unsigned utf8_byte_count(boost::uint8_t c)
{
boost::uint8_t mask = 0x80u;
unsigned result = 0;
while(c & mask)
{
++result;
mask >>= 1;
}
return (result == 0) ? 1 : ((result > 4) ? 4 : result);
}

inline unsigned utf8_trailing_byte_count(boost::uint8_t c)
{
return utf8_byte_count(c) - 1;
}

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4100)
#endif
#ifndef BOOST_NO_EXCEPTIONS
BOOST_NORETURN
#endif
inline void invalid_utf32_code_point(::boost::uint32_t val)
{
#ifndef BOOST_NO_STD_LOCALE
std::stringstream ss;
ss << "Invalid UTF-32 code point U+" << std::showbase << std::hex << val << " encountered while trying to encode UTF-16 sequence";
std::out_of_range e(ss.str());
#else
std::out_of_range e("Invalid UTF-32 code point encountered while trying to encode UTF-16 sequence");
#endif
boost::throw_exception(e);
}
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif


} 

template <class BaseIterator, class U16Type = ::boost::uint16_t>
class u32_to_u16_iterator
: public boost::iterator_facade<u32_to_u16_iterator<BaseIterator, U16Type>, U16Type, std::bidirectional_iterator_tag, const U16Type>
{
typedef boost::iterator_facade<u32_to_u16_iterator<BaseIterator, U16Type>, U16Type, std::bidirectional_iterator_tag, const U16Type> base_type;

#if !defined(BOOST_NO_STD_ITERATOR_TRAITS)
typedef typename std::iterator_traits<BaseIterator>::value_type base_value_type;

BOOST_STATIC_ASSERT(sizeof(base_value_type)*CHAR_BIT == 32);
BOOST_STATIC_ASSERT(sizeof(U16Type)*CHAR_BIT == 16);
#endif

public:
typename base_type::reference
dereference()const
{
if(m_current == 2)
extract_current();
return m_values[m_current];
}
bool equal(const u32_to_u16_iterator& that)const
{
if(m_position == that.m_position)
{
return (m_current + that.m_current) & 1u ? false : true;
}
return false;
}
void increment()
{
if(m_current == 2)
{
extract_current();
}
++m_current;
if(m_values[m_current] == 0)
{
m_current = 2;
++m_position;
}
}
void decrement()
{
if(m_current != 1)
{
--m_position;
extract_current();
m_current = m_values[1] ? 1 : 0;
}
else
{
m_current = 0;
}
}
BaseIterator base()const
{
return m_position;
}
u32_to_u16_iterator() : m_position(), m_current(0)
{
m_values[0] = 0;
m_values[1] = 0;
m_values[2] = 0;
}
u32_to_u16_iterator(BaseIterator b) : m_position(b), m_current(2)
{
m_values[0] = 0;
m_values[1] = 0;
m_values[2] = 0;
}
private:

void extract_current()const
{
::boost::uint32_t v = *m_position;
if(v >= 0x10000u)
{
if(v > 0x10FFFFu)
detail::invalid_utf32_code_point(*m_position);
m_values[0] = static_cast<U16Type>(v >> 10) + detail::high_surrogate_base;
m_values[1] = static_cast<U16Type>(v & detail::ten_bit_mask) + detail::low_surrogate_base;
m_current = 0;
BOOST_ASSERT(detail::is_high_surrogate(m_values[0]));
BOOST_ASSERT(detail::is_low_surrogate(m_values[1]));
}
else
{
m_values[0] = static_cast<U16Type>(*m_position);
m_values[1] = 0;
m_current = 0;
if(detail::is_surrogate(m_values[0]))
detail::invalid_utf32_code_point(*m_position);
}
}
BaseIterator m_position;
mutable U16Type m_values[3];
mutable unsigned m_current;
};

template <class BaseIterator, class U32Type = ::boost::uint32_t>
class u16_to_u32_iterator
: public boost::iterator_facade<u16_to_u32_iterator<BaseIterator, U32Type>, U32Type, std::bidirectional_iterator_tag, const U32Type>
{
typedef boost::iterator_facade<u16_to_u32_iterator<BaseIterator, U32Type>, U32Type, std::bidirectional_iterator_tag, const U32Type> base_type;
BOOST_STATIC_CONSTANT(U32Type, pending_read = 0xffffffffu);

#if !defined(BOOST_NO_STD_ITERATOR_TRAITS)
typedef typename std::iterator_traits<BaseIterator>::value_type base_value_type;

BOOST_STATIC_ASSERT(sizeof(base_value_type)*CHAR_BIT == 16);
BOOST_STATIC_ASSERT(sizeof(U32Type)*CHAR_BIT == 32);
#endif

public:
typename base_type::reference
dereference()const
{
if(m_value == pending_read)
extract_current();
return m_value;
}
bool equal(const u16_to_u32_iterator& that)const
{
return m_position == that.m_position;
}
void increment()
{
if(detail::is_high_surrogate(*m_position)) ++m_position;
++m_position;
m_value = pending_read;
}
void decrement()
{
--m_position;
if(detail::is_low_surrogate(*m_position)) 
--m_position;
m_value = pending_read;
}
BaseIterator base()const
{
return m_position;
}
u16_to_u32_iterator() : m_position()
{
m_value = pending_read;
}
u16_to_u32_iterator(BaseIterator b) : m_position(b)
{
m_value = pending_read;
}
u16_to_u32_iterator(BaseIterator b, BaseIterator start, BaseIterator end) : m_position(b)
{
m_value = pending_read;
boost::uint16_t val;
if(start != end)
{
if((b != start) && (b != end))
{
val = *b;
if(detail::is_surrogate(val) && ((val & 0xFC00u) == 0xDC00u))
invalid_code_point(val);
}
val = *start;
if(detail::is_surrogate(val) && ((val & 0xFC00u) == 0xDC00u))
invalid_code_point(val);
val = *--end;
if(detail::is_high_surrogate(val))
invalid_code_point(val);
}
}
private:
static void invalid_code_point(::boost::uint16_t val)
{
#ifndef BOOST_NO_STD_LOCALE
std::stringstream ss;
ss << "Misplaced UTF-16 surrogate U+" << std::showbase << std::hex << val << " encountered while trying to encode UTF-32 sequence";
std::out_of_range e(ss.str());
#else
std::out_of_range e("Misplaced UTF-16 surrogate encountered while trying to encode UTF-32 sequence");
#endif
boost::throw_exception(e);
}
void extract_current()const
{
m_value = static_cast<U32Type>(static_cast< ::boost::uint16_t>(*m_position));
if(detail::is_high_surrogate(*m_position))
{
BaseIterator next(m_position);
::boost::uint16_t t = *++next;
if((t & 0xFC00u) != 0xDC00u)
invalid_code_point(t);
m_value = (m_value - detail::high_surrogate_base) << 10;
m_value |= (static_cast<U32Type>(static_cast< ::boost::uint16_t>(t)) & detail::ten_bit_mask);
}
if(detail::is_surrogate(m_value))
invalid_code_point(static_cast< ::boost::uint16_t>(m_value));
}
BaseIterator m_position;
mutable U32Type m_value;
};

template <class BaseIterator, class U8Type = ::boost::uint8_t>
class u32_to_u8_iterator
: public boost::iterator_facade<u32_to_u8_iterator<BaseIterator, U8Type>, U8Type, std::bidirectional_iterator_tag, const U8Type>
{
typedef boost::iterator_facade<u32_to_u8_iterator<BaseIterator, U8Type>, U8Type, std::bidirectional_iterator_tag, const U8Type> base_type;

#if !defined(BOOST_NO_STD_ITERATOR_TRAITS)
typedef typename std::iterator_traits<BaseIterator>::value_type base_value_type;

BOOST_STATIC_ASSERT(sizeof(base_value_type)*CHAR_BIT == 32);
BOOST_STATIC_ASSERT(sizeof(U8Type)*CHAR_BIT == 8);
#endif

public:
typename base_type::reference
dereference()const
{
if(m_current == 4)
extract_current();
return m_values[m_current];
}
bool equal(const u32_to_u8_iterator& that)const
{
if(m_position == that.m_position)
{
return (m_current == that.m_current)
|| (((m_current | that.m_current) & 3) == 0);
}
return false;
}
void increment()
{
if(m_current == 4)
{
extract_current();
}
++m_current;
if(m_values[m_current] == 0)
{
m_current = 4;
++m_position;
}
}
void decrement()
{
if((m_current & 3) == 0)
{
--m_position;
extract_current();
m_current = 3;
while(m_current && (m_values[m_current] == 0))
--m_current;
}
else
--m_current;
}
BaseIterator base()const
{
return m_position;
}
u32_to_u8_iterator() : m_position(), m_current(0)
{
m_values[0] = 0;
m_values[1] = 0;
m_values[2] = 0;
m_values[3] = 0;
m_values[4] = 0;
}
u32_to_u8_iterator(BaseIterator b) : m_position(b), m_current(4)
{
m_values[0] = 0;
m_values[1] = 0;
m_values[2] = 0;
m_values[3] = 0;
m_values[4] = 0;
}
private:

void extract_current()const
{
boost::uint32_t c = *m_position;
if(c > 0x10FFFFu)
detail::invalid_utf32_code_point(c);
if(c < 0x80u)
{
m_values[0] = static_cast<unsigned char>(c);
m_values[1] = static_cast<unsigned char>(0u);
m_values[2] = static_cast<unsigned char>(0u);
m_values[3] = static_cast<unsigned char>(0u);
}
else if(c < 0x800u)
{
m_values[0] = static_cast<unsigned char>(0xC0u + (c >> 6));
m_values[1] = static_cast<unsigned char>(0x80u + (c & 0x3Fu));
m_values[2] = static_cast<unsigned char>(0u);
m_values[3] = static_cast<unsigned char>(0u);
}
else if(c < 0x10000u)
{
m_values[0] = static_cast<unsigned char>(0xE0u + (c >> 12));
m_values[1] = static_cast<unsigned char>(0x80u + ((c >> 6) & 0x3Fu));
m_values[2] = static_cast<unsigned char>(0x80u + (c & 0x3Fu));
m_values[3] = static_cast<unsigned char>(0u);
}
else
{
m_values[0] = static_cast<unsigned char>(0xF0u + (c >> 18));
m_values[1] = static_cast<unsigned char>(0x80u + ((c >> 12) & 0x3Fu));
m_values[2] = static_cast<unsigned char>(0x80u + ((c >> 6) & 0x3Fu));
m_values[3] = static_cast<unsigned char>(0x80u + (c & 0x3Fu));
}
m_current= 0;
}
BaseIterator m_position;
mutable U8Type m_values[5];
mutable unsigned m_current;
};

template <class BaseIterator, class U32Type = ::boost::uint32_t>
class u8_to_u32_iterator
: public boost::iterator_facade<u8_to_u32_iterator<BaseIterator, U32Type>, U32Type, std::bidirectional_iterator_tag, const U32Type>
{
typedef boost::iterator_facade<u8_to_u32_iterator<BaseIterator, U32Type>, U32Type, std::bidirectional_iterator_tag, const U32Type> base_type;
BOOST_STATIC_CONSTANT(U32Type, pending_read = 0xffffffffu);

#if !defined(BOOST_NO_STD_ITERATOR_TRAITS)
typedef typename std::iterator_traits<BaseIterator>::value_type base_value_type;

BOOST_STATIC_ASSERT(sizeof(base_value_type)*CHAR_BIT == 8);
BOOST_STATIC_ASSERT(sizeof(U32Type)*CHAR_BIT == 32);
#endif

public:
typename base_type::reference
dereference()const
{
if(m_value == pending_read)
extract_current();
return m_value;
}
bool equal(const u8_to_u32_iterator& that)const
{
return m_position == that.m_position;
}
void increment()
{
if((static_cast<boost::uint8_t>(*m_position) & 0xC0) == 0x80)
invalid_sequence();
unsigned c = detail::utf8_byte_count(*m_position);
if(m_value == pending_read)
{
for(unsigned i = 0; i < c; ++i)
{
++m_position;
if((i != c - 1) && ((static_cast<boost::uint8_t>(*m_position) & 0xC0) != 0x80))
invalid_sequence();
}
}
else
{
std::advance(m_position, c);
}
m_value = pending_read;
}
void decrement()
{
unsigned count = 0;
while((*--m_position & 0xC0u) == 0x80u) ++count;
if(count != detail::utf8_trailing_byte_count(*m_position))
invalid_sequence();
m_value = pending_read;
}
BaseIterator base()const
{
return m_position;
}
u8_to_u32_iterator() : m_position()
{
m_value = pending_read;
}
u8_to_u32_iterator(BaseIterator b) : m_position(b)
{
m_value = pending_read;
}
u8_to_u32_iterator(BaseIterator b, BaseIterator start, BaseIterator end) : m_position(b)
{
m_value = pending_read;
if(start != end)
{
unsigned char v = *start;
if((v & 0xC0u) == 0x80u)
invalid_sequence();
if((b != start) && (b != end) && ((*b & 0xC0u) == 0x80u))
invalid_sequence();
BaseIterator pos = end;
do
{
v = *--pos;
}
while((start != pos) && ((v & 0xC0u) == 0x80u));
std::ptrdiff_t extra = detail::utf8_byte_count(v);
if(std::distance(pos, end) < extra)
invalid_sequence();
}
}
private:
static void invalid_sequence()
{
std::out_of_range e("Invalid UTF-8 sequence encountered while trying to encode UTF-32 character");
boost::throw_exception(e);
}
void extract_current()const
{
m_value = static_cast<U32Type>(static_cast< ::boost::uint8_t>(*m_position));
if((m_value & 0xC0u) == 0x80u)
invalid_sequence();
unsigned extra = detail::utf8_trailing_byte_count(*m_position);
BaseIterator next(m_position);
for(unsigned c = 0; c < extra; ++c)
{
++next;
m_value <<= 6;
if((static_cast<boost::uint8_t>(*next) & 0xC0) != 0x80)
invalid_sequence();
m_value += static_cast<boost::uint8_t>(*next) & 0x3Fu;
}
static const boost::uint32_t masks[4] = 
{
0x7Fu,
0x7FFu,
0xFFFFu,
0x1FFFFFu,
};
m_value &= masks[extra];
if(m_value > static_cast<U32Type>(0x10FFFFu))
invalid_sequence();
if((m_value >= static_cast<U32Type>(0xD800)) && (m_value <= static_cast<U32Type>(0xDFFF)))
invalid_sequence();
if((extra > 0) && (m_value <= static_cast<U32Type>(masks[extra - 1])))
invalid_sequence();
}
BaseIterator m_position;
mutable U32Type m_value;
};

template <class BaseIterator>
class utf16_output_iterator
{
public:
typedef void                                   difference_type;
typedef void                                   value_type;
typedef boost::uint32_t*                       pointer;
typedef boost::uint32_t&                       reference;
typedef std::output_iterator_tag               iterator_category;

utf16_output_iterator(const BaseIterator& b)
: m_position(b){}
utf16_output_iterator(const utf16_output_iterator& that)
: m_position(that.m_position){}
utf16_output_iterator& operator=(const utf16_output_iterator& that)
{
m_position = that.m_position;
return *this;
}
const utf16_output_iterator& operator*()const
{
return *this;
}
void operator=(boost::uint32_t val)const
{
push(val);
}
utf16_output_iterator& operator++()
{
return *this;
}
utf16_output_iterator& operator++(int)
{
return *this;
}
BaseIterator base()const
{
return m_position;
}
private:
void push(boost::uint32_t v)const
{
if(v >= 0x10000u)
{
if(v > 0x10FFFFu)
detail::invalid_utf32_code_point(v);
*m_position++ = static_cast<boost::uint16_t>(v >> 10) + detail::high_surrogate_base;
*m_position++ = static_cast<boost::uint16_t>(v & detail::ten_bit_mask) + detail::low_surrogate_base;
}
else
{
if(detail::is_surrogate(v))
detail::invalid_utf32_code_point(v);
*m_position++ = static_cast<boost::uint16_t>(v);
}
}
mutable BaseIterator m_position;
};

template <class BaseIterator>
class utf8_output_iterator
{
public:
typedef void                                   difference_type;
typedef void                                   value_type;
typedef boost::uint32_t*                       pointer;
typedef boost::uint32_t&                       reference;
typedef std::output_iterator_tag               iterator_category;

utf8_output_iterator(const BaseIterator& b)
: m_position(b){}
utf8_output_iterator(const utf8_output_iterator& that)
: m_position(that.m_position){}
utf8_output_iterator& operator=(const utf8_output_iterator& that)
{
m_position = that.m_position;
return *this;
}
const utf8_output_iterator& operator*()const
{
return *this;
}
void operator=(boost::uint32_t val)const
{
push(val);
}
utf8_output_iterator& operator++()
{
return *this;
}
utf8_output_iterator& operator++(int)
{
return *this;
}
BaseIterator base()const
{
return m_position;
}
private:
void push(boost::uint32_t c)const
{
if(c > 0x10FFFFu)
detail::invalid_utf32_code_point(c);
if(c < 0x80u)
{
*m_position++ = static_cast<unsigned char>(c);
}
else if(c < 0x800u)
{
*m_position++ = static_cast<unsigned char>(0xC0u + (c >> 6));
*m_position++ = static_cast<unsigned char>(0x80u + (c & 0x3Fu));
}
else if(c < 0x10000u)
{
*m_position++ = static_cast<unsigned char>(0xE0u + (c >> 12));
*m_position++ = static_cast<unsigned char>(0x80u + ((c >> 6) & 0x3Fu));
*m_position++ = static_cast<unsigned char>(0x80u + (c & 0x3Fu));
}
else
{
*m_position++ = static_cast<unsigned char>(0xF0u + (c >> 18));
*m_position++ = static_cast<unsigned char>(0x80u + ((c >> 12) & 0x3Fu));
*m_position++ = static_cast<unsigned char>(0x80u + ((c >> 6) & 0x3Fu));
*m_position++ = static_cast<unsigned char>(0x80u + (c & 0x3Fu));
}
}
mutable BaseIterator m_position;
};

} 

#endif 

