#pragma once

#include <iterator> 
#include <type_traits> 

#include <nlohmann/detail/boolean_operators.hpp>
#include <nlohmann/detail/exceptions.hpp>
#include <nlohmann/detail/iterators/internal_iterator.hpp>
#include <nlohmann/detail/iterators/primitive_iterator.hpp>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/detail/meta/cpp_future.hpp>
#include <nlohmann/detail/meta/type_traits.hpp>
#include <nlohmann/detail/value_t.hpp>

namespace nlohmann
{
namespace detail
{
template<typename IteratorType> class iteration_proxy;
template<typename IteratorType> class iteration_proxy_value;


template<typename BasicJsonType>
class iter_impl
{
friend iter_impl<typename std::conditional<std::is_const<BasicJsonType>::value, typename std::remove_const<BasicJsonType>::type, const BasicJsonType>::type>;
friend BasicJsonType;
friend iteration_proxy<iter_impl>;
friend iteration_proxy_value<iter_impl>;

using object_t = typename BasicJsonType::object_t;
using array_t = typename BasicJsonType::array_t;
static_assert(is_basic_json<typename std::remove_const<BasicJsonType>::type>::value,
"iter_impl only accepts (const) basic_json");

public:

using iterator_category = std::bidirectional_iterator_tag;

using value_type = typename BasicJsonType::value_type;
using difference_type = typename BasicJsonType::difference_type;
using pointer = typename std::conditional<std::is_const<BasicJsonType>::value,
typename BasicJsonType::const_pointer,
typename BasicJsonType::pointer>::type;
using reference =
typename std::conditional<std::is_const<BasicJsonType>::value,
typename BasicJsonType::const_reference,
typename BasicJsonType::reference>::type;

iter_impl() = default;


explicit iter_impl(pointer object) noexcept : m_object(object)
{
assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
{
m_it.object_iterator = typename object_t::iterator();
break;
}

case value_t::array:
{
m_it.array_iterator = typename array_t::iterator();
break;
}

default:
{
m_it.primitive_iterator = primitive_iterator_t();
break;
}
}
}




iter_impl(const iter_impl<const BasicJsonType>& other) noexcept
: m_object(other.m_object), m_it(other.m_it)
{}


iter_impl& operator=(const iter_impl<const BasicJsonType>& other) noexcept
{
m_object = other.m_object;
m_it = other.m_it;
return *this;
}


iter_impl(const iter_impl<typename std::remove_const<BasicJsonType>::type>& other) noexcept
: m_object(other.m_object), m_it(other.m_it)
{}


iter_impl& operator=(const iter_impl<typename std::remove_const<BasicJsonType>::type>& other) noexcept
{
m_object = other.m_object;
m_it = other.m_it;
return *this;
}

private:

void set_begin() noexcept
{
assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
{
m_it.object_iterator = m_object->m_value.object->begin();
break;
}

case value_t::array:
{
m_it.array_iterator = m_object->m_value.array->begin();
break;
}

case value_t::null:
{
m_it.primitive_iterator.set_end();
break;
}

default:
{
m_it.primitive_iterator.set_begin();
break;
}
}
}


void set_end() noexcept
{
assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
{
m_it.object_iterator = m_object->m_value.object->end();
break;
}

case value_t::array:
{
m_it.array_iterator = m_object->m_value.array->end();
break;
}

default:
{
m_it.primitive_iterator.set_end();
break;
}
}
}

public:

reference operator*() const
{
assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
{
assert(m_it.object_iterator != m_object->m_value.object->end());
return m_it.object_iterator->second;
}

case value_t::array:
{
assert(m_it.array_iterator != m_object->m_value.array->end());
return *m_it.array_iterator;
}

case value_t::null:
JSON_THROW(invalid_iterator::create(214, "cannot get value"));

default:
{
if (JSON_HEDLEY_LIKELY(m_it.primitive_iterator.is_begin()))
{
return *m_object;
}

JSON_THROW(invalid_iterator::create(214, "cannot get value"));
}
}
}


pointer operator->() const
{
assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
{
assert(m_it.object_iterator != m_object->m_value.object->end());
return &(m_it.object_iterator->second);
}

case value_t::array:
{
assert(m_it.array_iterator != m_object->m_value.array->end());
return &*m_it.array_iterator;
}

default:
{
if (JSON_HEDLEY_LIKELY(m_it.primitive_iterator.is_begin()))
{
return m_object;
}

JSON_THROW(invalid_iterator::create(214, "cannot get value"));
}
}
}


iter_impl const operator++(int)
{
auto result = *this;
++(*this);
return result;
}


iter_impl& operator++()
{
assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
{
std::advance(m_it.object_iterator, 1);
break;
}

case value_t::array:
{
std::advance(m_it.array_iterator, 1);
break;
}

default:
{
++m_it.primitive_iterator;
break;
}
}

return *this;
}


iter_impl const operator--(int)
{
auto result = *this;
--(*this);
return result;
}


iter_impl& operator--()
{
assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
{
std::advance(m_it.object_iterator, -1);
break;
}

case value_t::array:
{
std::advance(m_it.array_iterator, -1);
break;
}

default:
{
--m_it.primitive_iterator;
break;
}
}

return *this;
}


bool operator==(const iter_impl& other) const
{
if (JSON_HEDLEY_UNLIKELY(m_object != other.m_object))
{
JSON_THROW(invalid_iterator::create(212, "cannot compare iterators of different containers"));
}

assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
return (m_it.object_iterator == other.m_it.object_iterator);

case value_t::array:
return (m_it.array_iterator == other.m_it.array_iterator);

default:
return (m_it.primitive_iterator == other.m_it.primitive_iterator);
}
}


bool operator!=(const iter_impl& other) const
{
return not operator==(other);
}


bool operator<(const iter_impl& other) const
{
if (JSON_HEDLEY_UNLIKELY(m_object != other.m_object))
{
JSON_THROW(invalid_iterator::create(212, "cannot compare iterators of different containers"));
}

assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
JSON_THROW(invalid_iterator::create(213, "cannot compare order of object iterators"));

case value_t::array:
return (m_it.array_iterator < other.m_it.array_iterator);

default:
return (m_it.primitive_iterator < other.m_it.primitive_iterator);
}
}


bool operator<=(const iter_impl& other) const
{
return not other.operator < (*this);
}


bool operator>(const iter_impl& other) const
{
return not operator<=(other);
}


bool operator>=(const iter_impl& other) const
{
return not operator<(other);
}


iter_impl& operator+=(difference_type i)
{
assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
JSON_THROW(invalid_iterator::create(209, "cannot use offsets with object iterators"));

case value_t::array:
{
std::advance(m_it.array_iterator, i);
break;
}

default:
{
m_it.primitive_iterator += i;
break;
}
}

return *this;
}


iter_impl& operator-=(difference_type i)
{
return operator+=(-i);
}


iter_impl operator+(difference_type i) const
{
auto result = *this;
result += i;
return result;
}


friend iter_impl operator+(difference_type i, const iter_impl& it)
{
auto result = it;
result += i;
return result;
}


iter_impl operator-(difference_type i) const
{
auto result = *this;
result -= i;
return result;
}


difference_type operator-(const iter_impl& other) const
{
assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
JSON_THROW(invalid_iterator::create(209, "cannot use offsets with object iterators"));

case value_t::array:
return m_it.array_iterator - other.m_it.array_iterator;

default:
return m_it.primitive_iterator - other.m_it.primitive_iterator;
}
}


reference operator[](difference_type n) const
{
assert(m_object != nullptr);

switch (m_object->m_type)
{
case value_t::object:
JSON_THROW(invalid_iterator::create(208, "cannot use operator[] for object iterators"));

case value_t::array:
return *std::next(m_it.array_iterator, n);

case value_t::null:
JSON_THROW(invalid_iterator::create(214, "cannot get value"));

default:
{
if (JSON_HEDLEY_LIKELY(m_it.primitive_iterator.get_value() == -n))
{
return *m_object;
}

JSON_THROW(invalid_iterator::create(214, "cannot get value"));
}
}
}


const typename object_t::key_type& key() const
{
assert(m_object != nullptr);

if (JSON_HEDLEY_LIKELY(m_object->is_object()))
{
return m_it.object_iterator->first;
}

JSON_THROW(invalid_iterator::create(207, "cannot use key() for non-object iterators"));
}


reference value() const
{
return operator*();
}

private:
pointer m_object = nullptr;
internal_iterator<typename std::remove_const<BasicJsonType>::type> m_it {};
};
} 
} 
