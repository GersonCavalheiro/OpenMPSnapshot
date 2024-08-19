
#ifndef ASIO_DETAIL_CALL_STACK_HPP
#define ASIO_DETAIL_CALL_STACK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/tss_ptr.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Key, typename Value = unsigned char>
class call_stack
{
public:
class context
: private noncopyable
{
public:
explicit context(Key* k)
: key_(k),
next_(call_stack<Key, Value>::top_)
{
value_ = reinterpret_cast<unsigned char*>(this);
call_stack<Key, Value>::top_ = this;
}

context(Key* k, Value& v)
: key_(k),
value_(&v),
next_(call_stack<Key, Value>::top_)
{
call_stack<Key, Value>::top_ = this;
}

~context()
{
call_stack<Key, Value>::top_ = next_;
}

Value* next_by_key() const
{
context* elem = next_;
while (elem)
{
if (elem->key_ == key_)
return elem->value_;
elem = elem->next_;
}
return 0;
}

private:
friend class call_stack<Key, Value>;

Key* key_;

Value* value_;

context* next_;
};

friend class context;

static Value* contains(Key* k)
{
context* elem = top_;
while (elem)
{
if (elem->key_ == k)
return elem->value_;
elem = elem->next_;
}
return 0;
}

static Value* top()
{
context* elem = top_;
return elem ? elem->value_ : 0;
}

private:
static tss_ptr<context> top_;
};

template <typename Key, typename Value>
tss_ptr<typename call_stack<Key, Value>::context>
call_stack<Key, Value>::top_;

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
