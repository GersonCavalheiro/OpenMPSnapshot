
#ifndef ASIO_DETAIL_HASH_MAP_HPP
#define ASIO_DETAIL_HASH_MAP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <list>
#include <utility>
#include "asio/detail/assert.hpp"
#include "asio/detail/noncopyable.hpp"

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
# include "asio/detail/socket_types.hpp"
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

inline std::size_t calculate_hash_value(int i)
{
return static_cast<std::size_t>(i);
}

inline std::size_t calculate_hash_value(void* p)
{
return reinterpret_cast<std::size_t>(p)
+ (reinterpret_cast<std::size_t>(p) >> 3);
}

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
inline std::size_t calculate_hash_value(SOCKET s)
{
return static_cast<std::size_t>(s);
}
#endif 

template <typename K, typename V>
class hash_map
: private noncopyable
{
public:
typedef std::pair<K, V> value_type;

typedef typename std::list<value_type>::iterator iterator;

typedef typename std::list<value_type>::const_iterator const_iterator;

hash_map()
: size_(0),
buckets_(0),
num_buckets_(0)
{
}

~hash_map()
{
delete[] buckets_;
}

iterator begin()
{
return values_.begin();
}

const_iterator begin() const
{
return values_.begin();
}

iterator end()
{
return values_.end();
}

const_iterator end() const
{
return values_.end();
}

bool empty() const
{
return values_.empty();
}

iterator find(const K& k)
{
if (num_buckets_)
{
size_t bucket = calculate_hash_value(k) % num_buckets_;
iterator it = buckets_[bucket].first;
if (it == values_.end())
return values_.end();
iterator end_it = buckets_[bucket].last;
++end_it;
while (it != end_it)
{
if (it->first == k)
return it;
++it;
}
}
return values_.end();
}

const_iterator find(const K& k) const
{
if (num_buckets_)
{
size_t bucket = calculate_hash_value(k) % num_buckets_;
const_iterator it = buckets_[bucket].first;
if (it == values_.end())
return it;
const_iterator end_it = buckets_[bucket].last;
++end_it;
while (it != end_it)
{
if (it->first == k)
return it;
++it;
}
}
return values_.end();
}

std::pair<iterator, bool> insert(const value_type& v)
{
if (size_ + 1 >= num_buckets_)
rehash(hash_size(size_ + 1));
size_t bucket = calculate_hash_value(v.first) % num_buckets_;
iterator it = buckets_[bucket].first;
if (it == values_.end())
{
buckets_[bucket].first = buckets_[bucket].last =
values_insert(values_.end(), v);
++size_;
return std::pair<iterator, bool>(buckets_[bucket].last, true);
}
iterator end_it = buckets_[bucket].last;
++end_it;
while (it != end_it)
{
if (it->first == v.first)
return std::pair<iterator, bool>(it, false);
++it;
}
buckets_[bucket].last = values_insert(end_it, v);
++size_;
return std::pair<iterator, bool>(buckets_[bucket].last, true);
}

void erase(iterator it)
{
ASIO_ASSERT(it != values_.end());
ASIO_ASSERT(num_buckets_ != 0);

size_t bucket = calculate_hash_value(it->first) % num_buckets_;
bool is_first = (it == buckets_[bucket].first);
bool is_last = (it == buckets_[bucket].last);
if (is_first && is_last)
buckets_[bucket].first = buckets_[bucket].last = values_.end();
else if (is_first)
++buckets_[bucket].first;
else if (is_last)
--buckets_[bucket].last;

values_erase(it);
--size_;
}

void erase(const K& k)
{
iterator it = find(k);
if (it != values_.end())
erase(it);
}

void clear()
{
values_.clear();
size_ = 0;

iterator end_it = values_.end();
for (size_t i = 0; i < num_buckets_; ++i)
buckets_[i].first = buckets_[i].last = end_it;
}

private:
static std::size_t hash_size(std::size_t num_elems)
{
static std::size_t sizes[] =
{
#if defined(ASIO_HASH_MAP_BUCKETS)
ASIO_HASH_MAP_BUCKETS
#else 
3, 13, 23, 53, 97, 193, 389, 769, 1543, 3079, 6151, 12289, 24593,
49157, 98317, 196613, 393241, 786433, 1572869, 3145739, 6291469,
12582917, 25165843
#endif 
};
const std::size_t nth_size = sizeof(sizes) / sizeof(std::size_t) - 1;
for (std::size_t i = 0; i < nth_size; ++i)
if (num_elems < sizes[i])
return sizes[i];
return sizes[nth_size];
}

void rehash(std::size_t num_buckets)
{
if (num_buckets == num_buckets_)
return;
ASIO_ASSERT(num_buckets != 0);

iterator end_iter = values_.end();

bucket_type* tmp = new bucket_type[num_buckets];
delete[] buckets_;
buckets_ = tmp;
num_buckets_ = num_buckets;
for (std::size_t i = 0; i < num_buckets_; ++i)
buckets_[i].first = buckets_[i].last = end_iter;

iterator iter = values_.begin();
while (iter != end_iter)
{
std::size_t bucket = calculate_hash_value(iter->first) % num_buckets_;
if (buckets_[bucket].last == end_iter)
{
buckets_[bucket].first = buckets_[bucket].last = iter++;
}
else if (++buckets_[bucket].last == iter)
{
++iter;
}
else
{
values_.splice(buckets_[bucket].last, values_, iter++);
--buckets_[bucket].last;
}
}
}

iterator values_insert(iterator it, const value_type& v)
{
if (spares_.empty())
{
return values_.insert(it, v);
}
else
{
spares_.front() = v;
values_.splice(it, spares_, spares_.begin());
return --it;
}
}

void values_erase(iterator it)
{
*it = value_type();
spares_.splice(spares_.begin(), values_, it);
}

std::size_t size_;

std::list<value_type> values_;

std::list<value_type> spares_;

struct bucket_type
{
iterator first;
iterator last;
};

bucket_type* buckets_;

std::size_t num_buckets_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
