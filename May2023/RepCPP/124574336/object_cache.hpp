



#ifndef BOOST_REGEX_OBJECT_CACHE_HPP
#define BOOST_REGEX_OBJECT_CACHE_HPP

#include <boost/config.hpp>
#include <boost/shared_ptr.hpp>
#include <map>
#include <list>
#include <stdexcept>
#include <string>
#ifdef BOOST_HAS_THREADS
#include <boost/regex/pending/static_mutex.hpp>
#endif

namespace boost{

template <class Key, class Object>
class object_cache
{
public:
typedef std::pair< ::boost::shared_ptr<Object const>, Key const*> value_type;
typedef std::list<value_type> list_type;
typedef typename list_type::iterator list_iterator;
typedef std::map<Key, list_iterator> map_type;
typedef typename map_type::iterator map_iterator;
typedef typename list_type::size_type size_type;
static boost::shared_ptr<Object const> get(const Key& k, size_type l_max_cache_size);

private:
static boost::shared_ptr<Object const> do_get(const Key& k, size_type l_max_cache_size);

struct data
{
list_type   cont;
map_type    index;
};

friend struct data;
};

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4702)
#endif
template <class Key, class Object>
boost::shared_ptr<Object const> object_cache<Key, Object>::get(const Key& k, size_type l_max_cache_size)
{
#ifdef BOOST_HAS_THREADS
static boost::static_mutex mut = BOOST_STATIC_MUTEX_INIT;

boost::static_mutex::scoped_lock l(mut);
if(l)
{
return do_get(k, l_max_cache_size);
}
::boost::throw_exception(std::runtime_error("Error in thread safety code: could not acquire a lock"));
#if defined(BOOST_NO_UNREACHABLE_RETURN_DETECTION) || defined(BOOST_NO_EXCEPTIONS)
return boost::shared_ptr<Object>();
#endif
#else
return do_get(k, l_max_cache_size);
#endif
}
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template <class Key, class Object>
boost::shared_ptr<Object const> object_cache<Key, Object>::do_get(const Key& k, size_type l_max_cache_size)
{
typedef typename object_cache<Key, Object>::data object_data;
typedef typename map_type::size_type map_size_type;
static object_data s_data;

map_iterator mpos = s_data.index.find(k);
if(mpos != s_data.index.end())
{
if(--(s_data.cont.end()) != mpos->second)
{
list_type temp;
temp.splice(temp.end(), s_data.cont, mpos->second);
s_data.cont.splice(s_data.cont.end(), temp, temp.begin());
BOOST_ASSERT(*(s_data.cont.back().second) == k);
mpos->second = --(s_data.cont.end());
BOOST_ASSERT(&(mpos->first) == mpos->second->second);
BOOST_ASSERT(&(mpos->first) == s_data.cont.back().second);
}
return s_data.cont.back().first;
}
boost::shared_ptr<Object const> result(new Object(k));
s_data.cont.push_back(value_type(result, static_cast<Key const*>(0)));
s_data.index.insert(std::make_pair(k, --(s_data.cont.end())));
s_data.cont.back().second = &(s_data.index.find(k)->first);
map_size_type s = s_data.index.size();
BOOST_ASSERT(s_data.index[k]->first.get() == result.get());
BOOST_ASSERT(&(s_data.index.find(k)->first) == s_data.cont.back().second);
BOOST_ASSERT(s_data.index.find(k)->first == k);
if(s > l_max_cache_size)
{
list_iterator pos = s_data.cont.begin();
list_iterator last = s_data.cont.end();
while((pos != last) && (s > l_max_cache_size))
{
if(pos->first.unique())
{
list_iterator condemmed(pos);
++pos;
BOOST_ASSERT(s_data.index.find(*(condemmed->second)) != s_data.index.end());
s_data.index.erase(*(condemmed->second));
s_data.cont.erase(condemmed); 
--s;
}
else
++pos;
}
BOOST_ASSERT(s_data.index[k]->first.get() == result.get());
BOOST_ASSERT(&(s_data.index.find(k)->first) == s_data.cont.back().second);
BOOST_ASSERT(s_data.index.find(k)->first == k);
}
return result;
}

}

#endif
