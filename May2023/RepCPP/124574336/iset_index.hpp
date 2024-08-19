
#ifndef BOOST_INTERPROCESS_ISET_INDEX_HPP
#define BOOST_INTERPROCESS_ISET_INDEX_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/intrusive/detail/minimal_pair_header.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/intrusive/detail/minimal_pair_header.hpp>         
#include <boost/intrusive/detail/minimal_less_equal_header.hpp>   
#include <boost/container/detail/minimal_char_traits_header.hpp>  
#include <boost/intrusive/set.hpp>


namespace boost {
namespace interprocess {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template <class MapConfig>
struct iset_index_aux
{
typedef typename
MapConfig::segment_manager_base                          segment_manager_base;

typedef typename
segment_manager_base::void_pointer                       void_pointer;
typedef typename bi::make_set_base_hook
< bi::void_pointer<void_pointer>
, bi::optimize_size<true>
>::type                                                  derivation_hook;

typedef typename MapConfig::template
intrusive_value_type<derivation_hook>::type              value_type;
typedef std::less<value_type>                               value_compare;
typedef typename bi::make_set
< value_type
, bi::base_hook<derivation_hook>
>::type                                                  index_t;
};
#endif   

template <class MapConfig>
class iset_index
:  public iset_index_aux<MapConfig>::index_t
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
typedef iset_index_aux<MapConfig>                     index_aux;
typedef typename index_aux::index_t                   index_type;
typedef typename MapConfig::
intrusive_compare_key_type                         intrusive_compare_key_type;
typedef typename MapConfig::char_type                 char_type;
#endif   

public:
typedef typename index_type::iterator                 iterator;
typedef typename index_type::const_iterator           const_iterator;
typedef typename index_type::insert_commit_data       insert_commit_data;
typedef typename index_type::value_type               value_type;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:

struct intrusive_key_value_less
{
bool operator()(const intrusive_compare_key_type &i, const value_type &b) const
{
std::size_t blen = b.name_length();
return (i.m_len < blen) ||
(i.m_len == blen &&
std::char_traits<char_type>::compare
(i.mp_str, b.name(), i.m_len) < 0);
}

bool operator()(const value_type &b, const intrusive_compare_key_type &i) const
{
std::size_t blen = b.name_length();
return (blen < i.m_len) ||
(blen == i.m_len &&
std::char_traits<char_type>::compare
(b.name(), i.mp_str, i.m_len) < 0);
}
};

#endif   

public:

iset_index(typename MapConfig::segment_manager_base *)
: index_type()
{}

void reserve(typename MapConfig::segment_manager_base::size_type)
{    }

void shrink_to_fit()
{     }

iterator find(const intrusive_compare_key_type &key)
{  return index_type::find(key, intrusive_key_value_less());  }

const_iterator find(const intrusive_compare_key_type &key) const
{  return index_type::find(key, intrusive_key_value_less());  }

std::pair<iterator, bool>insert_check
(const intrusive_compare_key_type &key, insert_commit_data &commit_data)
{  return index_type::insert_check(key, intrusive_key_value_less(), commit_data); }
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template<class MapConfig>
struct is_intrusive_index
<boost::interprocess::iset_index<MapConfig> >
{
static const bool value = true;
};
#endif   

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
