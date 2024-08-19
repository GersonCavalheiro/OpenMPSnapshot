

#ifndef BOOST_MULTI_INDEX_DETAIL_HASH_INDEX_ARGS_HPP
#define BOOST_MULTI_INDEX_DETAIL_HASH_INDEX_ARGS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/functional/hash.hpp>
#include <boost/mpl/aux_/na.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/if.hpp>
#include <boost/multi_index/tag.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <functional>

namespace boost{

namespace multi_index{

namespace detail{



template<typename KeyFromValue>
struct index_args_default_hash
{
typedef ::boost::hash<typename KeyFromValue::result_type> type;
};

template<typename KeyFromValue>
struct index_args_default_pred
{
typedef std::equal_to<typename KeyFromValue::result_type> type;
};

template<typename Arg1,typename Arg2,typename Arg3,typename Arg4>
struct hashed_index_args
{
typedef is_tag<Arg1> full_form;

typedef typename mpl::if_<
full_form,
Arg1,
tag< > >::type                                   tag_list_type;
typedef typename mpl::if_<
full_form,
Arg2,
Arg1>::type                                      key_from_value_type;
typedef typename mpl::if_<
full_form,
Arg3,
Arg2>::type                                      supplied_hash_type;
typedef typename mpl::eval_if<
mpl::is_na<supplied_hash_type>,
index_args_default_hash<key_from_value_type>,
mpl::identity<supplied_hash_type>
>::type                                            hash_type;
typedef typename mpl::if_<
full_form,
Arg4,
Arg3>::type                                      supplied_pred_type;
typedef typename mpl::eval_if<
mpl::is_na<supplied_pred_type>,
index_args_default_pred<key_from_value_type>,
mpl::identity<supplied_pred_type>
>::type                                            pred_type;

BOOST_STATIC_ASSERT(is_tag<tag_list_type>::value);
BOOST_STATIC_ASSERT(!mpl::is_na<key_from_value_type>::value);
BOOST_STATIC_ASSERT(!mpl::is_na<hash_type>::value);
BOOST_STATIC_ASSERT(!mpl::is_na<pred_type>::value);
};

} 

} 

} 

#endif
