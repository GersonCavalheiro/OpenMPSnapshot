

#ifndef BOOST_BIMAP_DETAIL_MANAGE_ADDITIONAL_PARAMETERS_HPP
#define BOOST_BIMAP_DETAIL_MANAGE_ADDITIONAL_PARAMETERS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <memory>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/aux_/na.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/bimap/detail/is_set_type_of.hpp>

namespace boost {
namespace bimaps {

template< class Type >
struct with_info
{
typedef Type value_type;
};

namespace detail {


template< class Type >
struct is_with_info : ::boost::mpl::false_ {};

template< class ValueType >
struct is_with_info< with_info<ValueType> > : ::boost::mpl::true_ {};



#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

template< class AP1, class AP2, class AP3 >
struct manage_additional_parameters
{

struct case_NNN
{
typedef left_based set_type_of_relation;
typedef std::allocator<char> allocator;
typedef ::boost::mpl::na additional_info;
};


struct case_ANN
{
typedef left_based set_type_of_relation;
typedef AP1 allocator;
typedef ::boost::mpl::na additional_info;
};


struct case_SNN
{
typedef AP1 set_type_of_relation;
typedef std::allocator<char> allocator;
typedef ::boost::mpl::na additional_info;
};


struct case_SAN
{
typedef AP1 set_type_of_relation;
typedef AP2 allocator;
typedef ::boost::mpl::na additional_info;
};


struct case_HNN
{
typedef left_based set_type_of_relation;
typedef std::allocator<char> allocator;
typedef BOOST_DEDUCED_TYPENAME AP1::value_type additional_info;
};


struct case_SHN
{
typedef AP1 set_type_of_relation;
typedef std::allocator<char> allocator;
typedef BOOST_DEDUCED_TYPENAME AP2::value_type additional_info;
};


struct case_HAN
{
typedef left_based set_type_of_relation;
typedef AP2 allocator;
typedef BOOST_DEDUCED_TYPENAME AP1::value_type additional_info;
};


struct case_SHA
{
typedef AP1 set_type_of_relation;
typedef AP2 allocator;
typedef BOOST_DEDUCED_TYPENAME AP2::value_type additional_info;
};


typedef BOOST_DEDUCED_TYPENAME mpl::if_
<
::boost::mpl::is_na<AP1>,
case_NNN, 
BOOST_DEDUCED_TYPENAME mpl::if_
<
::boost::mpl::is_na<AP2>,
BOOST_DEDUCED_TYPENAME mpl::if_
<
is_set_type_of_relation<AP1>,
case_SNN, 
BOOST_DEDUCED_TYPENAME mpl::if_
<
is_with_info<AP1>,
case_HNN, 
case_ANN  

>::type

>::type,
BOOST_DEDUCED_TYPENAME mpl::if_
<
::boost::mpl::is_na<AP3>,
BOOST_DEDUCED_TYPENAME mpl::if_
<
is_with_info<AP1>,
case_HAN, 
BOOST_DEDUCED_TYPENAME mpl::if_
<
is_with_info<AP2>,
case_SHN, 
case_SAN  

>::type

>::type,

case_SHA 

>::type

>::type

>::type type;

};

#endif 

} 
} 
} 


#endif 

