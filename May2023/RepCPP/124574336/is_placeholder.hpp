#ifndef BOOST_IS_PLACEHOLDER_HPP_INCLUDED
#define BOOST_IS_PLACEHOLDER_HPP_INCLUDED


#if defined( _MSC_VER ) && ( _MSC_VER >= 1020 )
# pragma once
#endif




namespace boost
{

template< class T > struct is_placeholder
{
enum _vt { value = 0 };
};

} 

#endif 
