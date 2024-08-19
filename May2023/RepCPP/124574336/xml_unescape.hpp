#ifndef BOOST_ARCHIVE_ITERATORS_XML_UNESCAPE_HPP
#define BOOST_ARCHIVE_ITERATORS_XML_UNESCAPE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/assert.hpp>

#include <boost/serialization/throw_exception.hpp>

#include <boost/archive/iterators/unescape.hpp>
#include <boost/archive/iterators/dataflow_exception.hpp>

namespace boost {
namespace archive {
namespace iterators {

template<class Base>
class xml_unescape
: public unescape<xml_unescape<Base>, Base>
{
friend class boost::iterator_core_access;
typedef xml_unescape<Base> this_t;
typedef unescape<this_t, Base> super_t;
typedef typename boost::iterator_reference<this_t> reference_type;

reference_type dereference() const {
return unescape<xml_unescape<Base>, Base>::dereference();
}
public:
#if BOOST_WORKAROUND(BOOST_MSVC, < 1900)
typedef int value_type;
#else
typedef typename super_t::value_type value_type;
#endif

void drain_residue(const char *literal);
value_type drain();

template<class T>
xml_unescape(T start) :
super_t(Base(static_cast< T >(start)))
{}
xml_unescape(const xml_unescape & rhs) :
super_t(rhs.base_reference())
{}
};

template<class Base>
void xml_unescape<Base>::drain_residue(const char * literal){
do{
if(* literal != * ++(this->base_reference()))
boost::serialization::throw_exception(
dataflow_exception(
dataflow_exception::invalid_xml_escape_sequence
)
);
}
while('\0' != * ++literal);
}

template<class Base>
typename xml_unescape<Base>::value_type
xml_unescape<Base>::drain(){
value_type retval = * this->base_reference();
if('&' != retval){
return retval;
}
retval = * ++(this->base_reference());
switch(retval){
case 'l': 
drain_residue("t;");
retval = '<';
break;
case 'g': 
drain_residue("t;");
retval = '>';
break;
case 'a':
retval = * ++(this->base_reference());
switch(retval){
case 'p': 
drain_residue("os;");
retval = '\'';
break;
case 'm': 
drain_residue("p;");
retval = '&';
break;
}
break;
case 'q':
drain_residue("uot;");
retval = '"';
break;
}
return retval;
}

} 
} 
} 

#endif 
