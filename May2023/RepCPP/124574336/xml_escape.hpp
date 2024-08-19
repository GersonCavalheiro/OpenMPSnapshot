#ifndef BOOST_ARCHIVE_ITERATORS_XML_ESCAPE_HPP
#define BOOST_ARCHIVE_ITERATORS_XML_ESCAPE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/assert.hpp>
#include <boost/archive/iterators/escape.hpp>

namespace boost {
namespace archive {
namespace iterators {


template<class Base>
class xml_escape
: public escape<xml_escape<Base>, Base>
{
friend class boost::iterator_core_access;

typedef escape<xml_escape<Base>, Base> super_t;

public:
char fill(const char * & bstart, const char * & bend);
wchar_t fill(const wchar_t * & bstart, const wchar_t * & bend);

template<class T>
xml_escape(T start) :
super_t(Base(static_cast< T >(start)))
{}
xml_escape(const xml_escape & rhs) :
super_t(rhs.base_reference())
{}
};

template<class Base>
char xml_escape<Base>::fill(
const char * & bstart,
const char * & bend
){
char current_value = * this->base_reference();
switch(current_value){
case '<':
bstart = "&lt;";
bend = bstart + 4;
break;
case '>':
bstart = "&gt;";
bend = bstart + 4;
break;
case '&':
bstart = "&amp;";
bend = bstart + 5;
break;
case '"':
bstart = "&quot;";
bend = bstart + 6;
break;
case '\'':
bstart = "&apos;";
bend = bstart + 6;
break;
default:
return current_value;
}
return *bstart;
}

template<class Base>
wchar_t xml_escape<Base>::fill(
const wchar_t * & bstart,
const wchar_t * & bend
){
wchar_t current_value = * this->base_reference();
switch(current_value){
case '<':
bstart = L"&lt;";
bend = bstart + 4;
break;
case '>':
bstart = L"&gt;";
bend = bstart + 4;
break;
case '&':
bstart = L"&amp;";
bend = bstart + 5;
break;
case '"':
bstart = L"&quot;";
bend = bstart + 6;
break;
case '\'':
bstart = L"&apos;";
bend = bstart + 6;
break;
default:
return current_value;
}
return *bstart;
}

} 
} 
} 

#endif 
