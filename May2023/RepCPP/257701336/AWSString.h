

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSAllocator.h>

#include <functional>
#include <string>

namespace Aws
{

#if defined(_GLIBCXX_FULLY_DYNAMIC_STRING) && _GLIBCXX_FULLY_DYNAMIC_STRING == 0 && defined(__ANDROID__)



using AndroidBasicString = std::basic_string< char, std::char_traits< char >, Aws::Allocator< char > >;

class String : public AndroidBasicString
{
public:
using Base = AndroidBasicString;

String() : Base("") {} 
String(const String& rhs ) : Base(rhs) {}
String(String&& rhs) : Base(rhs) {} 
String(const AndroidBasicString& rhs) : Base(rhs) {}
String(AndroidBasicString&& rhs) : Base(rhs) {} 
String(const char* str) : Base(str) {}
String(const char* str_begin, const char* str_end) : Base(str_begin, str_end) {}
String(const AndroidBasicString& str, size_type pos, size_type count) : Base(str, pos, count) {} 
String(const String& str, size_type pos, size_type count) : Base(str, pos, count) {}
String(const char* str, size_type count) : Base(str, count) {}
String(size_type count, char c) : Base(count, c) {}
String(std::initializer_list<char> __l) : Base(__l) {}

template<class _InputIterator>
String(_InputIterator __beg, _InputIterator __end) : Base(__beg, __end) {}

String& operator=(const String& rhs) { Base::operator=(rhs); return *this; }
String& operator=(String&& rhs) { Base::operator=(rhs); return *this; } 
String& operator=(const AndroidBasicString& rhs) { Base::operator=(rhs); return *this; }
String& operator=(AndroidBasicString&& rhs) { Base::operator=(rhs); return *this; } 
String& operator=(const char* str) { Base::operator=(str); return *this; }
};

#else

using String = std::basic_string< char, std::char_traits< char >, Aws::Allocator< char > >;

#ifdef _WIN32
using WString = std::basic_string< wchar_t, std::char_traits< wchar_t >, Aws::Allocator< wchar_t > >;
#endif

#endif 

} 



