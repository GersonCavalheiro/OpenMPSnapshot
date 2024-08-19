
#ifndef BOOST_INTERPROCESS_EXCEPTIONS_HPP
#define BOOST_INTERPROCESS_EXCEPTIONS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/errors.hpp>
#include <stdexcept>


namespace boost {

namespace interprocess {

class BOOST_SYMBOL_VISIBLE interprocess_exception : public std::exception
{
public:
interprocess_exception(const char *err)
:  m_err(other_error)
{
try   {  m_str = err; }
catch (...) {}
}

interprocess_exception(const error_info &err_info, const char *str = 0)
:  m_err(err_info)
{
try{
if(m_err.get_native_error() != 0){
fill_system_message(m_err.get_native_error(), m_str);
}
else if(str){
m_str = str;
}
else{
m_str = "boost::interprocess_exception::library_error";
}
}
catch(...){}
}

~interprocess_exception() BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE {}

const char * what() const BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE
{  return m_str.c_str();  }

native_error_t get_native_error()const { return m_err.get_native_error(); }

error_code_t   get_error_code()  const { return m_err.get_error_code(); }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
error_info        m_err;
std::string       m_str;
#endif   
};

class BOOST_SYMBOL_VISIBLE lock_exception : public interprocess_exception
{
public:
lock_exception()
:  interprocess_exception(lock_error)
{}

const char* what() const BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE
{  return "boost::interprocess::lock_exception";  }
};


class BOOST_SYMBOL_VISIBLE bad_alloc : public interprocess_exception
{
public:
bad_alloc() : interprocess_exception("::boost::interprocess::bad_alloc") {}

const char* what() const BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE
{  return "boost::interprocess::bad_alloc";  }
};

}  

}  

#include <boost/interprocess/detail/config_end.hpp>

#endif 
