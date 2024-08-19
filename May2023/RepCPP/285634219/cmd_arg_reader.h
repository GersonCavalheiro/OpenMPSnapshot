



#ifndef _CMDARGREADER_H_
#define _CMDARGREADER_H_

#include <map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <typeinfo>
#include "exception.h"



class CmdArgReader 
{
template<class> friend class TestCmdArgReader;

protected:

static  CmdArgReader*  self;

public:

static void  init( const int argc, const char** argv);

public:

template<class T>
static inline const T* getArg( const std::string& name);

static inline bool existArg( const std::string& name);

static inline int& getRArgc();

static inline char**& getRArgv();

public:

~CmdArgReader();

protected:

CmdArgReader();

private:


template<class T>
inline const T* getArgHelper( const std::string& name);

inline bool existArgHelper( const std::string& name) const;

void  createArgsMaps( const int argc, const char** argv);

template<class T>
static inline bool convertToT( const std::string& element, T& val);

public:


typedef std::pair< const std::type_info*, void*>  ValType;
typedef std::map< std::string, ValType >          ArgsMap;
typedef ArgsMap::iterator                         ArgsMapIter;
typedef ArgsMap::const_iterator                   ConstArgsMapIter;

typedef std::map< std::string, std::string>            UnpMap;
typedef std::map< std::string, std::string>::iterator  UnpMapIter;

private:

#ifdef _WIN32
#  pragma warning( disable: 4251)
#endif

static  int  rargc;

static char**  rargv;

ArgsMap     args;

UnpMap     unprocessed;

ArgsMapIter iter;

UnpMapIter iter_unprocessed;

#ifdef _WIN32
#  pragma warning( default: 4251)
#endif

private:

CmdArgReader( const CmdArgReader&);

CmdArgReader& operator=( const CmdArgReader&);
};



template<class T>
inline bool
CmdArgReader::convertToT( const std::string& element, T& val)
{
val.resize( std::count( element.begin(), element.end(), ',') + 1);

unsigned int i = 0;
std::string::size_type pos_start = 1;  
std::string::size_type pos_end = 0;

while( std::string::npos != ( pos_end = element.find(',', pos_end+1)) ) 
{
if ( ! convertToT< typename T::value_type >( 
std::string( element, pos_start, pos_end - pos_start), val[i])) 
{
return false;
}

pos_start = pos_end + 1;
++i;
}

std::string tmp1(  element, pos_start, element.length() - pos_start - 1);

if ( ! convertToT< typename T::value_type >( std::string( element,
pos_start,
element.length() - pos_start - 1),
val[i])) 
{
return false;
}

return true;
}

template<>
inline bool
CmdArgReader::convertToT<int>( const std::string& element, int& val) 
{
std::istringstream ios( element);
ios >> val;

bool ret_val = false;
if ( ios.eof()) 
{
ret_val = true;
}

return ret_val;
}

template<>
inline bool
CmdArgReader::convertToT<float>( const std::string& element, float& val) 
{
std::istringstream ios( element);
ios >> val;

bool ret_val = false;
if ( ios.eof()) 
{
ret_val = true;
}

return ret_val;
}

template<>
inline bool
CmdArgReader::convertToT<double>( const std::string& element, double& val) 
{
std::istringstream ios( element);
ios >> val;

bool ret_val = false;
if ( ios.eof()) 
{
ret_val = true;
}

return ret_val;
}

template<>
inline bool
CmdArgReader::convertToT<std::string>( const std::string& element, 
std::string& val)
{
val = element;
return true;
}

template<>
inline bool
CmdArgReader::convertToT<bool>( const std::string& element, bool& val) 
{
if ( "true" == element) 
{
val = true;
return true;
}
else if ( "false" == element) 
{
val = false;
return true;
}
else 
{
int tmp;
if ( convertToT<int>( element, tmp)) 
{
if ( 1 == tmp) 
{
val = true;
return true;
}
else if ( 0 == tmp) 
{
val = false;
return true;
}
}
}

return false;
}

template<class T>
const T*
CmdArgReader::getArg( const std::string& name) 
{
if( ! self) 
{
RUNTIME_EXCEPTION("CmdArgReader::getArg(): CmdArgReader not initialized.");
return NULL;
}

return self->getArgHelper<T>( name);
}

inline bool 
CmdArgReader::existArg( const std::string& name) 
{
if( ! self) 
{
RUNTIME_EXCEPTION("CmdArgReader::getArg(): CmdArgReader not initialized.");
return false;
}

return self->existArgHelper( name);
}

template<class T>
const T*
CmdArgReader::getArgHelper( const std::string& name) 
{
if ( args.end() != (iter = args.find( name))) 
{
if ( (*(iter->second.first)) == typeid( T) ) 
{
return (T*) iter->second.second;
}
}
else 
{
T* tmp = new T;

if ( unprocessed.end() != (iter_unprocessed = unprocessed.find( name))) 
{
if ( convertToT< T >( iter_unprocessed->second, *tmp)) 
{
args[name] = std::make_pair( &(typeid( T)), (void*) tmp);

return tmp;
}
}

delete tmp;
}

return NULL;
}

inline bool
CmdArgReader::existArgHelper( const std::string& name) const 
{
bool ret_val = false;

if( args.end() != args.find( name)) 
{
ret_val = true;
}
else 
{

if ( unprocessed.end() != unprocessed.find( name)) 
{
ret_val = true; 
}
}

return ret_val;
}

inline int&
CmdArgReader::getRArgc() 
{
if( ! self) 
{
RUNTIME_EXCEPTION("CmdArgReader::getRArgc(): CmdArgReader not initialized.");
}

return rargc;
}

inline char**&
CmdArgReader::getRArgv() 
{
if( ! self) 
{
RUNTIME_EXCEPTION("CmdArgReader::getRArgc(): CmdArgReader not initialized.");
}

return rargv;
}


#endif 

