#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept> 

#include "json/json.h"


namespace dg
{
namespace file
{


enum class error{
is_throw, 
is_warning, 
is_silent 
};

enum class comments{
are_kept, 
are_discarded, 
are_forbidden 
};


struct WrappedJsonValue
{
WrappedJsonValue() : m_js(0), m_mode( error::is_throw){}
WrappedJsonValue( error mode): m_js(0), m_mode( mode) {}
WrappedJsonValue(Json::Value js): m_js(js), m_mode( error::is_throw) {}
WrappedJsonValue(Json::Value js, error mode): m_js(js), m_mode( mode) {}
void set_mode( error new_mode){
m_mode = new_mode;
}
const Json::Value& asJson( ) const{ return m_js;}
Json::Value& asJson( ) { return m_js;}


std::string access_string() const {return m_access_str;}

WrappedJsonValue operator[](std::string key) const{
return get( key, Json::ValueType::objectValue, "empty object ");
}
WrappedJsonValue get( std::string key, const Json::Value& value) const{
std::stringstream default_str;
default_str << "value "<<value;
return get( key, value, default_str.str());
}
WrappedJsonValue operator[]( unsigned idx) const{
return get( idx, Json::ValueType::objectValue, "empty array");
}
WrappedJsonValue get( unsigned idx, const Json::Value& value) const{
std::stringstream default_str;
default_str << "value "<<value;
return get( idx, value, default_str.str());
}
unsigned size() const{
return m_js.size();
}
double asDouble( double value = 0) const{
if( m_js.isDouble())
return m_js.asDouble();
return type_error<double>( value, "a Double");
}
unsigned asUInt( unsigned value = 0) const{
if( m_js.isUInt())
return m_js.asUInt();
return type_error<unsigned>( value, "an Unsigned");
}
int asInt( int value = 0) const{
if( m_js.isInt())
return m_js.asInt();
return type_error<int>( value, "an Int");
}
bool asBool( bool value = false) const{
if( m_js.isBool())
return m_js.asBool();
return type_error<bool>( value, "a Bool");
}
std::string asString( std::string value = "") const{
if( m_js.isString())
return m_js.asString();
return type_error<std::string>( value, "a String");
}
private:
WrappedJsonValue(Json::Value js, error mode, std::string access):m_js(js), m_mode( mode), m_access_str(access) {}
WrappedJsonValue get( std::string key, const Json::Value& value, std::string default_str) const
{
std::string access = m_access_str + "\""+key+"\": ";
std::stringstream message;
if( !m_js.isObject( ) || !m_js.isMember(key))
{
message <<"*** Key error: "<<access<<" not found.";
raise_error( message.str(), default_str);
return WrappedJsonValue( value, m_mode, access);
}
return WrappedJsonValue(m_js[key], m_mode, access);
}
WrappedJsonValue get( unsigned idx, const Json::Value& value, std::string default_str) const
{
std::string access = m_access_str + "["+std::to_string(idx)+"] ";
if( !m_js.isArray() || !m_js.isValidIndex(idx))
{
std::stringstream message;
if( m_access_str.empty())
message <<"*** Index error: Index "<<idx<<" not present.";
else
message <<"*** Index error: Index "<<idx<<" not present in "<<m_access_str<<".";
raise_error( message.str(), default_str);
return WrappedJsonValue( value, m_mode, access);
}
return WrappedJsonValue(m_js[idx], m_mode, access);
}
template<class T>
T type_error( T value, std::string type) const
{
std::stringstream message, default_str;
default_str << "value "<<value;
message <<"*** Type error: "<<m_access_str<<" "<<m_js<<" is not "<<type<<".";
raise_error( message.str(), default_str.str());
return value;
}
void raise_error( std::string message, std::string default_str) const
{
if( error::is_throw == m_mode)
throw std::runtime_error( message);
else if ( error::is_warning == m_mode)
std::cerr <<"WARNING "<< message<<" Using default "<<default_str<<"\n";
else
;
}
Json::Value m_js;
error m_mode;
std::string m_access_str = "";
};


static inline void file2Json(std::string filename, Json::Value& js, enum comments comm = file::comments::are_discarded, enum error err = file::error::is_throw)
{
Json::CharReaderBuilder parser;
if( comments::are_forbidden == comm )
Json::CharReaderBuilder::strictMode( &parser.settings_);
else if( comments::are_discarded == comm )
{
Json::CharReaderBuilder::strictMode( &parser.settings_);
Json::Value js_true (true);
Json::Value js_false (false);
parser.settings_["allowComments"].swap( js_true);
parser.settings_["collectComments"].swap(js_false);
}
else
Json::CharReaderBuilder::setDefaults( &parser.settings_);

std::ifstream isI( filename);
if( !isI.good())
{
std::string message = "\nAn error occured while parsing "+filename+"\n";
message +=  "*** File does not exist! *** \n\n";
if( err == error::is_throw)
throw std::runtime_error( message);
else if (err == error::is_warning)
std::cerr << "WARNING: "<<message<<std::endl;
else
;
return;
}
std::string errs;
if( !parseFromStream( parser, isI, &js, &errs) )
{
std::string message = "An error occured while parsing "+filename+"\n"+errs;
if( err == error::is_throw)
throw std::runtime_error( message);
else if (err == error::is_warning)
std::cerr << "WARNING: "<<message<<std::endl;
else
;
return;
}
}

static inline void string2Json(std::string input, Json::Value& js, enum comments comm = file::comments::are_discarded, enum error err = file::error::is_throw)
{
Json::CharReaderBuilder parser;
if( comments::are_forbidden == comm )
Json::CharReaderBuilder::strictMode( &parser.settings_);
else if( comments::are_discarded == comm )
{
Json::CharReaderBuilder::strictMode( &parser.settings_);
parser.settings_["allowComments"] = true;
parser.settings_["collectComments"] = false;
}
else
Json::CharReaderBuilder::setDefaults( &parser.settings_);

std::string errs;
std::stringstream ss(input);
if( !parseFromStream( parser, ss, &js, &errs) )
{
if( err == error::is_throw)
throw std::runtime_error( errs);
else if (err == error::is_warning)
std::cerr << "WARNING: "<<errs<<std::endl;
else
;
return;
}
}

}
}
