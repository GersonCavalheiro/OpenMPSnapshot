#pragma once

#include <exception> 
#include <stdexcept> 
#include <string> 

#include <nlohmann/detail/input/position_t.hpp>
#include <nlohmann/detail/macro_scope.hpp>

namespace nlohmann
{
namespace detail
{


class exception : public std::exception
{
public:
JSON_HEDLEY_RETURNS_NON_NULL
const char* what() const noexcept override
{
return m.what();
}

const int id;

protected:
JSON_HEDLEY_NON_NULL(3)
exception(int id_, const char* what_arg) : id(id_), m(what_arg) {}

static std::string name(const std::string& ename, int id_)
{
return "[json.exception." + ename + "." + std::to_string(id_) + "] ";
}

private:
std::runtime_error m;
};


class parse_error : public exception
{
public:

static parse_error create(int id_, const position_t& pos, const std::string& what_arg)
{
std::string w = exception::name("parse_error", id_) + "parse error" +
position_string(pos) + ": " + what_arg;
return parse_error(id_, pos.chars_read_total, w.c_str());
}

static parse_error create(int id_, std::size_t byte_, const std::string& what_arg)
{
std::string w = exception::name("parse_error", id_) + "parse error" +
(byte_ != 0 ? (" at byte " + std::to_string(byte_)) : "") +
": " + what_arg;
return parse_error(id_, byte_, w.c_str());
}


const std::size_t byte;

private:
parse_error(int id_, std::size_t byte_, const char* what_arg)
: exception(id_, what_arg), byte(byte_) {}

static std::string position_string(const position_t& pos)
{
return " at line " + std::to_string(pos.lines_read + 1) +
", column " + std::to_string(pos.chars_read_current_line);
}
};


class invalid_iterator : public exception
{
public:
static invalid_iterator create(int id_, const std::string& what_arg)
{
std::string w = exception::name("invalid_iterator", id_) + what_arg;
return invalid_iterator(id_, w.c_str());
}

private:
JSON_HEDLEY_NON_NULL(3)
invalid_iterator(int id_, const char* what_arg)
: exception(id_, what_arg) {}
};


class type_error : public exception
{
public:
static type_error create(int id_, const std::string& what_arg)
{
std::string w = exception::name("type_error", id_) + what_arg;
return type_error(id_, w.c_str());
}

private:
JSON_HEDLEY_NON_NULL(3)
type_error(int id_, const char* what_arg) : exception(id_, what_arg) {}
};


class out_of_range : public exception
{
public:
static out_of_range create(int id_, const std::string& what_arg)
{
std::string w = exception::name("out_of_range", id_) + what_arg;
return out_of_range(id_, w.c_str());
}

private:
JSON_HEDLEY_NON_NULL(3)
out_of_range(int id_, const char* what_arg) : exception(id_, what_arg) {}
};


class other_error : public exception
{
public:
static other_error create(int id_, const std::string& what_arg)
{
std::string w = exception::name("other_error", id_) + what_arg;
return other_error(id_, w.c_str());
}

private:
JSON_HEDLEY_NON_NULL(3)
other_error(int id_, const char* what_arg) : exception(id_, what_arg) {}
};
}  
}  
