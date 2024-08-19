#pragma once

int constexpr STATUS_TEST_NOT_INCLUDED = -1;

#ifndef   STATUS_WITH_MESSAGE

typedef int status_t;

#else  

#include <cstdio> 
#include <utility> 
#include <cstdint> 

class status_t
{
private:
int32_t _code;
char _msg[124];
public:

status_t(int const code=0) : _code(code) {
if (code) std::sprintf(_msg, "%i", code); else _msg[0] = 0;
} 

template <class... Args>
status_t(char const *fmt, Args &&... args) : _code(1) {
auto const nc = std::sprintf(_msg, fmt, std::forward<Args>(args)...);
assert(nc <= 124); 
} 

char const * message() const { return _code ? _msg : nullptr; }
int  const      code() const { return _code; }

operator bool() const { return (0 != _code); };

operator int() const { return _code; }

status_t & operator += (int const & rhs) { _code += rhs; return *this; }
status_t & operator ++() { ++_code; return *this; }
bool operator > (int const rhs) { return _code > rhs; }

}; 

#define set_status status_t
inline char const * status_message(status_t const & status) { return status.message(); }

#endif 
