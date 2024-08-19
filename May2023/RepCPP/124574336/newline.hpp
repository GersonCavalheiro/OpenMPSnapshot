


#ifndef BOOST_IOSTREAMS_NEWLINE_FILTER_HPP_INCLUDED
#define BOOST_IOSTREAMS_NEWLINE_FILTER_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/assert.hpp>
#include <cstdio>
#include <stdexcept>                       
#include <boost/config.hpp>                
#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/detail/char_traits.hpp>
#include <boost/iostreams/detail/ios.hpp>  
#include <boost/iostreams/read.hpp>        
#include <boost/iostreams/write.hpp>       
#include <boost/iostreams/pipeline.hpp>
#include <boost/iostreams/putback.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits/is_convertible.hpp>

#include <boost/iostreams/detail/config/disable_warnings.hpp>

#define BOOST_IOSTREAMS_ASSERT_UNREACHABLE(val) \
(BOOST_ASSERT("unreachable code" == 0), val) \


namespace boost { namespace iostreams {

namespace newline {

const char CR                   = 0x0D;
const char LF                   = 0x0A;



const int posix             = 1;    
const int mac               = 2;    
const int dos               = 4;    
const int mixed             = 8;    
const int final_newline     = 16;
const int platform_mask     = posix | dos | mac;

} 

namespace detail {

class newline_base {
public:
bool is_posix() const
{
return !is_mixed() && (flags_ & newline::posix) != 0;
}
bool is_dos() const
{
return !is_mixed() && (flags_ & newline::dos) != 0;
}
bool is_mac() const
{
return !is_mixed() && (flags_ & newline::mac) != 0;
}
bool is_mixed_posix() const { return (flags_ & newline::posix) != 0; }
bool is_mixed_dos() const { return (flags_ & newline::dos) != 0; }
bool is_mixed_mac() const { return (flags_ & newline::mac) != 0; }
bool is_mixed() const
{
int platform =
(flags_ & newline::posix) != 0 ?
newline::posix :
(flags_ & newline::dos) != 0 ?
newline::dos :
(flags_ & newline::mac) != 0 ?
newline::mac :
0;
return (flags_ & ~platform & newline::platform_mask) != 0;
}
bool has_final_newline() const
{
return (flags_ & newline::final_newline) != 0;
}
protected:
newline_base(int flags) : flags_(flags) { }
int flags_;
};

} 

class newline_error
: public BOOST_IOSTREAMS_FAILURE, public detail::newline_base
{
private:
friend class newline_checker;
newline_error(int flags)
: BOOST_IOSTREAMS_FAILURE("bad line endings"),
detail::newline_base(flags)
{ }
};

class newline_filter {
public:
typedef char char_type;
struct category
: dual_use,
filter_tag,
closable_tag
{ };

explicit newline_filter(int target) : flags_(target)
{
if ( target != iostreams::newline::posix &&
target != iostreams::newline::dos &&
target != iostreams::newline::mac )
{
boost::throw_exception(std::logic_error("bad flags"));
}
}

template<typename Source>
int get(Source& src)
{
using iostreams::newline::CR;
using iostreams::newline::LF;

BOOST_ASSERT((flags_ & f_write) == 0);
flags_ |= f_read;

if (flags_ & (f_has_LF | f_has_EOF)) {
if (flags_ & f_has_LF)
return newline();
else
return EOF;
}

int c =
(flags_ & f_has_CR) == 0 ?
iostreams::get(src) :
CR;

if (c == WOULD_BLOCK )
return WOULD_BLOCK;

if (c == CR) {
flags_ |= f_has_CR;

int d;
if ((d = iostreams::get(src)) == WOULD_BLOCK)
return WOULD_BLOCK;

if (d == LF) {
flags_ &= ~f_has_CR;
return newline();
}

if (d == EOF) {
flags_ |= f_has_EOF;
} else {
iostreams::putback(src, d);
}

flags_ &= ~f_has_CR;
return newline();
}

if (c == LF)
return newline();

return c;
}

template<typename Sink>
bool put(Sink& dest, char c)
{
using iostreams::newline::CR;
using iostreams::newline::LF;

BOOST_ASSERT((flags_ & f_read) == 0);
flags_ |= f_write;

if ((flags_ & f_has_LF) != 0)
return c == LF ?
newline(dest) :
newline(dest) && this->put(dest, c);

if (c == LF)
return newline(dest);

if ((flags_ & f_has_CR) != 0)
return newline(dest) ?
this->put(dest, c) :
false;

if (c == CR) {
flags_ |= f_has_CR;
return true;
}

return iostreams::put(dest, c);
}

template<typename Sink>
void close(Sink& dest, BOOST_IOS::openmode)
{
if ((flags_ & f_write) != 0 && (flags_ & f_has_CR) != 0)
newline_if_sink(dest);
flags_ &= ~f_has_LF; 
}
private:

int newline()
{
using iostreams::newline::CR;
using iostreams::newline::LF;

switch (flags_ & iostreams::newline::platform_mask) {
case iostreams::newline::posix:
return LF;
case iostreams::newline::mac:
return CR;
case iostreams::newline::dos:
if (flags_ & f_has_LF) {
flags_ &= ~f_has_LF;
return LF;
} else {
flags_ |= f_has_LF;
return CR;
}
}
return BOOST_IOSTREAMS_ASSERT_UNREACHABLE(0);
}

template<typename Sink>
bool newline(Sink& dest)
{
using iostreams::newline::CR;
using iostreams::newline::LF;

bool success = false;
switch (flags_ & iostreams::newline::platform_mask) {
case iostreams::newline::posix:
success = boost::iostreams::put(dest, LF);
break;
case iostreams::newline::mac:
success = boost::iostreams::put(dest, CR);
break;
case iostreams::newline::dos:
if ((flags_ & f_has_LF) != 0) {
if ((success = boost::iostreams::put(dest, LF)))
flags_ &= ~f_has_LF;
} else if (boost::iostreams::put(dest, CR)) {
if (!(success = boost::iostreams::put(dest, LF)))
flags_ |= f_has_LF;
}
break;
}
if (success)
flags_ &= ~f_has_CR;
return success;
}

template<typename Device>
void newline_if_sink(Device& dest) 
{ 
typedef typename iostreams::category_of<Device>::type category;
newline_if_sink(dest, is_convertible<category, output>()); 
}

template<typename Sink>
void newline_if_sink(Sink& dest, mpl::true_) { newline(dest); }

template<typename Source>
void newline_if_sink(Source&, mpl::false_) { }

enum flags {
f_has_LF         = 32768,
f_has_CR         = f_has_LF << 1,
f_has_newline    = f_has_CR << 1,
f_has_EOF        = f_has_newline << 1,
f_read           = f_has_EOF << 1,
f_write          = f_read << 1
};
int       flags_;
};
BOOST_IOSTREAMS_PIPABLE(newline_filter, 0)

class newline_checker : public detail::newline_base {
public:
typedef char                 char_type;
struct category
: dual_use_filter_tag,
closable_tag
{ };
explicit newline_checker(int target = newline::mixed)
: detail::newline_base(0), target_(target), open_(false)
{ }
template<typename Source>
int get(Source& src)
{
using newline::CR;
using newline::LF;

if (!open_) {
open_ = true;
source() = 0;
}

int c;
if ((c = iostreams::get(src)) == WOULD_BLOCK)
return WOULD_BLOCK;

if (c != EOF)
source() &= ~f_line_complete;
if ((source() & f_has_CR) != 0) {
if (c == LF) {
source() |= newline::dos;
source() |= f_line_complete;
} else {
source() |= newline::mac;
if (c == EOF)
source() |= f_line_complete;
}
} else if (c == LF) {
source() |= newline::posix;
source() |= f_line_complete;
}
source() = (source() & ~f_has_CR) | (c == CR ? f_has_CR : 0);

if ( c == EOF &&
(target_ & newline::final_newline) != 0 &&
(source() & f_line_complete) == 0 )
{
fail();
}
if ( (target_ & newline::platform_mask) != 0 &&
(source() & ~target_ & newline::platform_mask) != 0 )
{
fail();
}

return c;
}

template<typename Sink>
bool put(Sink& dest, int c)
{
using iostreams::newline::CR;
using iostreams::newline::LF;

if (!open_) {
open_ = true;
source() = 0;
}

if (!iostreams::put(dest, c))
return false;

source() &= ~f_line_complete;
if ((source() & f_has_CR) != 0) {
if (c == LF) {
source() |= newline::dos;
source() |= f_line_complete;
} else {
source() |= newline::mac;
}
} else if (c == LF) {
source() |= newline::posix;
source() |= f_line_complete;
}
source() = (source() & ~f_has_CR) | (c == CR ? f_has_CR : 0);

if ( (target_ & newline::platform_mask) != 0 &&
(source() & ~target_ & newline::platform_mask) != 0 )
{
fail();
}

return true;
}

template<typename Sink>
void close(Sink&, BOOST_IOS::openmode)
{
using iostreams::newline::final_newline;

if ( (source() & f_has_CR) != 0 ||
(source() & f_line_complete) != 0 )
{
source() |= final_newline;
}

source() &= ~(f_has_CR | f_line_complete);

if ( (target_ & final_newline) != 0 &&
(source() & final_newline) == 0 )
{
fail();
}
}
private:
void fail() { boost::throw_exception(newline_error(source())); }
int& source() { return flags_; }
int source() const { return flags_; }

enum flags {
f_has_CR = 32768,
f_line_complete = f_has_CR << 1
};

int   target_;  
bool  open_;
};
BOOST_IOSTREAMS_PIPABLE(newline_checker, 0)

} } 

#include <boost/iostreams/detail/config/enable_warnings.hpp>

#endif 
