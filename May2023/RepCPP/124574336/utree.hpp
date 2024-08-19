
#if !defined(BOOST_SPIRIT_UTREE)
#define BOOST_SPIRIT_UTREE

#include <cstddef>
#include <algorithm>
#include <string>
#include <iostream>
#include <ios>
#include <sstream>
#include <typeinfo>

#include <boost/io/ios_state.hpp>
#include <boost/integer.hpp>
#include <boost/throw_exception.hpp>
#include <boost/assert.hpp>
#include <boost/noncopyable.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits/is_polymorphic.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/ref.hpp>
#include <boost/config.hpp>

#include <boost/spirit/home/support/utree/detail/utree_detail1.hpp>

#if defined(BOOST_MSVC)
# pragma warning(push)
# pragma warning(disable: 4804)
# pragma warning(disable: 4805)
# pragma warning(disable: 4244)
#endif

namespace boost { namespace spirit
{

struct BOOST_SYMBOL_VISIBLE utree_exception : std::exception {};


struct bad_type_exception ;


struct empty_exception ;


struct utree_type
{
enum info
{
invalid_type,       
nil_type,           
list_type,          
range_type,         
reference_type,     
any_type,           
function_type,      

bool_type,          
int_type,           
double_type,        

string_type,        
string_range_type,  
symbol_type,        

binary_type         
};
typedef boost::uint_t<sizeof(info)*8>::exact exact_integral_type; 
typedef boost::uint_t<sizeof(info)*8>::fast fast_integral_type; 
};

inline std::ostream& operator<<(std::ostream& out, utree_type::info t)
{
boost::io::ios_all_saver saver(out);
switch (t) {
case utree_type::invalid_type: { out << "invalid"; break; }
case utree_type::nil_type: { out << "nil"; break; }
case utree_type::list_type: { out << "list"; break; }
case utree_type::range_type: { out << "range"; break; }
case utree_type::reference_type: { out << "reference"; break; }
case utree_type::any_type: { out << "any"; break; }
case utree_type::function_type: { out << "function"; break; }
case utree_type::bool_type: { out << "bool"; break; }
case utree_type::int_type: { out << "int"; break; }
case utree_type::double_type: { out << "double"; break; }
case utree_type::string_type: { out << "string"; break; }
case utree_type::string_range_type: { out << "string_range"; break; }
case utree_type::symbol_type: { out << "symbol"; break; }
case utree_type::binary_type: { out << "binary"; break; }
default: { out << "unknown"; break; }
}
out << std::hex << "[0x"
<< static_cast<utree_type::fast_integral_type>(t) << "]";
return out;
}

struct bad_type_exception : utree_exception
{
std::string msg;

bad_type_exception(char const* error, utree_type::info got)
: msg()
{
std::ostringstream oss;
oss << "utree: " << error
<< " (got utree type '" << got << "')";
msg = oss.str();
}

bad_type_exception(char const* error, utree_type::info got1,
utree_type::info got2)
: msg()
{
std::ostringstream oss;
oss << "utree: " << error
<< " (got utree types '" << got1 << "' and '" << got2 << "')";
msg = oss.str();
}

virtual ~bad_type_exception() BOOST_NOEXCEPT_OR_NOTHROW {}

virtual char const* what() const BOOST_NOEXCEPT_OR_NOTHROW
{ return msg.c_str(); }
};

struct empty_exception : utree_exception
{
char const* msg;

empty_exception(char const* error) : msg(error) {}

virtual ~empty_exception() BOOST_NOEXCEPT_OR_NOTHROW {}

virtual char const* what() const BOOST_NOEXCEPT_OR_NOTHROW
{ return msg; }
};

template <typename Base, utree_type::info type_>
struct basic_string : Base
{
static utree_type::info const type = type_;

basic_string()
: Base() {}

basic_string(Base const& base)
: Base(base) {}

template <typename Iterator>
basic_string(Iterator bits, std::size_t len)
: Base(bits, bits + len) {}

template <typename Iterator>
basic_string(Iterator first, Iterator last)
: Base(first, last) {}

basic_string& operator=(Base const& other)
{
Base::operator=(other);
return *this;
}
};



typedef basic_string<
boost::iterator_range<char const*>, utree_type::binary_type
> binary_range_type;
typedef basic_string<
std::string, utree_type::binary_type
> binary_string_type;


typedef basic_string<
boost::iterator_range<char const*>, utree_type::string_type
> utf8_string_range_type;
typedef basic_string<
std::string, utree_type::string_type
> utf8_string_type;


typedef basic_string<
boost::iterator_range<char const*>, utree_type::symbol_type
> utf8_symbol_range_type;
typedef basic_string<
std::string, utree_type::symbol_type
> utf8_symbol_type;

class utree;

struct function_base
{
virtual ~function_base() {}
virtual utree operator()(utree const& env) const = 0;
virtual utree operator()(utree& env) const = 0;

virtual function_base* clone() const = 0; 
};

template <typename F>
struct stored_function : function_base
{
F f;
stored_function(F f = F());
virtual ~stored_function();
virtual utree operator()(utree const& env) const;
virtual utree operator()(utree& env) const;
virtual function_base* clone() const;
};

template <typename F>
struct referenced_function : function_base
{
F& f;
referenced_function(F& f);
virtual ~referenced_function();
virtual utree operator()(utree const& env) const;
virtual utree operator()(utree& env) const;
virtual function_base* clone() const;
};

struct shallow_tag {};
shallow_tag const shallow = {};

class any_ptr
{
public:
template <typename Ptr>
typename boost::disable_if<
boost::is_polymorphic<
typename boost::remove_pointer<Ptr>::type>,
Ptr>::type
get() const
{
if (*i == typeid(Ptr))
{
return static_cast<Ptr>(p);
}
boost::throw_exception(std::bad_cast());
}

template <typename T>
any_ptr(T* p)
: p(p), i(&typeid(T*))
{}

friend bool operator==(any_ptr const& a, any_ptr const& b)
{
return (a.p == b.p) && (*a.i == *b.i);
}

private:
any_ptr(void* p, std::type_info const* i)
: p(p), i(i) {}

template <typename UTreeX, typename UTreeY>
friend struct detail::visit_impl;

friend class utree;

void* p;
std::type_info const* i;
};

class utree {
public:
struct invalid_type {};

struct nil_type {};

struct list_type;

typedef utree value_type;
typedef utree& reference;
typedef utree const& const_reference;
typedef std::ptrdiff_t difference_type;
typedef std::size_t size_type;

typedef detail::list::node_iterator<utree> iterator;
typedef detail::list::node_iterator<utree const> const_iterator;

typedef detail::list::node_iterator<boost::reference_wrapper<utree> >
ref_iterator;

typedef boost::iterator_range<iterator> range;
typedef boost::iterator_range<const_iterator> const_range;

~utree();


utree(invalid_type = invalid_type());

utree(nil_type);
reference operator=(nil_type);

explicit utree(bool);
reference operator=(bool);

utree(unsigned int);
utree(int);
reference operator=(unsigned int);
reference operator=(int);

utree(double);
reference operator=(double);

utree(char);
utree(char const*);
utree(char const*, std::size_t);
utree(std::string const&);
reference operator=(char);
reference operator=(char const*);
reference operator=(std::string const&);

utree(utf8_string_range_type const&, shallow_tag);

utree(boost::reference_wrapper<utree>);
reference operator=(boost::reference_wrapper<utree>);

utree(any_ptr const&);
reference operator=(any_ptr const&);

template <class Iterator>
utree(boost::iterator_range<Iterator>);
template <class Iterator>
reference operator=(boost::iterator_range<Iterator>);

utree(function_base const&);
reference operator=(function_base const&);
utree(function_base*);
reference operator=(function_base*);

template <class Base, utree_type::info type_>
utree(basic_string<Base, type_> const&);
template <class Base, utree_type::info type_>
reference operator=(basic_string<Base, type_> const&);

utree(const_reference);
reference operator=(const_reference);

utree(range, shallow_tag);
utree(const_range, shallow_tag);

template <class Iterator>
void assign(Iterator, Iterator);



template <class F>
typename boost::result_of<F(utree const&)>::type
static visit(utree const&, F);

template <class F>
typename boost::result_of<F(utree&)>::type
static visit(utree&, F);

template <class F>
typename boost::result_of<F(utree const&, utree const&)>::type
static visit(utree const&, utree const&, F);

template <class F>
typename boost::result_of<F(utree&, utree const&)>::type
static visit(utree&, utree const&, F);

template <class F>
typename boost::result_of<F(utree const&, utree&)>::type
static visit(utree const&, utree&, F);

template <class F>
typename boost::result_of<F(utree&, utree&)>::type
static visit(utree&, utree&, F);



template <class T>
void push_back(T const&);
template <class T>
void push_front(T const&);
template <class T>
iterator insert(iterator, T const&);
template <class T>
void insert(iterator, std::size_t, T const&);
template <class Iterator>
void insert(iterator, Iterator, Iterator);

void pop_front();
void pop_back();
iterator erase(iterator);
iterator erase(iterator, iterator);

reference front();
const_reference front() const;
iterator begin();
const_iterator begin() const;
ref_iterator ref_begin();

reference back();
const_reference back() const;
iterator end();
const_iterator end() const;
ref_iterator ref_end();

void clear();

void swap(utree&);

bool empty() const;

size_type size() const;



utree_type::info which() const;

template <class T>
T get() const;

reference deref();
const_reference deref() const;

short tag() const;
void tag(short);

utree eval(utree const&) const;
utree eval(utree&) const;

utree operator() (utree const&) const;
utree operator() (utree&) const;
protected:
void ensure_list_type(char const* failed_in = "ensure_list_type()");

private:
typedef utree_type type;

template <class UTreeX, class UTreeY>
friend struct detail::visit_impl;
friend struct detail::index_impl;

type::info get_type() const;
void set_type(type::info);
void free();
void copy(const_reference);

union {
detail::fast_string s;
detail::list l;
detail::range r;
detail::string_range sr;
detail::void_ptr v;
bool b;
int i;
double d;
utree* p;
function_base* pf;
};
};

inline
utree::reference get(utree::reference, utree::size_type);
inline
utree::const_reference get(utree::const_reference, utree::size_type);


struct utree::list_type : utree
{
using utree::operator=;

list_type() : utree() { ensure_list_type("list_type()"); }

template <typename T0>
list_type(T0 t0) : utree(t0) {}

template <typename T0, typename T1>
list_type(T0 t0, T1 t1) : utree(t0, t1) {}
};

utree::invalid_type const invalid = {};
utree::nil_type const nil = {};
utree::list_type const empty_list = utree::list_type();
}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif

