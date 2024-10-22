



#ifndef BOOST_IOSTREAMS_DETAIL_CODECVT_HELPER_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_CODECVT_HELPER_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>  
#include <boost/detail/workaround.hpp>
#include <algorithm>         
#include <cstddef>           
#include <locale>            
#include <boost/iostreams/detail/config/codecvt.hpp>


namespace boost { namespace iostreams { namespace detail {

#if !BOOST_WORKAROUND(BOOST_DINKUMWARE_STDLIB, == 1) 

template<typename T>
struct codecvt_intern { typedef typename T::intern_type type; };

template<typename T>
struct codecvt_extern { typedef typename T::extern_type type; };

#else 

template<typename T>
struct codecvt_intern { typedef typename T::from_type type; };

template<typename T>
struct codecvt_extern { typedef typename T::to_type type; };

#endif 

template<typename T>
struct codecvt_state { typedef typename T::state_type type; };

} } } 


#if defined(BOOST_IOSTREAMS_NO_PRIMARY_CODECVT_DEFINITION) || \
defined(BOOST_IOSTREAMS_EMPTY_PRIMARY_CODECVT_DEFINITION) || \
defined(BOOST_IOSTREAMS_NO_LOCALE) \


namespace boost { namespace iostreams { namespace detail {

template<typename Intern, typename Extern, typename State>
struct codecvt_impl : std::locale::facet, std::codecvt_base {
public:
typedef Intern  intern_type;
typedef Extern  extern_type;
typedef State   state_type;

codecvt_impl(std::size_t refs = 0) : std::locale::facet(refs) { } 

std::codecvt_base::result
in( State& state, const Extern* first1, const Extern* last1,
const Extern*& next1, Intern* first2, Intern* last2,
Intern*& next2 ) const
{
return do_in(state, first1, last1, next1, first2, last2, next2);
}

std::codecvt_base::result
out( State& state, const Intern* first1, const Intern* last1,
const Intern*& next1, Extern* first2, Extern* last2,
Extern*& next2 ) const
{
return do_out(state, first1, last1, next1, first2, last2, next2);
}

std::codecvt_base::result
unshift(State& state, Extern* first2, Extern* last2, Extern*& next2) const
{
return do_unshift(state, first2, last2, next2);
}

bool always_noconv() const throw() { return do_always_noconv(); }

int max_length() const throw() { return do_max_length(); }

int encoding() const throw() { return do_encoding(); }

int length( BOOST_IOSTREAMS_CODECVT_CV_QUALIFIER State& state, 
const Extern* first1, const Extern* last1,
std::size_t len2 ) const throw()
{
return do_length(state, first1, last1, len2);
}
protected:
std::codecvt_base::result
virtual do_in( State&, const Extern*, const Extern*, const Extern*&, 
Intern*, Intern*, Intern*& ) const
{
return std::codecvt_base::noconv;
}

std::codecvt_base::result
virtual do_out( State&, const Intern*, const Intern*, const Intern*&, 
Extern*, Extern*, Extern*& ) const
{
return std::codecvt_base::noconv;
}

std::codecvt_base::result
virtual do_unshift(State&, Extern*, Extern*, Extern*&) const
{
return std::codecvt_base::ok;
}

virtual bool do_always_noconv() const throw() { return true; }

virtual int do_max_length() const throw() { return 1; }

virtual int do_encoding() const throw() { return 1; }

virtual int do_length( BOOST_IOSTREAMS_CODECVT_CV_QUALIFIER State&, 
const Extern* first1, const Extern* last1,
std::size_t len2 ) const throw()
{
return (std::min)(static_cast<std::size_t>(last1 - first1), len2);
}
};

} } } 

#endif 


#if defined(BOOST_IOSTREAMS_NO_PRIMARY_CODECVT_DEFINITION) || \
defined(BOOST_IOSTREAMS_EMPTY_PRIMARY_CODECVT_DEFINITION) \

#  define BOOST_IOSTREAMS_CODECVT_SPEC(state) \
namespace std { \
template<typename Intern, typename Extern> \
class codecvt<Intern, Extern, state> \
: public ::boost::iostreams::detail::codecvt_impl< \
Intern, Extern, state \
> \
{ \
public: \
codecvt(std::size_t refs = 0) \
: ::boost::iostreams::detail::codecvt_impl< \
Intern, Extern, state \
>(refs) \
{ } \
static std::locale::id id; \
}; \
template<typename Intern, typename Extern> \
std::locale::id codecvt<Intern, Extern, state>::id; \
} \

#else
# define BOOST_IOSTREAMS_CODECVT_SPEC(state)
#endif 

namespace boost { namespace iostreams { namespace detail {


template<typename Intern, typename Extern, typename State>
struct codecvt_helper : std::codecvt<Intern, Extern, State> { 
typedef Intern  intern_type;
typedef Extern  extern_type;
typedef State   state_type;
codecvt_helper(std::size_t refs = 0) 
#if !defined(BOOST_IOSTREAMS_NO_CODECVT_CTOR_FROM_SIZE_T)
: std::codecvt<Intern, Extern, State>(refs)
#else
: std::codecvt<Intern, Extern, State>()
#endif
{ }
#ifdef BOOST_IOSTREAMS_NO_CODECVT_MAX_LENGTH
int max_length() const throw() { return do_max_length(); }
protected:
virtual int do_max_length() const throw() { return 1; }
#endif
};

} } } 

#endif 
