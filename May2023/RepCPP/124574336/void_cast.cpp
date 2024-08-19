


#if (defined _MSC_VER) && (_MSC_VER == 1200)
# pragma warning (disable : 4786) 
#endif

#include <set>
#include <functional>
#include <algorithm>
#include <cstddef> 
#ifdef BOOST_SERIALIZATION_LOG
#include <iostream>
#endif

#include <boost/config.hpp>
#include <boost/assert.hpp>

#define BOOST_SERIALIZATION_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/void_cast.hpp>

namespace boost { 
namespace serialization {
namespace void_cast_detail {

bool void_caster::operator<(const void_caster & rhs) const {
if(m_derived != rhs.m_derived){
if(*m_derived < *rhs.m_derived)
return true;
if(*rhs.m_derived < *m_derived)
return false;
}
if(m_base != rhs.m_base)
return *m_base < *rhs.m_base;
else
return false;
}

struct void_caster_compare {
bool operator()(const void_caster * lhs, const void_caster * rhs) const {
return *lhs < *rhs;
}
};

typedef std::set<const void_caster *, void_caster_compare> set_type;
typedef boost::serialization::singleton<set_type> void_caster_registry;

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

class void_caster_shortcut : public void_caster
{
bool m_includes_virtual_base;

void const * 
vbc_upcast(
void const * const t
) const;
void const *
vbc_downcast(
void const * const t
) const;
void const *
upcast(void const * const t) const BOOST_OVERRIDE {
if(m_includes_virtual_base)
return vbc_upcast(t);
return static_cast<const char *> ( t ) - m_difference;
}
void const *
downcast(void const * const t) const BOOST_OVERRIDE {
if(m_includes_virtual_base)
return vbc_downcast(t);
return static_cast<const char *> ( t ) + m_difference;
}
virtual bool is_shortcut() const {
return true;
}
bool has_virtual_base() const BOOST_OVERRIDE {
return m_includes_virtual_base;
}
public:
void_caster_shortcut(
extended_type_info const * derived,
extended_type_info const * base,
std::ptrdiff_t difference,
bool includes_virtual_base,
void_caster const * const parent
) :
void_caster(derived, base, difference, parent),
m_includes_virtual_base(includes_virtual_base)
{
recursive_register(includes_virtual_base);
}
~void_caster_shortcut() BOOST_OVERRIDE {
recursive_unregister();
}
};

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

void const * 
void_caster_shortcut::vbc_downcast(
void const * const t
) const {
const void_cast_detail::set_type & s
= void_cast_detail::void_caster_registry::get_const_instance();
void_cast_detail::set_type::const_iterator it;
for(it = s.begin(); it != s.end(); ++it){
if ((*it)->m_derived == m_derived){
if ((*it)->m_base != m_base){
const void * t_new;
t_new = void_downcast(*(*it)->m_base, *m_base, t);
if(NULL != t_new){
const void_caster * vc = *it;
return vc->downcast(t_new);
}
}
}
}
return NULL;
}

void const * 
void_caster_shortcut::vbc_upcast(
void const * const t
) const {
const void_cast_detail::set_type & s
= void_cast_detail::void_caster_registry::get_const_instance();
void_cast_detail::set_type::const_iterator it;
for(it = s.begin(); it != s.end(); ++it){
if((*it)->m_base == m_base){
if ((*it)->m_derived != m_derived){
const void * t_new;
t_new = void_upcast(*m_derived, *(*it)->m_derived, t);
if(NULL != t_new)
return (*it)->upcast(t_new);
}
}
}
return NULL;
}

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

class void_caster_argument : public void_caster
{
void const *
upcast(void const * const ) const BOOST_OVERRIDE {
BOOST_ASSERT(false);
return NULL;
}
void const *
downcast( void const * const ) const BOOST_OVERRIDE {
BOOST_ASSERT(false);
return NULL;
}
bool has_virtual_base() const BOOST_OVERRIDE {
BOOST_ASSERT(false);
return false;
}
public:
void_caster_argument(
extended_type_info const * derived,
extended_type_info const * base
) :
void_caster(derived, base)
{}
~void_caster_argument() BOOST_OVERRIDE {}
};

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

BOOST_SERIALIZATION_DECL void
void_caster::recursive_register(bool includes_virtual_base) const {
void_cast_detail::set_type & s
= void_cast_detail::void_caster_registry::get_mutable_instance();

#ifdef BOOST_SERIALIZATION_LOG
std::clog << "recursive_register\n";
std::clog << m_derived->get_debug_info();
std::clog << "<-";
std::clog << m_base->get_debug_info();
std::clog << "\n";
#endif

std::pair<void_cast_detail::set_type::const_iterator, bool> result;
result = s.insert(this);

void_cast_detail::set_type::const_iterator it;
for(it = s.begin(); it != s.end(); ++it){
if(* m_derived == * (*it)->m_base){
const void_caster_argument vca(
(*it)->m_derived, 
m_base
);
void_cast_detail::set_type::const_iterator i;
i = s.find(& vca);
if(i == s.end()){
new void_caster_shortcut(
(*it)->m_derived, 
m_base,
m_difference + (*it)->m_difference,
(*it)->has_virtual_base() || includes_virtual_base,
this
);
}
}
if(* (*it)->m_derived == * m_base){
const void_caster_argument vca(
m_derived, 
(*it)->m_base
);
void_cast_detail::set_type::const_iterator i;
i = s.find(& vca);
if(i == s.end()){
new void_caster_shortcut(
m_derived, 
(*it)->m_base, 
m_difference + (*it)->m_difference,
(*it)->has_virtual_base() || includes_virtual_base,
this
);
}
}
}
}

BOOST_SERIALIZATION_DECL void
void_caster::recursive_unregister() const {
if(void_caster_registry::is_destroyed())
return;

#ifdef BOOST_SERIALIZATION_LOG
std::clog << "recursive_unregister\n";
std::clog << m_derived->get_debug_info();
std::clog << "<-";
std::clog << m_base->get_debug_info();
std::clog << "\n";
#endif

void_cast_detail::set_type & s 
= void_caster_registry::get_mutable_instance();

void_cast_detail::set_type::iterator it;
for(it = s.begin(); it != s.end();){
const void_caster * vc = *it;
if(vc == this){
s.erase(it++);
}
else
if(vc->m_parent == this){
s.erase(it);
delete vc;
it = s.begin();
}
else
it++;
}
}

} 

BOOST_SYMBOL_VISIBLE void const *
void_upcast(
extended_type_info const & derived,
extended_type_info const & base,
void const * const t
);

BOOST_SERIALIZATION_DECL void const *
void_upcast(
extended_type_info const & derived,
extended_type_info const & base,
void const * const t
){
if (derived == base)
return t;

const void_cast_detail::set_type & s
= void_cast_detail::void_caster_registry::get_const_instance();
const void_cast_detail::void_caster_argument ca(& derived, & base);

void_cast_detail::set_type::const_iterator it;
it = s.find(& ca);
if (s.end() != it)
return (*it)->upcast(t);

return NULL;
}

BOOST_SYMBOL_VISIBLE void const *
void_downcast(
extended_type_info const & derived,
extended_type_info const & base,
void const * const t
);

BOOST_SERIALIZATION_DECL void const *
void_downcast(
extended_type_info const & derived,
extended_type_info const & base,
void const * const t
){
if (derived == base)
return t;

const void_cast_detail::set_type & s
= void_cast_detail::void_caster_registry::get_const_instance();
const void_cast_detail::void_caster_argument ca(& derived, & base);

void_cast_detail::set_type::const_iterator it;
it = s.find(&ca);
if (s.end() != it)
return(*it)->downcast(t);

return NULL;
}

} 
} 
