


#include <boost/config.hpp> 

#include <boost/assert.hpp>
#include <set>
#include <cstddef> 

#include <boost/limits.hpp>

#define BOOST_ARCHIVE_SOURCE
#define BOOST_SERIALIZATION_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/serialization/state_saver.hpp>
#include <boost/serialization/throw_exception.hpp>
#include <boost/serialization/extended_type_info.hpp>

#include <boost/archive/detail/decl.hpp>
#include <boost/archive/basic_archive.hpp>
#include <boost/archive/detail/basic_oserializer.hpp>
#include <boost/archive/detail/basic_pointer_oserializer.hpp>
#include <boost/archive/detail/basic_oarchive.hpp>
#include <boost/archive/archive_exception.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4251 4231 4660 4275)
#endif

using namespace boost::serialization;

namespace boost {
namespace archive {
namespace detail {

class basic_oarchive_impl {
friend class basic_oarchive;
unsigned int m_flags;

struct aobject
{
const void * address;
class_id_type class_id;
object_id_type object_id;

bool operator<(const aobject &rhs) const
{
BOOST_ASSERT(NULL != address);
BOOST_ASSERT(NULL != rhs.address);
if( address < rhs.address )
return true;
if( address > rhs.address )
return false;
return class_id < rhs.class_id;
}
aobject & operator=(const aobject & rhs)
{
address = rhs.address;
class_id = rhs.class_id;
object_id = rhs.object_id;
return *this;
}
aobject(
const void *a,
class_id_type class_id_,
object_id_type object_id_
) :
address(a),
class_id(class_id_),
object_id(object_id_)
{}
aobject() : address(NULL){}
};
typedef std::set<aobject> object_set_type;
object_set_type object_set;

struct cobject_type
{
const basic_oserializer * m_bos_ptr;
const class_id_type m_class_id;
bool m_initialized;
cobject_type(
std::size_t class_id,
const basic_oserializer & bos
) :
m_bos_ptr(& bos),
m_class_id(class_id),
m_initialized(false)
{}
cobject_type(const basic_oserializer & bos) :
m_bos_ptr(& bos),
m_initialized(false)
{}
cobject_type(
const cobject_type & rhs
) :
m_bos_ptr(rhs.m_bos_ptr),
m_class_id(rhs.m_class_id),
m_initialized(rhs.m_initialized)
{}
cobject_type & operator=(const cobject_type &rhs);
bool operator<(const cobject_type &rhs) const {
return *m_bos_ptr < *(rhs.m_bos_ptr);
}
};
typedef std::set<cobject_type> cobject_info_set_type;
cobject_info_set_type cobject_info_set;

std::set<object_id_type> stored_pointers;

const void * pending_object;
const basic_oserializer * pending_bos;

basic_oarchive_impl(unsigned int flags) :
m_flags(flags),
pending_object(NULL),
pending_bos(NULL)
{}

const cobject_type &
find(const basic_oserializer & bos);
const basic_oserializer *  
find(const serialization::extended_type_info &ti) const;

const cobject_type &
register_type(const basic_oserializer & bos);
void save_object(
basic_oarchive & ar,
const void *t,
const basic_oserializer & bos
);
void save_pointer(
basic_oarchive & ar,
const void * t, 
const basic_pointer_oserializer * bpos
);
};


inline const basic_oserializer *
basic_oarchive_impl::find(const serialization::extended_type_info & ti) const {
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif
class bosarg : 
public basic_oserializer
{
bool class_info() const BOOST_OVERRIDE {
BOOST_ASSERT(false); 
return false;
}
bool tracking(const unsigned int) const BOOST_OVERRIDE {
BOOST_ASSERT(false);
return false;
}
version_type version() const BOOST_OVERRIDE {
BOOST_ASSERT(false);
return version_type(0);
}
bool is_polymorphic() const BOOST_OVERRIDE {
BOOST_ASSERT(false);
return false;
}
void save_object_data(      
basic_oarchive & , const void * 
) const BOOST_OVERRIDE {
BOOST_ASSERT(false);
}
public:
bosarg(const serialization::extended_type_info & eti) :
boost::archive::detail::basic_oserializer(eti)
{}
};
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
bosarg bos(ti);
cobject_info_set_type::const_iterator cit 
= cobject_info_set.find(cobject_type(bos));
if(cit == cobject_info_set.end()){
return NULL;
}
return cit->m_bos_ptr;
}

inline const basic_oarchive_impl::cobject_type &
basic_oarchive_impl::find(const basic_oserializer & bos)
{
std::pair<cobject_info_set_type::iterator, bool> cresult = 
cobject_info_set.insert(cobject_type(cobject_info_set.size(), bos));
return *(cresult.first);
}

inline const basic_oarchive_impl::cobject_type &
basic_oarchive_impl::register_type(
const basic_oserializer & bos
){
cobject_type co(cobject_info_set.size(), bos);
std::pair<cobject_info_set_type::const_iterator, bool>
result = cobject_info_set.insert(co);
return *(result.first);
}

inline void
basic_oarchive_impl::save_object(
basic_oarchive & ar,
const void *t,
const basic_oserializer & bos
){
if(t == pending_object && pending_bos == & bos){
ar.end_preamble();
(bos.save_object_data)(ar, t);
return;
}

const cobject_type & co = register_type(bos);
if(bos.class_info()){
if( ! co.m_initialized){
ar.vsave(class_id_optional_type(co.m_class_id));
ar.vsave(tracking_type(bos.tracking(m_flags)));
ar.vsave(version_type(bos.version()));
(const_cast<cobject_type &>(co)).m_initialized = true;
}
}

if(! bos.tracking(m_flags)){
ar.end_preamble();
(bos.save_object_data)(ar, t);
return;
}

object_id_type oid(object_set.size());
basic_oarchive_impl::aobject ao(t, co.m_class_id, oid);
std::pair<basic_oarchive_impl::object_set_type::const_iterator, bool>
aresult = object_set.insert(ao);
oid = aresult.first->object_id;

if(aresult.second){
ar.vsave(oid);
ar.end_preamble();
(bos.save_object_data)(ar, t);
return;
}

if(stored_pointers.end() != stored_pointers.find(oid)){
boost::serialization::throw_exception(
archive_exception(archive_exception::pointer_conflict)
);
}
ar.vsave(object_reference_type(oid));
ar.end_preamble();
}

inline void
basic_oarchive_impl::save_pointer(
basic_oarchive & ar,
const void * t, 
const basic_pointer_oserializer * bpos_ptr
){
const basic_oserializer & bos = bpos_ptr->get_basic_serializer();
std::size_t original_count = cobject_info_set.size();
const cobject_type & co = register_type(bos);
if(! co.m_initialized){
ar.vsave(co.m_class_id);
if((cobject_info_set.size() > original_count)){
if(bos.is_polymorphic()){
const serialization::extended_type_info *eti = & bos.get_eti();
const char * key = NULL;
if(NULL != eti)
key = eti->get_key();
if(NULL != key){
const class_name_type cn(key);
if(cn.size() > (BOOST_SERIALIZATION_MAX_KEY_SIZE - 1))
boost::serialization::throw_exception(
boost::archive::archive_exception(
boost::archive::archive_exception::
invalid_class_name)
);
ar.vsave(cn);
}
else
boost::serialization::throw_exception(
archive_exception(archive_exception::unregistered_class)
);
}
}
if(bos.class_info()){
ar.vsave(tracking_type(bos.tracking(m_flags)));
ar.vsave(version_type(bos.version()));
}
(const_cast<cobject_type &>(co)).m_initialized = true;
}
else{
ar.vsave(class_id_reference_type(co.m_class_id));
}

if(! bos.tracking(m_flags)){
ar.end_preamble();
serialization::state_saver<const void *> x(pending_object);
serialization::state_saver<const basic_oserializer *> y(pending_bos);
pending_object = t;
pending_bos = & bpos_ptr->get_basic_serializer();
bpos_ptr->save_object_ptr(ar, t);
return;
}

object_id_type oid(object_set.size());
basic_oarchive_impl::aobject ao(t, co.m_class_id, oid);
std::pair<basic_oarchive_impl::object_set_type::const_iterator, bool>
aresult = object_set.insert(ao);
oid = aresult.first->object_id;
if(! aresult.second){
ar.vsave(object_reference_type(oid));
ar.end_preamble();
return;
}

ar.vsave(oid);
ar.end_preamble();

serialization::state_saver<const void *> x(pending_object);
serialization::state_saver<const basic_oserializer *> y(pending_bos);
pending_object = t;
pending_bos = & bpos_ptr->get_basic_serializer();
bpos_ptr->save_object_ptr(ar, t);
stored_pointers.insert(oid);
}

} 
} 
} 


namespace boost {
namespace archive {
namespace detail {

BOOST_ARCHIVE_DECL 
basic_oarchive::basic_oarchive(unsigned int flags)
: pimpl(new basic_oarchive_impl(flags))
{}

BOOST_ARCHIVE_DECL 
basic_oarchive::~basic_oarchive()
{}

BOOST_ARCHIVE_DECL void 
basic_oarchive::save_object(
const void *x, 
const basic_oserializer & bos
){
pimpl->save_object(*this, x, bos);
}

BOOST_ARCHIVE_DECL void 
basic_oarchive::save_pointer(
const void * t, 
const basic_pointer_oserializer * bpos_ptr
){
pimpl->save_pointer(*this, t, bpos_ptr);
}

BOOST_ARCHIVE_DECL void 
basic_oarchive::register_basic_serializer(const basic_oserializer & bos){
pimpl->register_type(bos);
}

BOOST_ARCHIVE_DECL library_version_type
basic_oarchive::get_library_version() const{
return BOOST_ARCHIVE_VERSION();
}

BOOST_ARCHIVE_DECL unsigned int
basic_oarchive::get_flags() const{
return pimpl->m_flags;
}

BOOST_ARCHIVE_DECL void 
basic_oarchive::end_preamble(){
}

BOOST_ARCHIVE_DECL helper_collection &
basic_oarchive::get_helper_collection(){
return *this;
}

} 
} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
