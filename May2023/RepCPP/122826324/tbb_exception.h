

#ifndef __TBB_exception_H
#define __TBB_exception_H

#include "tbb_stddef.h"

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <stdexcept>
#include <string> 

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

namespace tbb {

class bad_last_alloc : public std::bad_alloc {
public:
const char* what() const throw();
#if __TBB_DEFAULT_DTOR_THROW_SPEC_BROKEN
~bad_last_alloc() throw() {}
#endif
};

class improper_lock : public std::exception {
public:
const char* what() const throw();
};

class user_abort : public std::exception {
public:
const char* what() const throw();
};

class missing_wait : public std::exception {
public:
const char* what() const throw();
};

class invalid_multiple_scheduling : public std::exception {
public:
const char* what() const throw();
};

namespace internal {
void __TBB_EXPORTED_FUNC throw_bad_last_alloc_exception_v4();

enum exception_id {
eid_bad_alloc = 1,
eid_bad_last_alloc,
eid_nonpositive_step,
eid_out_of_range,
eid_segment_range_error,
eid_index_range_error,
eid_missing_wait,
eid_invalid_multiple_scheduling,
eid_improper_lock,
eid_possible_deadlock,
eid_operation_not_permitted,
eid_condvar_wait_failed,
eid_invalid_load_factor,
eid_reserved, 
eid_invalid_swap,
eid_reservation_length_error,
eid_invalid_key,
eid_user_abort,

eid_max
};


void __TBB_EXPORTED_FUNC throw_exception_v4 ( exception_id );

inline void throw_exception ( exception_id eid ) { throw_exception_v4(eid); }

} 
} 

#if __TBB_TASK_GROUP_CONTEXT
#include "tbb_allocator.h"
#include <exception>
#include <typeinfo>
#include <new>

namespace tbb {


class tbb_exception : public std::exception
{

void* operator new ( size_t );

public:

virtual tbb_exception* move () throw() = 0;


virtual void destroy () throw() = 0;


virtual void throw_self () = 0;

virtual const char* name() const throw() = 0;

virtual const char* what() const throw() = 0;


void operator delete ( void* p ) {
internal::deallocate_via_handler_v3(p);
}
};


class captured_exception : public tbb_exception
{
public:
captured_exception ( const captured_exception& src )
: tbb_exception(src), my_dynamic(false)
{
set(src.my_exception_name, src.my_exception_info);
}

captured_exception ( const char* name_, const char* info )
: my_dynamic(false)
{
set(name_, info);
}

__TBB_EXPORTED_METHOD ~captured_exception () throw();

captured_exception& operator= ( const captured_exception& src ) {
if ( this != &src ) {
clear();
set(src.my_exception_name, src.my_exception_info);
}
return *this;
}


captured_exception* __TBB_EXPORTED_METHOD move () throw();


void __TBB_EXPORTED_METHOD destroy () throw();


void throw_self () { __TBB_THROW(*this); }


const char* __TBB_EXPORTED_METHOD name() const throw();


const char* __TBB_EXPORTED_METHOD what() const throw();

void __TBB_EXPORTED_METHOD set ( const char* name, const char* info ) throw();
void __TBB_EXPORTED_METHOD clear () throw();

private:
captured_exception() {}

static captured_exception* allocate ( const char* name, const char* info );

bool my_dynamic;
const char* my_exception_name;
const char* my_exception_info;
};


template<typename ExceptionData>
class movable_exception : public tbb_exception
{
typedef movable_exception<ExceptionData> self_type;

public:
movable_exception ( const ExceptionData& data_ )
: my_exception_data(data_)
, my_dynamic(false)
, my_exception_name(
#if TBB_USE_EXCEPTIONS
typeid(self_type).name()
#else 
"movable_exception"
#endif 
)
{}

movable_exception ( const movable_exception& src ) throw ()
: tbb_exception(src)
, my_exception_data(src.my_exception_data)
, my_dynamic(false)
, my_exception_name(src.my_exception_name)
{}

~movable_exception () throw() {}

const movable_exception& operator= ( const movable_exception& src ) {
if ( this != &src ) {
my_exception_data = src.my_exception_data;
my_exception_name = src.my_exception_name;
}
return *this;
}

ExceptionData& data () throw() { return my_exception_data; }

const ExceptionData& data () const throw() { return my_exception_data; }

const char* name () const throw() { return my_exception_name; }

const char* what () const throw() { return "tbb::movable_exception"; }


movable_exception* move () throw() {
void* e = internal::allocate_via_handler_v3(sizeof(movable_exception));
if ( e ) {
::new (e) movable_exception(*this);
((movable_exception*)e)->my_dynamic = true;
}
return (movable_exception*)e;
}

void destroy () throw() {
__TBB_ASSERT ( my_dynamic, "Method destroy can be called only on dynamically allocated movable_exceptions" );
if ( my_dynamic ) {
this->~movable_exception();
internal::deallocate_via_handler_v3(this);
}
}

void throw_self () { __TBB_THROW( *this ); }

protected:
ExceptionData  my_exception_data;

private:
bool my_dynamic;


const char* my_exception_name;
};

#if !TBB_USE_CAPTURED_EXCEPTION
namespace internal {


class tbb_exception_ptr {
std::exception_ptr  my_ptr;

public:
static tbb_exception_ptr* allocate ();
static tbb_exception_ptr* allocate ( const tbb_exception& tag );
static tbb_exception_ptr* allocate ( captured_exception& src );


void destroy () throw();

void throw_self () { std::rethrow_exception(my_ptr); }

private:
tbb_exception_ptr ( const std::exception_ptr& src ) : my_ptr(src) {}
tbb_exception_ptr ( const captured_exception& src ) :
#if __TBB_MAKE_EXCEPTION_PTR_PRESENT
my_ptr(std::make_exception_ptr(src))  
#else
my_ptr(std::copy_exception(src))      
#endif
{}
}; 

} 
#endif 

} 

#endif 

#endif 
