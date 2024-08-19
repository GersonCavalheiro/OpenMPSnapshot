#pragma once
namespace glm
{
enum profile
{
nice,
fast,
simd
};
typedef std::size_t sizeType;
namespace detail
{
template
<
typename VALTYPE, 
template <typename> class TYPE
>
struct genType
{
public:
enum ctor{null};
typedef VALTYPE value_type;
typedef VALTYPE & value_reference;
typedef VALTYPE * value_pointer;
typedef VALTYPE const * value_const_pointer;
typedef TYPE<bool> bool_type;
typedef sizeType size_type;
static bool is_vector();
static bool is_matrix();
typedef TYPE<VALTYPE> type;
typedef TYPE<VALTYPE> * pointer;
typedef TYPE<VALTYPE> const * const_pointer;
typedef TYPE<VALTYPE> const * const const_pointer_const;
typedef TYPE<VALTYPE> * const pointer_const;
typedef TYPE<VALTYPE> & reference;
typedef TYPE<VALTYPE> const & const_reference;
typedef TYPE<VALTYPE> const & param_type;
value_const_pointer value_address() const{return value_pointer(this);}
value_pointer value_address(){return value_pointer(this);}
};
template
<
typename VALTYPE, 
template <typename> class TYPE
>
bool genType<VALTYPE, TYPE>::is_vector()
{
return true;
}
}
}
