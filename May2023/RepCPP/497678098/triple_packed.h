
#ifndef SEQAN_INCLUDE_SEQAN_TRIPLE_PACKED_H_
#define SEQAN_INCLUDE_SEQAN_TRIPLE_PACKED_H_

namespace seqan {





#pragma pack(push,1)
template <typename T1, typename T2, typename T3>
struct Triple<T1, T2, T3, Pack>
{

T1 i1;
T2 i2;
T3 i3;


inline Triple() : i1(T1()), i2(T2()), i3(T3()) {}

inline Triple(Triple const &_p)
: i1(_p.i1), i2(_p.i2), i3(_p.i3) {}

inline Triple(T1 const &_i1, T2 const &_i2, T3 const &_i3)
: i1(_i1), i2(_i2), i3(_i3) {}

template <typename T1_, typename T2_, typename T3_, typename TSpec__>
inline Triple(Triple<T1_, T2_, T3_, TSpec__> const & _p)
: i1(getValueI1(_p)), i2(getValueI2(_p)), i3(getValueI3(_p)) {}
};
#pragma pack(pop)


template <typename T1, typename T2, typename T3, typename TSpec>
struct MakePacked< Triple<T1, T2, T3, TSpec> >
{
typedef Triple<T1, T2, T3, Pack> Type;
};




template <typename T1, typename T2, typename T3, typename T>
inline void
set(Triple<T1, T2, T3, Pack> & t1, Triple<T1, T2, T3, Pack> & t2)
{
t1 = t2;
}



template <typename T1, typename T2, typename T3, typename T>
inline void
move(Triple<T1, T2, T3, Pack> & t1, Triple<T1, T2, T3, Pack> & t2)
{
t1 = t2;
}



template <typename T1, typename T2, typename T3, typename T>
inline void setValueI1(Triple<T1, T2, T3, Pack> & triple, T const & _i)
{
triple.i1 = _i;
}

template <typename T1, typename T2, typename T3, typename T>
inline void setValueI2(Triple<T1, T2, T3, Pack> & triple, T const & _i)
{
triple.i2 = _i;
}

template <typename T1, typename T2, typename T3, typename T>
inline void setValueI3(Triple<T1, T2, T3, Pack> & triple, T const & _i)
{
triple.i3 = _i;
}



template <typename T1, typename T2, typename T3, typename T>
inline void moveValueI1(Triple<T1, T2, T3, Pack> & triple, T const & _i)
{
triple.i1 = _i;
}

template <typename T1, typename T2, typename T3, typename T>
inline void moveValueI2(Triple<T1, T2, T3, Pack> & triple, T const & _i)
{
triple.i2 = _i;
}

template <typename T1, typename T2, typename T3, typename T>
inline void moveValueI3(Triple<T1, T2, T3, Pack> & triple, T const & _i)
{
triple.i3 = _i;
}

}  

#endif  
