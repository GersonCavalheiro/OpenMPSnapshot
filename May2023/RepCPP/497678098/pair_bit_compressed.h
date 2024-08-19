
#ifndef SEQAN_INCLUDE_SEQAN_BASIC_PAIR_BIT_PACKED_H_
#define SEQAN_INCLUDE_SEQAN_BASIC_PAIR_BIT_PACKED_H_

namespace seqan {





#pragma pack(push,1)
template <typename T1, typename T2, unsigned BITSIZE1, unsigned BITSIZE2>
struct Pair<T1, T2, BitPacked<BITSIZE1, BITSIZE2> >
{

T1 i1:BITSIZE1;
T2 i2:BITSIZE2;


inline Pair() : i1(T1{}), i2(T2{}) {};
inline Pair(Pair const &) = default;
inline Pair(Pair &&) = default;
inline Pair & operator=(Pair const &) = default;
inline Pair & operator=(Pair &&) = default;
inline ~Pair() = default;

inline Pair(T1 const & _i1, T2 const & _i2) : i1(_i1), i2(_i2) {}

template <typename T1_, typename T2_, typename TSpec__>
inline Pair(Pair<T1_, T2_, TSpec__> const &_p) :
i1(getValueI1(_p)), i2(getValueI2(_p))
{}
};
#pragma pack(pop)





template <typename T1, typename T2, unsigned BITSIZE1, unsigned BITSIZE2>
inline void
set(Pair<T1, T2, BitPacked<BITSIZE1, BITSIZE2> > & p1, Pair<T1, T2, BitPacked<BITSIZE1, BITSIZE2> > & p2)
{
p1 = p2;
}



template <typename T1, typename T2, unsigned BITSIZE1, unsigned BITSIZE2>
inline void
move(Pair<T1, T2, BitPacked<BITSIZE1, BITSIZE2> > & p1, Pair<T1, T2, BitPacked<BITSIZE1, BITSIZE2> > & p2)
{
p1 = p2;
}



template <typename T1, typename T2, typename T, unsigned BITSIZE1, unsigned BITSIZE2>
inline void setValueI1(Pair<T1, T2, BitPacked<BITSIZE1, BITSIZE2> > & pair, T const & _i)
{
pair.i1 = _i;
}

template <typename T1, typename T2, typename T, unsigned BITSIZE1, unsigned BITSIZE2>
inline void setValueI2(Pair<T1, T2, BitPacked<BITSIZE1, BITSIZE2> > & pair, T const & _i)
{
pair.i2 = _i;
}



template <typename T1, typename T2, typename T, unsigned BITSIZE1, unsigned BITSIZE2>
inline void moveValueI1(Pair<T1, T2, BitPacked<BITSIZE1, BITSIZE2> > & pair, T & _i)
{
pair.i1 = _i;
}

template <typename T1, typename T2, typename T, unsigned BITSIZE1, unsigned BITSIZE2>
inline void moveValueI2(Pair<T1, T2, BitPacked<BITSIZE1, BITSIZE2> > & pair, T & _i)
{
pair.i2 = _i;
}

}  

#endif  
