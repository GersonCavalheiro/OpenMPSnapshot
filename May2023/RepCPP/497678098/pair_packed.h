
#ifndef SEQAN_INCLUDE_SEQAN_BASIC_PAIR_PACKED_H_
#define SEQAN_INCLUDE_SEQAN_BASIC_PAIR_PACKED_H_

namespace seqan {






#pragma pack(push,1)
template <typename T1, typename T2>
struct Pair<T1, T2, Pack>
{
T1 i1{};
T2 i2{};


#if defined(COMPILER_GCC) && (__GNUC__ <= 4)
Pair() : i1(T1()), i2(T2()) {};
#else
Pair() = default;
#endif

#if defined(COMPILER_LINTEL) || defined(COMPILER_WINTEL)
Pair(Pair const & p) : i1(p.i1), i2(p.i2) {};
#else
Pair(Pair const &) = default;
#endif
Pair(Pair &&) = default;
~Pair() = default;
Pair & operator=(Pair const &) = default;
Pair & operator=(Pair &&) = default;

Pair(T1 const & _i1, T2 const & _i2) : i1(_i1), i2(_i2) {}

template <typename T1_, typename T2_, typename TSpec__>
Pair(Pair<T1_, T2_, TSpec__> const &_p) :
i1(getValueI1(_p)), i2(getValueI2(_p))
{}
};
#pragma pack(pop)


template <typename T1, typename T2>
struct MakePacked< Pair<T1, T2> >
{
typedef Pair<T1, T2, Pack> Type;
};




template <typename T1, typename T2>
inline void
set(Pair<T1, T2, Pack> & p1, Pair<T1, T2, Pack> & p2)
{
p1 = p2;
}



template <typename T1, typename T2>
inline void
move(Pair<T1, T2, Pack> & p1, Pair<T1, T2, Pack> & p2)
{
p1 = p2;
}



template <typename T1, typename T2, typename T>
inline void setValueI1(Pair<T1, T2, Pack> & pair, T const & _i)
{
pair.i1 = _i;
}

template <typename T1, typename T2, typename T>
inline void setValueI2(Pair<T1, T2, Pack> & pair, T const & _i)
{
pair.i2 = _i;
}



template <typename T1, typename T2, typename T>
inline void moveValueI1(Pair<T1, T2, Pack> & pair, T & _i)
{
pair.i1 = _i;
}

template <typename T1, typename T2, typename T>
inline void moveValueI2(Pair<T1, T2, Pack> & pair, T & _i)
{
pair.i2 = _i;
}

}  

#endif  
