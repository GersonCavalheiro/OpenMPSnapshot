
#ifndef SEQAN_INCLUDE_SEQAN_BASIC_TUPLE_BIT_PACKED_H_
#define SEQAN_INCLUDE_SEQAN_BASIC_TUPLE_BIT_PACKED_H_

namespace seqan {


template <typename TValue>
inline bool testAllZeros(TValue const & val);

template <typename TValue>
inline bool testAllOnes(TValue const & val);




template <unsigned char SIZE>
struct BitVector_
{
typedef typename BitVector_<SIZE + 1>::Type Type;
};

template <> struct BitVector_<8> { typedef unsigned char Type; };
template <> struct BitVector_<16> { typedef unsigned short Type; };
template <> struct BitVector_<32> { typedef unsigned int Type; };
template <> struct BitVector_<64> { typedef uint64_t Type; };
template <> struct BitVector_<255>;


#pragma pack(push,1)
template <typename TValue, unsigned SIZE>
struct Tuple<TValue, SIZE, BitPacked<> >
{
typedef typename BitVector_<SIZE * BitsPerValue<TValue>::VALUE>::Type TBitVector;

static constexpr uint64_t BIT_MASK = ((1ull << (BitsPerValue<TValue>::VALUE - 1)       ) - 1ull) << 1 | 1ull;
static constexpr uint64_t MASK     = ((1ull << (SIZE * BitsPerValue<TValue>::VALUE - 1)) - 1ull) << 1 | 1ull;


TBitVector i;




template <typename TPos>
inline const TValue
operator[](TPos k) const
{
SEQAN_ASSERT_GEQ(static_cast<int64_t>(k), 0);
SEQAN_ASSERT_LT(static_cast<int64_t>(k), static_cast<int64_t>(SIZE));
return (i >> (SIZE - 1 - k) * BitsPerValue<TValue>::VALUE) & BIT_MASK;
}


template <unsigned size__>
inline Tuple & operator=(Tuple<TValue, size__, BitPacked<> > const & right)
{
i = right.i;
return *this;
}


template <typename TShiftSize>
inline TBitVector operator<<=(TShiftSize shift)
{
return i = (i << (shift * BitsPerValue<TValue>::VALUE)) & MASK;
}

template <typename TShiftSize>
inline TBitVector operator<<(TShiftSize shift) const
{
return (i << (shift * BitsPerValue<TValue>::VALUE)) & MASK;
}

template <typename TShiftSize>
inline TBitVector operator>>=(TShiftSize shift)
{
return i = (i >> (shift * BitsPerValue<TValue>::VALUE));
}

template <typename TShiftSize>
inline TBitVector operator>>(TShiftSize shift) const
{
return i >> (shift * BitsPerValue<TValue>::VALUE);
}

template <typename T>
inline void operator|=(T const & t)
{
i |= ordValue(t);
}

inline TBitVector* operator&()
{
return &i;
}

inline const TBitVector* operator&() const
{
return &i;
}

template <typename TPos, typename TValue2>
inline TValue2
assignValue(TPos k, TValue2 const source)
{
SEQAN_ASSERT_GEQ(static_cast<int64_t>(k), 0);
SEQAN_ASSERT_LT(static_cast<int64_t>(k), static_cast<int64_t>(SIZE));

unsigned shift = ((SIZE - 1 - k) * BitsPerValue<TValue>::VALUE);
i = (i & ~(BIT_MASK << shift)) | (TBitVector)ordValue(source) << shift;
return source;
}
};
#pragma pack(pop)




template <typename TValue, unsigned SIZE, typename TPos>
inline TValue
getValue(Tuple<TValue, SIZE, BitPacked<> > const & me,
TPos k)
{
SEQAN_ASSERT_GEQ(static_cast<int64_t>(k), 0);
SEQAN_ASSERT_LT(static_cast<int64_t>(k), static_cast<int64_t>(SIZE));

return (me.i >> (SIZE - 1 - k) * BitsPerValue<TValue>::VALUE) & me.BIT_MASK;
}

template <typename TValue, unsigned SIZE, typename TPos>
TValue
getValue(Tuple<TValue, SIZE, BitPacked<> > & me,
TPos k)
{
SEQAN_ASSERT_GEQ(static_cast<int64_t>(k), 0);
SEQAN_ASSERT_LT(static_cast<int64_t>(k), static_cast<int64_t>(SIZE));

return (me.i >> (SIZE - 1 - k) * BitsPerValue<TValue>::VALUE) & me.BIT_MASK;
}


template <typename TValue, unsigned SIZE, typename TValue2, typename TPos>
inline TValue2
assignValue(Tuple<TValue, SIZE, BitPacked<> > & me,
TPos k,
TValue2 const source)
{
typedef typename Tuple<TValue, SIZE, BitPacked<> >::TBitVector TBitVector;

SEQAN_ASSERT_GEQ(static_cast<int64_t>(k), 0);
SEQAN_ASSERT_LT(static_cast<int64_t>(k), static_cast<int64_t>(SIZE));

unsigned shift = ((SIZE - 1 - k) * BitsPerValue<TValue>::VALUE);
me.i = (me.i & ~(me.BIT_MASK << shift)) | (TBitVector)ordValue(source) << shift;
return source;
}


template <typename TValue, unsigned SIZE, typename TValue2, typename TPos>
inline TValue2
setValue(Tuple<TValue, SIZE, BitPacked<> > & me,
TPos k,
TValue2 const source)
{
return assignValue(me, k, source);
}


template <typename TValue, unsigned SIZE, typename TValue2, typename TPos>
inline TValue2
moveValue(Tuple<TValue, SIZE, BitPacked<> > & me,
TPos k,
TValue2 const source)
{
return assignValue(me, k, source);
}


template <typename TValue, unsigned SIZE>
inline void
move(Tuple<TValue, SIZE, BitPacked<> > & t1, Tuple<TValue, SIZE, BitPacked<> > & t2)
{
t1.i = t2.i;
}

template <typename TValue, unsigned SIZE>
inline void
set(Tuple<TValue, SIZE, BitPacked<> > & t1, Tuple<TValue, SIZE, BitPacked<> > const & t2)
{
t1.i = t2.i;
}

template <typename TValue, unsigned SIZE>
inline void
assign(Tuple<TValue, SIZE, BitPacked<> > & t1, Tuple<TValue, SIZE, BitPacked<> > const & t2)
{
t1.i = t2.i;
}


template <typename TValue, unsigned SIZE>
inline void shiftLeft(Tuple<TValue, SIZE, BitPacked<> > & me)
{
me <<= 1;
}


template <typename TValue, unsigned SIZE>
inline void shiftRight(Tuple<TValue, SIZE, BitPacked<> > & me)
{
me >>= 1;
}


template <typename TValue, unsigned SIZE>
inline bool testAllZeros(Tuple<TValue, SIZE, BitPacked<> > const & me)
{
return testAllZeros(me.i);
}


template <typename TValue, unsigned SIZE>
inline bool testAllOnes(Tuple<TValue, SIZE, BitPacked<> > const & me)
{
return testAllOnes(me.i);
}


template <typename TValue, unsigned SIZE>
inline void clear(Tuple<TValue, SIZE, BitPacked<> > & me)
{
me.i = 0;
}


template <typename TValue, unsigned SIZE>
inline Tuple<TValue, SIZE, BitPacked<> >
operator&(Tuple<TValue, SIZE, BitPacked<> > const & left,
Tuple<TValue, SIZE, BitPacked<> > const & right)
{
Tuple<TValue, SIZE, BitPacked<> > tmp;
tmp.i = left.i & right.i;
return tmp;
}

template <typename TValue, unsigned SIZE, typename T>
inline typename Tuple<TValue, SIZE, BitPacked<> >::TBitVector
operator&(Tuple<TValue, SIZE, BitPacked<> > const & left,
T const & right)
{
return left.i & right;
}


template <typename TValue, unsigned SIZE>
inline Tuple<TValue, SIZE, BitPacked<> >
operator|(Tuple<TValue, SIZE, BitPacked<> > const & left,
Tuple<TValue, SIZE, BitPacked<> > const & right)
{
Tuple<TValue, SIZE, BitPacked<> > tmp;
tmp.i = left.i | right.i;
return tmp;
}

template <typename TValue, unsigned SIZE, typename T>
inline typename Tuple<TValue, SIZE, BitPacked<> >::TBitVector
operator|(Tuple<TValue, SIZE, BitPacked<> > const & left,
T const & right)
{
return left.i | right;
}


template <typename TValue, unsigned SIZE>
inline Tuple<TValue, SIZE, BitPacked<> >
operator^(Tuple<TValue, SIZE, BitPacked<> > const & left,
Tuple<TValue, SIZE, BitPacked<> > const & right)
{
Tuple<TValue, SIZE, BitPacked<> > tmp;
tmp.i = left.i ^ right.i;
return tmp;
}

template <typename TValue, unsigned SIZE, typename T>
inline typename Tuple<TValue, SIZE, BitPacked<> >::TBitVector
operator^(Tuple<TValue, SIZE, BitPacked<> > const & left,
T const & right)
{
return left.i ^ right;
}


template <typename TValue, unsigned SIZE>
inline Tuple<TValue, SIZE, BitPacked<> >
operator~(Tuple<TValue, SIZE, BitPacked<> > const & val)
{
Tuple<TValue, SIZE, BitPacked<> > tmp;
tmp.i = ~val.i;
return tmp;
}


template <typename TValue, unsigned SIZE>
inline bool operator<(Tuple<TValue, SIZE, BitPacked<> > const & left,
Tuple<TValue, SIZE, BitPacked<> > const & right)
{
return left.i < right.i;
}

template <typename TValue, unsigned SIZE>
inline bool operator<(Tuple<TValue, SIZE, BitPacked<> > & left,
Tuple<TValue, SIZE, BitPacked<> > & right)
{
return left.i < right.i;
}


template <typename TValue, unsigned SIZE>
inline bool operator>(Tuple<TValue, SIZE, BitPacked<> > const & left,
Tuple<TValue, SIZE, BitPacked<> > const & right)
{
return left.i > right.i;
}

template <typename TValue, unsigned SIZE>
inline bool operator>(Tuple<TValue, SIZE, BitPacked<> > & left,
Tuple<TValue, SIZE, BitPacked<> > & right)
{
return left.i > right.i;
}


template <typename TValue, unsigned SIZE>
inline bool operator<=(Tuple<TValue, SIZE, BitPacked<> > const & left,
Tuple<TValue, SIZE, BitPacked<> > const & right)
{
return !operator>(left, right);
}

template <typename TValue, unsigned SIZE>
inline bool operator<=(Tuple<TValue, SIZE, BitPacked<> > & left,
Tuple<TValue, SIZE, BitPacked<> > & right)
{
return !operator>(left, right);
}


template <typename TValue, unsigned SIZE>
inline bool operator>=(Tuple<TValue, SIZE, BitPacked<> > const & left,
Tuple<TValue, SIZE, BitPacked<> > const & right)
{
return !operator<(left, right);
}

template <typename TValue, unsigned SIZE>
inline bool operator>=(Tuple<TValue, SIZE, BitPacked<> > & left,
Tuple<TValue, SIZE, BitPacked<> > & right)
{
return !operator<(left, right);
}


template <typename TValue, unsigned SIZE>
inline bool operator==(Tuple<TValue, SIZE, BitPacked<> > const & left,
Tuple<TValue, SIZE, BitPacked<> > const & right)
{
return left.i == right.i;
}

template <typename TValue, unsigned SIZE>
inline bool operator==(Tuple<TValue, SIZE, BitPacked<> > & left,
Tuple<TValue, SIZE, BitPacked<> > & right)
{
return left.i == right.i;
}


template <typename TValue, unsigned SIZE>
inline bool operator!=(Tuple<TValue, SIZE, BitPacked<> > const & left,
Tuple<TValue, SIZE, BitPacked<> > const & right)
{
return !operator==(left, right);
}

template <typename TValue, unsigned SIZE>
inline bool operator!=(Tuple<TValue, SIZE, BitPacked<> > & left,
Tuple<TValue, SIZE, BitPacked<> > & right)
{
return !operator==(left, right);
}

}  

#endif  
