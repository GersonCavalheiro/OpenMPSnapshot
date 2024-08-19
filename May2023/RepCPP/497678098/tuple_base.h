

#ifndef SEQAN_INCLUDE_SEQAN_BASIC_TUPLE_BASE_H_
#define SEQAN_INCLUDE_SEQAN_BASIC_TUPLE_BASE_H_

namespace seqan {



template <typename TValue>
struct StoredTupleValue_
{
typedef TValue Type;
};

template <typename TValue, typename TSpec>
struct StoredTupleValue_< SimpleType<TValue, TSpec> >
{
typedef TValue Type;
};





template <typename TValue, unsigned SIZE, typename TSpec = void>
struct Tuple
{

typename StoredTupleValue_<TValue>::Type i[SIZE];



template <typename TPos>
inline
typename StoredTupleValue_<TValue>::Type &
operator[](TPos k)
{
SEQAN_ASSERT_GEQ(static_cast<int64_t>(k), 0);
SEQAN_ASSERT_LT(static_cast<int64_t>(k), static_cast<int64_t>(SIZE));
return i[k];
}

template <typename TPos>
inline
typename StoredTupleValue_<TValue>::Type const &
operator[](TPos k) const
{
SEQAN_ASSERT_GEQ(static_cast<int64_t>(k), 0);
SEQAN_ASSERT_LT(static_cast<int64_t>(k), static_cast<int64_t>(SIZE));
return i[k];
}

template <typename TPos, typename TValue2>
inline TValue2
assignValue(TPos k, TValue2 const source)
{
return i[k] = source;
}
};


#pragma pack(push,1)
template <typename TValue, unsigned SIZE>
struct Tuple<TValue, SIZE, Pack>
{

typename StoredTupleValue_<TValue>::Type i[SIZE];



template <typename TPos>
inline typename StoredTupleValue_<TValue>::Type &
operator[](TPos k)
{
SEQAN_ASSERT_GEQ(static_cast<int64_t>(k), 0);
SEQAN_ASSERT_LT(static_cast<int64_t>(k), static_cast<int64_t>(SIZE));
return i[k];
}

template <typename TPos>
inline typename StoredTupleValue_<TValue>::Type const &
operator[](TPos k) const
{
SEQAN_ASSERT_GEQ(static_cast<int64_t>(k), 0);
SEQAN_ASSERT_LT(static_cast<int64_t>(k), static_cast<int64_t>(SIZE));
return i[k];
}

template <typename TPos, typename TValue2>
inline TValue2
assignValue(TPos k, TValue2 const source)
{
return i[k] = source;
}
};
#pragma pack(pop)






template <typename TValue, unsigned SIZE, typename TSpec>
struct LENGTH<Tuple<TValue, SIZE, TSpec> >
{
enum { VALUE = SIZE };
};




template <typename TValue, unsigned SIZE, typename TSpec>
struct Value<Tuple<TValue, SIZE, TSpec> >
{
typedef TValue Type;
};


template <typename TValue, unsigned SIZE, typename TSpec>
struct Spec<Tuple<TValue, SIZE, TSpec> >
{
typedef TSpec Type;
};



template <typename TTarget, typename TValue, unsigned SIZE, typename TSpec>
inline void
write(TTarget &target, Tuple<TValue, SIZE, TSpec> const &a)
{
writeValue(target, '[');
if (SIZE > 0)
write(target, (TValue)a[0]);
for (unsigned j = 1; j < SIZE; ++j)
{
writeValue(target, ' ');
write(target, (TValue)a[j]);
}
writeValue(target, ']');
}

template <typename TStream, typename TValue, unsigned SIZE, typename TSpec>
inline TStream &
operator<<(TStream & target,
Tuple<TValue, SIZE, TSpec> const & source)
{
typename DirectionIterator<TStream, Output>::Type it = directionIterator(target, Output());
write(it, source);
return target;
}


template <typename TTuple1, typename TTuple2>
struct TupleMoveSetWorkerContext_
{
TTuple1 & t1;
TTuple2 & t2;

TupleMoveSetWorkerContext_(TTuple1 & _t1, TTuple2 & _t2)
: t1(_t1), t2(_t2)
{}
};

struct TupleSetWorker_
{
template <typename TArg>
static inline void body(TArg & arg, unsigned I)
{
set(arg.t1.i[I - 1], arg.t2.i[I - 1]);
}
};

template <typename TValue, unsigned SIZE, typename TSpec>
inline void
set(Tuple<TValue, SIZE, TSpec> & t1, Tuple<TValue, SIZE, TSpec> const & t2)
{
typedef Tuple<TValue, SIZE, TSpec> TTuple1;
typedef Tuple<TValue, SIZE, TSpec> const TTuple2;
TupleMoveSetWorkerContext_<TTuple1, TTuple2> context(t1, t2);
Loop<TupleSetWorker_, SIZE>::run(context);
}

template <typename TValue, unsigned SIZE, typename TSpec>
inline void
set(Tuple<TValue, SIZE, TSpec> & t1, Tuple<TValue, SIZE, TSpec> & t2)
{
set(t1, const_cast<Tuple<TValue, SIZE, TSpec> const &>(t2));
}


struct TupleMoveWorker_
{
template <typename TArg>
static inline void body(TArg & arg, unsigned I)
{
move(arg.t1.i[I - 1], arg.t2.i[I - 1]);
}
};

template <typename TValue, unsigned SIZE, typename TSpec>
inline void
move(Tuple<TValue, SIZE, TSpec> & t1, Tuple<TValue, SIZE, TSpec> & t2)
{
typedef Tuple<TValue, SIZE, TSpec> TTuple1;
typedef Tuple<TValue, SIZE, TSpec> TTuple2;
TupleMoveSetWorkerContext_<TTuple1, TTuple2> context(t1, t2);
Loop<TupleMoveWorker_, SIZE>::run(context);
}


template <typename TValue, unsigned SIZE, typename TSpec, typename TPos, typename TValue2>
inline TValue2
assignValue(Tuple<TValue, SIZE, TSpec> & me, TPos k, TValue2 const source)
{
SEQAN_CHECK((unsigned(k) < SIZE), "Invalid position, k = %u, SIZE = %u.", unsigned(k), unsigned(SIZE));
return me.i[k] = source;
}


template <typename TValue, unsigned SIZE, typename TSpec, typename TPos>
inline TValue
getValue(Tuple<TValue, SIZE, TSpec> & me, TPos k)
{
SEQAN_CHECK((unsigned(k) < SIZE), "Invalid position, k = %u, SIZE = %u.", unsigned(k), unsigned(SIZE));
return me.i[k];
}

template <typename TValue, unsigned SIZE, typename TSpec, typename TPos>
inline TValue
getValue(Tuple<TValue, SIZE, TSpec> const & me, TPos k)
{
SEQAN_CHECK((unsigned(k) < SIZE), "Invalid position, k = %u, SIZE = %u.", unsigned(k), unsigned(SIZE));
return me.i[k];
}


template <typename TValue, unsigned SIZE, typename TSpec, typename TPos, typename TValue2>
inline void
setValue(Tuple<TValue, SIZE, TSpec> & me, TPos k, TValue2 const & source)
{
SEQAN_CHECK((unsigned(k) < SIZE), "Invalid position, k = %u, SIZE = %u.", unsigned(k), unsigned(SIZE));
set(me.i[k], source);
}


template <typename TValue, unsigned SIZE, typename TSpec, typename TPos, typename TValue2>
inline void
moveValue(Tuple<TValue, SIZE, TSpec> & me, TPos k, TValue2 & source)
{
SEQAN_CHECK((unsigned(k) < SIZE), "Invalid position, k = %u, SIZE = %u.", unsigned(k), unsigned(SIZE));
move(me.i[k], source);
}



struct TupleShiftLeftWorker_
{
template <typename TArg>
static inline void body(TArg & arg, unsigned I)
{
arg[I-1] = arg[I];  
}
};

template <typename TValue, unsigned SIZE, typename TSpec>
inline void shiftLeft(Tuple<TValue, SIZE, TSpec> &me)
{
Loop<TupleShiftLeftWorker_, SIZE - 1>::run(me.i);
}



struct TupleShiftRightWorker_
{
template <typename TArg>
static inline void body(TArg & arg, unsigned I)
{
arg[I] = arg[I - 1];  
}
};

template <typename TValue, unsigned SIZE, typename TSpec>
inline void shiftRight(Tuple<TValue, SIZE, TSpec> & me)
{
LoopReverse<TupleShiftRightWorker_, SIZE - 1>::run(me.i);
}


template <typename TValue, unsigned SIZE, typename TSpec>
inline unsigned length(Tuple<TValue, SIZE, TSpec> const &)
{
return SIZE;
}


template <typename TValue, unsigned SIZE, typename TSpec>
inline void clear(Tuple<TValue, SIZE, TSpec> & me)
{
memset<sizeof(me.i), 0>(&(me.i));
}


template <typename TTupleL, typename TTupleR>
struct ComparisonWorkerContext_
{
int result;
TTupleL const & left;
TTupleR const & right;

ComparisonWorkerContext_(int b, TTupleL const & l, TTupleR const & r)
: result(b), left(l), right(r)
{}
};

struct TupleComparisonWorkerEq_
{
template <typename TArg>
static inline void body(TArg & arg, unsigned I)
{
if (arg.result != 1)
return;
if (getValue(arg.left, I - 1) != getValue(arg.right, I - 1))
arg.result = 0;
}
};

template <typename TValue, unsigned SIZE, typename TSpecL, typename TSpecR>
inline bool
operator==(Tuple<TValue, SIZE, TSpecL> const & left,
Tuple<TValue, SIZE, TSpecR> const & right)
{
typedef Tuple<TValue, SIZE, TSpecL> TTupleL;
typedef Tuple<TValue, SIZE, TSpecR> TTupleR;
ComparisonWorkerContext_<TTupleL, TTupleR> context(1, left, right);
Loop<TupleComparisonWorkerEq_, SIZE>::run(context);
return context.result == 1;
}



template <typename TValue, unsigned SIZE, typename TSpecL, typename TSpecR>
inline bool
operator!=(Tuple<TValue, SIZE, TSpecL> const & left,
Tuple<TValue, SIZE, TSpecR> const & right)
{
return !operator==(left, right);
}


struct TupleComparisonWorkerLt_
{
template <typename TArg>
static inline void body(TArg & arg, unsigned I)
{
if (arg.result != -1)
return;
if (arg.left.i[I - 1] == arg.right.i[I - 1])
return;
if (arg.left.i[I - 1] < arg.right.i[I - 1])
arg.result = 1;
if (arg.left.i[I - 1] > arg.right.i[I - 1])
arg.result = 0;
}
};

template <typename TValue, unsigned SIZE, typename TSpecL, typename TSpecR>
inline bool
operator<(Tuple<TValue, SIZE, TSpecL> const & left,
Tuple<TValue, SIZE, TSpecR> const & right)
{
typedef Tuple<TValue, SIZE, TSpecL> TTupleL;
typedef Tuple<TValue, SIZE, TSpecR> TTupleR;
ComparisonWorkerContext_<TTupleL, TTupleR> context(-1, left, right);
Loop<TupleComparisonWorkerLt_, SIZE>::run(context);
return context.result == 1;
}


struct TupleComparisonWorkerGt_
{
template <typename TArg>
static inline void body(TArg & arg, unsigned I)
{
if (arg.result != -1)
return;
if (arg.left.i[I - 1] == arg.right.i[I - 1])
return;
if (arg.left.i[I - 1] > arg.right.i[I - 1])
arg.result = 1;
if (arg.left.i[I - 1] < arg.right.i[I - 1])
arg.result = 0;
}
};

template <typename TValue, unsigned SIZE, typename TSpecL, typename TSpecR>
inline bool
operator>(Tuple<TValue, SIZE, TSpecL> const & left,
Tuple<TValue, SIZE, TSpecR> const & right)
{
typedef Tuple<TValue, SIZE, TSpecL> TTupleL;
typedef Tuple<TValue, SIZE, TSpecR> TTupleR;
ComparisonWorkerContext_<TTupleL, TTupleR> context(-1, left, right);
Loop<TupleComparisonWorkerGt_, SIZE>::run(context);
return context.result == 1;
}


template <typename TValue, unsigned SIZE, typename TSpecL, typename TSpecR>
inline bool
operator<=(Tuple<TValue, SIZE, TSpecL> const & left,
Tuple<TValue, SIZE, TSpecR> const & right)
{
return !operator>(left, right);
}


template <typename TValue, unsigned SIZE, typename TSpecL, typename TSpecR>
inline bool
operator>=(Tuple<TValue, SIZE, TSpecL> const & left,
Tuple<TValue, SIZE, TSpecR> const & right)
{
return !operator<(left, right);
}


template <typename TValue, unsigned SIZE, typename TSpecL, typename TSpecR>
inline Tuple<TValue, SIZE, TSpecL>
operator+(Tuple<TValue, SIZE, TSpecL> const & left,
Tuple<TValue, SIZE, TSpecR> const & right)
{
Tuple<TValue, SIZE, TSpecL>  tuple;

for (unsigned j = 0; j < SIZE; ++j)
tuple[j] = left[j] + right[j];

return tuple;
}

template <typename TValue1, unsigned SIZE, typename TSpecL, typename TValue2, typename TSpecR>
inline Tuple<TValue1, SIZE, TSpecL>
operator+(Tuple<TValue1, SIZE, TSpecL> const & left,
Tuple<TValue2, SIZE, TSpecR> const & right)
{
Tuple<TValue1, SIZE, TSpecL>  tuple;

for (unsigned j = 0; j < SIZE; ++j)
tuple[j] = left[j] + right[j];

return tuple;
}

}  

#endif  
