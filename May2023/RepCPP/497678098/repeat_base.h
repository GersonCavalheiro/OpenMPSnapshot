
#ifndef SEQAN_HEADER_REPEAT_BASE_H
#define SEQAN_HEADER_REPEAT_BASE_H

#include <functional>

#if SEQAN_ENABLE_PARALLELISM
#include <seqan/parallel.h>
#endif  

namespace seqan {



template <typename TPos, typename TPeriod>
struct Repeat {
TPos        beginPosition;
TPos        endPosition;
TPeriod        period;
};

template <typename TPos, typename TPeriod>
struct Value< Repeat<TPos, TPeriod> > {
typedef TPos Type;
};

template <typename TPos, typename TPeriod>
struct Size< Repeat<TPos, TPeriod> > {
typedef TPeriod Type;
};


template <typename TSize>
struct RepeatFinderParams {
TSize minRepeatLen;
TSize maxPeriod;
};

struct TRepeatFinder;

template <typename TText>
struct Cargo<Index<TText, IndexWotd<TRepeatFinder> > >
{
typedef Index<TText, IndexWotd<TRepeatFinder> >    TIndex;
typedef typename Size<TIndex>::Type                    TSize;
typedef RepeatFinderParams<TSize>                    Type;
};


template <typename TText, typename TSpec>
bool nodePredicate(Iter<Index<TText, IndexWotd<TRepeatFinder> >, TSpec> &it)
{
return countOccurrences(it) * repLength(it) >= cargo(container(it)).minRepeatLen;
}

template <typename TText, typename TSpec>
bool nodeHullPredicate(Iter<Index<TText, IndexWotd<TRepeatFinder> >, TSpec> &it)
{
return repLength(it) <= cargo(container(it)).maxPeriod;
}

template <typename TPos>
struct RepeatLess_ : std::function<bool(TPos,TPos)>
{
inline bool operator() (TPos const &a, TPos const &b) const {
return posLess(a, b);
}
};

template <typename TValue>
inline bool _repeatMaskValue(TValue const &)
{
return false;
}

template <>
inline bool _repeatMaskValue(Dna5 const &val)
{
return val == unknownValue<Dna5>(); 
}

template <>
inline bool _repeatMaskValue(Dna5Q const &val)
{
return val == unknownValue<Dna5Q>(); 
}

template <>
inline bool _repeatMaskValue(Iupac const &val)
{
return val == unknownValue<Iupac>(); 
}



template <typename TRepeatStore, typename TString, typename TRepeatSize>
inline void findRepeats(TRepeatStore &repString, TString const &text, TRepeatSize minRepeatLen)
{
typedef typename Value<TRepeatStore>::Type    TRepeat;
typedef typename Iterator<TString const>::Type    TIterator;
typedef typename Size<TString>::Type        TSize;

#if SEQAN_ENABLE_PARALLELISM
typedef typename Value<TString>::Type        TValue;

if (length(text) > (TSize)(omp_get_max_threads() * 2 * minRepeatLen)) {


String<TSize> splitters;
String<TRepeatStore> threadLocalStores;

#pragma omp parallel
{
#pragma omp master
{
computeSplitters(splitters, length(text), omp_get_num_threads());
resize(threadLocalStores, omp_get_num_threads());
}  
#pragma omp barrier

int const t = omp_get_thread_num();
TRepeatStore & store = threadLocalStores[t];

TRepeat rep;
rep.beginPosition = 0;
rep.endPosition = 0;
rep.period = 1;

bool forceFirst = t > 0;
bool forceLast = (t + 1) < omp_get_num_threads();

TIterator it = iter(text, splitters[t], Standard());
TIterator itEnd = iter(text, splitters[t + 1], Standard());
if (it != itEnd)
{
TValue last = *it;
TSize repLeft = 0;
TSize repRight = 1;

for (++it; it != itEnd; ++it, ++repRight)
{
if (*it != last)
{
if (_repeatMaskValue(last) || (TRepeatSize)(repRight - repLeft) > minRepeatLen || forceFirst)
{
forceFirst = false;
rep.beginPosition = splitters[t] + repLeft;
rep.endPosition = splitters[t] + repRight;
appendValue(store, rep);
}
repLeft = repRight;
last = *it;
}
}
if (_repeatMaskValue(last) || (TRepeatSize)(repRight - repLeft) > minRepeatLen || forceLast)
{
if (empty(store) || (back(store).beginPosition != repLeft && back(store).endPosition != repRight))
{
rep.beginPosition = splitters[t] + repLeft;
rep.endPosition = splitters[t] + repRight;
appendValue(store, rep);
}
}
}
}  


String<Pair<TSize> > fromPositions;
resize(fromPositions, length(threadLocalStores));
for (unsigned i = 0; i < length(fromPositions); ++i)
{
fromPositions[i].i1 = 0;
fromPositions[i].i2 = length(threadLocalStores[i]);
}
bool anyChange;
do
{
anyChange = false;
int lastNonEmpty = -1;
for (unsigned i = 0; i < length(threadLocalStores); ++i)
{
if (fromPositions[i].i1 == fromPositions[i].i2)
continue;  

if (lastNonEmpty != -1)
{
bool const adjacent = back(threadLocalStores[lastNonEmpty]).endPosition == front(threadLocalStores[i]).beginPosition;
bool const charsEqual = text[back(threadLocalStores[lastNonEmpty]).beginPosition] == text[front(threadLocalStores[i]).beginPosition];
if (adjacent && charsEqual)
{
anyChange = true;
back(threadLocalStores[lastNonEmpty]).endPosition = front(threadLocalStores[i]).endPosition;
fromPositions[i].i1 += 1;
}
}

if (fromPositions[i].i1 != fromPositions[i].i2)
lastNonEmpty = i;
}
}
while (anyChange);
for (unsigned i = 0; i < length(threadLocalStores); ++i)
{
if (fromPositions[i].i1 == fromPositions[i].i2)
continue;
unsigned j = fromPositions[i].i1;
TRepeatSize len = threadLocalStores[i][j].endPosition - threadLocalStores[i][j].beginPosition;
if (!_repeatMaskValue(text[threadLocalStores[i][j].beginPosition]) &&  
len <= minRepeatLen)
fromPositions[i].i1 += 1;
if (fromPositions[i].i1 == fromPositions[i].i2)
continue;
j = fromPositions[i].i2 - 1;
len = threadLocalStores[i][j].endPosition - threadLocalStores[i][j].beginPosition;
if (!_repeatMaskValue(text[threadLocalStores[i][j].beginPosition]) &&  
len <= minRepeatLen)
fromPositions[i].i2 -= 1;
}
String<unsigned> outSplitters;
appendValue(outSplitters, 0);
for (unsigned i = 0; i < length(threadLocalStores); ++i)
appendValue(outSplitters, back(outSplitters) + fromPositions[i].i2 - fromPositions[i].i1);


clear(repString);
resize(repString, back(outSplitters));

unsigned nt = length(threadLocalStores);
(void) nt;  
#pragma omp parallel num_threads(nt)
{
int const t = omp_get_thread_num();
arrayCopy(iter(threadLocalStores[t], fromPositions[t].i1, Standard()),
iter(threadLocalStores[t], fromPositions[t].i2, Standard()),
iter(repString, outSplitters[t], Standard()));
}  
} else {
#endif  
TRepeat rep;
rep.period = 1;
clear(repString);

TIterator it = begin(text, Standard());
TIterator itEnd = end(text, Standard());
if (it == itEnd) return;

TSize repLen = 1;
for (++it; it != itEnd; ++it)
{
if (*it != *(it-1))
{
if (_repeatMaskValue(*(it-1)) || repLen > (TSize)minRepeatLen)
{
rep.endPosition = it - begin(text, Standard());
rep.beginPosition = rep.endPosition - repLen;
appendValue(repString, rep);
}
repLen = 1;
} else
++repLen;
}
if (_repeatMaskValue(*(it-1)) || repLen > (TSize)minRepeatLen)
{
rep.endPosition = length(text);
rep.beginPosition = rep.endPosition - repLen;
appendValue(repString, rep);
}
#if SEQAN_ENABLE_PARALLELISM
}
#endif  
}

template <typename TRepeatStore, typename TString, typename TSpec, typename TRepeatSize>
inline void findRepeats(TRepeatStore &repString, StringSet<TString, TSpec> const &text, TRepeatSize minRepeatLen)
{
typedef typename Value<TRepeatStore>::Type    TRepeat;
typedef typename Iterator<TString>::Type    TIterator;
typedef typename Value<TString>::Type        TValue;
typedef typename Size<TString>::Type        TSize;

TRepeat rep;
rep.period = 1;
clear(repString);

for (unsigned i = 0; i < length(text); ++i)
{
TIterator it = begin(text[i], Standard());
TIterator itEnd = end(text[i], Standard());
if (it == itEnd) continue;

TValue last = *it;
TSize repLeft = 0;
TSize repRight = 1;
rep.beginPosition.i1 = i;
rep.endPosition.i1 = i;

for (++it; it != itEnd; ++it, ++repRight)
{
if (last != *it)
{
if (_repeatMaskValue(last) || (TRepeatSize)(repRight - repLeft) > minRepeatLen)
{
rep.beginPosition.i2 = repLeft;
rep.endPosition.i2 = repRight;
appendValue(repString, rep);
}
repLeft = repRight;
last = *it;
}
}
if (_repeatMaskValue(last) || (TRepeatSize)(repRight - repLeft) > minRepeatLen)
{
rep.beginPosition.i2 = repLeft;
rep.endPosition.i2 = repRight;
appendValue(repString, rep);
}
}
}

template <typename TRepeatStore, typename TText, typename TRepeatSize, typename TPeriodSize>
void findRepeats(TRepeatStore &repString, TText const &text, TRepeatSize minRepeatLen, TPeriodSize maxPeriod)
{
typedef Index<TText, IndexWotd<TRepeatFinder> >                    TIndex;
typedef typename Size<TIndex>::Type                                    TSize;
typedef typename Iterator<TIndex, TopDown<ParentLinks<> > >::Type    TNodeIterator;
typedef typename Fibre<TIndex, FibreSA>::Type const                TSA;
typedef typename Infix<TSA>::Type                                    TOccString;
typedef typename Iterator<TOccString>::Type                            TOccIterator;

typedef typename Value<TRepeatStore>::Type                            TRepeat;
typedef typename Value<TOccString>::Type                            TOcc;

typedef std::map<TOcc,TRepeat,RepeatLess_<TOcc> >                    TRepeatList;

if (maxPeriod < 1) return;
if (maxPeriod == 1)
{
findRepeats(repString, text, minRepeatLen);
return;
}

TIndex        index(text);
TRepeatList list;

cargo(index).minRepeatLen = minRepeatLen;
cargo(index).maxPeriod = maxPeriod;

TNodeIterator nodeIt(index);
TOccIterator itA, itB, itRepBegin, itEnd;
TRepeat rep;
for (; !atEnd(nodeIt); goNext(nodeIt))
{
if (isRoot(nodeIt)) continue;

TOccString occ = getOccurrences(nodeIt);
itA = begin(occ, Standard());
itEnd = end(occ, Standard());
itRepBegin = itB = itA;

TSize repLen = repLength(nodeIt);        
if ((TSize)minRepeatLen <= repLen) continue;

TSize diff, period = 0;                    
TSize repeatLen = 0;                    
TSize minLen = minRepeatLen - repLen;    

for (++itB; itB != itEnd; ++itB)
{
diff = posSub(*itB, *itA);
if (diff != period || getSeqNo(*itA) != getSeqNo(*itB))
{
if (repeatLen >= minLen)
if (parentRepLength(nodeIt) < period && period <= repLen)
{
rep.beginPosition = *itRepBegin;
rep.endPosition = posAdd(*itA, period);
rep.period = period;
list.insert(std::pair<TOcc,TRepeat>(rep.beginPosition, rep));
}
itRepBegin = itA;
period = diff;
repeatLen = 0;
}
repeatLen += period;
itA = itB;
}

if (repeatLen >= minLen)
if (parentRepLength(nodeIt) < period && period <= repLen)
{
rep.beginPosition = *itRepBegin;
rep.endPosition = posAdd(*itA, period);
rep.period = period;
list.insert(std::pair<TOcc,TRepeat>(rep.beginPosition, rep));
}
}

clear(repString);
reserve(repString, list.size(), Exact());
typename TRepeatList::const_iterator lit = list.begin();
typename TRepeatList::const_iterator litEnd = list.end();
for (TSize i = 0; lit != litEnd; ++lit, ++i)
appendValue(repString, (*lit).second);
}

}    

#endif
