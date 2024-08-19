

#ifndef ABSL_STRINGS_INTERNAL_STR_JOIN_INTERNAL_H_
#define ABSL_STRINGS_INTERNAL_STR_JOIN_INTERNAL_H_

#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/strings/internal/ostringstream.h"
#include "absl/strings/internal/resize_uninitialized.h"
#include "absl/strings/str_cat.h"

namespace absl {
namespace strings_internal {


struct AlphaNumFormatterImpl {
template <typename T>
void operator()(std::string* out, const T& t) const {
StrAppend(out, AlphaNum(t));
}

void operator()(std::string* out, const AlphaNum& t) const {
StrAppend(out, t);
}
};


struct NoFormatter : public AlphaNumFormatterImpl {};

class StreamFormatterImpl {
public:
template <typename T>
void operator()(std::string* out, const T& t) {
if (strm_) {
strm_->clear();  
strm_->str(out);
} else {
strm_.reset(new strings_internal::OStringStream(out));
}
*strm_ << t;
}

private:
std::unique_ptr<strings_internal::OStringStream> strm_;
};

template <typename F1, typename F2>
class PairFormatterImpl {
public:
PairFormatterImpl(F1 f1, absl::string_view sep, F2 f2)
: f1_(std::move(f1)), sep_(sep), f2_(std::move(f2)) {}

template <typename T>
void operator()(std::string* out, const T& p) {
f1_(out, p.first);
out->append(sep_);
f2_(out, p.second);
}

template <typename T>
void operator()(std::string* out, const T& p) const {
f1_(out, p.first);
out->append(sep_);
f2_(out, p.second);
}

private:
F1 f1_;
std::string sep_;
F2 f2_;
};

template <typename Formatter>
class DereferenceFormatterImpl {
public:
DereferenceFormatterImpl() : f_() {}
explicit DereferenceFormatterImpl(Formatter&& f)
: f_(std::forward<Formatter>(f)) {}

template <typename T>
void operator()(std::string* out, const T& t) {
f_(out, *t);
}

template <typename T>
void operator()(std::string* out, const T& t) const {
f_(out, *t);
}

private:
Formatter f_;
};

template <typename ValueType>
struct DefaultFormatter {
typedef AlphaNumFormatterImpl Type;
};
template <>
struct DefaultFormatter<const char*> {
typedef AlphaNumFormatterImpl Type;
};
template <>
struct DefaultFormatter<char*> {
typedef AlphaNumFormatterImpl Type;
};
template <>
struct DefaultFormatter<std::string> {
typedef NoFormatter Type;
};
template <>
struct DefaultFormatter<absl::string_view> {
typedef NoFormatter Type;
};
template <typename ValueType>
struct DefaultFormatter<ValueType*> {
typedef DereferenceFormatterImpl<typename DefaultFormatter<ValueType>::Type>
Type;
};

template <typename ValueType>
struct DefaultFormatter<std::unique_ptr<ValueType>>
: public DefaultFormatter<ValueType*> {};


template <typename Iterator, typename Formatter>
std::string JoinAlgorithm(Iterator start, Iterator end, absl::string_view s,
Formatter&& f) {
std::string result;
absl::string_view sep("");
for (Iterator it = start; it != end; ++it) {
result.append(sep.data(), sep.size());
f(&result, *it);
sep = s;
}
return result;
}

template <typename Iterator,
typename = typename std::enable_if<std::is_convertible<
typename std::iterator_traits<Iterator>::iterator_category,
std::forward_iterator_tag>::value>::type>
std::string JoinAlgorithm(Iterator start, Iterator end, absl::string_view s,
NoFormatter) {
std::string result;
if (start != end) {
size_t result_size = start->size();
for (Iterator it = start; ++it != end;) {
result_size += s.size();
result_size += it->size();
}

if (result_size > 0) {
STLStringResizeUninitialized(&result, result_size);

char* result_buf = &*result.begin();
memcpy(result_buf, start->data(), start->size());
result_buf += start->size();
for (Iterator it = start; ++it != end;) {
memcpy(result_buf, s.data(), s.size());
result_buf += s.size();
memcpy(result_buf, it->data(), it->size());
result_buf += it->size();
}
}
}

return result;
}

template <size_t I, size_t N>
struct JoinTupleLoop {
template <typename Tup, typename Formatter>
void operator()(std::string* out, const Tup& tup, absl::string_view sep,
Formatter&& fmt) {
if (I > 0) out->append(sep.data(), sep.size());
fmt(out, std::get<I>(tup));
JoinTupleLoop<I + 1, N>()(out, tup, sep, fmt);
}
};
template <size_t N>
struct JoinTupleLoop<N, N> {
template <typename Tup, typename Formatter>
void operator()(std::string*, const Tup&, absl::string_view, Formatter&&) {}
};

template <typename... T, typename Formatter>
std::string JoinAlgorithm(const std::tuple<T...>& tup, absl::string_view sep,
Formatter&& fmt) {
std::string result;
JoinTupleLoop<0, sizeof...(T)>()(&result, tup, sep, fmt);
return result;
}

template <typename Iterator>
std::string JoinRange(Iterator first, Iterator last, absl::string_view separator) {
typedef typename std::iterator_traits<Iterator>::value_type ValueType;
typedef typename DefaultFormatter<ValueType>::Type Formatter;
return JoinAlgorithm(first, last, separator, Formatter());
}

template <typename Range, typename Formatter>
std::string JoinRange(const Range& range, absl::string_view separator,
Formatter&& fmt) {
using std::begin;
using std::end;
return JoinAlgorithm(begin(range), end(range), separator, fmt);
}

template <typename Range>
std::string JoinRange(const Range& range, absl::string_view separator) {
using std::begin;
using std::end;
return JoinRange(begin(range), end(range), separator);
}

}  
}  

#endif  
