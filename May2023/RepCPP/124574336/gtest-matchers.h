


#ifndef GTEST_INCLUDE_GTEST_GTEST_MATCHERS_H_
#define GTEST_INCLUDE_GTEST_GTEST_MATCHERS_H_

#include <memory>
#include <ostream>
#include <string>
#include <type_traits>

#include "gtest/gtest-printers.h"
#include "gtest/internal/gtest-internal.h"
#include "gtest/internal/gtest-port.h"

#if defined(_MSC_VER) && _MSC_VER >= 1915
#define GTEST_MAYBE_5046_ 5046
#else
#define GTEST_MAYBE_5046_
#endif

GTEST_DISABLE_MSC_WARNINGS_PUSH_(
4251 GTEST_MAYBE_5046_ 
)

namespace testing {


class MatchResultListener {
public:
explicit MatchResultListener(::std::ostream* os) : stream_(os) {}
virtual ~MatchResultListener() = 0;  

template <typename T>
MatchResultListener& operator<<(const T& x) {
if (stream_ != nullptr) *stream_ << x;
return *this;
}

::std::ostream* stream() { return stream_; }

bool IsInterested() const { return stream_ != nullptr; }

private:
::std::ostream* const stream_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(MatchResultListener);
};

inline MatchResultListener::~MatchResultListener() {
}

class MatcherDescriberInterface {
public:
virtual ~MatcherDescriberInterface() {}

virtual void DescribeTo(::std::ostream* os) const = 0;

virtual void DescribeNegationTo(::std::ostream* os) const {
*os << "not (";
DescribeTo(os);
*os << ")";
}
};

template <typename T>
class MatcherInterface : public MatcherDescriberInterface {
public:
virtual bool MatchAndExplain(T x, MatchResultListener* listener) const = 0;

};

namespace internal {

template <typename T>
class MatcherInterfaceAdapter : public MatcherInterface<const T&> {
public:
explicit MatcherInterfaceAdapter(const MatcherInterface<T>* impl)
: impl_(impl) {}
~MatcherInterfaceAdapter() override { delete impl_; }

void DescribeTo(::std::ostream* os) const override { impl_->DescribeTo(os); }

void DescribeNegationTo(::std::ostream* os) const override {
impl_->DescribeNegationTo(os);
}

bool MatchAndExplain(const T& x,
MatchResultListener* listener) const override {
return impl_->MatchAndExplain(x, listener);
}

private:
const MatcherInterface<T>* const impl_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(MatcherInterfaceAdapter);
};

struct AnyEq {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a == b; }
};
struct AnyNe {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a != b; }
};
struct AnyLt {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a < b; }
};
struct AnyGt {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a > b; }
};
struct AnyLe {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a <= b; }
};
struct AnyGe {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a >= b; }
};

class DummyMatchResultListener : public MatchResultListener {
public:
DummyMatchResultListener() : MatchResultListener(nullptr) {}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(DummyMatchResultListener);
};

class StreamMatchResultListener : public MatchResultListener {
public:
explicit StreamMatchResultListener(::std::ostream* os)
: MatchResultListener(os) {}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(StreamMatchResultListener);
};

template <typename T>
class MatcherBase {
public:
bool MatchAndExplain(const T& x, MatchResultListener* listener) const {
return impl_->MatchAndExplain(x, listener);
}

bool Matches(const T& x) const {
DummyMatchResultListener dummy;
return MatchAndExplain(x, &dummy);
}

void DescribeTo(::std::ostream* os) const { impl_->DescribeTo(os); }

void DescribeNegationTo(::std::ostream* os) const {
impl_->DescribeNegationTo(os);
}

void ExplainMatchResultTo(const T& x, ::std::ostream* os) const {
StreamMatchResultListener listener(os);
MatchAndExplain(x, &listener);
}

const MatcherDescriberInterface* GetDescriber() const {
return impl_.get();
}

protected:
MatcherBase() {}

explicit MatcherBase(const MatcherInterface<const T&>* impl) : impl_(impl) {}

template <typename U>
explicit MatcherBase(
const MatcherInterface<U>* impl,
typename std::enable_if<!std::is_same<U, const U&>::value>::type* =
nullptr)
: impl_(new internal::MatcherInterfaceAdapter<U>(impl)) {}

MatcherBase(const MatcherBase&) = default;
MatcherBase& operator=(const MatcherBase&) = default;
MatcherBase(MatcherBase&&) = default;
MatcherBase& operator=(MatcherBase&&) = default;

virtual ~MatcherBase() {}

private:
std::shared_ptr<const MatcherInterface<const T&>> impl_;
};

}  

template <typename T>
class Matcher : public internal::MatcherBase<T> {
public:
explicit Matcher() {}  

explicit Matcher(const MatcherInterface<const T&>* impl)
: internal::MatcherBase<T>(impl) {}

template <typename U>
explicit Matcher(
const MatcherInterface<U>* impl,
typename std::enable_if<!std::is_same<U, const U&>::value>::type* =
nullptr)
: internal::MatcherBase<T>(impl) {}

Matcher(T value);  
};

template <>
class GTEST_API_ Matcher<const std::string&>
: public internal::MatcherBase<const std::string&> {
public:
Matcher() {}

explicit Matcher(const MatcherInterface<const std::string&>* impl)
: internal::MatcherBase<const std::string&>(impl) {}

Matcher(const std::string& s);  

Matcher(const char* s);  
};

template <>
class GTEST_API_ Matcher<std::string>
: public internal::MatcherBase<std::string> {
public:
Matcher() {}

explicit Matcher(const MatcherInterface<const std::string&>* impl)
: internal::MatcherBase<std::string>(impl) {}
explicit Matcher(const MatcherInterface<std::string>* impl)
: internal::MatcherBase<std::string>(impl) {}

Matcher(const std::string& s);  

Matcher(const char* s);  
};

#if GTEST_INTERNAL_HAS_STRING_VIEW
template <>
class GTEST_API_ Matcher<const internal::StringView&>
: public internal::MatcherBase<const internal::StringView&> {
public:
Matcher() {}

explicit Matcher(const MatcherInterface<const internal::StringView&>* impl)
: internal::MatcherBase<const internal::StringView&>(impl) {}

Matcher(const std::string& s);  

Matcher(const char* s);  

Matcher(internal::StringView s);  
};

template <>
class GTEST_API_ Matcher<internal::StringView>
: public internal::MatcherBase<internal::StringView> {
public:
Matcher() {}

explicit Matcher(const MatcherInterface<const internal::StringView&>* impl)
: internal::MatcherBase<internal::StringView>(impl) {}
explicit Matcher(const MatcherInterface<internal::StringView>* impl)
: internal::MatcherBase<internal::StringView>(impl) {}

Matcher(const std::string& s);  

Matcher(const char* s);  

Matcher(internal::StringView s);  
};
#endif  

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matcher<T>& matcher) {
matcher.DescribeTo(&os);
return os;
}

template <class Impl>
class PolymorphicMatcher {
public:
explicit PolymorphicMatcher(const Impl& an_impl) : impl_(an_impl) {}

Impl& mutable_impl() { return impl_; }

const Impl& impl() const { return impl_; }

template <typename T>
operator Matcher<T>() const {
return Matcher<T>(new MonomorphicImpl<const T&>(impl_));
}

private:
template <typename T>
class MonomorphicImpl : public MatcherInterface<T> {
public:
explicit MonomorphicImpl(const Impl& impl) : impl_(impl) {}

void DescribeTo(::std::ostream* os) const override { impl_.DescribeTo(os); }

void DescribeNegationTo(::std::ostream* os) const override {
impl_.DescribeNegationTo(os);
}

bool MatchAndExplain(T x, MatchResultListener* listener) const override {
return impl_.MatchAndExplain(x, listener);
}

private:
const Impl impl_;
};

Impl impl_;
};

template <typename T>
inline Matcher<T> MakeMatcher(const MatcherInterface<T>* impl) {
return Matcher<T>(impl);
}

template <class Impl>
inline PolymorphicMatcher<Impl> MakePolymorphicMatcher(const Impl& impl) {
return PolymorphicMatcher<Impl>(impl);
}

namespace internal {
template <typename D, typename Rhs, typename Op>
class ComparisonBase {
public:
explicit ComparisonBase(const Rhs& rhs) : rhs_(rhs) {}
template <typename Lhs>
operator Matcher<Lhs>() const {
return Matcher<Lhs>(new Impl<const Lhs&>(rhs_));
}

private:
template <typename T>
static const T& Unwrap(const T& v) { return v; }
template <typename T>
static const T& Unwrap(std::reference_wrapper<T> v) { return v; }

template <typename Lhs, typename = Rhs>
class Impl : public MatcherInterface<Lhs> {
public:
explicit Impl(const Rhs& rhs) : rhs_(rhs) {}
bool MatchAndExplain(Lhs lhs,
MatchResultListener* ) const override {
return Op()(lhs, Unwrap(rhs_));
}
void DescribeTo(::std::ostream* os) const override {
*os << D::Desc() << " ";
UniversalPrint(Unwrap(rhs_), os);
}
void DescribeNegationTo(::std::ostream* os) const override {
*os << D::NegatedDesc() <<  " ";
UniversalPrint(Unwrap(rhs_), os);
}

private:
Rhs rhs_;
};
Rhs rhs_;
};

template <typename Rhs>
class EqMatcher : public ComparisonBase<EqMatcher<Rhs>, Rhs, AnyEq> {
public:
explicit EqMatcher(const Rhs& rhs)
: ComparisonBase<EqMatcher<Rhs>, Rhs, AnyEq>(rhs) { }
static const char* Desc() { return "is equal to"; }
static const char* NegatedDesc() { return "isn't equal to"; }
};
template <typename Rhs>
class NeMatcher : public ComparisonBase<NeMatcher<Rhs>, Rhs, AnyNe> {
public:
explicit NeMatcher(const Rhs& rhs)
: ComparisonBase<NeMatcher<Rhs>, Rhs, AnyNe>(rhs) { }
static const char* Desc() { return "isn't equal to"; }
static const char* NegatedDesc() { return "is equal to"; }
};
template <typename Rhs>
class LtMatcher : public ComparisonBase<LtMatcher<Rhs>, Rhs, AnyLt> {
public:
explicit LtMatcher(const Rhs& rhs)
: ComparisonBase<LtMatcher<Rhs>, Rhs, AnyLt>(rhs) { }
static const char* Desc() { return "is <"; }
static const char* NegatedDesc() { return "isn't <"; }
};
template <typename Rhs>
class GtMatcher : public ComparisonBase<GtMatcher<Rhs>, Rhs, AnyGt> {
public:
explicit GtMatcher(const Rhs& rhs)
: ComparisonBase<GtMatcher<Rhs>, Rhs, AnyGt>(rhs) { }
static const char* Desc() { return "is >"; }
static const char* NegatedDesc() { return "isn't >"; }
};
template <typename Rhs>
class LeMatcher : public ComparisonBase<LeMatcher<Rhs>, Rhs, AnyLe> {
public:
explicit LeMatcher(const Rhs& rhs)
: ComparisonBase<LeMatcher<Rhs>, Rhs, AnyLe>(rhs) { }
static const char* Desc() { return "is <="; }
static const char* NegatedDesc() { return "isn't <="; }
};
template <typename Rhs>
class GeMatcher : public ComparisonBase<GeMatcher<Rhs>, Rhs, AnyGe> {
public:
explicit GeMatcher(const Rhs& rhs)
: ComparisonBase<GeMatcher<Rhs>, Rhs, AnyGe>(rhs) { }
static const char* Desc() { return "is >="; }
static const char* NegatedDesc() { return "isn't >="; }
};

template <typename T, typename = typename std::enable_if<
std::is_constructible<std::string, T>::value>::type>
using StringLike = T;

class MatchesRegexMatcher {
public:
MatchesRegexMatcher(const RE* regex, bool full_match)
: regex_(regex), full_match_(full_match) {}

#if GTEST_INTERNAL_HAS_STRING_VIEW
bool MatchAndExplain(const internal::StringView& s,
MatchResultListener* listener) const {
return MatchAndExplain(std::string(s), listener);
}
#endif  

template <typename CharType>
bool MatchAndExplain(CharType* s, MatchResultListener* listener) const {
return s != nullptr && MatchAndExplain(std::string(s), listener);
}

template <class MatcheeStringType>
bool MatchAndExplain(const MatcheeStringType& s,
MatchResultListener* ) const {
const std::string& s2(s);
return full_match_ ? RE::FullMatch(s2, *regex_)
: RE::PartialMatch(s2, *regex_);
}

void DescribeTo(::std::ostream* os) const {
*os << (full_match_ ? "matches" : "contains") << " regular expression ";
UniversalPrinter<std::string>::Print(regex_->pattern(), os);
}

void DescribeNegationTo(::std::ostream* os) const {
*os << "doesn't " << (full_match_ ? "match" : "contain")
<< " regular expression ";
UniversalPrinter<std::string>::Print(regex_->pattern(), os);
}

private:
const std::shared_ptr<const RE> regex_;
const bool full_match_;
};
}  

inline PolymorphicMatcher<internal::MatchesRegexMatcher> MatchesRegex(
const internal::RE* regex) {
return MakePolymorphicMatcher(internal::MatchesRegexMatcher(regex, true));
}
template <typename T = std::string>
PolymorphicMatcher<internal::MatchesRegexMatcher> MatchesRegex(
const internal::StringLike<T>& regex) {
return MatchesRegex(new internal::RE(std::string(regex)));
}

inline PolymorphicMatcher<internal::MatchesRegexMatcher> ContainsRegex(
const internal::RE* regex) {
return MakePolymorphicMatcher(internal::MatchesRegexMatcher(regex, false));
}
template <typename T = std::string>
PolymorphicMatcher<internal::MatchesRegexMatcher> ContainsRegex(
const internal::StringLike<T>& regex) {
return ContainsRegex(new internal::RE(std::string(regex)));
}

template <typename T>
inline internal::EqMatcher<T> Eq(T x) { return internal::EqMatcher<T>(x); }

template <typename T>
Matcher<T>::Matcher(T value) { *this = Eq(value); }

template <typename Lhs, typename Rhs>
inline Matcher<Lhs> TypedEq(const Rhs& rhs) { return Eq(rhs); }

template <typename Rhs>
inline internal::GeMatcher<Rhs> Ge(Rhs x) {
return internal::GeMatcher<Rhs>(x);
}

template <typename Rhs>
inline internal::GtMatcher<Rhs> Gt(Rhs x) {
return internal::GtMatcher<Rhs>(x);
}

template <typename Rhs>
inline internal::LeMatcher<Rhs> Le(Rhs x) {
return internal::LeMatcher<Rhs>(x);
}

template <typename Rhs>
inline internal::LtMatcher<Rhs> Lt(Rhs x) {
return internal::LtMatcher<Rhs>(x);
}

template <typename Rhs>
inline internal::NeMatcher<Rhs> Ne(Rhs x) {
return internal::NeMatcher<Rhs>(x);
}
}  

GTEST_DISABLE_MSC_WARNINGS_POP_()  

#endif  
