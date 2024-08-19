



#ifndef GOOGLETEST_INCLUDE_GTEST_INTERNAL_GTEST_PARAM_UTIL_H_
#define GOOGLETEST_INCLUDE_GTEST_INTERNAL_GTEST_PARAM_UTIL_H_

#include <ctype.h>

#include <cassert>
#include <iterator>
#include <memory>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/internal/gtest-internal.h"
#include "gtest/internal/gtest-port.h"
#include "gtest/gtest-printers.h"
#include "gtest/gtest-test-part.h"

namespace testing {
template <class ParamType>
struct TestParamInfo {
TestParamInfo(const ParamType& a_param, size_t an_index) :
param(a_param),
index(an_index) {}
ParamType param;
size_t index;
};

struct PrintToStringParamName {
template <class ParamType>
std::string operator()(const TestParamInfo<ParamType>& info) const {
return PrintToString(info.param);
}
};

namespace internal {


GTEST_API_ void ReportInvalidTestSuiteType(const char* test_suite_name,
CodeLocation code_location);

template <typename> class ParamGeneratorInterface;
template <typename> class ParamGenerator;

template <typename T>
class ParamIteratorInterface {
public:
virtual ~ParamIteratorInterface() {}
virtual const ParamGeneratorInterface<T>* BaseGenerator() const = 0;
virtual void Advance() = 0;
virtual ParamIteratorInterface* Clone() const = 0;
virtual const T* Current() const = 0;
virtual bool Equals(const ParamIteratorInterface& other) const = 0;
};

template <typename T>
class ParamIterator {
public:
typedef T value_type;
typedef const T& reference;
typedef ptrdiff_t difference_type;

ParamIterator(const ParamIterator& other) : impl_(other.impl_->Clone()) {}
ParamIterator& operator=(const ParamIterator& other) {
if (this != &other)
impl_.reset(other.impl_->Clone());
return *this;
}

const T& operator*() const { return *impl_->Current(); }
const T* operator->() const { return impl_->Current(); }
ParamIterator& operator++() {
impl_->Advance();
return *this;
}
ParamIterator operator++(int ) {
ParamIteratorInterface<T>* clone = impl_->Clone();
impl_->Advance();
return ParamIterator(clone);
}
bool operator==(const ParamIterator& other) const {
return impl_.get() == other.impl_.get() || impl_->Equals(*other.impl_);
}
bool operator!=(const ParamIterator& other) const {
return !(*this == other);
}

private:
friend class ParamGenerator<T>;
explicit ParamIterator(ParamIteratorInterface<T>* impl) : impl_(impl) {}
std::unique_ptr<ParamIteratorInterface<T> > impl_;
};

template <typename T>
class ParamGeneratorInterface {
public:
typedef T ParamType;

virtual ~ParamGeneratorInterface() {}

virtual ParamIteratorInterface<T>* Begin() const = 0;
virtual ParamIteratorInterface<T>* End() const = 0;
};

template<typename T>
class ParamGenerator {
public:
typedef ParamIterator<T> iterator;

explicit ParamGenerator(ParamGeneratorInterface<T>* impl) : impl_(impl) {}
ParamGenerator(const ParamGenerator& other) : impl_(other.impl_) {}

ParamGenerator& operator=(const ParamGenerator& other) {
impl_ = other.impl_;
return *this;
}

iterator begin() const { return iterator(impl_->Begin()); }
iterator end() const { return iterator(impl_->End()); }

private:
std::shared_ptr<const ParamGeneratorInterface<T> > impl_;
};

template <typename T, typename IncrementT>
class RangeGenerator : public ParamGeneratorInterface<T> {
public:
RangeGenerator(T begin, T end, IncrementT step)
: begin_(begin), end_(end),
step_(step), end_index_(CalculateEndIndex(begin, end, step)) {}
~RangeGenerator() override {}

ParamIteratorInterface<T>* Begin() const override {
return new Iterator(this, begin_, 0, step_);
}
ParamIteratorInterface<T>* End() const override {
return new Iterator(this, end_, end_index_, step_);
}

private:
class Iterator : public ParamIteratorInterface<T> {
public:
Iterator(const ParamGeneratorInterface<T>* base, T value, int index,
IncrementT step)
: base_(base), value_(value), index_(index), step_(step) {}
~Iterator() override {}

const ParamGeneratorInterface<T>* BaseGenerator() const override {
return base_;
}
void Advance() override {
value_ = static_cast<T>(value_ + step_);
index_++;
}
ParamIteratorInterface<T>* Clone() const override {
return new Iterator(*this);
}
const T* Current() const override { return &value_; }
bool Equals(const ParamIteratorInterface<T>& other) const override {
GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
<< "The program attempted to compare iterators "
<< "from different generators." << std::endl;
const int other_index =
CheckedDowncastToActualType<const Iterator>(&other)->index_;
return index_ == other_index;
}

private:
Iterator(const Iterator& other)
: ParamIteratorInterface<T>(),
base_(other.base_), value_(other.value_), index_(other.index_),
step_(other.step_) {}

void operator=(const Iterator& other);

const ParamGeneratorInterface<T>* const base_;
T value_;
int index_;
const IncrementT step_;
};  

static int CalculateEndIndex(const T& begin,
const T& end,
const IncrementT& step) {
int end_index = 0;
for (T i = begin; i < end; i = static_cast<T>(i + step))
end_index++;
return end_index;
}

void operator=(const RangeGenerator& other);

const T begin_;
const T end_;
const IncrementT step_;
const int end_index_;
};  


template <typename T>
class ValuesInIteratorRangeGenerator : public ParamGeneratorInterface<T> {
public:
template <typename ForwardIterator>
ValuesInIteratorRangeGenerator(ForwardIterator begin, ForwardIterator end)
: container_(begin, end) {}
~ValuesInIteratorRangeGenerator() override {}

ParamIteratorInterface<T>* Begin() const override {
return new Iterator(this, container_.begin());
}
ParamIteratorInterface<T>* End() const override {
return new Iterator(this, container_.end());
}

private:
typedef typename ::std::vector<T> ContainerType;

class Iterator : public ParamIteratorInterface<T> {
public:
Iterator(const ParamGeneratorInterface<T>* base,
typename ContainerType::const_iterator iterator)
: base_(base), iterator_(iterator) {}
~Iterator() override {}

const ParamGeneratorInterface<T>* BaseGenerator() const override {
return base_;
}
void Advance() override {
++iterator_;
value_.reset();
}
ParamIteratorInterface<T>* Clone() const override {
return new Iterator(*this);
}
const T* Current() const override {
if (value_.get() == nullptr) value_.reset(new T(*iterator_));
return value_.get();
}
bool Equals(const ParamIteratorInterface<T>& other) const override {
GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
<< "The program attempted to compare iterators "
<< "from different generators." << std::endl;
return iterator_ ==
CheckedDowncastToActualType<const Iterator>(&other)->iterator_;
}

private:
Iterator(const Iterator& other)
: ParamIteratorInterface<T>(),
base_(other.base_),
iterator_(other.iterator_) {}

const ParamGeneratorInterface<T>* const base_;
typename ContainerType::const_iterator iterator_;
mutable std::unique_ptr<const T> value_;
};  

void operator=(const ValuesInIteratorRangeGenerator& other);

const ContainerType container_;
};  

template <class ParamType>
std::string DefaultParamName(const TestParamInfo<ParamType>& info) {
Message name_stream;
name_stream << info.index;
return name_stream.GetString();
}

template <typename T = int>
void TestNotEmpty() {
static_assert(sizeof(T) == 0, "Empty arguments are not allowed.");
}
template <typename T = int>
void TestNotEmpty(const T&) {}

template <class TestClass>
class ParameterizedTestFactory : public TestFactoryBase {
public:
typedef typename TestClass::ParamType ParamType;
explicit ParameterizedTestFactory(ParamType parameter) :
parameter_(parameter) {}
Test* CreateTest() override {
TestClass::SetParam(&parameter_);
return new TestClass();
}

private:
const ParamType parameter_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestFactory);
};

template <class ParamType>
class TestMetaFactoryBase {
public:
virtual ~TestMetaFactoryBase() {}

virtual TestFactoryBase* CreateTestFactory(ParamType parameter) = 0;
};

template <class TestSuite>
class TestMetaFactory
: public TestMetaFactoryBase<typename TestSuite::ParamType> {
public:
using ParamType = typename TestSuite::ParamType;

TestMetaFactory() {}

TestFactoryBase* CreateTestFactory(ParamType parameter) override {
return new ParameterizedTestFactory<TestSuite>(parameter);
}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(TestMetaFactory);
};

class ParameterizedTestSuiteInfoBase {
public:
virtual ~ParameterizedTestSuiteInfoBase() {}

virtual const std::string& GetTestSuiteName() const = 0;
virtual TypeId GetTestSuiteTypeId() const = 0;
virtual void RegisterTests() = 0;

protected:
ParameterizedTestSuiteInfoBase() {}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestSuiteInfoBase);
};

struct GTEST_API_ MarkAsIgnored {
explicit MarkAsIgnored(const char* test_suite);
};

GTEST_API_ void InsertSyntheticTestCase(const std::string& name,
CodeLocation location, bool has_test_p);

template <class TestSuite>
class ParameterizedTestSuiteInfo : public ParameterizedTestSuiteInfoBase {
public:
using ParamType = typename TestSuite::ParamType;
typedef ParamGenerator<ParamType>(GeneratorCreationFunc)();
using ParamNameGeneratorFunc = std::string(const TestParamInfo<ParamType>&);

explicit ParameterizedTestSuiteInfo(const char* name,
CodeLocation code_location)
: test_suite_name_(name), code_location_(code_location) {}

const std::string& GetTestSuiteName() const override {
return test_suite_name_;
}
TypeId GetTestSuiteTypeId() const override { return GetTypeId<TestSuite>(); }
void AddTestPattern(const char* test_suite_name, const char* test_base_name,
TestMetaFactoryBase<ParamType>* meta_factory,
CodeLocation code_location) {
tests_.push_back(std::shared_ptr<TestInfo>(new TestInfo(
test_suite_name, test_base_name, meta_factory, code_location)));
}
int AddTestSuiteInstantiation(const std::string& instantiation_name,
GeneratorCreationFunc* func,
ParamNameGeneratorFunc* name_func,
const char* file, int line) {
instantiations_.push_back(
InstantiationInfo(instantiation_name, func, name_func, file, line));
return 0;  
}
void RegisterTests() override {
bool generated_instantiations = false;

for (typename TestInfoContainer::iterator test_it = tests_.begin();
test_it != tests_.end(); ++test_it) {
std::shared_ptr<TestInfo> test_info = *test_it;
for (typename InstantiationContainer::iterator gen_it =
instantiations_.begin(); gen_it != instantiations_.end();
++gen_it) {
const std::string& instantiation_name = gen_it->name;
ParamGenerator<ParamType> generator((*gen_it->generator)());
ParamNameGeneratorFunc* name_func = gen_it->name_func;
const char* file = gen_it->file;
int line = gen_it->line;

std::string test_suite_name;
if ( !instantiation_name.empty() )
test_suite_name = instantiation_name + "/";
test_suite_name += test_info->test_suite_base_name;

size_t i = 0;
std::set<std::string> test_param_names;
for (typename ParamGenerator<ParamType>::iterator param_it =
generator.begin();
param_it != generator.end(); ++param_it, ++i) {
generated_instantiations = true;

Message test_name_stream;

std::string param_name = name_func(
TestParamInfo<ParamType>(*param_it, i));

GTEST_CHECK_(IsValidParamName(param_name))
<< "Parameterized test name '" << param_name
<< "' is invalid, in " << file
<< " line " << line << std::endl;

GTEST_CHECK_(test_param_names.count(param_name) == 0)
<< "Duplicate parameterized test name '" << param_name
<< "', in " << file << " line " << line << std::endl;

test_param_names.insert(param_name);

if (!test_info->test_base_name.empty()) {
test_name_stream << test_info->test_base_name << "/";
}
test_name_stream << param_name;
MakeAndRegisterTestInfo(
test_suite_name.c_str(), test_name_stream.GetString().c_str(),
nullptr,  
PrintToString(*param_it).c_str(), test_info->code_location,
GetTestSuiteTypeId(),
SuiteApiResolver<TestSuite>::GetSetUpCaseOrSuite(file, line),
SuiteApiResolver<TestSuite>::GetTearDownCaseOrSuite(file, line),
test_info->test_meta_factory->CreateTestFactory(*param_it));
}  
}  
}  

if (!generated_instantiations) {
InsertSyntheticTestCase(GetTestSuiteName(), code_location_,
!tests_.empty());
}
}    

private:
struct TestInfo {
TestInfo(const char* a_test_suite_base_name, const char* a_test_base_name,
TestMetaFactoryBase<ParamType>* a_test_meta_factory,
CodeLocation a_code_location)
: test_suite_base_name(a_test_suite_base_name),
test_base_name(a_test_base_name),
test_meta_factory(a_test_meta_factory),
code_location(a_code_location) {}

const std::string test_suite_base_name;
const std::string test_base_name;
const std::unique_ptr<TestMetaFactoryBase<ParamType> > test_meta_factory;
const CodeLocation code_location;
};
using TestInfoContainer = ::std::vector<std::shared_ptr<TestInfo> >;
struct InstantiationInfo {
InstantiationInfo(const std::string &name_in,
GeneratorCreationFunc* generator_in,
ParamNameGeneratorFunc* name_func_in,
const char* file_in,
int line_in)
: name(name_in),
generator(generator_in),
name_func(name_func_in),
file(file_in),
line(line_in) {}

std::string name;
GeneratorCreationFunc* generator;
ParamNameGeneratorFunc* name_func;
const char* file;
int line;
};
typedef ::std::vector<InstantiationInfo> InstantiationContainer;

static bool IsValidParamName(const std::string& name) {
if (name.empty())
return false;

for (std::string::size_type index = 0; index < name.size(); ++index) {
if (!IsAlNum(name[index]) && name[index] != '_')
return false;
}

return true;
}

const std::string test_suite_name_;
CodeLocation code_location_;
TestInfoContainer tests_;
InstantiationContainer instantiations_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestSuiteInfo);
};  

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
template <class TestCase>
using ParameterizedTestCaseInfo = ParameterizedTestSuiteInfo<TestCase>;
#endif  

class ParameterizedTestSuiteRegistry {
public:
ParameterizedTestSuiteRegistry() {}
~ParameterizedTestSuiteRegistry() {
for (auto& test_suite_info : test_suite_infos_) {
delete test_suite_info;
}
}

template <class TestSuite>
ParameterizedTestSuiteInfo<TestSuite>* GetTestSuitePatternHolder(
const char* test_suite_name, CodeLocation code_location) {
ParameterizedTestSuiteInfo<TestSuite>* typed_test_info = nullptr;
for (auto& test_suite_info : test_suite_infos_) {
if (test_suite_info->GetTestSuiteName() == test_suite_name) {
if (test_suite_info->GetTestSuiteTypeId() != GetTypeId<TestSuite>()) {
ReportInvalidTestSuiteType(test_suite_name, code_location);
posix::Abort();
} else {
typed_test_info = CheckedDowncastToActualType<
ParameterizedTestSuiteInfo<TestSuite> >(test_suite_info);
}
break;
}
}
if (typed_test_info == nullptr) {
typed_test_info = new ParameterizedTestSuiteInfo<TestSuite>(
test_suite_name, code_location);
test_suite_infos_.push_back(typed_test_info);
}
return typed_test_info;
}
void RegisterTests() {
for (auto& test_suite_info : test_suite_infos_) {
test_suite_info->RegisterTests();
}
}
#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
template <class TestCase>
ParameterizedTestCaseInfo<TestCase>* GetTestCasePatternHolder(
const char* test_case_name, CodeLocation code_location) {
return GetTestSuitePatternHolder<TestCase>(test_case_name, code_location);
}

#endif  

private:
using TestSuiteInfoContainer = ::std::vector<ParameterizedTestSuiteInfoBase*>;

TestSuiteInfoContainer test_suite_infos_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestSuiteRegistry);
};

class TypeParameterizedTestSuiteRegistry {
public:
void RegisterTestSuite(const char* test_suite_name,
CodeLocation code_location);

void RegisterInstantiation(const char* test_suite_name);

void CheckForInstantiations();

private:
struct TypeParameterizedTestSuiteInfo {
explicit TypeParameterizedTestSuiteInfo(CodeLocation c)
: code_location(c), instantiated(false) {}

CodeLocation code_location;
bool instantiated;
};

std::map<std::string, TypeParameterizedTestSuiteInfo> suites_;
};

}  

template <class Container>
internal::ParamGenerator<typename Container::value_type> ValuesIn(
const Container& container);

namespace internal {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

template <typename... Ts>
class ValueArray {
public:
explicit ValueArray(Ts... v) : v_(FlatTupleConstructTag{}, std::move(v)...) {}

template <typename T>
operator ParamGenerator<T>() const {  
return ValuesIn(MakeVector<T>(MakeIndexSequence<sizeof...(Ts)>()));
}

private:
template <typename T, size_t... I>
std::vector<T> MakeVector(IndexSequence<I...>) const {
return std::vector<T>{static_cast<T>(v_.template Get<I>())...};
}

FlatTuple<Ts...> v_;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

template <typename... T>
class CartesianProductGenerator
: public ParamGeneratorInterface<::std::tuple<T...>> {
public:
typedef ::std::tuple<T...> ParamType;

CartesianProductGenerator(const std::tuple<ParamGenerator<T>...>& g)
: generators_(g) {}
~CartesianProductGenerator() override {}

ParamIteratorInterface<ParamType>* Begin() const override {
return new Iterator(this, generators_, false);
}
ParamIteratorInterface<ParamType>* End() const override {
return new Iterator(this, generators_, true);
}

private:
template <class I>
class IteratorImpl;
template <size_t... I>
class IteratorImpl<IndexSequence<I...>>
: public ParamIteratorInterface<ParamType> {
public:
IteratorImpl(const ParamGeneratorInterface<ParamType>* base,
const std::tuple<ParamGenerator<T>...>& generators, bool is_end)
: base_(base),
begin_(std::get<I>(generators).begin()...),
end_(std::get<I>(generators).end()...),
current_(is_end ? end_ : begin_) {
ComputeCurrentValue();
}
~IteratorImpl() override {}

const ParamGeneratorInterface<ParamType>* BaseGenerator() const override {
return base_;
}
void Advance() override {
assert(!AtEnd());
++std::get<sizeof...(T) - 1>(current_);
AdvanceIfEnd<sizeof...(T) - 1>();
ComputeCurrentValue();
}
ParamIteratorInterface<ParamType>* Clone() const override {
return new IteratorImpl(*this);
}

const ParamType* Current() const override { return current_value_.get(); }

bool Equals(const ParamIteratorInterface<ParamType>& other) const override {
GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
<< "The program attempted to compare iterators "
<< "from different generators." << std::endl;
const IteratorImpl* typed_other =
CheckedDowncastToActualType<const IteratorImpl>(&other);

if (AtEnd() && typed_other->AtEnd()) return true;

bool same = true;
bool dummy[] = {
(same = same && std::get<I>(current_) ==
std::get<I>(typed_other->current_))...};
(void)dummy;
return same;
}

private:
template <size_t ThisI>
void AdvanceIfEnd() {
if (std::get<ThisI>(current_) != std::get<ThisI>(end_)) return;

bool last = ThisI == 0;
if (last) {
return;
}

constexpr size_t NextI = ThisI - (ThisI != 0);
std::get<ThisI>(current_) = std::get<ThisI>(begin_);
++std::get<NextI>(current_);
AdvanceIfEnd<NextI>();
}

void ComputeCurrentValue() {
if (!AtEnd())
current_value_ = std::make_shared<ParamType>(*std::get<I>(current_)...);
}
bool AtEnd() const {
bool at_end = false;
bool dummy[] = {
(at_end = at_end || std::get<I>(current_) == std::get<I>(end_))...};
(void)dummy;
return at_end;
}

const ParamGeneratorInterface<ParamType>* const base_;
std::tuple<typename ParamGenerator<T>::iterator...> begin_;
std::tuple<typename ParamGenerator<T>::iterator...> end_;
std::tuple<typename ParamGenerator<T>::iterator...> current_;
std::shared_ptr<ParamType> current_value_;
};

using Iterator = IteratorImpl<typename MakeIndexSequence<sizeof...(T)>::type>;

std::tuple<ParamGenerator<T>...> generators_;
};

template <class... Gen>
class CartesianProductHolder {
public:
CartesianProductHolder(const Gen&... g) : generators_(g...) {}
template <typename... T>
operator ParamGenerator<::std::tuple<T...>>() const {
return ParamGenerator<::std::tuple<T...>>(
new CartesianProductGenerator<T...>(generators_));
}

private:
std::tuple<Gen...> generators_;
};

}  
}  

#endif  
