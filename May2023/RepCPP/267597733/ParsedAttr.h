
#ifndef LLVM_CLANG_SEMA_ATTRIBUTELIST_H
#define LLVM_CLANG_SEMA_ATTRIBUTELIST_H

#include "clang/Basic/AttrSubjectMatchRules.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Ownership.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/VersionTuple.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <utility>

namespace clang {

class ASTContext;
class Decl;
class Expr;
class IdentifierInfo;
class LangOptions;

struct AvailabilityChange {
SourceLocation KeywordLoc;

VersionTuple Version;

SourceRange VersionRange;

bool isValid() const { return !Version.empty(); }
};

namespace {

enum AvailabilitySlot {
IntroducedSlot, DeprecatedSlot, ObsoletedSlot, NumAvailabilitySlots
};

struct AvailabilityData {
AvailabilityChange Changes[NumAvailabilitySlots];
SourceLocation StrictLoc;
const Expr *Replacement;

AvailabilityData(const AvailabilityChange &Introduced,
const AvailabilityChange &Deprecated,
const AvailabilityChange &Obsoleted,
SourceLocation Strict, const Expr *ReplaceExpr)
: StrictLoc(Strict), Replacement(ReplaceExpr) {
Changes[IntroducedSlot] = Introduced;
Changes[DeprecatedSlot] = Deprecated;
Changes[ObsoletedSlot] = Obsoleted;
}
};

} 

struct IdentifierLoc {
SourceLocation Loc;
IdentifierInfo *Ident;

static IdentifierLoc *create(ASTContext &Ctx, SourceLocation Loc,
IdentifierInfo *Ident);
};

using ArgsUnion = llvm::PointerUnion<Expr *, IdentifierLoc *>;
using ArgsVector = llvm::SmallVector<ArgsUnion, 12U>;

class ParsedAttr { 
public:
enum Syntax {
AS_GNU,

AS_CXX11,

AS_C2x,

AS_Declspec,

AS_Microsoft,

AS_Keyword,

AS_Pragma,

AS_ContextSensitiveKeyword,
};

private:
IdentifierInfo *AttrName;
IdentifierInfo *ScopeName;
SourceRange AttrRange;
SourceLocation ScopeLoc;
SourceLocation EllipsisLoc;

unsigned AttrKind : 16;

unsigned NumArgs : 16;

unsigned SyntaxUsed : 3;

mutable unsigned Invalid : 1;

mutable unsigned UsedAsTypeAttr : 1;

unsigned IsAvailability : 1;

unsigned IsTypeTagForDatatype : 1;

unsigned IsProperty : 1;

unsigned HasParsedType : 1;

mutable unsigned HasProcessingCache : 1;

mutable unsigned ProcessingCache : 8;

SourceLocation UnavailableLoc;

const Expr *MessageExpr;

ArgsUnion *getArgsBuffer() { return reinterpret_cast<ArgsUnion *>(this + 1); }
ArgsUnion const *getArgsBuffer() const {
return reinterpret_cast<ArgsUnion const *>(this + 1);
}

AvailabilityData *getAvailabilityData() {
return reinterpret_cast<AvailabilityData*>(getArgsBuffer() + NumArgs);
}
const AvailabilityData *getAvailabilityData() const {
return reinterpret_cast<const AvailabilityData*>(getArgsBuffer() + NumArgs);
}

public:
struct TypeTagForDatatypeData {
ParsedType *MatchingCType;
unsigned LayoutCompatible : 1;
unsigned MustBeNull : 1;
};
struct PropertyData {
IdentifierInfo *GetterId, *SetterId;

PropertyData(IdentifierInfo *getterId, IdentifierInfo *setterId)
: GetterId(getterId), SetterId(setterId) {}
};

private:
friend class AttributeFactory;
friend class AttributePool;

ParsedAttr(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
ArgsUnion *args, unsigned numArgs, Syntax syntaxUsed,
SourceLocation ellipsisLoc)
: AttrName(attrName), ScopeName(scopeName), AttrRange(attrRange),
ScopeLoc(scopeLoc), EllipsisLoc(ellipsisLoc), NumArgs(numArgs),
SyntaxUsed(syntaxUsed), Invalid(false), UsedAsTypeAttr(false),
IsAvailability(false), IsTypeTagForDatatype(false), IsProperty(false),
HasParsedType(false), HasProcessingCache(false) {
if (numArgs) memcpy(getArgsBuffer(), args, numArgs * sizeof(ArgsUnion));
AttrKind = getKind(getName(), getScopeName(), syntaxUsed);
}

ParsedAttr(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierLoc *Parm, const AvailabilityChange &introduced,
const AvailabilityChange &deprecated,
const AvailabilityChange &obsoleted, SourceLocation unavailable,
const Expr *messageExpr, Syntax syntaxUsed, SourceLocation strict,
const Expr *replacementExpr)
: AttrName(attrName), ScopeName(scopeName), AttrRange(attrRange),
ScopeLoc(scopeLoc), NumArgs(1), SyntaxUsed(syntaxUsed), Invalid(false),
UsedAsTypeAttr(false), IsAvailability(true),
IsTypeTagForDatatype(false), IsProperty(false), HasParsedType(false),
HasProcessingCache(false), UnavailableLoc(unavailable),
MessageExpr(messageExpr) {
ArgsUnion PVal(Parm);
memcpy(getArgsBuffer(), &PVal, sizeof(ArgsUnion));
new (getAvailabilityData()) AvailabilityData(
introduced, deprecated, obsoleted, strict, replacementExpr);
AttrKind = getKind(getName(), getScopeName(), syntaxUsed);
}

ParsedAttr(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierLoc *Parm1, IdentifierLoc *Parm2, IdentifierLoc *Parm3,
Syntax syntaxUsed)
: AttrName(attrName), ScopeName(scopeName), AttrRange(attrRange),
ScopeLoc(scopeLoc), NumArgs(3), SyntaxUsed(syntaxUsed), Invalid(false),
UsedAsTypeAttr(false), IsAvailability(false),
IsTypeTagForDatatype(false), IsProperty(false), HasParsedType(false),
HasProcessingCache(false) {
ArgsUnion *Args = getArgsBuffer();
Args[0] = Parm1;
Args[1] = Parm2;
Args[2] = Parm3;
AttrKind = getKind(getName(), getScopeName(), syntaxUsed);
}

ParsedAttr(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierLoc *ArgKind, ParsedType matchingCType,
bool layoutCompatible, bool mustBeNull, Syntax syntaxUsed)
: AttrName(attrName), ScopeName(scopeName), AttrRange(attrRange),
ScopeLoc(scopeLoc), NumArgs(1), SyntaxUsed(syntaxUsed), Invalid(false),
UsedAsTypeAttr(false), IsAvailability(false),
IsTypeTagForDatatype(true), IsProperty(false), HasParsedType(false),
HasProcessingCache(false) {
ArgsUnion PVal(ArgKind);
memcpy(getArgsBuffer(), &PVal, sizeof(ArgsUnion));
TypeTagForDatatypeData &ExtraData = getTypeTagForDatatypeDataSlot();
new (&ExtraData.MatchingCType) ParsedType(matchingCType);
ExtraData.LayoutCompatible = layoutCompatible;
ExtraData.MustBeNull = mustBeNull;
AttrKind = getKind(getName(), getScopeName(), syntaxUsed);
}

ParsedAttr(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
ParsedType typeArg, Syntax syntaxUsed)
: AttrName(attrName), ScopeName(scopeName), AttrRange(attrRange),
ScopeLoc(scopeLoc), NumArgs(0), SyntaxUsed(syntaxUsed), Invalid(false),
UsedAsTypeAttr(false), IsAvailability(false),
IsTypeTagForDatatype(false), IsProperty(false), HasParsedType(true),
HasProcessingCache(false) {
new (&getTypeBuffer()) ParsedType(typeArg);
AttrKind = getKind(getName(), getScopeName(), syntaxUsed);
}

ParsedAttr(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierInfo *getterId, IdentifierInfo *setterId,
Syntax syntaxUsed)
: AttrName(attrName), ScopeName(scopeName), AttrRange(attrRange),
ScopeLoc(scopeLoc), NumArgs(0), SyntaxUsed(syntaxUsed), Invalid(false),
UsedAsTypeAttr(false), IsAvailability(false),
IsTypeTagForDatatype(false), IsProperty(true), HasParsedType(false),
HasProcessingCache(false) {
new (&getPropertyDataBuffer()) PropertyData(getterId, setterId);
AttrKind = getKind(getName(), getScopeName(), syntaxUsed);
}

TypeTagForDatatypeData &getTypeTagForDatatypeDataSlot() {
return *reinterpret_cast<TypeTagForDatatypeData*>(getArgsBuffer()+NumArgs);
}
const TypeTagForDatatypeData &getTypeTagForDatatypeDataSlot() const {
return *reinterpret_cast<const TypeTagForDatatypeData*>(getArgsBuffer()
+ NumArgs);
}

ParsedType &getTypeBuffer() {
return *reinterpret_cast<ParsedType *>(this + 1);
}
const ParsedType &getTypeBuffer() const {
return *reinterpret_cast<const ParsedType *>(this + 1);
}

PropertyData &getPropertyDataBuffer() {
assert(IsProperty);
return *reinterpret_cast<PropertyData*>(this + 1);
}
const PropertyData &getPropertyDataBuffer() const {
assert(IsProperty);
return *reinterpret_cast<const PropertyData*>(this + 1);
}

size_t allocated_size() const;

public:
ParsedAttr(const ParsedAttr &) = delete;
ParsedAttr &operator=(const ParsedAttr &) = delete;
~ParsedAttr() = delete;

void operator delete(void *) = delete;

enum Kind {
#define PARSED_ATTR(NAME) AT_##NAME,
#include "clang/Sema/AttrParsedAttrList.inc"
#undef PARSED_ATTR
IgnoredAttribute,
UnknownAttribute
};

IdentifierInfo *getName() const { return AttrName; }
SourceLocation getLoc() const { return AttrRange.getBegin(); }
SourceRange getRange() const { return AttrRange; }

bool hasScope() const { return ScopeName; }
IdentifierInfo *getScopeName() const { return ScopeName; }
SourceLocation getScopeLoc() const { return ScopeLoc; }

bool hasParsedType() const { return HasParsedType; }

bool isDeclspecPropertyAttribute() const  {
return IsProperty;
}

bool isAlignasAttribute() const {
return getKind() == AT_Aligned && isKeywordAttribute();
}

bool isDeclspecAttribute() const { return SyntaxUsed == AS_Declspec; }
bool isMicrosoftAttribute() const { return SyntaxUsed == AS_Microsoft; }

bool isCXX11Attribute() const {
return SyntaxUsed == AS_CXX11 || isAlignasAttribute();
}

bool isC2xAttribute() const {
return SyntaxUsed == AS_C2x;
}

bool isKeywordAttribute() const {
return SyntaxUsed == AS_Keyword || SyntaxUsed == AS_ContextSensitiveKeyword;
}

bool isContextSensitiveKeywordAttribute() const {
return SyntaxUsed == AS_ContextSensitiveKeyword;
}

bool isInvalid() const { return Invalid; }
void setInvalid(bool b = true) const { Invalid = b; }

bool hasProcessingCache() const { return HasProcessingCache; }

unsigned getProcessingCache() const {
assert(hasProcessingCache());
return ProcessingCache;
}

void setProcessingCache(unsigned value) const {
ProcessingCache = value;
HasProcessingCache = true;
}

bool isUsedAsTypeAttr() const { return UsedAsTypeAttr; }
void setUsedAsTypeAttr() { UsedAsTypeAttr = true; }

bool isPackExpansion() const { return EllipsisLoc.isValid(); }
SourceLocation getEllipsisLoc() const { return EllipsisLoc; }

Kind getKind() const { return Kind(AttrKind); }
static Kind getKind(const IdentifierInfo *Name, const IdentifierInfo *Scope,
Syntax SyntaxUsed);

unsigned getNumArgs() const { return NumArgs; }

ArgsUnion getArg(unsigned Arg) const {
assert(Arg < NumArgs && "Arg access out of range!");
return getArgsBuffer()[Arg];
}

bool isArgExpr(unsigned Arg) const {
return Arg < NumArgs && getArg(Arg).is<Expr*>();
}

Expr *getArgAsExpr(unsigned Arg) const {
return getArg(Arg).get<Expr*>();
}

bool isArgIdent(unsigned Arg) const {
return Arg < NumArgs && getArg(Arg).is<IdentifierLoc*>();
}

IdentifierLoc *getArgAsIdent(unsigned Arg) const {
return getArg(Arg).get<IdentifierLoc*>();
}

const AvailabilityChange &getAvailabilityIntroduced() const {
assert(getKind() == AT_Availability && "Not an availability attribute");
return getAvailabilityData()->Changes[IntroducedSlot];
}

const AvailabilityChange &getAvailabilityDeprecated() const {
assert(getKind() == AT_Availability && "Not an availability attribute");
return getAvailabilityData()->Changes[DeprecatedSlot];
}

const AvailabilityChange &getAvailabilityObsoleted() const {
assert(getKind() == AT_Availability && "Not an availability attribute");
return getAvailabilityData()->Changes[ObsoletedSlot];
}

SourceLocation getStrictLoc() const {
assert(getKind() == AT_Availability && "Not an availability attribute");
return getAvailabilityData()->StrictLoc;
}

SourceLocation getUnavailableLoc() const {
assert(getKind() == AT_Availability && "Not an availability attribute");
return UnavailableLoc;
}

const Expr * getMessageExpr() const {
assert(getKind() == AT_Availability && "Not an availability attribute");
return MessageExpr;
}

const Expr *getReplacementExpr() const {
assert(getKind() == AT_Availability && "Not an availability attribute");
return getAvailabilityData()->Replacement;
}

const ParsedType &getMatchingCType() const {
assert(getKind() == AT_TypeTagForDatatype &&
"Not a type_tag_for_datatype attribute");
return *getTypeTagForDatatypeDataSlot().MatchingCType;
}

bool getLayoutCompatible() const {
assert(getKind() == AT_TypeTagForDatatype &&
"Not a type_tag_for_datatype attribute");
return getTypeTagForDatatypeDataSlot().LayoutCompatible;
}

bool getMustBeNull() const {
assert(getKind() == AT_TypeTagForDatatype &&
"Not a type_tag_for_datatype attribute");
return getTypeTagForDatatypeDataSlot().MustBeNull;
}

const ParsedType &getTypeArg() const {
assert(HasParsedType && "Not a type attribute");
return getTypeBuffer();
}

const PropertyData &getPropertyData() const {
assert(isDeclspecPropertyAttribute() && "Not a __delcspec(property) attribute");
return getPropertyDataBuffer();
}

unsigned getAttributeSpellingListIndex() const;

bool isTargetSpecificAttr() const;
bool isTypeAttr() const;
bool isStmtAttr() const;

bool hasCustomParsing() const;
unsigned getMinArgs() const;
unsigned getMaxArgs() const;
bool hasVariadicArg() const;
bool diagnoseAppertainsTo(class Sema &S, const Decl *D) const;
bool appliesToDecl(const Decl *D, attr::SubjectMatchRule MatchRule) const;
void getMatchRules(const LangOptions &LangOpts,
SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>>
&MatchRules) const;
bool diagnoseLangOpts(class Sema &S) const;
bool existsInTarget(const TargetInfo &Target) const;
bool isKnownToGCC() const;
bool isSupportedByPragmaAttribute() const;

unsigned getSemanticSpelling() const;
};

class AttributePool;
class AttributeFactory {
public:
enum {
AvailabilityAllocSize =
sizeof(ParsedAttr) +
((sizeof(AvailabilityData) + sizeof(void *) + sizeof(ArgsUnion) - 1) /
sizeof(void *) * sizeof(void *)),
TypeTagForDatatypeAllocSize = sizeof(ParsedAttr) +
(sizeof(ParsedAttr::TypeTagForDatatypeData) +
sizeof(void *) + sizeof(ArgsUnion) - 1) /
sizeof(void *) * sizeof(void *),
PropertyAllocSize =
sizeof(ParsedAttr) +
(sizeof(ParsedAttr::PropertyData) + sizeof(void *) - 1) /
sizeof(void *) * sizeof(void *)
};

private:
enum {
InlineFreeListsCapacity =
1 + (AvailabilityAllocSize - sizeof(ParsedAttr)) / sizeof(void *)
};

llvm::BumpPtrAllocator Alloc;

SmallVector<SmallVector<ParsedAttr *, 8>, InlineFreeListsCapacity> FreeLists;

friend class AttributePool;

void *allocate(size_t size);

void deallocate(ParsedAttr *AL);

void reclaimPool(AttributePool &head);

public:
AttributeFactory();
~AttributeFactory();
};

class AttributePool {
friend class AttributeFactory;
AttributeFactory &Factory;
llvm::TinyPtrVector<ParsedAttr *> Attrs;

void *allocate(size_t size) {
return Factory.allocate(size);
}

ParsedAttr *add(ParsedAttr *attr) {
Attrs.push_back(attr);
return attr;
}

void remove(ParsedAttr *attr) {
assert(llvm::is_contained(Attrs, attr) &&
"Can't take attribute from a pool that doesn't own it!");
Attrs.erase(llvm::find(Attrs, attr));
}

void takePool(AttributePool &pool);

public:
AttributePool(AttributeFactory &factory) : Factory(factory) {}

AttributePool(const AttributePool &) = delete;

~AttributePool() { Factory.reclaimPool(*this); }

AttributePool(AttributePool &&pool) = default;

AttributeFactory &getFactory() const { return Factory; }

void clear() {
Factory.reclaimPool(*this);
Attrs.clear();
}

void takeAllFrom(AttributePool &pool) {
takePool(pool);
pool.Attrs.clear();
}

ParsedAttr *create(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
ArgsUnion *args, unsigned numArgs,
ParsedAttr::Syntax syntax,
SourceLocation ellipsisLoc = SourceLocation()) {
void *memory = allocate(sizeof(ParsedAttr) + numArgs * sizeof(ArgsUnion));
return add(new (memory) ParsedAttr(attrName, attrRange, scopeName, scopeLoc,
args, numArgs, syntax, ellipsisLoc));
}

ParsedAttr *create(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierLoc *Param, const AvailabilityChange &introduced,
const AvailabilityChange &deprecated,
const AvailabilityChange &obsoleted,
SourceLocation unavailable, const Expr *MessageExpr,
ParsedAttr::Syntax syntax, SourceLocation strict,
const Expr *ReplacementExpr) {
void *memory = allocate(AttributeFactory::AvailabilityAllocSize);
return add(new (memory) ParsedAttr(
attrName, attrRange, scopeName, scopeLoc, Param, introduced, deprecated,
obsoleted, unavailable, MessageExpr, syntax, strict, ReplacementExpr));
}

ParsedAttr *create(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierLoc *Param1, IdentifierLoc *Param2,
IdentifierLoc *Param3, ParsedAttr::Syntax syntax) {
size_t size = sizeof(ParsedAttr) + 3 * sizeof(ArgsUnion);
void *memory = allocate(size);
return add(new (memory) ParsedAttr(attrName, attrRange, scopeName, scopeLoc,
Param1, Param2, Param3, syntax));
}

ParsedAttr *
createTypeTagForDatatype(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierLoc *argumentKind,
ParsedType matchingCType, bool layoutCompatible,
bool mustBeNull, ParsedAttr::Syntax syntax) {
void *memory = allocate(AttributeFactory::TypeTagForDatatypeAllocSize);
return add(new (memory) ParsedAttr(attrName, attrRange, scopeName, scopeLoc,
argumentKind, matchingCType,
layoutCompatible, mustBeNull, syntax));
}

ParsedAttr *createTypeAttribute(IdentifierInfo *attrName,
SourceRange attrRange,
IdentifierInfo *scopeName,
SourceLocation scopeLoc, ParsedType typeArg,
ParsedAttr::Syntax syntaxUsed) {
void *memory = allocate(sizeof(ParsedAttr) + sizeof(void *));
return add(new (memory) ParsedAttr(attrName, attrRange, scopeName, scopeLoc,
typeArg, syntaxUsed));
}

ParsedAttr *
createPropertyAttribute(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierInfo *getterId, IdentifierInfo *setterId,
ParsedAttr::Syntax syntaxUsed) {
void *memory = allocate(AttributeFactory::PropertyAllocSize);
return add(new (memory) ParsedAttr(attrName, attrRange, scopeName, scopeLoc,
getterId, setterId, syntaxUsed));
}
};

class ParsedAttributesView {
using VecTy = llvm::TinyPtrVector<ParsedAttr *>;
using SizeType = decltype(std::declval<VecTy>().size());

public:
bool empty() const { return AttrList.empty(); }
SizeType size() const { return AttrList.size(); }
ParsedAttr &operator[](SizeType pos) { return *AttrList[pos]; }
const ParsedAttr &operator[](SizeType pos) const { return *AttrList[pos]; }

void addAtStart(ParsedAttr *newAttr) {
assert(newAttr);
AttrList.insert(AttrList.begin(), newAttr);
}
void addAtEnd(ParsedAttr *newAttr) {
assert(newAttr);
AttrList.push_back(newAttr);
}

void remove(ParsedAttr *ToBeRemoved) {
assert(is_contained(AttrList, ToBeRemoved) &&
"Cannot remove attribute that isn't in the list");
AttrList.erase(llvm::find(AttrList, ToBeRemoved));
}

void clearListOnly() { AttrList.clear(); }

struct iterator : llvm::iterator_adaptor_base<iterator, VecTy::iterator,
std::random_access_iterator_tag,
ParsedAttr> {
iterator() : iterator_adaptor_base(nullptr) {}
iterator(VecTy::iterator I) : iterator_adaptor_base(I) {}
reference operator*() { return **I; }
friend class ParsedAttributesView;
};
struct const_iterator
: llvm::iterator_adaptor_base<const_iterator, VecTy::const_iterator,
std::random_access_iterator_tag,
ParsedAttr> {
const_iterator() : iterator_adaptor_base(nullptr) {}
const_iterator(VecTy::const_iterator I) : iterator_adaptor_base(I) {}

reference operator*() const { return **I; }
friend class ParsedAttributesView;
};

void addAll(iterator B, iterator E) {
AttrList.insert(AttrList.begin(), B.I, E.I);
}

void addAll(const_iterator B, const_iterator E) {
AttrList.insert(AttrList.begin(), B.I, E.I);
}

void addAllAtEnd(iterator B, iterator E) {
AttrList.insert(AttrList.end(), B.I, E.I);
}

void addAllAtEnd(const_iterator B, const_iterator E) {
AttrList.insert(AttrList.end(), B.I, E.I);
}

iterator begin() { return iterator(AttrList.begin()); }
const_iterator begin() const { return const_iterator(AttrList.begin()); }
iterator end() { return iterator(AttrList.end()); }
const_iterator end() const { return const_iterator(AttrList.end()); }

bool hasAttribute(ParsedAttr::Kind K) const {
return llvm::any_of(
AttrList, [K](const ParsedAttr *AL) { return AL->getKind() == K; });
}

private:
VecTy AttrList;
};

class ParsedAttributes : public ParsedAttributesView {
public:
ParsedAttributes(AttributeFactory &factory) : pool(factory) {}
ParsedAttributes(const ParsedAttributes &) = delete;

AttributePool &getPool() const { return pool; }

void takeAllFrom(ParsedAttributes &attrs) {
addAll(attrs.begin(), attrs.end());
attrs.clearListOnly();
pool.takeAllFrom(attrs.pool);
}

void clear() {
clearListOnly();
pool.clear();
}

ParsedAttr *addNew(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
ArgsUnion *args, unsigned numArgs,
ParsedAttr::Syntax syntax,
SourceLocation ellipsisLoc = SourceLocation()) {
ParsedAttr *attr = pool.create(attrName, attrRange, scopeName, scopeLoc,
args, numArgs, syntax, ellipsisLoc);
addAtStart(attr);
return attr;
}

ParsedAttr *addNew(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierLoc *Param, const AvailabilityChange &introduced,
const AvailabilityChange &deprecated,
const AvailabilityChange &obsoleted,
SourceLocation unavailable, const Expr *MessageExpr,
ParsedAttr::Syntax syntax, SourceLocation strict,
const Expr *ReplacementExpr) {
ParsedAttr *attr = pool.create(
attrName, attrRange, scopeName, scopeLoc, Param, introduced, deprecated,
obsoleted, unavailable, MessageExpr, syntax, strict, ReplacementExpr);
addAtStart(attr);
return attr;
}

ParsedAttr *addNew(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierLoc *Param1, IdentifierLoc *Param2,
IdentifierLoc *Param3, ParsedAttr::Syntax syntax) {
ParsedAttr *attr = pool.create(attrName, attrRange, scopeName, scopeLoc,
Param1, Param2, Param3, syntax);
addAtStart(attr);
return attr;
}

ParsedAttr *
addNewTypeTagForDatatype(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierLoc *argumentKind,
ParsedType matchingCType, bool layoutCompatible,
bool mustBeNull, ParsedAttr::Syntax syntax) {
ParsedAttr *attr = pool.createTypeTagForDatatype(
attrName, attrRange, scopeName, scopeLoc, argumentKind, matchingCType,
layoutCompatible, mustBeNull, syntax);
addAtStart(attr);
return attr;
}

ParsedAttr *addNewTypeAttr(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
ParsedType typeArg,
ParsedAttr::Syntax syntaxUsed) {
ParsedAttr *attr = pool.createTypeAttribute(attrName, attrRange, scopeName,
scopeLoc, typeArg, syntaxUsed);
addAtStart(attr);
return attr;
}

ParsedAttr *
addNewPropertyAttr(IdentifierInfo *attrName, SourceRange attrRange,
IdentifierInfo *scopeName, SourceLocation scopeLoc,
IdentifierInfo *getterId, IdentifierInfo *setterId,
ParsedAttr::Syntax syntaxUsed) {
ParsedAttr *attr =
pool.createPropertyAttribute(attrName, attrRange, scopeName, scopeLoc,
getterId, setterId, syntaxUsed);
addAtStart(attr);
return attr;
}

private:
mutable AttributePool pool;
};

enum AttributeArgumentNType {
AANT_ArgumentIntOrBool,
AANT_ArgumentIntegerConstant,
AANT_ArgumentString,
AANT_ArgumentIdentifier
};

enum AttributeDeclKind {
ExpectedFunction,
ExpectedUnion,
ExpectedVariableOrFunction,
ExpectedFunctionOrMethod,
ExpectedFunctionMethodOrBlock,
ExpectedFunctionMethodOrParameter,
ExpectedVariable,
ExpectedVariableOrField,
ExpectedVariableFieldOrTag,
ExpectedTypeOrNamespace,
ExpectedFunctionVariableOrClass,
ExpectedKernelFunction,
ExpectedFunctionWithProtoType,
};

} 

#endif 
