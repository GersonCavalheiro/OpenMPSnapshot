
#ifndef LLVM_CLANG_AST_DECL_H
#define LLVM_CLANG_AST_DECL_H

#include "clang/AST/APValue.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Redeclarable.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Linkage.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/PragmaKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/Visibility.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace clang {

class ASTContext;
struct ASTTemplateArgumentListInfo;
class Attr;
class CompoundStmt;
class DependentFunctionTemplateSpecializationInfo;
class EnumDecl;
class Expr;
class FunctionTemplateDecl;
class FunctionTemplateSpecializationInfo;
class LabelStmt;
class MemberSpecializationInfo;
class Module;
class NamespaceDecl;
class ParmVarDecl;
class RecordDecl;
class Stmt;
class StringLiteral;
class TagDecl;
class TemplateArgumentList;
class TemplateArgumentListInfo;
class TemplateParameterList;
class TypeAliasTemplateDecl;
class TypeLoc;
class UnresolvedSetImpl;
class VarTemplateDecl;

class alignas(8) TypeSourceInfo {
friend class ASTContext;

QualType Ty;

TypeSourceInfo(QualType ty) : Ty(ty) {}

public:
QualType getType() const { return Ty; }

TypeLoc getTypeLoc() const; 

void overrideType(QualType T) { Ty = T; }
};

class TranslationUnitDecl : public Decl, public DeclContext {
ASTContext &Ctx;

NamespaceDecl *AnonymousNamespace = nullptr;

explicit TranslationUnitDecl(ASTContext &ctx);

virtual void anchor();

public:
ASTContext &getASTContext() const { return Ctx; }

NamespaceDecl *getAnonymousNamespace() const { return AnonymousNamespace; }
void setAnonymousNamespace(NamespaceDecl *D) { AnonymousNamespace = D; }

static TranslationUnitDecl *Create(ASTContext &C);

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == TranslationUnit; }
static DeclContext *castToDeclContext(const TranslationUnitDecl *D) {
return static_cast<DeclContext *>(const_cast<TranslationUnitDecl*>(D));
}
static TranslationUnitDecl *castFromDeclContext(const DeclContext *DC) {
return static_cast<TranslationUnitDecl *>(const_cast<DeclContext*>(DC));
}
};

class PragmaCommentDecl final
: public Decl,
private llvm::TrailingObjects<PragmaCommentDecl, char> {
friend class ASTDeclReader;
friend class ASTDeclWriter;
friend TrailingObjects;

PragmaMSCommentKind CommentKind;

PragmaCommentDecl(TranslationUnitDecl *TU, SourceLocation CommentLoc,
PragmaMSCommentKind CommentKind)
: Decl(PragmaComment, TU, CommentLoc), CommentKind(CommentKind) {}

virtual void anchor();

public:
static PragmaCommentDecl *Create(const ASTContext &C, TranslationUnitDecl *DC,
SourceLocation CommentLoc,
PragmaMSCommentKind CommentKind,
StringRef Arg);
static PragmaCommentDecl *CreateDeserialized(ASTContext &C, unsigned ID,
unsigned ArgSize);

PragmaMSCommentKind getCommentKind() const { return CommentKind; }

StringRef getArg() const { return getTrailingObjects<char>(); }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == PragmaComment; }
};

class PragmaDetectMismatchDecl final
: public Decl,
private llvm::TrailingObjects<PragmaDetectMismatchDecl, char> {
friend class ASTDeclReader;
friend class ASTDeclWriter;
friend TrailingObjects;

size_t ValueStart;

PragmaDetectMismatchDecl(TranslationUnitDecl *TU, SourceLocation Loc,
size_t ValueStart)
: Decl(PragmaDetectMismatch, TU, Loc), ValueStart(ValueStart) {}

virtual void anchor();

public:
static PragmaDetectMismatchDecl *Create(const ASTContext &C,
TranslationUnitDecl *DC,
SourceLocation Loc, StringRef Name,
StringRef Value);
static PragmaDetectMismatchDecl *
CreateDeserialized(ASTContext &C, unsigned ID, unsigned NameValueSize);

StringRef getName() const { return getTrailingObjects<char>(); }
StringRef getValue() const { return getTrailingObjects<char>() + ValueStart; }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == PragmaDetectMismatch; }
};

class ExternCContextDecl : public Decl, public DeclContext {
explicit ExternCContextDecl(TranslationUnitDecl *TU)
: Decl(ExternCContext, TU, SourceLocation()),
DeclContext(ExternCContext) {}

virtual void anchor();

public:
static ExternCContextDecl *Create(const ASTContext &C,
TranslationUnitDecl *TU);

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == ExternCContext; }
static DeclContext *castToDeclContext(const ExternCContextDecl *D) {
return static_cast<DeclContext *>(const_cast<ExternCContextDecl*>(D));
}
static ExternCContextDecl *castFromDeclContext(const DeclContext *DC) {
return static_cast<ExternCContextDecl *>(const_cast<DeclContext*>(DC));
}
};

class NamedDecl : public Decl {
DeclarationName Name;

virtual void anchor();

private:
NamedDecl *getUnderlyingDeclImpl() LLVM_READONLY;

protected:
NamedDecl(Kind DK, DeclContext *DC, SourceLocation L, DeclarationName N)
: Decl(DK, DC, L), Name(N) {}

public:
IdentifierInfo *getIdentifier() const { return Name.getAsIdentifierInfo(); }

StringRef getName() const {
assert(Name.isIdentifier() && "Name is not a simple identifier");
return getIdentifier() ? getIdentifier()->getName() : "";
}

std::string getNameAsString() const { return Name.getAsString(); }

virtual void printName(raw_ostream &os) const;

DeclarationName getDeclName() const { return Name; }

void setDeclName(DeclarationName N) { Name = N; }

void printQualifiedName(raw_ostream &OS) const;
void printQualifiedName(raw_ostream &OS, const PrintingPolicy &Policy) const;

std::string getQualifiedNameAsString() const;

virtual void getNameForDiagnostic(raw_ostream &OS,
const PrintingPolicy &Policy,
bool Qualified) const;

bool declarationReplaces(NamedDecl *OldD, bool IsKnownNewer = true) const;

bool hasLinkage() const;

using Decl::isModulePrivate;
using Decl::setModulePrivate;

bool isCXXClassMember() const {
const DeclContext *DC = getDeclContext();

if (isa<EnumDecl>(DC))
DC = DC->getRedeclContext();

return DC->isRecord();
}

bool isCXXInstanceMember() const;

Linkage getLinkageInternal() const;

Linkage getFormalLinkage() const {
return clang::getFormalLinkage(getLinkageInternal());
}

bool hasExternalFormalLinkage() const {
return isExternalFormalLinkage(getLinkageInternal());
}

bool isExternallyVisible() const {
return clang::isExternallyVisible(getLinkageInternal());
}

bool isExternallyDeclarable() const {
return isExternallyVisible() && !getOwningModuleForLinkage();
}

Visibility getVisibility() const {
return getLinkageAndVisibility().getVisibility();
}

LinkageInfo getLinkageAndVisibility() const;

enum ExplicitVisibilityKind {
VisibilityForType,

VisibilityForValue
};

Optional<Visibility>
getExplicitVisibility(ExplicitVisibilityKind kind) const;

bool isLinkageValid() const;

bool hasLinkageBeenComputed() const {
return hasCachedLinkage();
}

NamedDecl *getUnderlyingDecl() {
if (this->getKind() != UsingShadow &&
this->getKind() != ConstructorUsingShadow &&
this->getKind() != ObjCCompatibleAlias &&
this->getKind() != NamespaceAlias)
return this;

return getUnderlyingDeclImpl();
}
const NamedDecl *getUnderlyingDecl() const {
return const_cast<NamedDecl*>(this)->getUnderlyingDecl();
}

NamedDecl *getMostRecentDecl() {
return cast<NamedDecl>(static_cast<Decl *>(this)->getMostRecentDecl());
}
const NamedDecl *getMostRecentDecl() const {
return const_cast<NamedDecl*>(this)->getMostRecentDecl();
}

ObjCStringFormatFamily getObjCFStringFormattingFamily() const;

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K >= firstNamed && K <= lastNamed; }
};

inline raw_ostream &operator<<(raw_ostream &OS, const NamedDecl &ND) {
ND.printName(OS);
return OS;
}

class LabelDecl : public NamedDecl {
LabelStmt *TheStmt;
StringRef MSAsmName;
bool MSAsmNameResolved = false;

SourceLocation LocStart;

LabelDecl(DeclContext *DC, SourceLocation IdentL, IdentifierInfo *II,
LabelStmt *S, SourceLocation StartL)
: NamedDecl(Label, DC, IdentL, II), TheStmt(S), LocStart(StartL) {}

void anchor() override;

public:
static LabelDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation IdentL, IdentifierInfo *II);
static LabelDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation IdentL, IdentifierInfo *II,
SourceLocation GnuLabelL);
static LabelDecl *CreateDeserialized(ASTContext &C, unsigned ID);

LabelStmt *getStmt() const { return TheStmt; }
void setStmt(LabelStmt *T) { TheStmt = T; }

bool isGnuLocal() const { return LocStart != getLocation(); }
void setLocStart(SourceLocation L) { LocStart = L; }

SourceRange getSourceRange() const override LLVM_READONLY {
return SourceRange(LocStart, getLocation());
}

bool isMSAsmLabel() const { return !MSAsmName.empty(); }
bool isResolvedMSAsmLabel() const { return isMSAsmLabel() && MSAsmNameResolved; }
void setMSAsmLabel(StringRef Name);
StringRef getMSAsmLabel() const { return MSAsmName; }
void setMSAsmLabelResolved() { MSAsmNameResolved = true; }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == Label; }
};

class NamespaceDecl : public NamedDecl, public DeclContext,
public Redeclarable<NamespaceDecl>
{
SourceLocation LocStart;

SourceLocation RBraceLoc;

llvm::PointerIntPair<NamespaceDecl *, 1, bool> AnonOrFirstNamespaceAndInline;

NamespaceDecl(ASTContext &C, DeclContext *DC, bool Inline,
SourceLocation StartLoc, SourceLocation IdLoc,
IdentifierInfo *Id, NamespaceDecl *PrevDecl);

using redeclarable_base = Redeclarable<NamespaceDecl>;

NamespaceDecl *getNextRedeclarationImpl() override;
NamespaceDecl *getPreviousDeclImpl() override;
NamespaceDecl *getMostRecentDeclImpl() override;

public:
friend class ASTDeclReader;
friend class ASTDeclWriter;

static NamespaceDecl *Create(ASTContext &C, DeclContext *DC,
bool Inline, SourceLocation StartLoc,
SourceLocation IdLoc, IdentifierInfo *Id,
NamespaceDecl *PrevDecl);

static NamespaceDecl *CreateDeserialized(ASTContext &C, unsigned ID);

using redecl_range = redeclarable_base::redecl_range;
using redecl_iterator = redeclarable_base::redecl_iterator;

using redeclarable_base::redecls_begin;
using redeclarable_base::redecls_end;
using redeclarable_base::redecls;
using redeclarable_base::getPreviousDecl;
using redeclarable_base::getMostRecentDecl;
using redeclarable_base::isFirstDecl;

bool isAnonymousNamespace() const {
return !getIdentifier();
}

bool isInline() const {
return AnonOrFirstNamespaceAndInline.getInt();
}

void setInline(bool Inline) {
AnonOrFirstNamespaceAndInline.setInt(Inline);
}

NamespaceDecl *getOriginalNamespace();

const NamespaceDecl *getOriginalNamespace() const;

bool isOriginalNamespace() const;

NamespaceDecl *getAnonymousNamespace() const {
return getOriginalNamespace()->AnonOrFirstNamespaceAndInline.getPointer();
}

void setAnonymousNamespace(NamespaceDecl *D) {
getOriginalNamespace()->AnonOrFirstNamespaceAndInline.setPointer(D);
}

NamespaceDecl *getCanonicalDecl() override {
return getOriginalNamespace();
}
const NamespaceDecl *getCanonicalDecl() const {
return getOriginalNamespace();
}

SourceRange getSourceRange() const override LLVM_READONLY {
return SourceRange(LocStart, RBraceLoc);
}

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return LocStart; }
SourceLocation getRBraceLoc() const { return RBraceLoc; }
void setLocStart(SourceLocation L) { LocStart = L; }
void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == Namespace; }
static DeclContext *castToDeclContext(const NamespaceDecl *D) {
return static_cast<DeclContext *>(const_cast<NamespaceDecl*>(D));
}
static NamespaceDecl *castFromDeclContext(const DeclContext *DC) {
return static_cast<NamespaceDecl *>(const_cast<DeclContext*>(DC));
}
};

class ValueDecl : public NamedDecl {
QualType DeclType;

void anchor() override;

protected:
ValueDecl(Kind DK, DeclContext *DC, SourceLocation L,
DeclarationName N, QualType T)
: NamedDecl(DK, DC, L, N), DeclType(T) {}

public:
QualType getType() const { return DeclType; }
void setType(QualType newType) { DeclType = newType; }

bool isWeak() const;

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K >= firstValue && K <= lastValue; }
};

struct QualifierInfo {
NestedNameSpecifierLoc QualifierLoc;

unsigned NumTemplParamLists = 0;

TemplateParameterList** TemplParamLists = nullptr;

QualifierInfo() = default;
QualifierInfo(const QualifierInfo &) = delete;
QualifierInfo& operator=(const QualifierInfo &) = delete;

void setTemplateParameterListsInfo(ASTContext &Context,
ArrayRef<TemplateParameterList *> TPLists);
};

class DeclaratorDecl : public ValueDecl {
struct ExtInfo : public QualifierInfo {
TypeSourceInfo *TInfo;
};

llvm::PointerUnion<TypeSourceInfo *, ExtInfo *> DeclInfo;

SourceLocation InnerLocStart;

bool hasExtInfo() const { return DeclInfo.is<ExtInfo*>(); }
ExtInfo *getExtInfo() { return DeclInfo.get<ExtInfo*>(); }
const ExtInfo *getExtInfo() const { return DeclInfo.get<ExtInfo*>(); }

protected:
DeclaratorDecl(Kind DK, DeclContext *DC, SourceLocation L,
DeclarationName N, QualType T, TypeSourceInfo *TInfo,
SourceLocation StartL)
: ValueDecl(DK, DC, L, N, T), DeclInfo(TInfo), InnerLocStart(StartL) {}

public:
friend class ASTDeclReader;
friend class ASTDeclWriter;

TypeSourceInfo *getTypeSourceInfo() const {
return hasExtInfo()
? getExtInfo()->TInfo
: DeclInfo.get<TypeSourceInfo*>();
}

void setTypeSourceInfo(TypeSourceInfo *TI) {
if (hasExtInfo())
getExtInfo()->TInfo = TI;
else
DeclInfo = TI;
}

SourceLocation getInnerLocStart() const { return InnerLocStart; }
void setInnerLocStart(SourceLocation L) { InnerLocStart = L; }

SourceLocation getOuterLocStart() const;

SourceRange getSourceRange() const override LLVM_READONLY;

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY {
return getOuterLocStart();
}

NestedNameSpecifier *getQualifier() const {
return hasExtInfo() ? getExtInfo()->QualifierLoc.getNestedNameSpecifier()
: nullptr;
}

NestedNameSpecifierLoc getQualifierLoc() const {
return hasExtInfo() ? getExtInfo()->QualifierLoc
: NestedNameSpecifierLoc();
}

void setQualifierInfo(NestedNameSpecifierLoc QualifierLoc);

unsigned getNumTemplateParameterLists() const {
return hasExtInfo() ? getExtInfo()->NumTemplParamLists : 0;
}

TemplateParameterList *getTemplateParameterList(unsigned index) const {
assert(index < getNumTemplateParameterLists());
return getExtInfo()->TemplParamLists[index];
}

void setTemplateParameterListsInfo(ASTContext &Context,
ArrayRef<TemplateParameterList *> TPLists);

SourceLocation getTypeSpecStartLoc() const;

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) {
return K >= firstDeclarator && K <= lastDeclarator;
}
};

struct EvaluatedStmt {
bool WasEvaluated : 1;

bool IsEvaluating : 1;

bool CheckedICE : 1;

bool CheckingICE : 1;

bool IsICE : 1;

Stmt *Value;
APValue Evaluated;

EvaluatedStmt() : WasEvaluated(false), IsEvaluating(false), CheckedICE(false),
CheckingICE(false), IsICE(false) {}

};

class VarDecl : public DeclaratorDecl, public Redeclarable<VarDecl> {
public:
enum InitializationStyle {
CInit,

CallInit,

ListInit
};

enum TLSKind {
TLS_None,

TLS_Static,

TLS_Dynamic
};

static const char *getStorageClassSpecifierString(StorageClass SC);

protected:
using InitType = llvm::PointerUnion<Stmt *, EvaluatedStmt *>;

mutable InitType Init;

private:
friend class ASTDeclReader;
friend class ASTNodeImporter;
friend class StmtIteratorBase;

class VarDeclBitfields {
friend class ASTDeclReader;
friend class VarDecl;

unsigned SClass : 3;
unsigned TSCSpec : 2;
unsigned InitStyle : 2;
};
enum { NumVarDeclBits = 7 };

protected:
enum { NumParameterIndexBits = 8 };

enum DefaultArgKind {
DAK_None,
DAK_Unparsed,
DAK_Uninstantiated,
DAK_Normal
};

class ParmVarDeclBitfields {
friend class ASTDeclReader;
friend class ParmVarDecl;

unsigned : NumVarDeclBits;

unsigned HasInheritedDefaultArg : 1;

unsigned DefaultArgKind : 2;

unsigned IsKNRPromoted : 1;

unsigned IsObjCMethodParam : 1;

unsigned ScopeDepthOrObjCQuals : 7;

unsigned ParameterIndex : NumParameterIndexBits;
};

class NonParmVarDeclBitfields {
friend class ASTDeclReader;
friend class ImplicitParamDecl;
friend class VarDecl;

unsigned : NumVarDeclBits;

unsigned IsThisDeclarationADemotedDefinition : 1;

unsigned ExceptionVar : 1;

unsigned NRVOVariable : 1;

unsigned CXXForRangeDecl : 1;

unsigned ObjCForDecl : 1;

unsigned ARCPseudoStrong : 1;

unsigned IsInline : 1;

unsigned IsInlineSpecified : 1;

unsigned IsConstexpr : 1;

unsigned IsInitCapture : 1;

unsigned PreviousDeclInSameBlockScope : 1;

unsigned ImplicitParamKind : 3;
};

union {
unsigned AllBits;
VarDeclBitfields VarDeclBits;
ParmVarDeclBitfields ParmVarDeclBits;
NonParmVarDeclBitfields NonParmVarDeclBits;
};

VarDecl(Kind DK, ASTContext &C, DeclContext *DC, SourceLocation StartLoc,
SourceLocation IdLoc, IdentifierInfo *Id, QualType T,
TypeSourceInfo *TInfo, StorageClass SC);

using redeclarable_base = Redeclarable<VarDecl>;

VarDecl *getNextRedeclarationImpl() override {
return getNextRedeclaration();
}

VarDecl *getPreviousDeclImpl() override {
return getPreviousDecl();
}

VarDecl *getMostRecentDeclImpl() override {
return getMostRecentDecl();
}

public:
using redecl_range = redeclarable_base::redecl_range;
using redecl_iterator = redeclarable_base::redecl_iterator;

using redeclarable_base::redecls_begin;
using redeclarable_base::redecls_end;
using redeclarable_base::redecls;
using redeclarable_base::getPreviousDecl;
using redeclarable_base::getMostRecentDecl;
using redeclarable_base::isFirstDecl;

static VarDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation StartLoc, SourceLocation IdLoc,
IdentifierInfo *Id, QualType T, TypeSourceInfo *TInfo,
StorageClass S);

static VarDecl *CreateDeserialized(ASTContext &C, unsigned ID);

SourceRange getSourceRange() const override LLVM_READONLY;

StorageClass getStorageClass() const {
return (StorageClass) VarDeclBits.SClass;
}
void setStorageClass(StorageClass SC);

void setTSCSpec(ThreadStorageClassSpecifier TSC) {
VarDeclBits.TSCSpec = TSC;
assert(VarDeclBits.TSCSpec == TSC && "truncation");
}
ThreadStorageClassSpecifier getTSCSpec() const {
return static_cast<ThreadStorageClassSpecifier>(VarDeclBits.TSCSpec);
}
TLSKind getTLSKind() const;

bool hasLocalStorage() const {
if (getStorageClass() == SC_None) {
if (getType().getAddressSpace() == LangAS::opencl_constant)
return false;
return !isFileVarDecl() && getTSCSpec() == TSCS_unspecified;
}

if (getStorageClass() == SC_Register && !isLocalVarDeclOrParm())
return false;


return getStorageClass() >= SC_Auto;
}

bool isStaticLocal() const {
return (getStorageClass() == SC_Static ||
(getStorageClass() == SC_None && getTSCSpec() == TSCS_thread_local))
&& !isFileVarDecl();
}

bool hasExternalStorage() const {
return getStorageClass() == SC_Extern ||
getStorageClass() == SC_PrivateExtern;
}

bool hasGlobalStorage() const { return !hasLocalStorage(); }

StorageDuration getStorageDuration() const {
return hasLocalStorage() ? SD_Automatic :
getTSCSpec() ? SD_Thread : SD_Static;
}

LanguageLinkage getLanguageLinkage() const;

bool isExternC() const;

bool isInExternCContext() const;

bool isInExternCXXContext() const;

bool isLocalVarDecl() const {
if (getKind() != Decl::Var && getKind() != Decl::Decomposition)
return false;
if (const DeclContext *DC = getLexicalDeclContext())
return DC->getRedeclContext()->isFunctionOrMethod();
return false;
}

bool isLocalVarDeclOrParm() const {
return isLocalVarDecl() || getKind() == Decl::ParmVar;
}

bool isFunctionOrMethodVarDecl() const {
if (getKind() != Decl::Var && getKind() != Decl::Decomposition)
return false;
const DeclContext *DC = getLexicalDeclContext()->getRedeclContext();
return DC->isFunctionOrMethod() && DC->getDeclKind() != Decl::Block;
}

bool isStaticDataMember() const {
return getKind() != Decl::ParmVar && getDeclContext()->isRecord();
}

VarDecl *getCanonicalDecl() override;
const VarDecl *getCanonicalDecl() const {
return const_cast<VarDecl*>(this)->getCanonicalDecl();
}

enum DefinitionKind {
DeclarationOnly,

TentativeDefinition,

Definition
};

DefinitionKind isThisDeclarationADefinition(ASTContext &) const;
DefinitionKind isThisDeclarationADefinition() const {
return isThisDeclarationADefinition(getASTContext());
}

DefinitionKind hasDefinition(ASTContext &) const;
DefinitionKind hasDefinition() const {
return hasDefinition(getASTContext());
}

VarDecl *getActingDefinition();
const VarDecl *getActingDefinition() const {
return const_cast<VarDecl*>(this)->getActingDefinition();
}

VarDecl *getDefinition(ASTContext &);
const VarDecl *getDefinition(ASTContext &C) const {
return const_cast<VarDecl*>(this)->getDefinition(C);
}
VarDecl *getDefinition() {
return getDefinition(getASTContext());
}
const VarDecl *getDefinition() const {
return const_cast<VarDecl*>(this)->getDefinition();
}

bool isOutOfLine() const override;

bool isFileVarDecl() const {
Kind K = getKind();
if (K == ParmVar || K == ImplicitParam)
return false;

if (getLexicalDeclContext()->getRedeclContext()->isFileContext())
return true;

if (isStaticDataMember())
return true;

return false;
}

const Expr *getAnyInitializer() const {
const VarDecl *D;
return getAnyInitializer(D);
}

const Expr *getAnyInitializer(const VarDecl *&D) const;

bool hasInit() const;
const Expr *getInit() const {
return const_cast<VarDecl *>(this)->getInit();
}
Expr *getInit();

Stmt **getInitAddress();

void setInit(Expr *I);

bool isUsableInConstantExpressions(ASTContext &C) const;

EvaluatedStmt *ensureEvaluatedStmt() const;

APValue *evaluateValue() const;
APValue *evaluateValue(SmallVectorImpl<PartialDiagnosticAt> &Notes) const;

APValue *getEvaluatedValue() const;

bool isInitKnownICE() const;

bool isInitICE() const;

bool checkInitIsICE() const;

void setInitStyle(InitializationStyle Style) {
VarDeclBits.InitStyle = Style;
}

InitializationStyle getInitStyle() const {
return static_cast<InitializationStyle>(VarDeclBits.InitStyle);
}

bool isDirectInit() const {
return getInitStyle() != CInit;
}

bool isThisDeclarationADemotedDefinition() const {
return isa<ParmVarDecl>(this) ? false :
NonParmVarDeclBits.IsThisDeclarationADemotedDefinition;
}

void demoteThisDefinitionToDeclaration() {
assert(isThisDeclarationADefinition() && "Not a definition!");
assert(!isa<ParmVarDecl>(this) && "Cannot demote ParmVarDecls!");
NonParmVarDeclBits.IsThisDeclarationADemotedDefinition = 1;
}

bool isExceptionVariable() const {
return isa<ParmVarDecl>(this) ? false : NonParmVarDeclBits.ExceptionVar;
}
void setExceptionVariable(bool EV) {
assert(!isa<ParmVarDecl>(this));
NonParmVarDeclBits.ExceptionVar = EV;
}

bool isNRVOVariable() const {
return isa<ParmVarDecl>(this) ? false : NonParmVarDeclBits.NRVOVariable;
}
void setNRVOVariable(bool NRVO) {
assert(!isa<ParmVarDecl>(this));
NonParmVarDeclBits.NRVOVariable = NRVO;
}

bool isCXXForRangeDecl() const {
return isa<ParmVarDecl>(this) ? false : NonParmVarDeclBits.CXXForRangeDecl;
}
void setCXXForRangeDecl(bool FRD) {
assert(!isa<ParmVarDecl>(this));
NonParmVarDeclBits.CXXForRangeDecl = FRD;
}

bool isObjCForDecl() const {
return NonParmVarDeclBits.ObjCForDecl;
}

void setObjCForDecl(bool FRD) {
NonParmVarDeclBits.ObjCForDecl = FRD;
}

bool isARCPseudoStrong() const {
return isa<ParmVarDecl>(this) ? false : NonParmVarDeclBits.ARCPseudoStrong;
}
void setARCPseudoStrong(bool ps) {
assert(!isa<ParmVarDecl>(this));
NonParmVarDeclBits.ARCPseudoStrong = ps;
}

bool isInline() const {
return isa<ParmVarDecl>(this) ? false : NonParmVarDeclBits.IsInline;
}
bool isInlineSpecified() const {
return isa<ParmVarDecl>(this) ? false
: NonParmVarDeclBits.IsInlineSpecified;
}
void setInlineSpecified() {
assert(!isa<ParmVarDecl>(this));
NonParmVarDeclBits.IsInline = true;
NonParmVarDeclBits.IsInlineSpecified = true;
}
void setImplicitlyInline() {
assert(!isa<ParmVarDecl>(this));
NonParmVarDeclBits.IsInline = true;
}

bool isConstexpr() const {
return isa<ParmVarDecl>(this) ? false : NonParmVarDeclBits.IsConstexpr;
}
void setConstexpr(bool IC) {
assert(!isa<ParmVarDecl>(this));
NonParmVarDeclBits.IsConstexpr = IC;
}

bool isInitCapture() const {
return isa<ParmVarDecl>(this) ? false : NonParmVarDeclBits.IsInitCapture;
}
void setInitCapture(bool IC) {
assert(!isa<ParmVarDecl>(this));
NonParmVarDeclBits.IsInitCapture = IC;
}

bool isPreviousDeclInSameBlockScope() const {
return isa<ParmVarDecl>(this)
? false
: NonParmVarDeclBits.PreviousDeclInSameBlockScope;
}
void setPreviousDeclInSameBlockScope(bool Same) {
assert(!isa<ParmVarDecl>(this));
NonParmVarDeclBits.PreviousDeclInSameBlockScope = Same;
}

VarDecl *getTemplateInstantiationPattern() const;

VarDecl *getInstantiatedFromStaticDataMember() const;

TemplateSpecializationKind getTemplateSpecializationKind() const;

SourceLocation getPointOfInstantiation() const;

MemberSpecializationInfo *getMemberSpecializationInfo() const;

void setTemplateSpecializationKind(TemplateSpecializationKind TSK,
SourceLocation PointOfInstantiation = SourceLocation());

void setInstantiationOfStaticDataMember(VarDecl *VD,
TemplateSpecializationKind TSK);

VarTemplateDecl *getDescribedVarTemplate() const;

void setDescribedVarTemplate(VarTemplateDecl *Template);

bool isKnownToBeDefined() const;

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K >= firstVar && K <= lastVar; }
};

class ImplicitParamDecl : public VarDecl {
void anchor() override;

public:
enum ImplicitParamKind : unsigned {
ObjCSelf,

ObjCCmd,

CXXThis,

CXXVTT,

CapturedContext,

Other,
};

static ImplicitParamDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation IdLoc, IdentifierInfo *Id,
QualType T, ImplicitParamKind ParamKind);
static ImplicitParamDecl *Create(ASTContext &C, QualType T,
ImplicitParamKind ParamKind);

static ImplicitParamDecl *CreateDeserialized(ASTContext &C, unsigned ID);

ImplicitParamDecl(ASTContext &C, DeclContext *DC, SourceLocation IdLoc,
IdentifierInfo *Id, QualType Type,
ImplicitParamKind ParamKind)
: VarDecl(ImplicitParam, C, DC, IdLoc, IdLoc, Id, Type,
nullptr, SC_None) {
NonParmVarDeclBits.ImplicitParamKind = ParamKind;
setImplicit();
}

ImplicitParamDecl(ASTContext &C, QualType Type, ImplicitParamKind ParamKind)
: VarDecl(ImplicitParam, C, nullptr, SourceLocation(),
SourceLocation(), nullptr, Type,
nullptr, SC_None) {
NonParmVarDeclBits.ImplicitParamKind = ParamKind;
setImplicit();
}

ImplicitParamKind getParameterKind() const {
return static_cast<ImplicitParamKind>(NonParmVarDeclBits.ImplicitParamKind);
}

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == ImplicitParam; }
};

class ParmVarDecl : public VarDecl {
public:
enum { MaxFunctionScopeDepth = 255 };
enum { MaxFunctionScopeIndex = 255 };

protected:
ParmVarDecl(Kind DK, ASTContext &C, DeclContext *DC, SourceLocation StartLoc,
SourceLocation IdLoc, IdentifierInfo *Id, QualType T,
TypeSourceInfo *TInfo, StorageClass S, Expr *DefArg)
: VarDecl(DK, C, DC, StartLoc, IdLoc, Id, T, TInfo, S) {
assert(ParmVarDeclBits.HasInheritedDefaultArg == false);
assert(ParmVarDeclBits.DefaultArgKind == DAK_None);
assert(ParmVarDeclBits.IsKNRPromoted == false);
assert(ParmVarDeclBits.IsObjCMethodParam == false);
setDefaultArg(DefArg);
}

public:
static ParmVarDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation StartLoc,
SourceLocation IdLoc, IdentifierInfo *Id,
QualType T, TypeSourceInfo *TInfo,
StorageClass S, Expr *DefArg);

static ParmVarDecl *CreateDeserialized(ASTContext &C, unsigned ID);

SourceRange getSourceRange() const override LLVM_READONLY;

void setObjCMethodScopeInfo(unsigned parameterIndex) {
ParmVarDeclBits.IsObjCMethodParam = true;
setParameterIndex(parameterIndex);
}

void setScopeInfo(unsigned scopeDepth, unsigned parameterIndex) {
assert(!ParmVarDeclBits.IsObjCMethodParam);

ParmVarDeclBits.ScopeDepthOrObjCQuals = scopeDepth;
assert(ParmVarDeclBits.ScopeDepthOrObjCQuals == scopeDepth
&& "truncation!");

setParameterIndex(parameterIndex);
}

bool isObjCMethodParameter() const {
return ParmVarDeclBits.IsObjCMethodParam;
}

unsigned getFunctionScopeDepth() const {
if (ParmVarDeclBits.IsObjCMethodParam) return 0;
return ParmVarDeclBits.ScopeDepthOrObjCQuals;
}

unsigned getFunctionScopeIndex() const {
return getParameterIndex();
}

ObjCDeclQualifier getObjCDeclQualifier() const {
if (!ParmVarDeclBits.IsObjCMethodParam) return OBJC_TQ_None;
return ObjCDeclQualifier(ParmVarDeclBits.ScopeDepthOrObjCQuals);
}
void setObjCDeclQualifier(ObjCDeclQualifier QTVal) {
assert(ParmVarDeclBits.IsObjCMethodParam);
ParmVarDeclBits.ScopeDepthOrObjCQuals = QTVal;
}

bool isKNRPromoted() const {
return ParmVarDeclBits.IsKNRPromoted;
}
void setKNRPromoted(bool promoted) {
ParmVarDeclBits.IsKNRPromoted = promoted;
}

Expr *getDefaultArg();
const Expr *getDefaultArg() const {
return const_cast<ParmVarDecl *>(this)->getDefaultArg();
}

void setDefaultArg(Expr *defarg);

SourceRange getDefaultArgRange() const;
void setUninstantiatedDefaultArg(Expr *arg);
Expr *getUninstantiatedDefaultArg();
const Expr *getUninstantiatedDefaultArg() const {
return const_cast<ParmVarDecl *>(this)->getUninstantiatedDefaultArg();
}

bool hasDefaultArg() const;

bool hasUnparsedDefaultArg() const {
return ParmVarDeclBits.DefaultArgKind == DAK_Unparsed;
}

bool hasUninstantiatedDefaultArg() const {
return ParmVarDeclBits.DefaultArgKind == DAK_Uninstantiated;
}

void setUnparsedDefaultArg() {
ParmVarDeclBits.DefaultArgKind = DAK_Unparsed;
}

bool hasInheritedDefaultArg() const {
return ParmVarDeclBits.HasInheritedDefaultArg;
}

void setHasInheritedDefaultArg(bool I = true) {
ParmVarDeclBits.HasInheritedDefaultArg = I;
}

QualType getOriginalType() const;

bool isParameterPack() const;

void setOwningFunction(DeclContext *FD) { setDeclContext(FD); }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == ParmVar; }

private:
enum { ParameterIndexSentinel = (1 << NumParameterIndexBits) - 1 };

void setParameterIndex(unsigned parameterIndex) {
if (parameterIndex >= ParameterIndexSentinel) {
setParameterIndexLarge(parameterIndex);
return;
}

ParmVarDeclBits.ParameterIndex = parameterIndex;
assert(ParmVarDeclBits.ParameterIndex == parameterIndex && "truncation!");
}
unsigned getParameterIndex() const {
unsigned d = ParmVarDeclBits.ParameterIndex;
return d == ParameterIndexSentinel ? getParameterIndexLarge() : d;
}

void setParameterIndexLarge(unsigned parameterIndex);
unsigned getParameterIndexLarge() const;
};

class FunctionDecl : public DeclaratorDecl, public DeclContext,
public Redeclarable<FunctionDecl> {
public:
enum TemplatedKind {
TK_NonTemplate,
TK_FunctionTemplate,
TK_MemberSpecialization,
TK_FunctionTemplateSpecialization,
TK_DependentFunctionTemplateSpecialization
};

private:
ParmVarDecl **ParamInfo = nullptr;

LazyDeclStmtPtr Body;

unsigned SClass : 3;
unsigned IsInline : 1;
unsigned IsInlineSpecified : 1;

protected:
unsigned IsExplicitSpecified : 1;

private:
unsigned IsVirtualAsWritten : 1;
unsigned IsPure : 1;
unsigned HasInheritedPrototype : 1;
unsigned HasWrittenPrototype : 1;
unsigned IsDeleted : 1;
unsigned IsTrivial : 1; 

unsigned IsTrivialForCall : 1;

unsigned IsDefaulted : 1; 
unsigned IsExplicitlyDefaulted : 1; 
unsigned HasImplicitReturnZero : 1;
unsigned IsLateTemplateParsed : 1;
unsigned IsConstexpr : 1;
unsigned InstantiationIsPending : 1;

unsigned UsesSEHTry : 1;

unsigned HasSkippedBody : 1;

unsigned WillHaveBody : 1;

unsigned IsMultiVersion : 1;

protected:
unsigned IsCopyDeductionCandidate : 1;

private:

unsigned HasODRHash : 1;
unsigned ODRHash;

SourceLocation EndRangeLoc;

llvm::PointerUnion4<FunctionTemplateDecl *,
MemberSpecializationInfo *,
FunctionTemplateSpecializationInfo *,
DependentFunctionTemplateSpecializationInfo *>
TemplateOrSpecialization;

DeclarationNameLoc DNLoc;

void setFunctionTemplateSpecialization(ASTContext &C,
FunctionTemplateDecl *Template,
const TemplateArgumentList *TemplateArgs,
void *InsertPos,
TemplateSpecializationKind TSK,
const TemplateArgumentListInfo *TemplateArgsAsWritten,
SourceLocation PointOfInstantiation);

void setInstantiationOfMemberFunction(ASTContext &C, FunctionDecl *FD,
TemplateSpecializationKind TSK);

void setParams(ASTContext &C, ArrayRef<ParmVarDecl *> NewParamInfo);

protected:
FunctionDecl(Kind DK, ASTContext &C, DeclContext *DC, SourceLocation StartLoc,
const DeclarationNameInfo &NameInfo, QualType T,
TypeSourceInfo *TInfo, StorageClass S, bool isInlineSpecified,
bool isConstexprSpecified)
: DeclaratorDecl(DK, DC, NameInfo.getLoc(), NameInfo.getName(), T, TInfo,
StartLoc),
DeclContext(DK), redeclarable_base(C), SClass(S),
IsInline(isInlineSpecified), IsInlineSpecified(isInlineSpecified),
IsExplicitSpecified(false), IsVirtualAsWritten(false), IsPure(false),
HasInheritedPrototype(false), HasWrittenPrototype(true),
IsDeleted(false), IsTrivial(false), IsTrivialForCall(false),
IsDefaulted(false),
IsExplicitlyDefaulted(false), HasImplicitReturnZero(false),
IsLateTemplateParsed(false), IsConstexpr(isConstexprSpecified),
InstantiationIsPending(false), UsesSEHTry(false), HasSkippedBody(false),
WillHaveBody(false), IsMultiVersion(false),
IsCopyDeductionCandidate(false), HasODRHash(false), ODRHash(0),
EndRangeLoc(NameInfo.getEndLoc()), DNLoc(NameInfo.getInfo()) {}

using redeclarable_base = Redeclarable<FunctionDecl>;

FunctionDecl *getNextRedeclarationImpl() override {
return getNextRedeclaration();
}

FunctionDecl *getPreviousDeclImpl() override {
return getPreviousDecl();
}

FunctionDecl *getMostRecentDeclImpl() override {
return getMostRecentDecl();
}

public:
friend class ASTDeclReader;
friend class ASTDeclWriter;

using redecl_range = redeclarable_base::redecl_range;
using redecl_iterator = redeclarable_base::redecl_iterator;

using redeclarable_base::redecls_begin;
using redeclarable_base::redecls_end;
using redeclarable_base::redecls;
using redeclarable_base::getPreviousDecl;
using redeclarable_base::getMostRecentDecl;
using redeclarable_base::isFirstDecl;

static FunctionDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation StartLoc, SourceLocation NLoc,
DeclarationName N, QualType T,
TypeSourceInfo *TInfo,
StorageClass SC,
bool isInlineSpecified = false,
bool hasWrittenPrototype = true,
bool isConstexprSpecified = false) {
DeclarationNameInfo NameInfo(N, NLoc);
return FunctionDecl::Create(C, DC, StartLoc, NameInfo, T, TInfo,
SC,
isInlineSpecified, hasWrittenPrototype,
isConstexprSpecified);
}

static FunctionDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation StartLoc,
const DeclarationNameInfo &NameInfo,
QualType T, TypeSourceInfo *TInfo,
StorageClass SC,
bool isInlineSpecified,
bool hasWrittenPrototype,
bool isConstexprSpecified = false);

static FunctionDecl *CreateDeserialized(ASTContext &C, unsigned ID);

DeclarationNameInfo getNameInfo() const {
return DeclarationNameInfo(getDeclName(), getLocation(), DNLoc);
}

void getNameForDiagnostic(raw_ostream &OS, const PrintingPolicy &Policy,
bool Qualified) const override;

void setRangeEnd(SourceLocation E) { EndRangeLoc = E; }

SourceRange getSourceRange() const override LLVM_READONLY;


bool hasBody(const FunctionDecl *&Definition) const;

bool hasBody() const override {
const FunctionDecl* Definition;
return hasBody(Definition);
}

bool hasTrivialBody() const;

bool isDefined(const FunctionDecl *&Definition) const;

virtual bool isDefined() const {
const FunctionDecl* Definition;
return isDefined(Definition);
}

FunctionDecl *getDefinition() {
const FunctionDecl *Definition;
if (isDefined(Definition))
return const_cast<FunctionDecl *>(Definition);
return nullptr;
}
const FunctionDecl *getDefinition() const {
return const_cast<FunctionDecl *>(this)->getDefinition();
}

Stmt *getBody(const FunctionDecl *&Definition) const;

Stmt *getBody() const override {
const FunctionDecl* Definition;
return getBody(Definition);
}

bool isThisDeclarationADefinition() const {
return IsDeleted || IsDefaulted || Body || HasSkippedBody ||
IsLateTemplateParsed || WillHaveBody || hasDefiningAttr();
}

bool doesThisDeclarationHaveABody() const {
return Body || IsLateTemplateParsed;
}

void setBody(Stmt *B);
void setLazyBody(uint64_t Offset) { Body = Offset; }

bool isVariadic() const;

bool isVirtualAsWritten() const { return IsVirtualAsWritten; }
void setVirtualAsWritten(bool V) { IsVirtualAsWritten = V; }

bool isPure() const { return IsPure; }
void setPure(bool P = true);

bool isLateTemplateParsed() const { return IsLateTemplateParsed; }
void setLateTemplateParsed(bool ILT = true) { IsLateTemplateParsed = ILT; }

bool isTrivial() const { return IsTrivial; }
void setTrivial(bool IT) { IsTrivial = IT; }

bool isTrivialForCall() const { return IsTrivialForCall; }
void setTrivialForCall(bool IT) { IsTrivialForCall = IT; }

bool isDefaulted() const { return IsDefaulted; }
void setDefaulted(bool D = true) { IsDefaulted = D; }

bool isExplicitlyDefaulted() const { return IsExplicitlyDefaulted; }
void setExplicitlyDefaulted(bool ED = true) { IsExplicitlyDefaulted = ED; }

bool hasImplicitReturnZero() const { return HasImplicitReturnZero; }
void setHasImplicitReturnZero(bool IRZ) { HasImplicitReturnZero = IRZ; }

bool hasPrototype() const {
return HasWrittenPrototype || HasInheritedPrototype;
}

bool hasWrittenPrototype() const { return HasWrittenPrototype; }

bool hasInheritedPrototype() const { return HasInheritedPrototype; }
void setHasInheritedPrototype(bool P = true) { HasInheritedPrototype = P; }

bool isConstexpr() const { return IsConstexpr; }
void setConstexpr(bool IC) { IsConstexpr = IC; }

bool instantiationIsPending() const { return InstantiationIsPending; }
void setInstantiationIsPending(bool IC) { InstantiationIsPending = IC; }

bool usesSEHTry() const { return UsesSEHTry; }
void setUsesSEHTry(bool UST) { UsesSEHTry = UST; }

bool isDeleted() const { return getCanonicalDecl()->IsDeleted; }
bool isDeletedAsWritten() const { return IsDeleted && !IsDefaulted; }
void setDeletedAsWritten(bool D = true) { IsDeleted = D; }

bool isMain() const;

bool isMSVCRTEntryPoint() const;

bool isReservedGlobalPlacementOperator() const;

bool isReplaceableGlobalAllocationFunction(bool *IsAligned = nullptr) const;

bool isDestroyingOperatorDelete() const;

LanguageLinkage getLanguageLinkage() const;

bool isExternC() const;

bool isInExternCContext() const;

bool isInExternCXXContext() const;

bool isGlobal() const;

bool isNoReturn() const;

bool hasSkippedBody() const { return HasSkippedBody; }
void setHasSkippedBody(bool Skipped = true) { HasSkippedBody = Skipped; }

bool willHaveBody() const { return WillHaveBody; }
void setWillHaveBody(bool V = true) { WillHaveBody = V; }

bool isMultiVersion() const { return getCanonicalDecl()->IsMultiVersion; }

void setIsMultiVersion(bool V = true) {
getCanonicalDecl()->IsMultiVersion = V;
}

bool isCPUDispatchMultiVersion() const;
bool isCPUSpecificMultiVersion() const;

void setPreviousDeclaration(FunctionDecl * PrevDecl);

FunctionDecl *getCanonicalDecl() override;
const FunctionDecl *getCanonicalDecl() const {
return const_cast<FunctionDecl*>(this)->getCanonicalDecl();
}

unsigned getBuiltinID() const;

ArrayRef<ParmVarDecl *> parameters() const {
return {ParamInfo, getNumParams()};
}
MutableArrayRef<ParmVarDecl *> parameters() {
return {ParamInfo, getNumParams()};
}

using param_iterator = MutableArrayRef<ParmVarDecl *>::iterator;
using param_const_iterator = ArrayRef<ParmVarDecl *>::const_iterator;

bool param_empty() const { return parameters().empty(); }
param_iterator param_begin() { return parameters().begin(); }
param_iterator param_end() { return parameters().end(); }
param_const_iterator param_begin() const { return parameters().begin(); }
param_const_iterator param_end() const { return parameters().end(); }
size_t param_size() const { return parameters().size(); }

unsigned getNumParams() const;

const ParmVarDecl *getParamDecl(unsigned i) const {
assert(i < getNumParams() && "Illegal param #");
return ParamInfo[i];
}
ParmVarDecl *getParamDecl(unsigned i) {
assert(i < getNumParams() && "Illegal param #");
return ParamInfo[i];
}
void setParams(ArrayRef<ParmVarDecl *> NewParamInfo) {
setParams(getASTContext(), NewParamInfo);
}

unsigned getMinRequiredArguments() const;

QualType getReturnType() const {
return getType()->castAs<FunctionType>()->getReturnType();
}

SourceRange getReturnTypeSourceRange() const;

QualType getDeclaredReturnType() const {
auto *TSI = getTypeSourceInfo();
QualType T = TSI ? TSI->getType() : getType();
return T->castAs<FunctionType>()->getReturnType();
}

SourceRange getExceptionSpecSourceRange() const;

QualType getCallResultType() const {
return getType()->castAs<FunctionType>()->getCallResultType(
getASTContext());
}

const Attr *getUnusedResultAttr() const;

bool hasUnusedResultAttr() const { return getUnusedResultAttr() != nullptr; }

StorageClass getStorageClass() const { return StorageClass(SClass); }

bool isInlineSpecified() const { return IsInlineSpecified; }

void setInlineSpecified(bool I) {
IsInlineSpecified = I;
IsInline = I;
}

void setImplicitlyInline() {
IsInline = true;
}

bool isInlined() const { return IsInline; }

bool isInlineDefinitionExternallyVisible() const;

bool isMSExternInline() const;

bool doesDeclarationForceExternallyVisibleDefinition() const;

bool isOverloadedOperator() const {
return getOverloadedOperator() != OO_None;
}

OverloadedOperatorKind getOverloadedOperator() const;

const IdentifierInfo *getLiteralIdentifier() const;

FunctionDecl *getInstantiatedFromMemberFunction() const;

TemplatedKind getTemplatedKind() const;

MemberSpecializationInfo *getMemberSpecializationInfo() const;

void setInstantiationOfMemberFunction(FunctionDecl *FD,
TemplateSpecializationKind TSK) {
setInstantiationOfMemberFunction(getASTContext(), FD, TSK);
}

FunctionTemplateDecl *getDescribedFunctionTemplate() const;

void setDescribedFunctionTemplate(FunctionTemplateDecl *Template);

bool isFunctionTemplateSpecialization() const {
return getPrimaryTemplate() != nullptr;
}

FunctionDecl *getClassScopeSpecializationPattern() const;

FunctionTemplateSpecializationInfo *getTemplateSpecializationInfo() const;

bool isImplicitlyInstantiable() const;

bool isTemplateInstantiation() const;

FunctionDecl *getTemplateInstantiationPattern() const;

FunctionTemplateDecl *getPrimaryTemplate() const;

const TemplateArgumentList *getTemplateSpecializationArgs() const;

const ASTTemplateArgumentListInfo*
getTemplateSpecializationArgsAsWritten() const;

void setFunctionTemplateSpecialization(FunctionTemplateDecl *Template,
const TemplateArgumentList *TemplateArgs,
void *InsertPos,
TemplateSpecializationKind TSK = TSK_ImplicitInstantiation,
const TemplateArgumentListInfo *TemplateArgsAsWritten = nullptr,
SourceLocation PointOfInstantiation = SourceLocation()) {
setFunctionTemplateSpecialization(getASTContext(), Template, TemplateArgs,
InsertPos, TSK, TemplateArgsAsWritten,
PointOfInstantiation);
}

void setDependentTemplateSpecialization(ASTContext &Context,
const UnresolvedSetImpl &Templates,
const TemplateArgumentListInfo &TemplateArgs);

DependentFunctionTemplateSpecializationInfo *
getDependentSpecializationInfo() const;

TemplateSpecializationKind getTemplateSpecializationKind() const;

void setTemplateSpecializationKind(TemplateSpecializationKind TSK,
SourceLocation PointOfInstantiation = SourceLocation());

SourceLocation getPointOfInstantiation() const;

bool isOutOfLine() const override;

unsigned getMemoryFunctionKind() const;

unsigned getODRHash();

unsigned getODRHash() const;

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) {
return K >= firstFunction && K <= lastFunction;
}
static DeclContext *castToDeclContext(const FunctionDecl *D) {
return static_cast<DeclContext *>(const_cast<FunctionDecl*>(D));
}
static FunctionDecl *castFromDeclContext(const DeclContext *DC) {
return static_cast<FunctionDecl *>(const_cast<DeclContext*>(DC));
}
};

class FieldDecl : public DeclaratorDecl, public Mergeable<FieldDecl> {
unsigned BitField : 1;
unsigned Mutable : 1;
mutable unsigned CachedFieldIndex : 30;

enum InitStorageKind {
ISK_NoInit = (unsigned) ICIS_NoInit,

ISK_InClassCopyInit = (unsigned) ICIS_CopyInit,

ISK_InClassListInit = (unsigned) ICIS_ListInit,

ISK_CapturedVLAType,
};

struct InitAndBitWidth {
Expr *Init;
Expr *BitWidth;
};

llvm::PointerIntPair<void *, 2, InitStorageKind> InitStorage;

protected:
FieldDecl(Kind DK, DeclContext *DC, SourceLocation StartLoc,
SourceLocation IdLoc, IdentifierInfo *Id,
QualType T, TypeSourceInfo *TInfo, Expr *BW, bool Mutable,
InClassInitStyle InitStyle)
: DeclaratorDecl(DK, DC, IdLoc, Id, T, TInfo, StartLoc),
BitField(false), Mutable(Mutable), CachedFieldIndex(0),
InitStorage(nullptr, (InitStorageKind) InitStyle) {
if (BW)
setBitWidth(BW);
}

public:
friend class ASTDeclReader;
friend class ASTDeclWriter;

static FieldDecl *Create(const ASTContext &C, DeclContext *DC,
SourceLocation StartLoc, SourceLocation IdLoc,
IdentifierInfo *Id, QualType T,
TypeSourceInfo *TInfo, Expr *BW, bool Mutable,
InClassInitStyle InitStyle);

static FieldDecl *CreateDeserialized(ASTContext &C, unsigned ID);

unsigned getFieldIndex() const;

bool isMutable() const { return Mutable; }

bool isBitField() const { return BitField; }

bool isUnnamedBitfield() const { return isBitField() && !getDeclName(); }

bool isAnonymousStructOrUnion() const;

Expr *getBitWidth() const {
if (!BitField)
return nullptr;
void *Ptr = InitStorage.getPointer();
if (getInClassInitStyle())
return static_cast<InitAndBitWidth*>(Ptr)->BitWidth;
return static_cast<Expr*>(Ptr);
}

unsigned getBitWidthValue(const ASTContext &Ctx) const;

void setBitWidth(Expr *Width) {
assert(!hasCapturedVLAType() && !BitField &&
"bit width or captured type already set");
assert(Width && "no bit width specified");
InitStorage.setPointer(
InitStorage.getInt()
? new (getASTContext())
InitAndBitWidth{getInClassInitializer(), Width}
: static_cast<void*>(Width));
BitField = true;
}

void removeBitWidth() {
assert(isBitField() && "no bitfield width to remove");
InitStorage.setPointer(getInClassInitializer());
BitField = false;
}

bool isZeroLengthBitField(const ASTContext &Ctx) const;

InClassInitStyle getInClassInitStyle() const {
InitStorageKind storageKind = InitStorage.getInt();
return (storageKind == ISK_CapturedVLAType
? ICIS_NoInit : (InClassInitStyle) storageKind);
}

bool hasInClassInitializer() const {
return getInClassInitStyle() != ICIS_NoInit;
}

Expr *getInClassInitializer() const {
if (!hasInClassInitializer())
return nullptr;
void *Ptr = InitStorage.getPointer();
if (BitField)
return static_cast<InitAndBitWidth*>(Ptr)->Init;
return static_cast<Expr*>(Ptr);
}

void setInClassInitializer(Expr *Init) {
assert(hasInClassInitializer() && !getInClassInitializer());
if (BitField)
static_cast<InitAndBitWidth*>(InitStorage.getPointer())->Init = Init;
else
InitStorage.setPointer(Init);
}

void removeInClassInitializer() {
assert(hasInClassInitializer() && "no initializer to remove");
InitStorage.setPointerAndInt(getBitWidth(), ISK_NoInit);
}

bool hasCapturedVLAType() const {
return InitStorage.getInt() == ISK_CapturedVLAType;
}

const VariableArrayType *getCapturedVLAType() const {
return hasCapturedVLAType() ? static_cast<const VariableArrayType *>(
InitStorage.getPointer())
: nullptr;
}

void setCapturedVLAType(const VariableArrayType *VLAType);

const RecordDecl *getParent() const {
return cast<RecordDecl>(getDeclContext());
}

RecordDecl *getParent() {
return cast<RecordDecl>(getDeclContext());
}

SourceRange getSourceRange() const override LLVM_READONLY;

FieldDecl *getCanonicalDecl() override { return getFirstDecl(); }
const FieldDecl *getCanonicalDecl() const { return getFirstDecl(); }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K >= firstField && K <= lastField; }
};

class EnumConstantDecl : public ValueDecl, public Mergeable<EnumConstantDecl> {
Stmt *Init; 
llvm::APSInt Val; 

protected:
EnumConstantDecl(DeclContext *DC, SourceLocation L,
IdentifierInfo *Id, QualType T, Expr *E,
const llvm::APSInt &V)
: ValueDecl(EnumConstant, DC, L, Id, T), Init((Stmt*)E), Val(V) {}

public:
friend class StmtIteratorBase;

static EnumConstantDecl *Create(ASTContext &C, EnumDecl *DC,
SourceLocation L, IdentifierInfo *Id,
QualType T, Expr *E,
const llvm::APSInt &V);
static EnumConstantDecl *CreateDeserialized(ASTContext &C, unsigned ID);

const Expr *getInitExpr() const { return (const Expr*) Init; }
Expr *getInitExpr() { return (Expr*) Init; }
const llvm::APSInt &getInitVal() const { return Val; }

void setInitExpr(Expr *E) { Init = (Stmt*) E; }
void setInitVal(const llvm::APSInt &V) { Val = V; }

SourceRange getSourceRange() const override LLVM_READONLY;

EnumConstantDecl *getCanonicalDecl() override { return getFirstDecl(); }
const EnumConstantDecl *getCanonicalDecl() const { return getFirstDecl(); }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == EnumConstant; }
};

class IndirectFieldDecl : public ValueDecl,
public Mergeable<IndirectFieldDecl> {
NamedDecl **Chaining;
unsigned ChainingSize;

IndirectFieldDecl(ASTContext &C, DeclContext *DC, SourceLocation L,
DeclarationName N, QualType T,
MutableArrayRef<NamedDecl *> CH);

void anchor() override;

public:
friend class ASTDeclReader;

static IndirectFieldDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation L, IdentifierInfo *Id,
QualType T, llvm::MutableArrayRef<NamedDecl *> CH);

static IndirectFieldDecl *CreateDeserialized(ASTContext &C, unsigned ID);

using chain_iterator = ArrayRef<NamedDecl *>::const_iterator;

ArrayRef<NamedDecl *> chain() const {
return llvm::makeArrayRef(Chaining, ChainingSize);
}
chain_iterator chain_begin() const { return chain().begin(); }
chain_iterator chain_end() const { return chain().end(); }

unsigned getChainingSize() const { return ChainingSize; }

FieldDecl *getAnonField() const {
assert(chain().size() >= 2);
return cast<FieldDecl>(chain().back());
}

VarDecl *getVarDecl() const {
assert(chain().size() >= 2);
return dyn_cast<VarDecl>(chain().front());
}

IndirectFieldDecl *getCanonicalDecl() override { return getFirstDecl(); }
const IndirectFieldDecl *getCanonicalDecl() const { return getFirstDecl(); }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == IndirectField; }
};

class TypeDecl : public NamedDecl {
friend class ASTContext;

mutable const Type *TypeForDecl = nullptr;

SourceLocation LocStart;

void anchor() override;

protected:
TypeDecl(Kind DK, DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
SourceLocation StartL = SourceLocation())
: NamedDecl(DK, DC, L, Id), LocStart(StartL) {}

public:
const Type *getTypeForDecl() const { return TypeForDecl; }
void setTypeForDecl(const Type *TD) { TypeForDecl = TD; }

SourceLocation getLocStart() const LLVM_READONLY { return getBeginLoc(); }
SourceLocation getBeginLoc() const LLVM_READONLY { return LocStart; }
void setLocStart(SourceLocation L) { LocStart = L; }
SourceRange getSourceRange() const override LLVM_READONLY {
if (LocStart.isValid())
return SourceRange(LocStart, getLocation());
else
return SourceRange(getLocation());
}

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K >= firstType && K <= lastType; }
};

class TypedefNameDecl : public TypeDecl, public Redeclarable<TypedefNameDecl> {
struct alignas(8) ModedTInfo {
TypeSourceInfo *first;
QualType second;
};

mutable llvm::PointerIntPair<
llvm::PointerUnion<TypeSourceInfo *, ModedTInfo *>, 2>
MaybeModedTInfo;

void anchor() override;

protected:
TypedefNameDecl(Kind DK, ASTContext &C, DeclContext *DC,
SourceLocation StartLoc, SourceLocation IdLoc,
IdentifierInfo *Id, TypeSourceInfo *TInfo)
: TypeDecl(DK, DC, IdLoc, Id, StartLoc), redeclarable_base(C),
MaybeModedTInfo(TInfo, 0) {}

using redeclarable_base = Redeclarable<TypedefNameDecl>;

TypedefNameDecl *getNextRedeclarationImpl() override {
return getNextRedeclaration();
}

TypedefNameDecl *getPreviousDeclImpl() override {
return getPreviousDecl();
}

TypedefNameDecl *getMostRecentDeclImpl() override {
return getMostRecentDecl();
}

public:
using redecl_range = redeclarable_base::redecl_range;
using redecl_iterator = redeclarable_base::redecl_iterator;

using redeclarable_base::redecls_begin;
using redeclarable_base::redecls_end;
using redeclarable_base::redecls;
using redeclarable_base::getPreviousDecl;
using redeclarable_base::getMostRecentDecl;
using redeclarable_base::isFirstDecl;

bool isModed() const {
return MaybeModedTInfo.getPointer().is<ModedTInfo *>();
}

TypeSourceInfo *getTypeSourceInfo() const {
return isModed() ? MaybeModedTInfo.getPointer().get<ModedTInfo *>()->first
: MaybeModedTInfo.getPointer().get<TypeSourceInfo *>();
}

QualType getUnderlyingType() const {
return isModed() ? MaybeModedTInfo.getPointer().get<ModedTInfo *>()->second
: MaybeModedTInfo.getPointer()
.get<TypeSourceInfo *>()
->getType();
}

void setTypeSourceInfo(TypeSourceInfo *newType) {
MaybeModedTInfo.setPointer(newType);
}

void setModedTypeSourceInfo(TypeSourceInfo *unmodedTSI, QualType modedTy) {
MaybeModedTInfo.setPointer(new (getASTContext(), 8)
ModedTInfo({unmodedTSI, modedTy}));
}

TypedefNameDecl *getCanonicalDecl() override { return getFirstDecl(); }
const TypedefNameDecl *getCanonicalDecl() const { return getFirstDecl(); }

TagDecl *getAnonDeclWithTypedefName(bool AnyRedecl = false) const;

bool isTransparentTag() const {
if (MaybeModedTInfo.getInt())
return MaybeModedTInfo.getInt() & 0x2;
return isTransparentTagSlow();
}

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) {
return K >= firstTypedefName && K <= lastTypedefName;
}

private:
bool isTransparentTagSlow() const;
};

class TypedefDecl : public TypedefNameDecl {
TypedefDecl(ASTContext &C, DeclContext *DC, SourceLocation StartLoc,
SourceLocation IdLoc, IdentifierInfo *Id, TypeSourceInfo *TInfo)
: TypedefNameDecl(Typedef, C, DC, StartLoc, IdLoc, Id, TInfo) {}

public:
static TypedefDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation StartLoc, SourceLocation IdLoc,
IdentifierInfo *Id, TypeSourceInfo *TInfo);
static TypedefDecl *CreateDeserialized(ASTContext &C, unsigned ID);

SourceRange getSourceRange() const override LLVM_READONLY;

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == Typedef; }
};

class TypeAliasDecl : public TypedefNameDecl {
TypeAliasTemplateDecl *Template;

TypeAliasDecl(ASTContext &C, DeclContext *DC, SourceLocation StartLoc,
SourceLocation IdLoc, IdentifierInfo *Id, TypeSourceInfo *TInfo)
: TypedefNameDecl(TypeAlias, C, DC, StartLoc, IdLoc, Id, TInfo),
Template(nullptr) {}

public:
static TypeAliasDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation StartLoc, SourceLocation IdLoc,
IdentifierInfo *Id, TypeSourceInfo *TInfo);
static TypeAliasDecl *CreateDeserialized(ASTContext &C, unsigned ID);

SourceRange getSourceRange() const override LLVM_READONLY;

TypeAliasTemplateDecl *getDescribedAliasTemplate() const { return Template; }
void setDescribedAliasTemplate(TypeAliasTemplateDecl *TAT) { Template = TAT; }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == TypeAlias; }
};

class TagDecl
: public TypeDecl, public DeclContext, public Redeclarable<TagDecl> {
public:
using TagKind = TagTypeKind;

private:
unsigned TagDeclKind : 3;

unsigned IsCompleteDefinition : 1;

protected:
unsigned IsBeingDefined : 1;

private:
unsigned IsEmbeddedInDeclarator : 1;

unsigned IsFreeStanding : 1;

protected:
unsigned NumPositiveBits : 8;
unsigned NumNegativeBits : 8;

unsigned IsScoped : 1;

unsigned IsScopedUsingClassTag : 1;

unsigned IsFixed : 1;

unsigned MayHaveOutOfDateDef : 1;

unsigned IsCompleteDefinitionRequired : 1;

private:
SourceRange BraceRange;

using ExtInfo = QualifierInfo;

llvm::PointerUnion<TypedefNameDecl *, ExtInfo *> TypedefNameDeclOrQualifier;

bool hasExtInfo() const { return TypedefNameDeclOrQualifier.is<ExtInfo *>(); }
ExtInfo *getExtInfo() { return TypedefNameDeclOrQualifier.get<ExtInfo *>(); }
const ExtInfo *getExtInfo() const {
return TypedefNameDeclOrQualifier.get<ExtInfo *>();
}

protected:
TagDecl(Kind DK, TagKind TK, const ASTContext &C, DeclContext *DC,
SourceLocation L, IdentifierInfo *Id, TagDecl *PrevDecl,
SourceLocation StartL)
: TypeDecl(DK, DC, L, Id, StartL), DeclContext(DK), redeclarable_base(C),
TagDeclKind(TK), IsCompleteDefinition(false), IsBeingDefined(false),
IsEmbeddedInDeclarator(false), IsFreeStanding(false),
IsCompleteDefinitionRequired(false),
TypedefNameDeclOrQualifier((TypedefNameDecl *)nullptr) {
assert((DK != Enum || TK == TTK_Enum) &&
"EnumDecl not matched with TTK_Enum");
setPreviousDecl(PrevDecl);
}

using redeclarable_base = Redeclarable<TagDecl>;

TagDecl *getNextRedeclarationImpl() override {
return getNextRedeclaration();
}

TagDecl *getPreviousDeclImpl() override {
return getPreviousDecl();
}

TagDecl *getMostRecentDeclImpl() override {
return getMostRecentDecl();
}

void completeDefinition();

public:
friend class ASTDeclReader;
friend class ASTDeclWriter;

using redecl_range = redeclarable_base::redecl_range;
using redecl_iterator = redeclarable_base::redecl_iterator;

using redeclarable_base::redecls_begin;
using redeclarable_base::redecls_end;
using redeclarable_base::redecls;
using redeclarable_base::getPreviousDecl;
using redeclarable_base::getMostRecentDecl;
using redeclarable_base::isFirstDecl;

SourceRange getBraceRange() const { return BraceRange; }
void setBraceRange(SourceRange R) { BraceRange = R; }

SourceLocation getInnerLocStart() const { return getLocStart(); }

SourceLocation getOuterLocStart() const;
SourceRange getSourceRange() const override LLVM_READONLY;

TagDecl *getCanonicalDecl() override;
const TagDecl *getCanonicalDecl() const {
return const_cast<TagDecl*>(this)->getCanonicalDecl();
}

bool isThisDeclarationADefinition() const {
return isCompleteDefinition();
}

bool isCompleteDefinition() const {
return IsCompleteDefinition;
}

bool isCompleteDefinitionRequired() const {
return IsCompleteDefinitionRequired;
}

bool isBeingDefined() const {
return IsBeingDefined;
}

bool isEmbeddedInDeclarator() const {
return IsEmbeddedInDeclarator;
}
void setEmbeddedInDeclarator(bool isInDeclarator) {
IsEmbeddedInDeclarator = isInDeclarator;
}

bool isFreeStanding() const { return IsFreeStanding; }
void setFreeStanding(bool isFreeStanding = true) {
IsFreeStanding = isFreeStanding;
}

bool isDependentType() const { return isDependentContext(); }

void startDefinition();

TagDecl *getDefinition() const;

void setCompleteDefinition(bool V) { IsCompleteDefinition = V; }

void setCompleteDefinitionRequired(bool V = true) {
IsCompleteDefinitionRequired = V;
}

StringRef getKindName() const {
return TypeWithKeyword::getTagTypeKindName(getTagKind());
}

TagKind getTagKind() const {
return TagKind(TagDeclKind);
}

void setTagKind(TagKind TK) { TagDeclKind = TK; }

bool isStruct() const { return getTagKind() == TTK_Struct; }
bool isInterface() const { return getTagKind() == TTK_Interface; }
bool isClass()  const { return getTagKind() == TTK_Class; }
bool isUnion()  const { return getTagKind() == TTK_Union; }
bool isEnum()   const { return getTagKind() == TTK_Enum; }

bool hasNameForLinkage() const {
return (getDeclName() || getTypedefNameForAnonDecl());
}

TypedefNameDecl *getTypedefNameForAnonDecl() const {
return hasExtInfo() ? nullptr
: TypedefNameDeclOrQualifier.get<TypedefNameDecl *>();
}

void setTypedefNameForAnonDecl(TypedefNameDecl *TDD);

NestedNameSpecifier *getQualifier() const {
return hasExtInfo() ? getExtInfo()->QualifierLoc.getNestedNameSpecifier()
: nullptr;
}

NestedNameSpecifierLoc getQualifierLoc() const {
return hasExtInfo() ? getExtInfo()->QualifierLoc
: NestedNameSpecifierLoc();
}

void setQualifierInfo(NestedNameSpecifierLoc QualifierLoc);

unsigned getNumTemplateParameterLists() const {
return hasExtInfo() ? getExtInfo()->NumTemplParamLists : 0;
}

TemplateParameterList *getTemplateParameterList(unsigned i) const {
assert(i < getNumTemplateParameterLists());
return getExtInfo()->TemplParamLists[i];
}

void setTemplateParameterListsInfo(ASTContext &Context,
ArrayRef<TemplateParameterList *> TPLists);

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K >= firstTag && K <= lastTag; }

static DeclContext *castToDeclContext(const TagDecl *D) {
return static_cast<DeclContext *>(const_cast<TagDecl*>(D));
}

static TagDecl *castFromDeclContext(const DeclContext *DC) {
return static_cast<TagDecl *>(const_cast<DeclContext*>(DC));
}
};

class EnumDecl : public TagDecl {
llvm::PointerUnion<const Type *, TypeSourceInfo *> IntegerType;

QualType PromotionType;

MemberSpecializationInfo *SpecializationInfo = nullptr;

unsigned HasODRHash : 1;
unsigned ODRHash;

EnumDecl(ASTContext &C, DeclContext *DC, SourceLocation StartLoc,
SourceLocation IdLoc, IdentifierInfo *Id, EnumDecl *PrevDecl,
bool Scoped, bool ScopedUsingClassTag, bool Fixed)
: TagDecl(Enum, TTK_Enum, C, DC, IdLoc, Id, PrevDecl, StartLoc) {
assert(Scoped || !ScopedUsingClassTag);
IntegerType = (const Type *)nullptr;
NumNegativeBits = 0;
NumPositiveBits = 0;
IsScoped = Scoped;
IsScopedUsingClassTag = ScopedUsingClassTag;
IsFixed = Fixed;
HasODRHash = false;
ODRHash = 0;
}

void anchor() override;

void setInstantiationOfMemberEnum(ASTContext &C, EnumDecl *ED,
TemplateSpecializationKind TSK);
public:
friend class ASTDeclReader;

EnumDecl *getCanonicalDecl() override {
return cast<EnumDecl>(TagDecl::getCanonicalDecl());
}
const EnumDecl *getCanonicalDecl() const {
return const_cast<EnumDecl*>(this)->getCanonicalDecl();
}

EnumDecl *getPreviousDecl() {
return cast_or_null<EnumDecl>(
static_cast<TagDecl *>(this)->getPreviousDecl());
}
const EnumDecl *getPreviousDecl() const {
return const_cast<EnumDecl*>(this)->getPreviousDecl();
}

EnumDecl *getMostRecentDecl() {
return cast<EnumDecl>(static_cast<TagDecl *>(this)->getMostRecentDecl());
}
const EnumDecl *getMostRecentDecl() const {
return const_cast<EnumDecl*>(this)->getMostRecentDecl();
}

EnumDecl *getDefinition() const {
return cast_or_null<EnumDecl>(TagDecl::getDefinition());
}

static EnumDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation StartLoc, SourceLocation IdLoc,
IdentifierInfo *Id, EnumDecl *PrevDecl,
bool IsScoped, bool IsScopedUsingClassTag,
bool IsFixed);
static EnumDecl *CreateDeserialized(ASTContext &C, unsigned ID);

void completeDefinition(QualType NewType,
QualType PromotionType,
unsigned NumPositiveBits,
unsigned NumNegativeBits);

using enumerator_iterator = specific_decl_iterator<EnumConstantDecl>;
using enumerator_range =
llvm::iterator_range<specific_decl_iterator<EnumConstantDecl>>;

enumerator_range enumerators() const {
return enumerator_range(enumerator_begin(), enumerator_end());
}

enumerator_iterator enumerator_begin() const {
const EnumDecl *E = getDefinition();
if (!E)
E = this;
return enumerator_iterator(E->decls_begin());
}

enumerator_iterator enumerator_end() const {
const EnumDecl *E = getDefinition();
if (!E)
E = this;
return enumerator_iterator(E->decls_end());
}

QualType getPromotionType() const { return PromotionType; }

void setPromotionType(QualType T) { PromotionType = T; }

QualType getIntegerType() const {
if (!IntegerType)
return QualType();
if (const Type *T = IntegerType.dyn_cast<const Type*>())
return QualType(T, 0);
return IntegerType.get<TypeSourceInfo*>()->getType().getUnqualifiedType();
}

void setIntegerType(QualType T) { IntegerType = T.getTypePtrOrNull(); }

void setIntegerTypeSourceInfo(TypeSourceInfo *TInfo) { IntegerType = TInfo; }

TypeSourceInfo *getIntegerTypeSourceInfo() const {
return IntegerType.dyn_cast<TypeSourceInfo*>();
}

SourceRange getIntegerTypeRange() const LLVM_READONLY;

unsigned getNumPositiveBits() const {
return NumPositiveBits;
}
void setNumPositiveBits(unsigned Num) {
NumPositiveBits = Num;
assert(NumPositiveBits == Num && "can't store this bitcount");
}

unsigned getNumNegativeBits() const {
return NumNegativeBits;
}
void setNumNegativeBits(unsigned Num) {
NumNegativeBits = Num;
}

bool isScoped() const {
return IsScoped;
}

bool isScopedUsingClassTag() const {
return IsScopedUsingClassTag;
}

bool isFixed() const {
return IsFixed;
}

unsigned getODRHash();

bool isComplete() const {
return isCompleteDefinition() || IntegerType;
}

bool isClosed() const;

bool isClosedFlag() const;

bool isClosedNonFlag() const;

EnumDecl *getTemplateInstantiationPattern() const;

EnumDecl *getInstantiatedFromMemberEnum() const;

TemplateSpecializationKind getTemplateSpecializationKind() const;

void setTemplateSpecializationKind(TemplateSpecializationKind TSK,
SourceLocation PointOfInstantiation = SourceLocation());

MemberSpecializationInfo *getMemberSpecializationInfo() const {
return SpecializationInfo;
}

void setInstantiationOfMemberEnum(EnumDecl *ED,
TemplateSpecializationKind TSK) {
setInstantiationOfMemberEnum(getASTContext(), ED, TSK);
}

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == Enum; }
};

class RecordDecl : public TagDecl {
public:
enum ArgPassingKind : unsigned {
APK_CanPassInRegs,

APK_CannotPassInRegs,

APK_CanNeverPassInRegs
};

private:
friend class DeclContext;

unsigned HasFlexibleArrayMember : 1;

unsigned AnonymousStructOrUnion : 1;

unsigned HasObjectMember : 1;

unsigned HasVolatileMember : 1;

mutable unsigned LoadedFieldsFromExternalStorage : 1;

unsigned NonTrivialToPrimitiveDefaultInitialize : 1;
unsigned NonTrivialToPrimitiveCopy : 1;
unsigned NonTrivialToPrimitiveDestroy : 1;

unsigned ParamDestroyedInCallee : 1;

unsigned ArgPassingRestrictions : 2;

protected:
RecordDecl(Kind DK, TagKind TK, const ASTContext &C, DeclContext *DC,
SourceLocation StartLoc, SourceLocation IdLoc,
IdentifierInfo *Id, RecordDecl *PrevDecl);

public:
static RecordDecl *Create(const ASTContext &C, TagKind TK, DeclContext *DC,
SourceLocation StartLoc, SourceLocation IdLoc,
IdentifierInfo *Id, RecordDecl* PrevDecl = nullptr);
static RecordDecl *CreateDeserialized(const ASTContext &C, unsigned ID);

RecordDecl *getPreviousDecl() {
return cast_or_null<RecordDecl>(
static_cast<TagDecl *>(this)->getPreviousDecl());
}
const RecordDecl *getPreviousDecl() const {
return const_cast<RecordDecl*>(this)->getPreviousDecl();
}

RecordDecl *getMostRecentDecl() {
return cast<RecordDecl>(static_cast<TagDecl *>(this)->getMostRecentDecl());
}
const RecordDecl *getMostRecentDecl() const {
return const_cast<RecordDecl*>(this)->getMostRecentDecl();
}

bool hasFlexibleArrayMember() const { return HasFlexibleArrayMember; }
void setHasFlexibleArrayMember(bool V) { HasFlexibleArrayMember = V; }

bool isAnonymousStructOrUnion() const { return AnonymousStructOrUnion; }
void setAnonymousStructOrUnion(bool Anon) {
AnonymousStructOrUnion = Anon;
}

bool hasObjectMember() const { return HasObjectMember; }
void setHasObjectMember (bool val) { HasObjectMember = val; }

bool hasVolatileMember() const { return HasVolatileMember; }
void setHasVolatileMember (bool val) { HasVolatileMember = val; }

bool hasLoadedFieldsFromExternalStorage() const {
return LoadedFieldsFromExternalStorage;
}
void setHasLoadedFieldsFromExternalStorage(bool val) {
LoadedFieldsFromExternalStorage = val;
}

bool isNonTrivialToPrimitiveDefaultInitialize() const {
return NonTrivialToPrimitiveDefaultInitialize;
}

void setNonTrivialToPrimitiveDefaultInitialize(bool V) {
NonTrivialToPrimitiveDefaultInitialize = V;
}

bool isNonTrivialToPrimitiveCopy() const {
return NonTrivialToPrimitiveCopy;
}

void setNonTrivialToPrimitiveCopy(bool V) {
NonTrivialToPrimitiveCopy = V;
}

bool isNonTrivialToPrimitiveDestroy() const {
return NonTrivialToPrimitiveDestroy;
}

void setNonTrivialToPrimitiveDestroy(bool V) {
NonTrivialToPrimitiveDestroy = V;
}

bool canPassInRegisters() const {
return getArgPassingRestrictions() == APK_CanPassInRegs;
}

ArgPassingKind getArgPassingRestrictions() const {
return static_cast<ArgPassingKind>(ArgPassingRestrictions);
}

void setArgPassingRestrictions(ArgPassingKind Kind) {
ArgPassingRestrictions = static_cast<uint8_t>(Kind);
}

bool isParamDestroyedInCallee() const {
return ParamDestroyedInCallee;
}

void setParamDestroyedInCallee(bool V) {
ParamDestroyedInCallee = V;
}

bool isInjectedClassName() const;

bool isLambda() const;

bool isCapturedRecord() const;

void setCapturedRecord();

RecordDecl *getDefinition() const {
return cast_or_null<RecordDecl>(TagDecl::getDefinition());
}

using field_iterator = specific_decl_iterator<FieldDecl>;
using field_range = llvm::iterator_range<specific_decl_iterator<FieldDecl>>;

field_range fields() const { return field_range(field_begin(), field_end()); }
field_iterator field_begin() const;

field_iterator field_end() const {
return field_iterator(decl_iterator());
}

bool field_empty() const {
return field_begin() == field_end();
}

virtual void completeDefinition();

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) {
return K >= firstRecord && K <= lastRecord;
}

bool isMsStruct(const ASTContext &C) const;

bool mayInsertExtraPadding(bool EmitRemark = false) const;

const FieldDecl *findFirstNamedDataMember() const;

private:
void LoadFieldsFromExternalStorage() const;
};

class FileScopeAsmDecl : public Decl {
StringLiteral *AsmString;
SourceLocation RParenLoc;

FileScopeAsmDecl(DeclContext *DC, StringLiteral *asmstring,
SourceLocation StartL, SourceLocation EndL)
: Decl(FileScopeAsm, DC, StartL), AsmString(asmstring), RParenLoc(EndL) {}

virtual void anchor();

public:
static FileScopeAsmDecl *Create(ASTContext &C, DeclContext *DC,
StringLiteral *Str, SourceLocation AsmLoc,
SourceLocation RParenLoc);

static FileScopeAsmDecl *CreateDeserialized(ASTContext &C, unsigned ID);

SourceLocation getAsmLoc() const { return getLocation(); }
SourceLocation getRParenLoc() const { return RParenLoc; }
void setRParenLoc(SourceLocation L) { RParenLoc = L; }
SourceRange getSourceRange() const override LLVM_READONLY {
return SourceRange(getAsmLoc(), getRParenLoc());
}

const StringLiteral *getAsmString() const { return AsmString; }
StringLiteral *getAsmString() { return AsmString; }
void setAsmString(StringLiteral *Asm) { AsmString = Asm; }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == FileScopeAsm; }
};

class BlockDecl : public Decl, public DeclContext {
public:
class Capture {
enum {
flag_isByRef = 0x1,
flag_isNested = 0x2
};

llvm::PointerIntPair<VarDecl*, 2> VariableAndFlags;

Expr *CopyExpr;

public:
Capture(VarDecl *variable, bool byRef, bool nested, Expr *copy)
: VariableAndFlags(variable,
(byRef ? flag_isByRef : 0) | (nested ? flag_isNested : 0)),
CopyExpr(copy) {}

VarDecl *getVariable() const { return VariableAndFlags.getPointer(); }

bool isByRef() const { return VariableAndFlags.getInt() & flag_isByRef; }

bool isNested() const { return VariableAndFlags.getInt() & flag_isNested; }

bool hasCopyExpr() const { return CopyExpr != nullptr; }
Expr *getCopyExpr() const { return CopyExpr; }
void setCopyExpr(Expr *e) { CopyExpr = e; }
};

private:
bool IsVariadic : 1;
bool CapturesCXXThis : 1;
bool BlockMissingReturnType : 1;
bool IsConversionFromLambda : 1;

bool DoesNotEscape : 1;

ParmVarDecl **ParamInfo = nullptr;
unsigned NumParams = 0;

Stmt *Body = nullptr;
TypeSourceInfo *SignatureAsWritten = nullptr;

const Capture *Captures = nullptr;
unsigned NumCaptures = 0;

unsigned ManglingNumber = 0;
Decl *ManglingContextDecl = nullptr;

protected:
BlockDecl(DeclContext *DC, SourceLocation CaretLoc)
: Decl(Block, DC, CaretLoc), DeclContext(Block), IsVariadic(false),
CapturesCXXThis(false), BlockMissingReturnType(true),
IsConversionFromLambda(false), DoesNotEscape(false) {}

public:
static BlockDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L);
static BlockDecl *CreateDeserialized(ASTContext &C, unsigned ID);

SourceLocation getCaretLocation() const { return getLocation(); }

bool isVariadic() const { return IsVariadic; }
void setIsVariadic(bool value) { IsVariadic = value; }

CompoundStmt *getCompoundBody() const { return (CompoundStmt*) Body; }
Stmt *getBody() const override { return (Stmt*) Body; }
void setBody(CompoundStmt *B) { Body = (Stmt*) B; }

void setSignatureAsWritten(TypeSourceInfo *Sig) { SignatureAsWritten = Sig; }
TypeSourceInfo *getSignatureAsWritten() const { return SignatureAsWritten; }

ArrayRef<ParmVarDecl *> parameters() const {
return {ParamInfo, getNumParams()};
}
MutableArrayRef<ParmVarDecl *> parameters() {
return {ParamInfo, getNumParams()};
}

using param_iterator = MutableArrayRef<ParmVarDecl *>::iterator;
using param_const_iterator = ArrayRef<ParmVarDecl *>::const_iterator;

bool param_empty() const { return parameters().empty(); }
param_iterator param_begin() { return parameters().begin(); }
param_iterator param_end() { return parameters().end(); }
param_const_iterator param_begin() const { return parameters().begin(); }
param_const_iterator param_end() const { return parameters().end(); }
size_t param_size() const { return parameters().size(); }

unsigned getNumParams() const { return NumParams; }

const ParmVarDecl *getParamDecl(unsigned i) const {
assert(i < getNumParams() && "Illegal param #");
return ParamInfo[i];
}
ParmVarDecl *getParamDecl(unsigned i) {
assert(i < getNumParams() && "Illegal param #");
return ParamInfo[i];
}

void setParams(ArrayRef<ParmVarDecl *> NewParamInfo);

bool hasCaptures() const { return NumCaptures != 0 || CapturesCXXThis; }

unsigned getNumCaptures() const { return NumCaptures; }

using capture_const_iterator = ArrayRef<Capture>::const_iterator;

ArrayRef<Capture> captures() const { return {Captures, NumCaptures}; }

capture_const_iterator capture_begin() const { return captures().begin(); }
capture_const_iterator capture_end() const { return captures().end(); }

bool capturesCXXThis() const { return CapturesCXXThis; }
bool blockMissingReturnType() const { return BlockMissingReturnType; }
void setBlockMissingReturnType(bool val) { BlockMissingReturnType = val; }

bool isConversionFromLambda() const { return IsConversionFromLambda; }
void setIsConversionFromLambda(bool val) { IsConversionFromLambda = val; }

bool doesNotEscape() const { return DoesNotEscape; }
void setDoesNotEscape() { DoesNotEscape = true; }

bool capturesVariable(const VarDecl *var) const;

void setCaptures(ASTContext &Context, ArrayRef<Capture> Captures,
bool CapturesCXXThis);

unsigned getBlockManglingNumber() const {
return ManglingNumber;
}

Decl *getBlockManglingContextDecl() const {
return ManglingContextDecl;
}

void setBlockMangling(unsigned Number, Decl *Ctx) {
ManglingNumber = Number;
ManglingContextDecl = Ctx;
}

SourceRange getSourceRange() const override LLVM_READONLY;

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == Block; }
static DeclContext *castToDeclContext(const BlockDecl *D) {
return static_cast<DeclContext *>(const_cast<BlockDecl*>(D));
}
static BlockDecl *castFromDeclContext(const DeclContext *DC) {
return static_cast<BlockDecl *>(const_cast<DeclContext*>(DC));
}
};

class CapturedDecl final
: public Decl,
public DeclContext,
private llvm::TrailingObjects<CapturedDecl, ImplicitParamDecl *> {
protected:
size_t numTrailingObjects(OverloadToken<ImplicitParamDecl>) {
return NumParams;
}

private:
unsigned NumParams;

unsigned ContextParam;

llvm::PointerIntPair<Stmt *, 1, bool> BodyAndNothrow;

explicit CapturedDecl(DeclContext *DC, unsigned NumParams);

ImplicitParamDecl *const *getParams() const {
return getTrailingObjects<ImplicitParamDecl *>();
}

ImplicitParamDecl **getParams() {
return getTrailingObjects<ImplicitParamDecl *>();
}

public:
friend class ASTDeclReader;
friend class ASTDeclWriter;
friend TrailingObjects;

static CapturedDecl *Create(ASTContext &C, DeclContext *DC,
unsigned NumParams);
static CapturedDecl *CreateDeserialized(ASTContext &C, unsigned ID,
unsigned NumParams);

Stmt *getBody() const override;
void setBody(Stmt *B);

bool isNothrow() const;
void setNothrow(bool Nothrow = true);

unsigned getNumParams() const { return NumParams; }

ImplicitParamDecl *getParam(unsigned i) const {
assert(i < NumParams);
return getParams()[i];
}
void setParam(unsigned i, ImplicitParamDecl *P) {
assert(i < NumParams);
getParams()[i] = P;
}

ArrayRef<ImplicitParamDecl *> parameters() const {
return {getParams(), getNumParams()};
}
MutableArrayRef<ImplicitParamDecl *> parameters() {
return {getParams(), getNumParams()};
}

ImplicitParamDecl *getContextParam() const {
assert(ContextParam < NumParams);
return getParam(ContextParam);
}
void setContextParam(unsigned i, ImplicitParamDecl *P) {
assert(i < NumParams);
ContextParam = i;
setParam(i, P);
}
unsigned getContextParamPosition() const { return ContextParam; }

using param_iterator = ImplicitParamDecl *const *;
using param_range = llvm::iterator_range<param_iterator>;

param_iterator param_begin() const { return getParams(); }
param_iterator param_end() const { return getParams() + NumParams; }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == Captured; }
static DeclContext *castToDeclContext(const CapturedDecl *D) {
return static_cast<DeclContext *>(const_cast<CapturedDecl *>(D));
}
static CapturedDecl *castFromDeclContext(const DeclContext *DC) {
return static_cast<CapturedDecl *>(const_cast<DeclContext *>(DC));
}
};

class ImportDecl final : public Decl,
llvm::TrailingObjects<ImportDecl, SourceLocation> {
friend class ASTContext;
friend class ASTDeclReader;
friend class ASTReader;
friend TrailingObjects;

llvm::PointerIntPair<Module *, 1, bool> ImportedAndComplete;

ImportDecl *NextLocalImport = nullptr;

ImportDecl(DeclContext *DC, SourceLocation StartLoc, Module *Imported,
ArrayRef<SourceLocation> IdentifierLocs);

ImportDecl(DeclContext *DC, SourceLocation StartLoc, Module *Imported,
SourceLocation EndLoc);

ImportDecl(EmptyShell Empty) : Decl(Import, Empty) {}

public:
static ImportDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation StartLoc, Module *Imported,
ArrayRef<SourceLocation> IdentifierLocs);

static ImportDecl *CreateImplicit(ASTContext &C, DeclContext *DC,
SourceLocation StartLoc, Module *Imported,
SourceLocation EndLoc);

static ImportDecl *CreateDeserialized(ASTContext &C, unsigned ID,
unsigned NumLocations);

Module *getImportedModule() const { return ImportedAndComplete.getPointer(); }

ArrayRef<SourceLocation> getIdentifierLocs() const;

SourceRange getSourceRange() const override LLVM_READONLY;

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == Import; }
};

class ExportDecl final : public Decl, public DeclContext {
virtual void anchor();

private:
friend class ASTDeclReader;

SourceLocation RBraceLoc;

ExportDecl(DeclContext *DC, SourceLocation ExportLoc)
: Decl(Export, DC, ExportLoc), DeclContext(Export),
RBraceLoc(SourceLocation()) {}

public:
static ExportDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation ExportLoc);
static ExportDecl *CreateDeserialized(ASTContext &C, unsigned ID);

SourceLocation getExportLoc() const { return getLocation(); }
SourceLocation getRBraceLoc() const { return RBraceLoc; }
void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }

SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }
SourceLocation getEndLoc() const LLVM_READONLY {
if (RBraceLoc.isValid())
return RBraceLoc;
return decls_empty() ? getLocation() : decls_begin()->getLocEnd();
}

SourceRange getSourceRange() const override LLVM_READONLY {
return SourceRange(getLocation(), getLocEnd());
}

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == Export; }
static DeclContext *castToDeclContext(const ExportDecl *D) {
return static_cast<DeclContext *>(const_cast<ExportDecl*>(D));
}
static ExportDecl *castFromDeclContext(const DeclContext *DC) {
return static_cast<ExportDecl *>(const_cast<DeclContext*>(DC));
}
};

class EmptyDecl : public Decl {
EmptyDecl(DeclContext *DC, SourceLocation L) : Decl(Empty, DC, L) {}

virtual void anchor();

public:
static EmptyDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation L);
static EmptyDecl *CreateDeserialized(ASTContext &C, unsigned ID);

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == Empty; }
};

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
const NamedDecl* ND) {
DB.AddTaggedVal(reinterpret_cast<intptr_t>(ND),
DiagnosticsEngine::ak_nameddecl);
return DB;
}
inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
const NamedDecl* ND) {
PD.AddTaggedVal(reinterpret_cast<intptr_t>(ND),
DiagnosticsEngine::ak_nameddecl);
return PD;
}

template<typename decl_type>
void Redeclarable<decl_type>::setPreviousDecl(decl_type *PrevDecl) {
assert(RedeclLink.isFirst() &&
"setPreviousDecl on a decl already in a redeclaration chain");

if (PrevDecl) {
First = PrevDecl->getFirstDecl();
assert(First->RedeclLink.isFirst() && "Expected first");
decl_type *MostRecent = First->getNextRedeclaration();
RedeclLink = PreviousDeclLink(cast<decl_type>(MostRecent));

static_cast<decl_type*>(this)->IdentifierNamespace |=
MostRecent->getIdentifierNamespace() &
(Decl::IDNS_Ordinary | Decl::IDNS_Tag | Decl::IDNS_Type);
} else {
First = static_cast<decl_type*>(this);
}

First->RedeclLink.setLatest(static_cast<decl_type*>(this));

assert(!isa<NamedDecl>(static_cast<decl_type*>(this)) ||
cast<NamedDecl>(static_cast<decl_type*>(this))->isLinkageValid());
}


inline bool IsEnumDeclComplete(EnumDecl *ED) {
return ED->isComplete();
}

inline bool IsEnumDeclScoped(EnumDecl *ED) {
return ED->isScoped();
}

} 

#endif 
