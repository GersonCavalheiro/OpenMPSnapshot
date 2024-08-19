
#ifndef LLVM_CLANG_PARSE_PARSER_H
#define LLVM_CLANG_PARSE_PARSER_H

#include "clang/AST/Availability.h"
#include "clang/Basic/BitmaskEnum.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Lex/CodeCompletionHandler.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/LoopHint.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SaveAndRestore.h"
#include <memory>
#include <stack>

namespace clang {
class PragmaHandler;
class Scope;
class BalancedDelimiterTracker;
class CorrectionCandidateCallback;
class DeclGroupRef;
class DiagnosticBuilder;
class Parser;
class ParsingDeclRAIIObject;
class ParsingDeclSpec;
class ParsingDeclarator;
class ParsingFieldDeclarator;
class ColonProtectionRAIIObject;
class InMessageExpressionRAIIObject;
class PoisonSEHIdentifiersRAIIObject;
class OMPClause;
class ObjCTypeParamList;
class ObjCTypeParameter;

class Parser : public CodeCompletionHandler {
friend class ColonProtectionRAIIObject;
friend class InMessageExpressionRAIIObject;
friend class PoisonSEHIdentifiersRAIIObject;
friend class ObjCDeclContextSwitch;
friend class ParenBraceBracketBalancer;
friend class BalancedDelimiterTracker;

Preprocessor &PP;

Token Tok;

SourceLocation PrevTokLocation;

unsigned short ParenCount = 0, BracketCount = 0, BraceCount = 0;
unsigned short MisplacedModuleBeginCount = 0;

Sema &Actions;

DiagnosticsEngine &Diags;

enum { ScopeCacheSize = 16 };
unsigned NumCachedScopes;
Scope *ScopeCache[ScopeCacheSize];

IdentifierInfo *Ident__exception_code,
*Ident___exception_code,
*Ident_GetExceptionCode;
IdentifierInfo *Ident__exception_info,
*Ident___exception_info,
*Ident_GetExceptionInfo;
IdentifierInfo *Ident__abnormal_termination,
*Ident___abnormal_termination,
*Ident_AbnormalTermination;

IdentifierInfo *Ident__except;
mutable IdentifierInfo *Ident_sealed;

IdentifierInfo *Ident_super;
IdentifierInfo *Ident_vector;
IdentifierInfo *Ident_bool;
IdentifierInfo *Ident_pixel;

mutable IdentifierInfo *Ident_instancetype;

IdentifierInfo *Ident_introduced;

IdentifierInfo *Ident_deprecated;

IdentifierInfo *Ident_obsoleted;

IdentifierInfo *Ident_unavailable;

IdentifierInfo *Ident_message;

IdentifierInfo *Ident_strict;

IdentifierInfo *Ident_replacement;

IdentifierInfo *Ident_language, *Ident_defined_in,
*Ident_generated_declaration;

mutable IdentifierInfo *Ident_final;
mutable IdentifierInfo *Ident_GNU_final;
mutable IdentifierInfo *Ident_override;

llvm::SmallDenseMap<IdentifierInfo *, tok::TokenKind> RevertibleTypeTraits;

std::unique_ptr<PragmaHandler> AlignHandler;
std::unique_ptr<PragmaHandler> GCCVisibilityHandler;
std::unique_ptr<PragmaHandler> OptionsHandler;
std::unique_ptr<PragmaHandler> PackHandler;
std::unique_ptr<PragmaHandler> MSStructHandler;
std::unique_ptr<PragmaHandler> UnusedHandler;
std::unique_ptr<PragmaHandler> WeakHandler;
std::unique_ptr<PragmaHandler> RedefineExtnameHandler;
std::unique_ptr<PragmaHandler> FPContractHandler;
std::unique_ptr<PragmaHandler> OpenCLExtensionHandler;
std::unique_ptr<PragmaHandler> OpenMPHandler;
std::unique_ptr<PragmaHandler> PCSectionHandler;
std::unique_ptr<PragmaHandler> MSCommentHandler;
std::unique_ptr<PragmaHandler> MSDetectMismatchHandler;
std::unique_ptr<PragmaHandler> MSPointersToMembers;
std::unique_ptr<PragmaHandler> MSVtorDisp;
std::unique_ptr<PragmaHandler> MSInitSeg;
std::unique_ptr<PragmaHandler> MSDataSeg;
std::unique_ptr<PragmaHandler> MSBSSSeg;
std::unique_ptr<PragmaHandler> MSConstSeg;
std::unique_ptr<PragmaHandler> MSCodeSeg;
std::unique_ptr<PragmaHandler> MSSection;
std::unique_ptr<PragmaHandler> MSRuntimeChecks;
std::unique_ptr<PragmaHandler> MSIntrinsic;
std::unique_ptr<PragmaHandler> MSOptimize;
std::unique_ptr<PragmaHandler> CUDAForceHostDeviceHandler;
std::unique_ptr<PragmaHandler> OptimizeHandler;
std::unique_ptr<PragmaHandler> LoopHintHandler;
std::unique_ptr<PragmaHandler> UnrollHintHandler;
std::unique_ptr<PragmaHandler> NoUnrollHintHandler;
std::unique_ptr<PragmaHandler> FPHandler;
std::unique_ptr<PragmaHandler> STDCFENVHandler;
std::unique_ptr<PragmaHandler> STDCCXLIMITHandler;
std::unique_ptr<PragmaHandler> STDCUnknownHandler;
std::unique_ptr<PragmaHandler> AttributePragmaHandler;

std::unique_ptr<CommentHandler> CommentSemaHandler;

bool GreaterThanIsOperator;

bool ColonIsSacred;

bool InMessageExpression;

unsigned TemplateParameterDepth;

class TemplateParameterDepthRAII {
unsigned &Depth;
unsigned AddedLevels;
public:
explicit TemplateParameterDepthRAII(unsigned &Depth)
: Depth(Depth), AddedLevels(0) {}

~TemplateParameterDepthRAII() {
Depth -= AddedLevels;
}

void operator++() {
++Depth;
++AddedLevels;
}
void addDepth(unsigned D) {
Depth += D;
AddedLevels += D;
}
unsigned getDepth() const { return Depth; }
};

AttributeFactory AttrFactory;

SmallVector<TemplateIdAnnotation *, 16> TemplateIds;

SmallVector<IdentifierInfo *, 8> TentativelyDeclaredIdentifiers;

struct AngleBracketTracker {
enum Priority : unsigned short {
PotentialTypo = 0x0,
DependentName = 0x2,

SpaceBeforeLess = 0x0,
NoSpaceBeforeLess = 0x1,

LLVM_MARK_AS_BITMASK_ENUM( DependentName)
};

struct Loc {
Expr *TemplateName;
SourceLocation LessLoc;
AngleBracketTracker::Priority Priority;
unsigned short ParenCount, BracketCount, BraceCount;

bool isActive(Parser &P) const {
return P.ParenCount == ParenCount && P.BracketCount == BracketCount &&
P.BraceCount == BraceCount;
}

bool isActiveOrNested(Parser &P) const {
return isActive(P) || P.ParenCount > ParenCount ||
P.BracketCount > BracketCount || P.BraceCount > BraceCount;
}
};

SmallVector<Loc, 8> Locs;

void add(Parser &P, Expr *TemplateName, SourceLocation LessLoc,
Priority Prio) {
if (!Locs.empty() && Locs.back().isActive(P)) {
if (Locs.back().Priority <= Prio) {
Locs.back().TemplateName = TemplateName;
Locs.back().LessLoc = LessLoc;
Locs.back().Priority = Prio;
}
} else {
Locs.push_back({TemplateName, LessLoc, Prio,
P.ParenCount, P.BracketCount, P.BraceCount});
}
}

void clear(Parser &P) {
while (!Locs.empty() && Locs.back().isActiveOrNested(P))
Locs.pop_back();
}

Loc *getCurrent(Parser &P) {
if (!Locs.empty() && Locs.back().isActive(P))
return &Locs.back();
return nullptr;
}
};

AngleBracketTracker AngleBrackets;

IdentifierInfo *getSEHExceptKeyword();

bool ParsingInObjCContainer;

bool SkipFunctionBodies;

SourceLocation ExprStatementTokLoc;

public:
Parser(Preprocessor &PP, Sema &Actions, bool SkipFunctionBodies);
~Parser() override;

const LangOptions &getLangOpts() const { return PP.getLangOpts(); }
const TargetInfo &getTargetInfo() const { return PP.getTargetInfo(); }
Preprocessor &getPreprocessor() const { return PP; }
Sema &getActions() const { return Actions; }
AttributeFactory &getAttrFactory() { return AttrFactory; }

const Token &getCurToken() const { return Tok; }
Scope *getCurScope() const { return Actions.getCurScope(); }
void incrementMSManglingNumber() const {
return Actions.incrementMSManglingNumber();
}

Decl  *getObjCDeclContext() const { return Actions.getObjCDeclContext(); }

typedef OpaquePtr<DeclGroupRef> DeclGroupPtrTy;
typedef OpaquePtr<TemplateName> TemplateTy;

typedef SmallVector<TemplateParameterList *, 4> TemplateParameterLists;

typedef Sema::FullExprArg FullExprArg;


void Initialize();

bool ParseFirstTopLevelDecl(DeclGroupPtrTy &Result);

bool ParseTopLevelDecl(DeclGroupPtrTy &Result);
bool ParseTopLevelDecl() {
DeclGroupPtrTy Result;
return ParseTopLevelDecl(Result);
}

SourceLocation ConsumeToken() {
assert(!isTokenSpecial() &&
"Should consume special tokens with Consume*Token");
PrevTokLocation = Tok.getLocation();
PP.Lex(Tok);
return PrevTokLocation;
}

bool TryConsumeToken(tok::TokenKind Expected) {
if (Tok.isNot(Expected))
return false;
assert(!isTokenSpecial() &&
"Should consume special tokens with Consume*Token");
PrevTokLocation = Tok.getLocation();
PP.Lex(Tok);
return true;
}

bool TryConsumeToken(tok::TokenKind Expected, SourceLocation &Loc) {
if (!TryConsumeToken(Expected))
return false;
Loc = PrevTokLocation;
return true;
}

SourceLocation ConsumeAnyToken(bool ConsumeCodeCompletionTok = false) {
if (isTokenParen())
return ConsumeParen();
if (isTokenBracket())
return ConsumeBracket();
if (isTokenBrace())
return ConsumeBrace();
if (isTokenStringLiteral())
return ConsumeStringToken();
if (Tok.is(tok::code_completion))
return ConsumeCodeCompletionTok ? ConsumeCodeCompletionToken()
: handleUnexpectedCodeCompletionToken();
if (Tok.isAnnotation())
return ConsumeAnnotationToken();
return ConsumeToken();
}


SourceLocation getEndOfPreviousToken() {
return PP.getLocForEndOfToken(PrevTokLocation);
}

IdentifierInfo *getNullabilityKeyword(NullabilityKind nullability) {
return Actions.getNullabilityKeyword(nullability);
}

private:

bool isTokenParen() const {
return Tok.isOneOf(tok::l_paren, tok::r_paren);
}
bool isTokenBracket() const {
return Tok.isOneOf(tok::l_square, tok::r_square);
}
bool isTokenBrace() const {
return Tok.isOneOf(tok::l_brace, tok::r_brace);
}
bool isTokenStringLiteral() const {
return tok::isStringLiteral(Tok.getKind());
}
bool isTokenSpecial() const {
return isTokenStringLiteral() || isTokenParen() || isTokenBracket() ||
isTokenBrace() || Tok.is(tok::code_completion) || Tok.isAnnotation();
}

bool isTokenEqualOrEqualTypo();

void UnconsumeToken(Token &Consumed) {
Token Next = Tok;
PP.EnterToken(Consumed);
PP.Lex(Tok);
PP.EnterToken(Next);
}

SourceLocation ConsumeAnnotationToken() {
assert(Tok.isAnnotation() && "wrong consume method");
SourceLocation Loc = Tok.getLocation();
PrevTokLocation = Tok.getAnnotationEndLoc();
PP.Lex(Tok);
return Loc;
}

SourceLocation ConsumeParen() {
assert(isTokenParen() && "wrong consume method");
if (Tok.getKind() == tok::l_paren)
++ParenCount;
else if (ParenCount) {
AngleBrackets.clear(*this);
--ParenCount;       
}
PrevTokLocation = Tok.getLocation();
PP.Lex(Tok);
return PrevTokLocation;
}

SourceLocation ConsumeBracket() {
assert(isTokenBracket() && "wrong consume method");
if (Tok.getKind() == tok::l_square)
++BracketCount;
else if (BracketCount) {
AngleBrackets.clear(*this);
--BracketCount;     
}

PrevTokLocation = Tok.getLocation();
PP.Lex(Tok);
return PrevTokLocation;
}

SourceLocation ConsumeBrace() {
assert(isTokenBrace() && "wrong consume method");
if (Tok.getKind() == tok::l_brace)
++BraceCount;
else if (BraceCount) {
AngleBrackets.clear(*this);
--BraceCount;     
}

PrevTokLocation = Tok.getLocation();
PP.Lex(Tok);
return PrevTokLocation;
}

SourceLocation ConsumeStringToken() {
assert(isTokenStringLiteral() &&
"Should only consume string literals with this method");
PrevTokLocation = Tok.getLocation();
PP.Lex(Tok);
return PrevTokLocation;
}

SourceLocation ConsumeCodeCompletionToken() {
assert(Tok.is(tok::code_completion));
PrevTokLocation = Tok.getLocation();
PP.Lex(Tok);
return PrevTokLocation;
}

SourceLocation handleUnexpectedCodeCompletionToken();

void cutOffParsing() {
if (PP.isCodeCompletionEnabled())
PP.setCodeCompletionReached();
Tok.setKind(tok::eof);
}

bool isEofOrEom() {
tok::TokenKind Kind = Tok.getKind();
return Kind == tok::eof || Kind == tok::annot_module_begin ||
Kind == tok::annot_module_end || Kind == tok::annot_module_include;
}

bool isFoldOperator(prec::Level Level) const;

bool isFoldOperator(tok::TokenKind Kind) const;

void initializePragmaHandlers();

void resetPragmaHandlers();

void HandlePragmaUnused();

void HandlePragmaVisibility();

void HandlePragmaPack();

void HandlePragmaMSStruct();

void HandlePragmaMSComment();

void HandlePragmaMSPointersToMembers();

void HandlePragmaMSVtorDisp();

void HandlePragmaMSPragma();
bool HandlePragmaMSSection(StringRef PragmaName,
SourceLocation PragmaLocation);
bool HandlePragmaMSSegment(StringRef PragmaName,
SourceLocation PragmaLocation);
bool HandlePragmaMSInitSeg(StringRef PragmaName,
SourceLocation PragmaLocation);

void HandlePragmaAlign();

void HandlePragmaDump();

void HandlePragmaWeak();

void HandlePragmaWeakAlias();

void HandlePragmaRedefineExtname();

void HandlePragmaFPContract();

void HandlePragmaFP();

void HandlePragmaOpenCLExtension();

StmtResult HandlePragmaCaptured();

bool HandlePragmaLoopHint(LoopHint &Hint);

bool ParsePragmaAttributeSubjectMatchRuleSet(
attr::ParsedSubjectMatchRuleSet &SubjectMatchRules,
SourceLocation &AnyLoc, SourceLocation &LastMatchRuleEndLoc);

void HandlePragmaAttribute();

const Token &GetLookAheadToken(unsigned N) {
if (N == 0 || Tok.is(tok::eof)) return Tok;
return PP.LookAhead(N-1);
}

public:
const Token &NextToken() {
return PP.LookAhead(0);
}

static ParsedType getTypeAnnotation(const Token &Tok) {
return ParsedType::getFromOpaquePtr(Tok.getAnnotationValue());
}

private:
static void setTypeAnnotation(Token &Tok, ParsedType T) {
Tok.setAnnotationValue(T.getAsOpaquePtr());
}

static ExprResult getExprAnnotation(const Token &Tok) {
return ExprResult::getFromOpaquePointer(Tok.getAnnotationValue());
}

static void setExprAnnotation(Token &Tok, ExprResult ER) {
Tok.setAnnotationValue(ER.getAsOpaquePointer());
}

public:
bool TryAnnotateTypeOrScopeToken();
bool TryAnnotateTypeOrScopeTokenAfterScopeSpec(CXXScopeSpec &SS,
bool IsNewScope);
bool TryAnnotateCXXScopeToken(bool EnteringContext = false);

private:
enum AnnotatedNameKind {
ANK_Error,
ANK_TentativeDecl,
ANK_TemplateName,
ANK_Unresolved,
ANK_Success
};
AnnotatedNameKind
TryAnnotateName(bool IsAddressOfOperand,
std::unique_ptr<CorrectionCandidateCallback> CCC = nullptr);

void AnnotateScopeToken(CXXScopeSpec &SS, bool IsNewAnnotation);

bool TryAltiVecToken(DeclSpec &DS, SourceLocation Loc,
const char *&PrevSpec, unsigned &DiagID,
bool &isInvalid) {
if (!getLangOpts().AltiVec && !getLangOpts().ZVector)
return false;

if (Tok.getIdentifierInfo() != Ident_vector &&
Tok.getIdentifierInfo() != Ident_bool &&
(!getLangOpts().AltiVec || Tok.getIdentifierInfo() != Ident_pixel))
return false;

return TryAltiVecTokenOutOfLine(DS, Loc, PrevSpec, DiagID, isInvalid);
}

bool TryAltiVecVectorToken() {
if ((!getLangOpts().AltiVec && !getLangOpts().ZVector) ||
Tok.getIdentifierInfo() != Ident_vector) return false;
return TryAltiVecVectorTokenOutOfLine();
}

bool TryAltiVecVectorTokenOutOfLine();
bool TryAltiVecTokenOutOfLine(DeclSpec &DS, SourceLocation Loc,
const char *&PrevSpec, unsigned &DiagID,
bool &isInvalid);

bool isObjCInstancetype() {
assert(getLangOpts().ObjC1);
if (Tok.isAnnotation())
return false;
if (!Ident_instancetype)
Ident_instancetype = PP.getIdentifierInfo("instancetype");
return Tok.getIdentifierInfo() == Ident_instancetype;
}

bool TryKeywordIdentFallback(bool DisableKeyword);

TemplateIdAnnotation *takeTemplateIdAnnotation(const Token &tok);

class TentativeParsingAction {
Parser &P;
Token PrevTok;
size_t PrevTentativelyDeclaredIdentifierCount;
unsigned short PrevParenCount, PrevBracketCount, PrevBraceCount;
bool isActive;

public:
explicit TentativeParsingAction(Parser& p) : P(p) {
PrevTok = P.Tok;
PrevTentativelyDeclaredIdentifierCount =
P.TentativelyDeclaredIdentifiers.size();
PrevParenCount = P.ParenCount;
PrevBracketCount = P.BracketCount;
PrevBraceCount = P.BraceCount;
P.PP.EnableBacktrackAtThisPos();
isActive = true;
}
void Commit() {
assert(isActive && "Parsing action was finished!");
P.TentativelyDeclaredIdentifiers.resize(
PrevTentativelyDeclaredIdentifierCount);
P.PP.CommitBacktrackedTokens();
isActive = false;
}
void Revert() {
assert(isActive && "Parsing action was finished!");
P.PP.Backtrack();
P.Tok = PrevTok;
P.TentativelyDeclaredIdentifiers.resize(
PrevTentativelyDeclaredIdentifierCount);
P.ParenCount = PrevParenCount;
P.BracketCount = PrevBracketCount;
P.BraceCount = PrevBraceCount;
isActive = false;
}
~TentativeParsingAction() {
assert(!isActive && "Forgot to call Commit or Revert!");
}
};
class RevertingTentativeParsingAction
: private Parser::TentativeParsingAction {
public:
RevertingTentativeParsingAction(Parser &P)
: Parser::TentativeParsingAction(P) {}
~RevertingTentativeParsingAction() { Revert(); }
};

class UnannotatedTentativeParsingAction;

class ObjCDeclContextSwitch {
Parser &P;
Decl *DC;
SaveAndRestore<bool> WithinObjCContainer;
public:
explicit ObjCDeclContextSwitch(Parser &p)
: P(p), DC(p.getObjCDeclContext()),
WithinObjCContainer(P.ParsingInObjCContainer, DC != nullptr) {
if (DC)
P.Actions.ActOnObjCTemporaryExitContainerContext(cast<DeclContext>(DC));
}
~ObjCDeclContextSwitch() {
if (DC)
P.Actions.ActOnObjCReenterContainerContext(cast<DeclContext>(DC));
}
};

bool ExpectAndConsume(tok::TokenKind ExpectedTok,
unsigned Diag = diag::err_expected,
StringRef DiagMsg = "");

bool ExpectAndConsumeSemi(unsigned DiagID);

enum ExtraSemiKind {
OutsideFunction = 0,
InsideStruct = 1,
InstanceVariableList = 2,
AfterMemberFunctionDefinition = 3
};

void ConsumeExtraSemi(ExtraSemiKind Kind, unsigned TST = TST_unspecified);

bool expectIdentifier();

public:

class ParseScope {
Parser *Self;
ParseScope(const ParseScope &) = delete;
void operator=(const ParseScope &) = delete;

public:
ParseScope(Parser *Self, unsigned ScopeFlags, bool EnteredScope = true,
bool BeforeCompoundStmt = false)
: Self(Self) {
if (EnteredScope && !BeforeCompoundStmt)
Self->EnterScope(ScopeFlags);
else {
if (BeforeCompoundStmt)
Self->incrementMSManglingNumber();

this->Self = nullptr;
}
}

void Exit() {
if (Self) {
Self->ExitScope();
Self = nullptr;
}
}

~ParseScope() {
Exit();
}
};

void EnterScope(unsigned ScopeFlags);

void ExitScope();

private:
class ParseScopeFlags {
Scope *CurScope;
unsigned OldFlags;
ParseScopeFlags(const ParseScopeFlags &) = delete;
void operator=(const ParseScopeFlags &) = delete;

public:
ParseScopeFlags(Parser *Self, unsigned ScopeFlags, bool ManageFlags = true);
~ParseScopeFlags();
};


public:
DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);
DiagnosticBuilder Diag(const Token &Tok, unsigned DiagID);
DiagnosticBuilder Diag(unsigned DiagID) {
return Diag(Tok, DiagID);
}

private:
void SuggestParentheses(SourceLocation Loc, unsigned DK,
SourceRange ParenRange);
void CheckNestedObjCContexts(SourceLocation AtLoc);

public:

enum SkipUntilFlags {
StopAtSemi = 1 << 0,  
StopBeforeMatch = 1 << 1,
StopAtCodeCompletion = 1 << 2 
};

friend constexpr SkipUntilFlags operator|(SkipUntilFlags L,
SkipUntilFlags R) {
return static_cast<SkipUntilFlags>(static_cast<unsigned>(L) |
static_cast<unsigned>(R));
}

bool SkipUntil(tok::TokenKind T,
SkipUntilFlags Flags = static_cast<SkipUntilFlags>(0)) {
return SkipUntil(llvm::makeArrayRef(T), Flags);
}
bool SkipUntil(tok::TokenKind T1, tok::TokenKind T2,
SkipUntilFlags Flags = static_cast<SkipUntilFlags>(0)) {
tok::TokenKind TokArray[] = {T1, T2};
return SkipUntil(TokArray, Flags);
}
bool SkipUntil(tok::TokenKind T1, tok::TokenKind T2, tok::TokenKind T3,
SkipUntilFlags Flags = static_cast<SkipUntilFlags>(0)) {
tok::TokenKind TokArray[] = {T1, T2, T3};
return SkipUntil(TokArray, Flags);
}
bool SkipUntil(ArrayRef<tok::TokenKind> Toks,
SkipUntilFlags Flags = static_cast<SkipUntilFlags>(0));

void SkipMalformedDecl();

private:

struct ParsingClass;

class LateParsedDeclaration {
public:
virtual ~LateParsedDeclaration();

virtual void ParseLexedMethodDeclarations();
virtual void ParseLexedMemberInitializers();
virtual void ParseLexedMethodDefs();
virtual void ParseLexedAttributes();
};

class LateParsedClass : public LateParsedDeclaration {
public:
LateParsedClass(Parser *P, ParsingClass *C);
~LateParsedClass() override;

void ParseLexedMethodDeclarations() override;
void ParseLexedMemberInitializers() override;
void ParseLexedMethodDefs() override;
void ParseLexedAttributes() override;

private:
Parser *Self;
ParsingClass *Class;
};

struct LateParsedAttribute : public LateParsedDeclaration {
Parser *Self;
CachedTokens Toks;
IdentifierInfo &AttrName;
SourceLocation AttrNameLoc;
SmallVector<Decl*, 2> Decls;

explicit LateParsedAttribute(Parser *P, IdentifierInfo &Name,
SourceLocation Loc)
: Self(P), AttrName(Name), AttrNameLoc(Loc) {}

void ParseLexedAttributes() override;

void addDecl(Decl *D) { Decls.push_back(D); }
};

class LateParsedAttrList: public SmallVector<LateParsedAttribute *, 2> {
public:
LateParsedAttrList(bool PSoon = false) : ParseSoon(PSoon) { }

bool parseSoon() { return ParseSoon; }

private:
bool ParseSoon;  
};

struct LexedMethod : public LateParsedDeclaration {
Parser *Self;
Decl *D;
CachedTokens Toks;

bool TemplateScope;

explicit LexedMethod(Parser* P, Decl *MD)
: Self(P), D(MD), TemplateScope(false) {}

void ParseLexedMethodDefs() override;
};

struct LateParsedDefaultArgument {
explicit LateParsedDefaultArgument(Decl *P,
std::unique_ptr<CachedTokens> Toks = nullptr)
: Param(P), Toks(std::move(Toks)) { }

Decl *Param;

std::unique_ptr<CachedTokens> Toks;
};

struct LateParsedMethodDeclaration : public LateParsedDeclaration {
explicit LateParsedMethodDeclaration(Parser *P, Decl *M)
: Self(P), Method(M), TemplateScope(false),
ExceptionSpecTokens(nullptr) {}

void ParseLexedMethodDeclarations() override;

Parser* Self;

Decl *Method;

bool TemplateScope;

SmallVector<LateParsedDefaultArgument, 8> DefaultArgs;

CachedTokens *ExceptionSpecTokens;
};

struct LateParsedMemberInitializer : public LateParsedDeclaration {
LateParsedMemberInitializer(Parser *P, Decl *FD)
: Self(P), Field(FD) { }

void ParseLexedMemberInitializers() override;

Parser *Self;

Decl *Field;

CachedTokens Toks;
};

typedef SmallVector<LateParsedDeclaration*,2> LateParsedDeclarationsContainer;

struct ParsingClass {
ParsingClass(Decl *TagOrTemplate, bool TopLevelClass, bool IsInterface)
: TopLevelClass(TopLevelClass), TemplateScope(false),
IsInterface(IsInterface), TagOrTemplate(TagOrTemplate) { }

bool TopLevelClass : 1;

bool TemplateScope : 1;

bool IsInterface : 1;

Decl *TagOrTemplate;

LateParsedDeclarationsContainer LateParsedDeclarations;
};

std::stack<ParsingClass *> ClassStack;

ParsingClass &getCurrentClass() {
assert(!ClassStack.empty() && "No lexed method stacks!");
return *ClassStack.top();
}

class ParsingClassDefinition {
Parser &P;
bool Popped;
Sema::ParsingClassState State;

public:
ParsingClassDefinition(Parser &P, Decl *TagOrTemplate, bool TopLevelClass,
bool IsInterface)
: P(P), Popped(false),
State(P.PushParsingClass(TagOrTemplate, TopLevelClass, IsInterface)) {
}

void Pop() {
assert(!Popped && "Nested class has already been popped");
Popped = true;
P.PopParsingClass(State);
}

~ParsingClassDefinition() {
if (!Popped)
P.PopParsingClass(State);
}
};

struct ParsedTemplateInfo {
ParsedTemplateInfo()
: Kind(NonTemplate), TemplateParams(nullptr), TemplateLoc() { }

ParsedTemplateInfo(TemplateParameterLists *TemplateParams,
bool isSpecialization,
bool lastParameterListWasEmpty = false)
: Kind(isSpecialization? ExplicitSpecialization : Template),
TemplateParams(TemplateParams),
LastParameterListWasEmpty(lastParameterListWasEmpty) { }

explicit ParsedTemplateInfo(SourceLocation ExternLoc,
SourceLocation TemplateLoc)
: Kind(ExplicitInstantiation), TemplateParams(nullptr),
ExternLoc(ExternLoc), TemplateLoc(TemplateLoc),
LastParameterListWasEmpty(false){ }

enum {
NonTemplate = 0,
Template,
ExplicitSpecialization,
ExplicitInstantiation
} Kind;

TemplateParameterLists *TemplateParams;

SourceLocation ExternLoc;

SourceLocation TemplateLoc;

bool LastParameterListWasEmpty;

SourceRange getSourceRange() const LLVM_READONLY;
};

void LexTemplateFunctionForLateParsing(CachedTokens &Toks);
void ParseLateTemplatedFuncDef(LateParsedTemplate &LPT);

static void LateTemplateParserCallback(void *P, LateParsedTemplate &LPT);
static void LateTemplateParserCleanupCallback(void *P);

Sema::ParsingClassState
PushParsingClass(Decl *TagOrTemplate, bool TopLevelClass, bool IsInterface);
void DeallocateParsedClasses(ParsingClass *Class);
void PopParsingClass(Sema::ParsingClassState);

enum CachedInitKind {
CIK_DefaultArgument,
CIK_DefaultInitializer
};

NamedDecl *ParseCXXInlineMethodDef(AccessSpecifier AS,
ParsedAttributes &AccessAttrs,
ParsingDeclarator &D,
const ParsedTemplateInfo &TemplateInfo,
const VirtSpecifiers &VS,
SourceLocation PureSpecLoc);
void ParseCXXNonStaticMemberInitializer(Decl *VarD);
void ParseLexedAttributes(ParsingClass &Class);
void ParseLexedAttributeList(LateParsedAttrList &LAs, Decl *D,
bool EnterScope, bool OnDefinition);
void ParseLexedAttribute(LateParsedAttribute &LA,
bool EnterScope, bool OnDefinition);
void ParseLexedMethodDeclarations(ParsingClass &Class);
void ParseLexedMethodDeclaration(LateParsedMethodDeclaration &LM);
void ParseLexedMethodDefs(ParsingClass &Class);
void ParseLexedMethodDef(LexedMethod &LM);
void ParseLexedMemberInitializers(ParsingClass &Class);
void ParseLexedMemberInitializer(LateParsedMemberInitializer &MI);
void ParseLexedObjCMethodDefs(LexedMethod &LM, bool parseMethod);
bool ConsumeAndStoreFunctionPrologue(CachedTokens &Toks);
bool ConsumeAndStoreInitializer(CachedTokens &Toks, CachedInitKind CIK);
bool ConsumeAndStoreConditional(CachedTokens &Toks);
bool ConsumeAndStoreUntil(tok::TokenKind T1,
CachedTokens &Toks,
bool StopAtSemi = true,
bool ConsumeFinalToken = true) {
return ConsumeAndStoreUntil(T1, T1, Toks, StopAtSemi, ConsumeFinalToken);
}
bool ConsumeAndStoreUntil(tok::TokenKind T1, tok::TokenKind T2,
CachedTokens &Toks,
bool StopAtSemi = true,
bool ConsumeFinalToken = true);

struct ParsedAttributesWithRange : ParsedAttributes {
ParsedAttributesWithRange(AttributeFactory &factory)
: ParsedAttributes(factory) {}

void clear() {
ParsedAttributes::clear();
Range = SourceRange();
}

SourceRange Range;
};
struct ParsedAttributesViewWithRange : ParsedAttributesView {
ParsedAttributesViewWithRange() : ParsedAttributesView() {}
void clearListOnly() {
ParsedAttributesView::clearListOnly();
Range = SourceRange();
}

SourceRange Range;
};

DeclGroupPtrTy ParseExternalDeclaration(ParsedAttributesWithRange &attrs,
ParsingDeclSpec *DS = nullptr);
bool isDeclarationAfterDeclarator();
bool isStartOfFunctionDefinition(const ParsingDeclarator &Declarator);
DeclGroupPtrTy ParseDeclarationOrFunctionDefinition(
ParsedAttributesWithRange &attrs,
ParsingDeclSpec *DS = nullptr,
AccessSpecifier AS = AS_none);
DeclGroupPtrTy ParseDeclOrFunctionDefInternal(ParsedAttributesWithRange &attrs,
ParsingDeclSpec &DS,
AccessSpecifier AS);

void SkipFunctionBody();
Decl *ParseFunctionDefinition(ParsingDeclarator &D,
const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
LateParsedAttrList *LateParsedAttrs = nullptr);
void ParseKNRParamDeclarations(Declarator &D);
ExprResult ParseSimpleAsm(SourceLocation *EndLoc = nullptr);
ExprResult ParseAsmStringLiteral();

void MaybeSkipAttributes(tok::ObjCKeywordKind Kind);
DeclGroupPtrTy ParseObjCAtDirectives(ParsedAttributesWithRange &Attrs);
DeclGroupPtrTy ParseObjCAtClassDeclaration(SourceLocation atLoc);
Decl *ParseObjCAtInterfaceDeclaration(SourceLocation AtLoc,
ParsedAttributes &prefixAttrs);
class ObjCTypeParamListScope;
ObjCTypeParamList *parseObjCTypeParamList();
ObjCTypeParamList *parseObjCTypeParamListOrProtocolRefs(
ObjCTypeParamListScope &Scope, SourceLocation &lAngleLoc,
SmallVectorImpl<IdentifierLocPair> &protocolIdents,
SourceLocation &rAngleLoc, bool mayBeProtocolList = true);

void HelperActionsForIvarDeclarations(Decl *interfaceDecl, SourceLocation atLoc,
BalancedDelimiterTracker &T,
SmallVectorImpl<Decl *> &AllIvarDecls,
bool RBraceMissing);
void ParseObjCClassInstanceVariables(Decl *interfaceDecl,
tok::ObjCKeywordKind visibility,
SourceLocation atLoc);
bool ParseObjCProtocolReferences(SmallVectorImpl<Decl *> &P,
SmallVectorImpl<SourceLocation> &PLocs,
bool WarnOnDeclarations,
bool ForObjCContainer,
SourceLocation &LAngleLoc,
SourceLocation &EndProtoLoc,
bool consumeLastToken);

void parseObjCTypeArgsOrProtocolQualifiers(
ParsedType baseType,
SourceLocation &typeArgsLAngleLoc,
SmallVectorImpl<ParsedType> &typeArgs,
SourceLocation &typeArgsRAngleLoc,
SourceLocation &protocolLAngleLoc,
SmallVectorImpl<Decl *> &protocols,
SmallVectorImpl<SourceLocation> &protocolLocs,
SourceLocation &protocolRAngleLoc,
bool consumeLastToken,
bool warnOnIncompleteProtocols);

void parseObjCTypeArgsAndProtocolQualifiers(
ParsedType baseType,
SourceLocation &typeArgsLAngleLoc,
SmallVectorImpl<ParsedType> &typeArgs,
SourceLocation &typeArgsRAngleLoc,
SourceLocation &protocolLAngleLoc,
SmallVectorImpl<Decl *> &protocols,
SmallVectorImpl<SourceLocation> &protocolLocs,
SourceLocation &protocolRAngleLoc,
bool consumeLastToken);

TypeResult parseObjCProtocolQualifierType(SourceLocation &rAngleLoc);

TypeResult parseObjCTypeArgsAndProtocolQualifiers(SourceLocation loc,
ParsedType type,
bool consumeLastToken,
SourceLocation &endLoc);

void ParseObjCInterfaceDeclList(tok::ObjCKeywordKind contextKey,
Decl *CDecl);
DeclGroupPtrTy ParseObjCAtProtocolDeclaration(SourceLocation atLoc,
ParsedAttributes &prefixAttrs);

struct ObjCImplParsingDataRAII {
Parser &P;
Decl *Dcl;
bool HasCFunction;
typedef SmallVector<LexedMethod*, 8> LateParsedObjCMethodContainer;
LateParsedObjCMethodContainer LateParsedObjCMethods;

ObjCImplParsingDataRAII(Parser &parser, Decl *D)
: P(parser), Dcl(D), HasCFunction(false) {
P.CurParsedObjCImpl = this;
Finished = false;
}
~ObjCImplParsingDataRAII();

void finish(SourceRange AtEnd);
bool isFinished() const { return Finished; }

private:
bool Finished;
};
ObjCImplParsingDataRAII *CurParsedObjCImpl;
void StashAwayMethodOrFunctionBodyTokens(Decl *MDecl);

DeclGroupPtrTy ParseObjCAtImplementationDeclaration(SourceLocation AtLoc);
DeclGroupPtrTy ParseObjCAtEndDeclaration(SourceRange atEnd);
Decl *ParseObjCAtAliasDeclaration(SourceLocation atLoc);
Decl *ParseObjCPropertySynthesize(SourceLocation atLoc);
Decl *ParseObjCPropertyDynamic(SourceLocation atLoc);

IdentifierInfo *ParseObjCSelectorPiece(SourceLocation &MethodLocation);
enum ObjCTypeQual {
objc_in=0, objc_out, objc_inout, objc_oneway, objc_bycopy, objc_byref,
objc_nonnull, objc_nullable, objc_null_unspecified,
objc_NumQuals
};
IdentifierInfo *ObjCTypeQuals[objc_NumQuals];

bool isTokIdentifier_in() const;

ParsedType ParseObjCTypeName(ObjCDeclSpec &DS, DeclaratorContext Ctx,
ParsedAttributes *ParamAttrs);
void ParseObjCMethodRequirement();
Decl *ParseObjCMethodPrototype(
tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword,
bool MethodDefinition = true);
Decl *ParseObjCMethodDecl(SourceLocation mLoc, tok::TokenKind mType,
tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword,
bool MethodDefinition=true);
void ParseObjCPropertyAttribute(ObjCDeclSpec &DS);

Decl *ParseObjCMethodDefinition();

public:

enum TypeCastState {
NotTypeCast = 0,
MaybeTypeCast,
IsTypeCast
};

ExprResult ParseExpression(TypeCastState isTypeCast = NotTypeCast);
ExprResult ParseConstantExpressionInExprEvalContext(
TypeCastState isTypeCast = NotTypeCast);
ExprResult ParseConstantExpression(TypeCastState isTypeCast = NotTypeCast);
ExprResult ParseCaseExpression(SourceLocation CaseLoc);
ExprResult ParseConstraintExpression();
ExprResult ParseAssignmentExpression(TypeCastState isTypeCast = NotTypeCast);

ExprResult ParseMSAsmIdentifier(llvm::SmallVectorImpl<Token> &LineToks,
unsigned &NumLineToksConsumed,
bool IsUnevaluated);

private:
ExprResult ParseExpressionWithLeadingAt(SourceLocation AtLoc);

ExprResult ParseExpressionWithLeadingExtension(SourceLocation ExtLoc);

ExprResult ParseRHSOfBinaryExpression(ExprResult LHS,
prec::Level MinPrec);
ExprResult ParseCastExpression(bool isUnaryExpression,
bool isAddressOfOperand,
bool &NotCastExpr,
TypeCastState isTypeCast,
bool isVectorLiteral = false);
ExprResult ParseCastExpression(bool isUnaryExpression,
bool isAddressOfOperand = false,
TypeCastState isTypeCast = NotTypeCast,
bool isVectorLiteral = false);

bool isNotExpressionStart();

bool isPostfixExpressionSuffixStart() {
tok::TokenKind K = Tok.getKind();
return (K == tok::l_square || K == tok::l_paren ||
K == tok::period || K == tok::arrow ||
K == tok::plusplus || K == tok::minusminus);
}

bool diagnoseUnknownTemplateId(ExprResult TemplateName, SourceLocation Less);
void checkPotentialAngleBracket(ExprResult &PotentialTemplateName);
bool checkPotentialAngleBracketDelimiter(const AngleBracketTracker::Loc &,
const Token &OpToken);
bool checkPotentialAngleBracketDelimiter(const Token &OpToken) {
if (auto *Info = AngleBrackets.getCurrent(*this))
return checkPotentialAngleBracketDelimiter(*Info, OpToken);
return false;
}

ExprResult ParsePostfixExpressionSuffix(ExprResult LHS);
ExprResult ParseUnaryExprOrTypeTraitExpression();
ExprResult ParseBuiltinPrimaryExpression();

ExprResult ParseExprAfterUnaryExprOrTypeTrait(const Token &OpTok,
bool &isCastExpr,
ParsedType &CastTy,
SourceRange &CastRange);

typedef SmallVector<Expr*, 20> ExprListTy;
typedef SmallVector<SourceLocation, 20> CommaLocsTy;

bool ParseExpressionList(
SmallVectorImpl<Expr *> &Exprs,
SmallVectorImpl<SourceLocation> &CommaLocs,
llvm::function_ref<void()> Completer = llvm::function_ref<void()>());

bool ParseSimpleExpressionList(SmallVectorImpl<Expr*> &Exprs,
SmallVectorImpl<SourceLocation> &CommaLocs);


enum ParenParseOption {
SimpleExpr,      
FoldExpr,        
CompoundStmt,    
CompoundLiteral, 
CastExpr         
};
ExprResult ParseParenExpression(ParenParseOption &ExprType,
bool stopIfCastExpr,
bool isTypeCast,
ParsedType &CastTy,
SourceLocation &RParenLoc);

ExprResult ParseCXXAmbiguousParenExpression(
ParenParseOption &ExprType, ParsedType &CastTy,
BalancedDelimiterTracker &Tracker, ColonProtectionRAIIObject &ColonProt);
ExprResult ParseCompoundLiteralExpression(ParsedType Ty,
SourceLocation LParenLoc,
SourceLocation RParenLoc);

ExprResult ParseStringLiteralExpression(bool AllowUserDefinedLiteral = false);

ExprResult ParseGenericSelectionExpression();

ExprResult ParseObjCBoolLiteral();

ExprResult ParseFoldExpression(ExprResult LHS, BalancedDelimiterTracker &T);

ExprResult tryParseCXXIdExpression(CXXScopeSpec &SS, bool isAddressOfOperand,
Token &Replacement);
ExprResult ParseCXXIdExpression(bool isAddressOfOperand = false);

bool areTokensAdjacent(const Token &A, const Token &B);

void CheckForTemplateAndDigraph(Token &Next, ParsedType ObjectTypePtr,
bool EnteringContext, IdentifierInfo &II,
CXXScopeSpec &SS);

bool ParseOptionalCXXScopeSpecifier(CXXScopeSpec &SS,
ParsedType ObjectType,
bool EnteringContext,
bool *MayBePseudoDestructor = nullptr,
bool IsTypename = false,
IdentifierInfo **LastII = nullptr,
bool OnlyNamespace = false);


ExprResult ParseLambdaExpression();
ExprResult TryParseLambdaExpression();
Optional<unsigned> ParseLambdaIntroducer(LambdaIntroducer &Intro,
bool *SkippedInits = nullptr);
bool TryParseLambdaIntroducer(LambdaIntroducer &Intro);
ExprResult ParseLambdaExpressionAfterIntroducer(
LambdaIntroducer &Intro);

ExprResult ParseCXXCasts();

ExprResult ParseCXXTypeid();

ExprResult ParseCXXUuidof();

ExprResult ParseCXXPseudoDestructor(Expr *Base, SourceLocation OpLoc,
tok::TokenKind OpKind,
CXXScopeSpec &SS,
ParsedType ObjectType);

ExprResult ParseCXXThis();

ExprResult ParseThrowExpression();

ExceptionSpecificationType tryParseExceptionSpecification(
bool Delayed,
SourceRange &SpecificationRange,
SmallVectorImpl<ParsedType> &DynamicExceptions,
SmallVectorImpl<SourceRange> &DynamicExceptionRanges,
ExprResult &NoexceptExpr,
CachedTokens *&ExceptionSpecTokens);

ExceptionSpecificationType ParseDynamicExceptionSpecification(
SourceRange &SpecificationRange,
SmallVectorImpl<ParsedType> &Exceptions,
SmallVectorImpl<SourceRange> &Ranges);

TypeResult ParseTrailingReturnType(SourceRange &Range,
bool MayBeFollowedByDirectInit);

ExprResult ParseCXXBoolLiteral();

ExprResult ParseCXXTypeConstructExpression(const DeclSpec &DS);

void ParseCXXSimpleTypeSpecifier(DeclSpec &DS);

bool ParseCXXTypeSpecifierSeq(DeclSpec &DS);

bool ParseExpressionListOrTypeId(SmallVectorImpl<Expr*> &Exprs,
Declarator &D);
void ParseDirectNewDeclarator(Declarator &D);
ExprResult ParseCXXNewExpression(bool UseGlobal, SourceLocation Start);
ExprResult ParseCXXDeleteExpression(bool UseGlobal,
SourceLocation Start);

Sema::ConditionResult ParseCXXCondition(StmtResult *InitStmt,
SourceLocation Loc,
Sema::ConditionKind CK);


ExprResult ParseCoyieldExpression();


ExprResult ParseInitializer() {
if (Tok.isNot(tok::l_brace))
return ParseAssignmentExpression();
return ParseBraceInitializer();
}
bool MayBeDesignationStart();
ExprResult ParseBraceInitializer();
ExprResult ParseInitializerWithPotentialDesignator();


ExprResult ParseBlockLiteralExpression();  

ExprResult ParseObjCAtExpression(SourceLocation AtLocation);
ExprResult ParseObjCStringLiteral(SourceLocation AtLoc);
ExprResult ParseObjCCharacterLiteral(SourceLocation AtLoc);
ExprResult ParseObjCNumericLiteral(SourceLocation AtLoc);
ExprResult ParseObjCBooleanLiteral(SourceLocation AtLoc, bool ArgValue);
ExprResult ParseObjCArrayLiteral(SourceLocation AtLoc);
ExprResult ParseObjCDictionaryLiteral(SourceLocation AtLoc);
ExprResult ParseObjCBoxedExpr(SourceLocation AtLoc);
ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc);
ExprResult ParseObjCSelectorExpression(SourceLocation AtLoc);
ExprResult ParseObjCProtocolExpression(SourceLocation AtLoc);
bool isSimpleObjCMessageExpression();
ExprResult ParseObjCMessageExpression();
ExprResult ParseObjCMessageExpressionBody(SourceLocation LBracloc,
SourceLocation SuperLoc,
ParsedType ReceiverType,
Expr *ReceiverExpr);
ExprResult ParseAssignmentExprWithObjCMessageExprStart(
SourceLocation LBracloc, SourceLocation SuperLoc,
ParsedType ReceiverType, Expr *ReceiverExpr);
bool ParseObjCXXMessageReceiver(bool &IsExpr, void *&TypeOrExpr);


typedef SmallVector<Stmt*, 32> StmtVector;
typedef SmallVector<Expr*, 12> ExprVector;
typedef SmallVector<ParsedType, 12> TypeVector;

StmtResult ParseStatement(SourceLocation *TrailingElseLoc = nullptr,
bool AllowOpenMPStandalone = false);
enum AllowedConstructsKind {
ACK_Any,
ACK_StatementsOpenMPNonStandalone,
ACK_StatementsOpenMPAnyExecutable
};
StmtResult
ParseStatementOrDeclaration(StmtVector &Stmts, AllowedConstructsKind Allowed,
SourceLocation *TrailingElseLoc = nullptr);
StmtResult ParseStatementOrDeclarationAfterAttributes(
StmtVector &Stmts,
AllowedConstructsKind Allowed,
SourceLocation *TrailingElseLoc,
ParsedAttributesWithRange &Attrs);
StmtResult ParseExprStatement();
StmtResult ParseLabeledStatement(ParsedAttributesWithRange &attrs);
StmtResult ParseCaseStatement(bool MissingCase = false,
ExprResult Expr = ExprResult());
StmtResult ParseDefaultStatement();
StmtResult ParseCompoundStatement(bool isStmtExpr = false);
StmtResult ParseCompoundStatement(bool isStmtExpr,
unsigned ScopeFlags);
void ParseCompoundStatementLeadingPragmas();
StmtResult ParseCompoundStatementBody(bool isStmtExpr = false);
bool ParseParenExprOrCondition(StmtResult *InitStmt,
Sema::ConditionResult &CondResult,
SourceLocation Loc,
Sema::ConditionKind CK);
StmtResult ParseIfStatement(SourceLocation *TrailingElseLoc);
StmtResult ParseSwitchStatement(SourceLocation *TrailingElseLoc);
StmtResult ParseWhileStatement(SourceLocation *TrailingElseLoc);
StmtResult ParseDoStatement();
StmtResult ParseForStatement(SourceLocation *TrailingElseLoc);
StmtResult ParseGotoStatement();
StmtResult ParseContinueStatement();
StmtResult ParseBreakStatement();
StmtResult ParseReturnStatement();
StmtResult ParseAsmStatement(bool &msAsm);
StmtResult ParseMicrosoftAsmStatement(SourceLocation AsmLoc);
StmtResult ParsePragmaLoopHint(StmtVector &Stmts,
AllowedConstructsKind Allowed,
SourceLocation *TrailingElseLoc,
ParsedAttributesWithRange &Attrs);

enum IfExistsBehavior {
IEB_Parse,
IEB_Skip,
IEB_Dependent
};

struct IfExistsCondition {
SourceLocation KeywordLoc;
bool IsIfExists;

CXXScopeSpec SS;

UnqualifiedId Name;

IfExistsBehavior Behavior;
};

bool ParseMicrosoftIfExistsCondition(IfExistsCondition& Result);
void ParseMicrosoftIfExistsStatement(StmtVector &Stmts);
void ParseMicrosoftIfExistsExternalDeclaration();
void ParseMicrosoftIfExistsClassDeclaration(DeclSpec::TST TagType,
ParsedAttributes &AccessAttrs,
AccessSpecifier &CurAS);
bool ParseMicrosoftIfExistsBraceInitializer(ExprVector &InitExprs,
bool &InitExprsOk);
bool ParseAsmOperandsOpt(SmallVectorImpl<IdentifierInfo *> &Names,
SmallVectorImpl<Expr *> &Constraints,
SmallVectorImpl<Expr *> &Exprs);


StmtResult ParseCXXTryBlock();
StmtResult ParseCXXTryBlockCommon(SourceLocation TryLoc, bool FnTry = false);
StmtResult ParseCXXCatchBlock(bool FnCatch = false);


StmtResult ParseSEHTryBlock();
StmtResult ParseSEHExceptBlock(SourceLocation Loc);
StmtResult ParseSEHFinallyBlock(SourceLocation Loc);
StmtResult ParseSEHLeaveStatement();


StmtResult ParseObjCAtStatement(SourceLocation atLoc);
StmtResult ParseObjCTryStmt(SourceLocation atLoc);
StmtResult ParseObjCThrowStmt(SourceLocation atLoc);
StmtResult ParseObjCSynchronizedStmt(SourceLocation atLoc);
StmtResult ParseObjCAutoreleasePoolStmt(SourceLocation atLoc);



enum class DeclSpecContext {
DSC_normal, 
DSC_class,  
DSC_type_specifier, 
DSC_trailing, 
DSC_alias_declaration, 
DSC_top_level, 
DSC_template_param, 
DSC_template_type_arg, 
DSC_objc_method_result, 
DSC_condition 
};

static bool isTypeSpecifier(DeclSpecContext DSC) {
switch (DSC) {
case DeclSpecContext::DSC_normal:
case DeclSpecContext::DSC_template_param:
case DeclSpecContext::DSC_class:
case DeclSpecContext::DSC_top_level:
case DeclSpecContext::DSC_objc_method_result:
case DeclSpecContext::DSC_condition:
return false;

case DeclSpecContext::DSC_template_type_arg:
case DeclSpecContext::DSC_type_specifier:
case DeclSpecContext::DSC_trailing:
case DeclSpecContext::DSC_alias_declaration:
return true;
}
llvm_unreachable("Missing DeclSpecContext case");
}

static bool isClassTemplateDeductionContext(DeclSpecContext DSC) {
switch (DSC) {
case DeclSpecContext::DSC_normal:
case DeclSpecContext::DSC_template_param:
case DeclSpecContext::DSC_class:
case DeclSpecContext::DSC_top_level:
case DeclSpecContext::DSC_condition:
case DeclSpecContext::DSC_type_specifier:
return true;

case DeclSpecContext::DSC_objc_method_result:
case DeclSpecContext::DSC_template_type_arg:
case DeclSpecContext::DSC_trailing:
case DeclSpecContext::DSC_alias_declaration:
return false;
}
llvm_unreachable("Missing DeclSpecContext case");
}

struct ForRangeInit {
SourceLocation ColonLoc;
ExprResult RangeExpr;

bool ParsedForRangeDecl() { return !ColonLoc.isInvalid(); }
};

DeclGroupPtrTy ParseDeclaration(DeclaratorContext Context,
SourceLocation &DeclEnd,
ParsedAttributesWithRange &attrs);
DeclGroupPtrTy ParseSimpleDeclaration(DeclaratorContext Context,
SourceLocation &DeclEnd,
ParsedAttributesWithRange &attrs,
bool RequireSemi,
ForRangeInit *FRI = nullptr);
bool MightBeDeclarator(DeclaratorContext Context);
DeclGroupPtrTy ParseDeclGroup(ParsingDeclSpec &DS, DeclaratorContext Context,
SourceLocation *DeclEnd = nullptr,
ForRangeInit *FRI = nullptr);
Decl *ParseDeclarationAfterDeclarator(Declarator &D,
const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo());
bool ParseAsmAttributesAfterDeclarator(Declarator &D);
Decl *ParseDeclarationAfterDeclaratorAndAttributes(
Declarator &D,
const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
ForRangeInit *FRI = nullptr);
Decl *ParseFunctionStatementBody(Decl *Decl, ParseScope &BodyScope);
Decl *ParseFunctionTryBlock(Decl *Decl, ParseScope &BodyScope);

bool trySkippingFunctionBody();

bool ParseImplicitInt(DeclSpec &DS, CXXScopeSpec *SS,
const ParsedTemplateInfo &TemplateInfo,
AccessSpecifier AS, DeclSpecContext DSC,
ParsedAttributesWithRange &Attrs);
DeclSpecContext
getDeclSpecContextFromDeclaratorContext(DeclaratorContext Context);
void ParseDeclarationSpecifiers(
DeclSpec &DS,
const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
AccessSpecifier AS = AS_none,
DeclSpecContext DSC = DeclSpecContext::DSC_normal,
LateParsedAttrList *LateAttrs = nullptr);
bool DiagnoseMissingSemiAfterTagDefinition(
DeclSpec &DS, AccessSpecifier AS, DeclSpecContext DSContext,
LateParsedAttrList *LateAttrs = nullptr);

void ParseSpecifierQualifierList(
DeclSpec &DS, AccessSpecifier AS = AS_none,
DeclSpecContext DSC = DeclSpecContext::DSC_normal);

void ParseObjCTypeQualifierList(ObjCDeclSpec &DS,
DeclaratorContext Context);

void ParseEnumSpecifier(SourceLocation TagLoc, DeclSpec &DS,
const ParsedTemplateInfo &TemplateInfo,
AccessSpecifier AS, DeclSpecContext DSC);
void ParseEnumBody(SourceLocation StartLoc, Decl *TagDecl);
void ParseStructUnionBody(SourceLocation StartLoc, unsigned TagType,
Decl *TagDecl);

void ParseStructDeclaration(
ParsingDeclSpec &DS,
llvm::function_ref<void(ParsingFieldDeclarator &)> FieldsCallback);

bool isDeclarationSpecifier(bool DisambiguatingWithExpression = false);
bool isTypeSpecifierQualifier();

bool isKnownToBeTypeSpecifier(const Token &Tok) const;

bool isKnownToBeDeclarationSpecifier() {
if (getLangOpts().CPlusPlus)
return isCXXDeclarationSpecifier() == TPResult::True;
return isDeclarationSpecifier(true);
}

bool isDeclarationStatement() {
if (getLangOpts().CPlusPlus)
return isCXXDeclarationStatement();
return isDeclarationSpecifier(true);
}

bool isForInitDeclaration() {
if (getLangOpts().CPlusPlus)
return isCXXSimpleDeclaration(true);
return isDeclarationSpecifier(true);
}

bool isForRangeIdentifier();

bool isStartOfObjCClassMessageMissingOpenBracket();

bool isConstructorDeclarator(bool Unqualified, bool DeductionGuide = false);

enum TentativeCXXTypeIdContext {
TypeIdInParens,
TypeIdUnambiguous,
TypeIdAsTemplateArgument
};


bool isTypeIdInParens(bool &isAmbiguous) {
if (getLangOpts().CPlusPlus)
return isCXXTypeId(TypeIdInParens, isAmbiguous);
isAmbiguous = false;
return isTypeSpecifierQualifier();
}
bool isTypeIdInParens() {
bool isAmbiguous;
return isTypeIdInParens(isAmbiguous);
}

bool isTypeIdUnambiguously() {
bool IsAmbiguous;
if (getLangOpts().CPlusPlus)
return isCXXTypeId(TypeIdUnambiguous, IsAmbiguous);
return isTypeSpecifierQualifier();
}

bool isCXXDeclarationStatement();

bool isCXXSimpleDeclaration(bool AllowForRangeDecl);

bool isCXXFunctionDeclarator(bool *IsAmbiguous = nullptr);

struct ConditionDeclarationOrInitStatementState;
enum class ConditionOrInitStatement {
Expression,    
ConditionDecl, 
InitStmtDecl,  
Error          
};
ConditionOrInitStatement
isCXXConditionDeclarationOrInitStatement(bool CanBeInitStmt);

bool isCXXTypeId(TentativeCXXTypeIdContext Context, bool &isAmbiguous);
bool isCXXTypeId(TentativeCXXTypeIdContext Context) {
bool isAmbiguous;
return isCXXTypeId(Context, isAmbiguous);
}

enum class TPResult {
True, False, Ambiguous, Error
};

TPResult isExpressionOrTypeSpecifierSimple(tok::TokenKind Kind);

TPResult
isCXXDeclarationSpecifier(TPResult BracedCastResult = TPResult::False,
bool *HasMissingTypename = nullptr);

bool isCXXDeclarationSpecifierAType();

bool isTentativelyDeclared(IdentifierInfo *II);


TPResult TryParseSimpleDeclaration(bool AllowForRangeDecl);
TPResult TryParseTypeofSpecifier();
TPResult TryParseProtocolQualifiers();
TPResult TryParsePtrOperatorSeq();
TPResult TryParseOperatorId();
TPResult TryParseInitDeclaratorList();
TPResult TryParseDeclarator(bool mayBeAbstract, bool mayHaveIdentifier = true,
bool mayHaveDirectInit = false);
TPResult
TryParseParameterDeclarationClause(bool *InvalidAsDeclaration = nullptr,
bool VersusTemplateArg = false);
TPResult TryParseFunctionDeclarator();
TPResult TryParseBracketDeclarator();
TPResult TryConsumeDeclarationSpecifier();

public:
TypeResult ParseTypeName(SourceRange *Range = nullptr,
DeclaratorContext Context
= DeclaratorContext::TypeNameContext,
AccessSpecifier AS = AS_none,
Decl **OwnedType = nullptr,
ParsedAttributes *Attrs = nullptr);

private:
void ParseBlockId(SourceLocation CaretLoc);

bool standardAttributesAllowed() const {
const LangOptions &LO = getLangOpts();
return LO.DoubleSquareBracketAttributes;
}

bool CheckProhibitedCXX11Attribute() {
assert(Tok.is(tok::l_square));
if (!standardAttributesAllowed() || NextToken().isNot(tok::l_square))
return false;
return DiagnoseProhibitedCXX11Attribute();
}

bool DiagnoseProhibitedCXX11Attribute();
void CheckMisplacedCXX11Attribute(ParsedAttributesWithRange &Attrs,
SourceLocation CorrectLocation) {
if (!standardAttributesAllowed())
return;
if ((Tok.isNot(tok::l_square) || NextToken().isNot(tok::l_square)) &&
Tok.isNot(tok::kw_alignas))
return;
DiagnoseMisplacedCXX11Attribute(Attrs, CorrectLocation);
}
void DiagnoseMisplacedCXX11Attribute(ParsedAttributesWithRange &Attrs,
SourceLocation CorrectLocation);

void stripTypeAttributesOffDeclSpec(ParsedAttributesWithRange &Attrs,
DeclSpec &DS, Sema::TagUseKind TUK);

void ProhibitAttributes(ParsedAttributesWithRange &Attrs,
SourceLocation FixItLoc = SourceLocation()) {
if (Attrs.Range.isInvalid())
return;
DiagnoseProhibitedAttributes(Attrs.Range, FixItLoc);
Attrs.clear();
}

void ProhibitAttributes(ParsedAttributesViewWithRange &Attrs,
SourceLocation FixItLoc = SourceLocation()) {
if (Attrs.Range.isInvalid())
return;
DiagnoseProhibitedAttributes(Attrs.Range, FixItLoc);
Attrs.clearListOnly();
}
void DiagnoseProhibitedAttributes(const SourceRange &Range,
SourceLocation FixItLoc);

void ProhibitCXX11Attributes(ParsedAttributesWithRange &Attrs,
unsigned DiagID);

SourceLocation SkipCXX11Attributes();

void DiagnoseAndSkipCXX11Attributes();

unsigned
ParseAttributeArgsCommon(IdentifierInfo *AttrName, SourceLocation AttrNameLoc,
ParsedAttributes &Attrs, SourceLocation *EndLoc,
IdentifierInfo *ScopeName, SourceLocation ScopeLoc,
ParsedAttr::Syntax Syntax);

void MaybeParseGNUAttributes(Declarator &D,
LateParsedAttrList *LateAttrs = nullptr) {
if (Tok.is(tok::kw___attribute)) {
ParsedAttributes attrs(AttrFactory);
SourceLocation endLoc;
ParseGNUAttributes(attrs, &endLoc, LateAttrs, &D);
D.takeAttributes(attrs, endLoc);
}
}
void MaybeParseGNUAttributes(ParsedAttributes &attrs,
SourceLocation *endLoc = nullptr,
LateParsedAttrList *LateAttrs = nullptr) {
if (Tok.is(tok::kw___attribute))
ParseGNUAttributes(attrs, endLoc, LateAttrs);
}
void ParseGNUAttributes(ParsedAttributes &attrs,
SourceLocation *endLoc = nullptr,
LateParsedAttrList *LateAttrs = nullptr,
Declarator *D = nullptr);
void ParseGNUAttributeArgs(IdentifierInfo *AttrName,
SourceLocation AttrNameLoc,
ParsedAttributes &Attrs, SourceLocation *EndLoc,
IdentifierInfo *ScopeName, SourceLocation ScopeLoc,
ParsedAttr::Syntax Syntax, Declarator *D);
IdentifierLoc *ParseIdentifierLoc();

unsigned
ParseClangAttributeArgs(IdentifierInfo *AttrName, SourceLocation AttrNameLoc,
ParsedAttributes &Attrs, SourceLocation *EndLoc,
IdentifierInfo *ScopeName, SourceLocation ScopeLoc,
ParsedAttr::Syntax Syntax);

void MaybeParseCXX11Attributes(Declarator &D) {
if (standardAttributesAllowed() && isCXX11AttributeSpecifier()) {
ParsedAttributesWithRange attrs(AttrFactory);
SourceLocation endLoc;
ParseCXX11Attributes(attrs, &endLoc);
D.takeAttributes(attrs, endLoc);
}
}
void MaybeParseCXX11Attributes(ParsedAttributes &attrs,
SourceLocation *endLoc = nullptr) {
if (standardAttributesAllowed() && isCXX11AttributeSpecifier()) {
ParsedAttributesWithRange attrsWithRange(AttrFactory);
ParseCXX11Attributes(attrsWithRange, endLoc);
attrs.takeAllFrom(attrsWithRange);
}
}
void MaybeParseCXX11Attributes(ParsedAttributesWithRange &attrs,
SourceLocation *endLoc = nullptr,
bool OuterMightBeMessageSend = false) {
if (standardAttributesAllowed() &&
isCXX11AttributeSpecifier(false, OuterMightBeMessageSend))
ParseCXX11Attributes(attrs, endLoc);
}

void ParseCXX11AttributeSpecifier(ParsedAttributes &attrs,
SourceLocation *EndLoc = nullptr);
void ParseCXX11Attributes(ParsedAttributesWithRange &attrs,
SourceLocation *EndLoc = nullptr);
bool ParseCXX11AttributeArgs(IdentifierInfo *AttrName,
SourceLocation AttrNameLoc,
ParsedAttributes &Attrs, SourceLocation *EndLoc,
IdentifierInfo *ScopeName,
SourceLocation ScopeLoc);

IdentifierInfo *TryParseCXX11AttributeIdentifier(SourceLocation &Loc);

void MaybeParseMicrosoftAttributes(ParsedAttributes &attrs,
SourceLocation *endLoc = nullptr) {
if (getLangOpts().MicrosoftExt && Tok.is(tok::l_square))
ParseMicrosoftAttributes(attrs, endLoc);
}
void ParseMicrosoftUuidAttributeArgs(ParsedAttributes &Attrs);
void ParseMicrosoftAttributes(ParsedAttributes &attrs,
SourceLocation *endLoc = nullptr);
void MaybeParseMicrosoftDeclSpecs(ParsedAttributes &Attrs,
SourceLocation *End = nullptr) {
const auto &LO = getLangOpts();
if (LO.DeclSpecKeyword && Tok.is(tok::kw___declspec))
ParseMicrosoftDeclSpecs(Attrs, End);
}
void ParseMicrosoftDeclSpecs(ParsedAttributes &Attrs,
SourceLocation *End = nullptr);
bool ParseMicrosoftDeclSpecArgs(IdentifierInfo *AttrName,
SourceLocation AttrNameLoc,
ParsedAttributes &Attrs);
void ParseMicrosoftTypeAttributes(ParsedAttributes &attrs);
void DiagnoseAndSkipExtendedMicrosoftTypeAttributes();
SourceLocation SkipExtendedMicrosoftTypeAttributes();
void ParseMicrosoftInheritanceClassAttributes(ParsedAttributes &attrs);
void ParseBorlandTypeAttributes(ParsedAttributes &attrs);
void ParseOpenCLKernelAttributes(ParsedAttributes &attrs);
void ParseOpenCLQualifiers(ParsedAttributes &Attrs);
bool MaybeParseOpenCLUnrollHintAttribute(ParsedAttributes &Attrs) {
if (getLangOpts().OpenCL)
return ParseOpenCLUnrollHintAttribute(Attrs);
return true;
}
bool ParseOpenCLUnrollHintAttribute(ParsedAttributes &Attrs);
void ParseNullabilityTypeSpecifiers(ParsedAttributes &attrs);

VersionTuple ParseVersionTuple(SourceRange &Range);
void ParseAvailabilityAttribute(IdentifierInfo &Availability,
SourceLocation AvailabilityLoc,
ParsedAttributes &attrs,
SourceLocation *endLoc,
IdentifierInfo *ScopeName,
SourceLocation ScopeLoc,
ParsedAttr::Syntax Syntax);

Optional<AvailabilitySpec> ParseAvailabilitySpec();
ExprResult ParseAvailabilityCheckExpr(SourceLocation StartLoc);

void ParseExternalSourceSymbolAttribute(IdentifierInfo &ExternalSourceSymbol,
SourceLocation Loc,
ParsedAttributes &Attrs,
SourceLocation *EndLoc,
IdentifierInfo *ScopeName,
SourceLocation ScopeLoc,
ParsedAttr::Syntax Syntax);

void ParseObjCBridgeRelatedAttribute(IdentifierInfo &ObjCBridgeRelated,
SourceLocation ObjCBridgeRelatedLoc,
ParsedAttributes &attrs,
SourceLocation *endLoc,
IdentifierInfo *ScopeName,
SourceLocation ScopeLoc,
ParsedAttr::Syntax Syntax);

void ParseTypeTagForDatatypeAttribute(IdentifierInfo &AttrName,
SourceLocation AttrNameLoc,
ParsedAttributes &Attrs,
SourceLocation *EndLoc,
IdentifierInfo *ScopeName,
SourceLocation ScopeLoc,
ParsedAttr::Syntax Syntax);

void
ParseAttributeWithTypeArg(IdentifierInfo &AttrName,
SourceLocation AttrNameLoc, ParsedAttributes &Attrs,
SourceLocation *EndLoc, IdentifierInfo *ScopeName,
SourceLocation ScopeLoc, ParsedAttr::Syntax Syntax);

void ParseTypeofSpecifier(DeclSpec &DS);
SourceLocation ParseDecltypeSpecifier(DeclSpec &DS);
void AnnotateExistingDecltypeSpecifier(const DeclSpec &DS,
SourceLocation StartLoc,
SourceLocation EndLoc);
void ParseUnderlyingTypeSpecifier(DeclSpec &DS);
void ParseAtomicSpecifier(DeclSpec &DS);

ExprResult ParseAlignArgument(SourceLocation Start,
SourceLocation &EllipsisLoc);
void ParseAlignmentSpecifier(ParsedAttributes &Attrs,
SourceLocation *endLoc = nullptr);

VirtSpecifiers::Specifier isCXX11VirtSpecifier(const Token &Tok) const;
VirtSpecifiers::Specifier isCXX11VirtSpecifier() const {
return isCXX11VirtSpecifier(Tok);
}
void ParseOptionalCXX11VirtSpecifierSeq(VirtSpecifiers &VS, bool IsInterface,
SourceLocation FriendLoc);

bool isCXX11FinalKeyword() const;

class DeclaratorScopeObj {
Parser &P;
CXXScopeSpec &SS;
bool EnteredScope;
bool CreatedScope;
public:
DeclaratorScopeObj(Parser &p, CXXScopeSpec &ss)
: P(p), SS(ss), EnteredScope(false), CreatedScope(false) {}

void EnterDeclaratorScope() {
assert(!EnteredScope && "Already entered the scope!");
assert(SS.isSet() && "C++ scope was not set!");

CreatedScope = true;
P.EnterScope(0); 

if (!P.Actions.ActOnCXXEnterDeclaratorScope(P.getCurScope(), SS))
EnteredScope = true;
}

~DeclaratorScopeObj() {
if (EnteredScope) {
assert(SS.isSet() && "C++ scope was cleared ?");
P.Actions.ActOnCXXExitDeclaratorScope(P.getCurScope(), SS);
}
if (CreatedScope)
P.ExitScope();
}
};

void ParseDeclarator(Declarator &D);
typedef void (Parser::*DirectDeclParseFunction)(Declarator&);
void ParseDeclaratorInternal(Declarator &D,
DirectDeclParseFunction DirectDeclParser);

enum AttrRequirements {
AR_NoAttributesParsed = 0, 
AR_GNUAttributesParsedAndRejected = 1 << 0, 
AR_GNUAttributesParsed = 1 << 1,
AR_CXX11AttributesParsed = 1 << 2,
AR_DeclspecAttributesParsed = 1 << 3,
AR_AllAttributesParsed = AR_GNUAttributesParsed |
AR_CXX11AttributesParsed |
AR_DeclspecAttributesParsed,
AR_VendorAttributesParsed = AR_GNUAttributesParsed |
AR_DeclspecAttributesParsed
};

void ParseTypeQualifierListOpt(
DeclSpec &DS, unsigned AttrReqs = AR_AllAttributesParsed,
bool AtomicAllowed = true, bool IdentifierRequired = false,
Optional<llvm::function_ref<void()>> CodeCompletionHandler = None);
void ParseDirectDeclarator(Declarator &D);
void ParseDecompositionDeclarator(Declarator &D);
void ParseParenDeclarator(Declarator &D);
void ParseFunctionDeclarator(Declarator &D,
ParsedAttributes &attrs,
BalancedDelimiterTracker &Tracker,
bool IsAmbiguous,
bool RequiresArg = false);
bool ParseRefQualifier(bool &RefQualifierIsLValueRef,
SourceLocation &RefQualifierLoc);
bool isFunctionDeclaratorIdentifierList();
void ParseFunctionDeclaratorIdentifierList(
Declarator &D,
SmallVectorImpl<DeclaratorChunk::ParamInfo> &ParamInfo);
void ParseParameterDeclarationClause(
Declarator &D,
ParsedAttributes &attrs,
SmallVectorImpl<DeclaratorChunk::ParamInfo> &ParamInfo,
SourceLocation &EllipsisLoc);
void ParseBracketDeclarator(Declarator &D);
void ParseMisplacedBracketDeclarator(Declarator &D);


enum CXX11AttributeKind {
CAK_NotAttributeSpecifier,
CAK_AttributeSpecifier,
CAK_InvalidAttributeSpecifier
};
CXX11AttributeKind
isCXX11AttributeSpecifier(bool Disambiguate = false,
bool OuterMightBeMessageSend = false);

void DiagnoseUnexpectedNamespace(NamedDecl *Context);

DeclGroupPtrTy ParseNamespace(DeclaratorContext Context,
SourceLocation &DeclEnd,
SourceLocation InlineLoc = SourceLocation());
void ParseInnerNamespace(std::vector<SourceLocation> &IdentLoc,
std::vector<IdentifierInfo *> &Ident,
std::vector<SourceLocation> &NamespaceLoc,
unsigned int index, SourceLocation &InlineLoc,
ParsedAttributes &attrs,
BalancedDelimiterTracker &Tracker);
Decl *ParseLinkage(ParsingDeclSpec &DS, DeclaratorContext Context);
Decl *ParseExportDeclaration();
DeclGroupPtrTy ParseUsingDirectiveOrDeclaration(
DeclaratorContext Context, const ParsedTemplateInfo &TemplateInfo,
SourceLocation &DeclEnd, ParsedAttributesWithRange &attrs);
Decl *ParseUsingDirective(DeclaratorContext Context,
SourceLocation UsingLoc,
SourceLocation &DeclEnd,
ParsedAttributes &attrs);

struct UsingDeclarator {
SourceLocation TypenameLoc;
CXXScopeSpec SS;
UnqualifiedId Name;
SourceLocation EllipsisLoc;

void clear() {
TypenameLoc = EllipsisLoc = SourceLocation();
SS.clear();
Name.clear();
}
};

bool ParseUsingDeclarator(DeclaratorContext Context, UsingDeclarator &D);
DeclGroupPtrTy ParseUsingDeclaration(DeclaratorContext Context,
const ParsedTemplateInfo &TemplateInfo,
SourceLocation UsingLoc,
SourceLocation &DeclEnd,
AccessSpecifier AS = AS_none);
Decl *ParseAliasDeclarationAfterDeclarator(
const ParsedTemplateInfo &TemplateInfo, SourceLocation UsingLoc,
UsingDeclarator &D, SourceLocation &DeclEnd, AccessSpecifier AS,
ParsedAttributes &Attrs, Decl **OwnedType = nullptr);

Decl *ParseStaticAssertDeclaration(SourceLocation &DeclEnd);
Decl *ParseNamespaceAlias(SourceLocation NamespaceLoc,
SourceLocation AliasLoc, IdentifierInfo *Alias,
SourceLocation &DeclEnd);

bool isValidAfterTypeSpecifier(bool CouldBeBitfield);
void ParseClassSpecifier(tok::TokenKind TagTokKind, SourceLocation TagLoc,
DeclSpec &DS, const ParsedTemplateInfo &TemplateInfo,
AccessSpecifier AS, bool EnteringContext,
DeclSpecContext DSC,
ParsedAttributesWithRange &Attributes);
void SkipCXXMemberSpecification(SourceLocation StartLoc,
SourceLocation AttrFixitLoc,
unsigned TagType,
Decl *TagDecl);
void ParseCXXMemberSpecification(SourceLocation StartLoc,
SourceLocation AttrFixitLoc,
ParsedAttributesWithRange &Attrs,
unsigned TagType,
Decl *TagDecl);
ExprResult ParseCXXMemberInitializer(Decl *D, bool IsFunction,
SourceLocation &EqualLoc);
bool ParseCXXMemberDeclaratorBeforeInitializer(Declarator &DeclaratorInfo,
VirtSpecifiers &VS,
ExprResult &BitfieldSize,
LateParsedAttrList &LateAttrs);
void MaybeParseAndDiagnoseDeclSpecAfterCXX11VirtSpecifierSeq(Declarator &D,
VirtSpecifiers &VS);
DeclGroupPtrTy ParseCXXClassMemberDeclaration(
AccessSpecifier AS, ParsedAttributes &Attr,
const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
ParsingDeclRAIIObject *DiagsFromTParams = nullptr);
DeclGroupPtrTy ParseCXXClassMemberDeclarationWithPragmas(
AccessSpecifier &AS, ParsedAttributesWithRange &AccessAttrs,
DeclSpec::TST TagType, Decl *Tag);
void ParseConstructorInitializer(Decl *ConstructorDecl);
MemInitResult ParseMemInitializer(Decl *ConstructorDecl);
void HandleMemberFunctionDeclDelays(Declarator& DeclaratorInfo,
Decl *ThisDecl);

TypeResult ParseBaseTypeSpecifier(SourceLocation &BaseLoc,
SourceLocation &EndLocation);
void ParseBaseClause(Decl *ClassDecl);
BaseResult ParseBaseSpecifier(Decl *ClassDecl);
AccessSpecifier getAccessSpecifierIfPresent() const;

bool ParseUnqualifiedIdTemplateId(CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
IdentifierInfo *Name,
SourceLocation NameLoc,
bool EnteringContext,
ParsedType ObjectType,
UnqualifiedId &Id,
bool AssumeTemplateId);
bool ParseUnqualifiedIdOperator(CXXScopeSpec &SS, bool EnteringContext,
ParsedType ObjectType,
UnqualifiedId &Result);

DeclGroupPtrTy ParseOMPDeclareSimdClauses(DeclGroupPtrTy Ptr,
CachedTokens &Toks,
SourceLocation Loc);
DeclGroupPtrTy ParseOpenMPDeclarativeDirectiveWithExtDecl(
AccessSpecifier &AS, ParsedAttributesWithRange &Attrs,
DeclSpec::TST TagType = DeclSpec::TST_unspecified,
Decl *TagDecl = nullptr);
DeclGroupPtrTy ParseOpenMPDeclareReductionDirective(AccessSpecifier AS);
void ParseOpenMPReductionInitializerForDecl(VarDecl *OmpPrivParm);

bool ParseOpenMPSimpleVarList(
OpenMPDirectiveKind Kind,
const llvm::function_ref<void(CXXScopeSpec &, DeclarationNameInfo)> &
Callback,
bool AllowScopeSpecifier);
StmtResult
ParseOpenMPDeclarativeOrExecutableDirective(AllowedConstructsKind Allowed);
OMPClause *ParseOpenMPClause(OpenMPDirectiveKind DKind,
OpenMPClauseKind CKind, bool FirstClause);
OMPClause *ParseOpenMPSingleExprClause(OpenMPClauseKind Kind,
bool ParseOnly);
OMPClause *ParseOpenMPSimpleClause(OpenMPClauseKind Kind, bool ParseOnly);
OMPClause *ParseOpenMPSingleExprWithArgClause(OpenMPClauseKind Kind,
bool ParseOnly);
OMPClause *ParseOpenMPClause(OpenMPClauseKind Kind, bool ParseOnly = false);
OMPClause *ParseOpenMPVarListClause(OpenMPDirectiveKind DKind,
OpenMPClauseKind Kind, bool ParseOnly);

public:
ExprResult ParseOpenMPParensExpr(StringRef ClauseName, SourceLocation &RLoc);

struct OpenMPVarListDataTy {
Expr *TailExpr = nullptr;
SourceLocation ColonLoc;
SourceLocation RLoc;
CXXScopeSpec ReductionIdScopeSpec;
DeclarationNameInfo ReductionId;
OpenMPDependClauseKind DepKind = OMPC_DEPEND_unknown;
OpenMPLinearClauseKind LinKind = OMPC_LINEAR_val;
OpenMPMapClauseKind MapTypeModifier = OMPC_MAP_unknown;
OpenMPMapClauseKind MapType = OMPC_MAP_unknown;
bool IsMapTypeImplicit = false;
SourceLocation DepLinMapLoc;
};

bool ParseOpenMPVarList(OpenMPDirectiveKind DKind, OpenMPClauseKind Kind,
SmallVectorImpl<Expr *> &Vars,
OpenMPVarListDataTy &Data);
bool ParseUnqualifiedId(CXXScopeSpec &SS, bool EnteringContext,
bool AllowDestructorName,
bool AllowConstructorName,
bool AllowDeductionGuide,
ParsedType ObjectType,
SourceLocation *TemplateKWLoc,
UnqualifiedId &Result);

private:

Decl *ParseDeclarationStartingWithTemplate(DeclaratorContext Context,
SourceLocation &DeclEnd,
ParsedAttributes &AccessAttrs,
AccessSpecifier AS = AS_none);
Decl *ParseTemplateDeclarationOrSpecialization(DeclaratorContext Context,
SourceLocation &DeclEnd,
ParsedAttributes &AccessAttrs,
AccessSpecifier AS);
Decl *ParseSingleDeclarationAfterTemplate(
DeclaratorContext Context, const ParsedTemplateInfo &TemplateInfo,
ParsingDeclRAIIObject &DiagsFromParams, SourceLocation &DeclEnd,
ParsedAttributes &AccessAttrs, AccessSpecifier AS = AS_none);
bool ParseTemplateParameters(unsigned Depth,
SmallVectorImpl<NamedDecl *> &TemplateParams,
SourceLocation &LAngleLoc,
SourceLocation &RAngleLoc);
bool ParseTemplateParameterList(unsigned Depth,
SmallVectorImpl<NamedDecl*> &TemplateParams);
bool isStartOfTemplateTypeParameter();
NamedDecl *ParseTemplateParameter(unsigned Depth, unsigned Position);
NamedDecl *ParseTypeParameter(unsigned Depth, unsigned Position);
NamedDecl *ParseTemplateTemplateParameter(unsigned Depth, unsigned Position);
NamedDecl *ParseNonTypeTemplateParameter(unsigned Depth, unsigned Position);
void DiagnoseMisplacedEllipsis(SourceLocation EllipsisLoc,
SourceLocation CorrectLoc,
bool AlreadyHasEllipsis,
bool IdentifierHasName);
void DiagnoseMisplacedEllipsisInDeclarator(SourceLocation EllipsisLoc,
Declarator &D);
typedef SmallVector<ParsedTemplateArgument, 16> TemplateArgList;

bool ParseGreaterThanInTemplateList(SourceLocation &RAngleLoc,
bool ConsumeLastToken,
bool ObjCGenericList);
bool ParseTemplateIdAfterTemplateName(bool ConsumeLastToken,
SourceLocation &LAngleLoc,
TemplateArgList &TemplateArgs,
SourceLocation &RAngleLoc);

bool AnnotateTemplateIdToken(TemplateTy Template, TemplateNameKind TNK,
CXXScopeSpec &SS,
SourceLocation TemplateKWLoc,
UnqualifiedId &TemplateName,
bool AllowTypeAnnotation = true);
void AnnotateTemplateIdTokenAsType(bool IsClassName = false);
bool IsTemplateArgumentList(unsigned Skip = 0);
bool ParseTemplateArgumentList(TemplateArgList &TemplateArgs);
ParsedTemplateArgument ParseTemplateTemplateArgument();
ParsedTemplateArgument ParseTemplateArgument();
Decl *ParseExplicitInstantiation(DeclaratorContext Context,
SourceLocation ExternLoc,
SourceLocation TemplateLoc,
SourceLocation &DeclEnd,
ParsedAttributes &AccessAttrs,
AccessSpecifier AS = AS_none);

DeclGroupPtrTy ParseModuleDecl();
Decl *ParseModuleImport(SourceLocation AtLoc);
bool parseMisplacedModuleImport();
bool tryParseMisplacedModuleImport() {
tok::TokenKind Kind = Tok.getKind();
if (Kind == tok::annot_module_begin || Kind == tok::annot_module_end ||
Kind == tok::annot_module_include)
return parseMisplacedModuleImport();
return false;
}

bool ParseModuleName(
SourceLocation UseLoc,
SmallVectorImpl<std::pair<IdentifierInfo *, SourceLocation>> &Path,
bool IsImport);

ExprResult ParseTypeTrait();

ExprResult ParseArrayTypeTrait();
ExprResult ParseExpressionTrait();

void CodeCompleteDirective(bool InConditional) override;
void CodeCompleteInConditionalExclusion() override;
void CodeCompleteMacroName(bool IsDefinition) override;
void CodeCompletePreprocessorExpression() override;
void CodeCompleteMacroArgument(IdentifierInfo *Macro, MacroInfo *MacroInfo,
unsigned ArgumentIndex) override;
void CodeCompleteNaturalLanguage() override;
};

}  

#endif
