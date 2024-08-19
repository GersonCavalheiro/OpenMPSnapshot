
#ifndef LLVM_CLANG_AST_DECLOPENMP_H
#define LLVM_CLANG_AST_DECLOPENMP_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"

namespace clang {

class OMPThreadPrivateDecl final
: public Decl,
private llvm::TrailingObjects<OMPThreadPrivateDecl, Expr *> {
friend class ASTDeclReader;
friend TrailingObjects;

unsigned NumVars;

virtual void anchor();

OMPThreadPrivateDecl(Kind DK, DeclContext *DC, SourceLocation L) :
Decl(DK, DC, L), NumVars(0) { }

ArrayRef<const Expr *> getVars() const {
return llvm::makeArrayRef(getTrailingObjects<Expr *>(), NumVars);
}

MutableArrayRef<Expr *> getVars() {
return MutableArrayRef<Expr *>(getTrailingObjects<Expr *>(), NumVars);
}

void setVars(ArrayRef<Expr *> VL);

public:
static OMPThreadPrivateDecl *Create(ASTContext &C, DeclContext *DC,
SourceLocation L,
ArrayRef<Expr *> VL);
static OMPThreadPrivateDecl *CreateDeserialized(ASTContext &C,
unsigned ID, unsigned N);

typedef MutableArrayRef<Expr *>::iterator varlist_iterator;
typedef ArrayRef<const Expr *>::iterator varlist_const_iterator;
typedef llvm::iterator_range<varlist_iterator> varlist_range;
typedef llvm::iterator_range<varlist_const_iterator> varlist_const_range;

unsigned varlist_size() const { return NumVars; }
bool varlist_empty() const { return NumVars == 0; }

varlist_range varlists() {
return varlist_range(varlist_begin(), varlist_end());
}
varlist_const_range varlists() const {
return varlist_const_range(varlist_begin(), varlist_end());
}
varlist_iterator varlist_begin() { return getVars().begin(); }
varlist_iterator varlist_end() { return getVars().end(); }
varlist_const_iterator varlist_begin() const { return getVars().begin(); }
varlist_const_iterator varlist_end() const { return getVars().end(); }

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == OMPThreadPrivate; }
};

class OMPDeclareReductionDecl final : public ValueDecl, public DeclContext {
public:
enum InitKind {
CallInit,   
DirectInit, 
CopyInit    
};

private:
friend class ASTDeclReader;
Expr *Combiner;
Expr *Initializer;
InitKind InitializerKind = CallInit;

LazyDeclPtr PrevDeclInScope;

virtual void anchor();

OMPDeclareReductionDecl(Kind DK, DeclContext *DC, SourceLocation L,
DeclarationName Name, QualType Ty,
OMPDeclareReductionDecl *PrevDeclInScope)
: ValueDecl(DK, DC, L, Name, Ty), DeclContext(DK), Combiner(nullptr),
Initializer(nullptr), InitializerKind(CallInit),
PrevDeclInScope(PrevDeclInScope) {}

void setPrevDeclInScope(OMPDeclareReductionDecl *Prev) {
PrevDeclInScope = Prev;
}

public:
static OMPDeclareReductionDecl *
Create(ASTContext &C, DeclContext *DC, SourceLocation L, DeclarationName Name,
QualType T, OMPDeclareReductionDecl *PrevDeclInScope);
static OMPDeclareReductionDecl *CreateDeserialized(ASTContext &C,
unsigned ID);

Expr *getCombiner() { return Combiner; }
const Expr *getCombiner() const { return Combiner; }
void setCombiner(Expr *E) { Combiner = E; }

Expr *getInitializer() { return Initializer; }
const Expr *getInitializer() const { return Initializer; }
InitKind getInitializerKind() const { return InitializerKind; }
void setInitializer(Expr *E, InitKind IK) {
Initializer = E;
InitializerKind = IK;
}

OMPDeclareReductionDecl *getPrevDeclInScope();
const OMPDeclareReductionDecl *getPrevDeclInScope() const;

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == OMPDeclareReduction; }
static DeclContext *castToDeclContext(const OMPDeclareReductionDecl *D) {
return static_cast<DeclContext *>(const_cast<OMPDeclareReductionDecl *>(D));
}
static OMPDeclareReductionDecl *castFromDeclContext(const DeclContext *DC) {
return static_cast<OMPDeclareReductionDecl *>(
const_cast<DeclContext *>(DC));
}
};

class OMPCapturedExprDecl final : public VarDecl {
friend class ASTDeclReader;
void anchor() override;

OMPCapturedExprDecl(ASTContext &C, DeclContext *DC, IdentifierInfo *Id,
QualType Type, TypeSourceInfo *TInfo,
SourceLocation StartLoc)
: VarDecl(OMPCapturedExpr, C, DC, StartLoc, StartLoc, Id, Type, TInfo,
SC_None) {
setImplicit();
}

public:
static OMPCapturedExprDecl *Create(ASTContext &C, DeclContext *DC,
IdentifierInfo *Id, QualType T,
SourceLocation StartLoc);

static OMPCapturedExprDecl *CreateDeserialized(ASTContext &C, unsigned ID);

SourceRange getSourceRange() const override LLVM_READONLY;

static bool classof(const Decl *D) { return classofKind(D->getKind()); }
static bool classofKind(Kind K) { return K == OMPCapturedExpr; }
};

} 

#endif
