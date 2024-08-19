
#ifndef LLVM_CLANG_LEX_PRAGMA_H
#define LLVM_CLANG_LEX_PRAGMA_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang {

class PragmaNamespace;
class Preprocessor;
class Token;


enum PragmaIntroducerKind {

PIK_HashPragma,


PIK__Pragma,


PIK___pragma
};

class PragmaHandler {
std::string Name;

public:
PragmaHandler() = default;
explicit PragmaHandler(StringRef name) : Name(name) {}
virtual ~PragmaHandler();

StringRef getName() const { return Name; }
virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
Token &FirstToken) = 0;

virtual PragmaNamespace *getIfNamespace() { return nullptr; }
};

class EmptyPragmaHandler : public PragmaHandler {
public:
explicit EmptyPragmaHandler(StringRef Name = StringRef());

void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
Token &FirstToken) override;
};

class PragmaNamespace : public PragmaHandler {
llvm::StringMap<PragmaHandler *> Handlers;

public:
explicit PragmaNamespace(StringRef Name) : PragmaHandler(Name) {}
~PragmaNamespace() override;

PragmaHandler *FindHandler(StringRef Name,
bool IgnoreNull = true) const;

void AddPragma(PragmaHandler *Handler);

void RemovePragmaHandler(PragmaHandler *Handler);

bool IsEmpty() const { return Handlers.empty(); }

void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
Token &FirstToken) override;

PragmaNamespace *getIfNamespace() override { return this; }
};

} 

#endif 
