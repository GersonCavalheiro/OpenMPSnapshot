
#ifndef LLVM_CLANG_LEX_LEXER_H
#define LLVM_CLANG_LEX_LEXER_H

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/PreprocessorLexer.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cstdint>
#include <string>

namespace llvm {

class MemoryBuffer;

} 

namespace clang {

class DiagnosticBuilder;
class Preprocessor;
class SourceManager;

enum ConflictMarkerKind {
CMK_None,

CMK_Normal,

CMK_Perforce
};

struct PreambleBounds {
unsigned Size;

bool PreambleEndsAtStartOfLine;

PreambleBounds(unsigned Size, bool PreambleEndsAtStartOfLine)
: Size(Size), PreambleEndsAtStartOfLine(PreambleEndsAtStartOfLine) {}
};

class Lexer : public PreprocessorLexer {
friend class Preprocessor;

void anchor() override;


const char *BufferStart;

const char *BufferEnd;

SourceLocation FileLoc;

LangOptions LangOpts;

bool Is_PragmaLexer;


unsigned char ExtendedTokenMode;


const char *BufferPtr;

bool IsAtStartOfLine;

bool IsAtPhysicalStartOfLine;

bool HasLeadingSpace;

bool HasLeadingEmptyMacro;

ConflictMarkerKind CurrentConflictMarkerState;

void InitLexer(const char *BufStart, const char *BufPtr, const char *BufEnd);

public:
Lexer(FileID FID, const llvm::MemoryBuffer *InputBuffer, Preprocessor &PP);

Lexer(SourceLocation FileLoc, const LangOptions &LangOpts,
const char *BufStart, const char *BufPtr, const char *BufEnd);

Lexer(FileID FID, const llvm::MemoryBuffer *InputBuffer,
const SourceManager &SM, const LangOptions &LangOpts);

Lexer(const Lexer &) = delete;
Lexer &operator=(const Lexer &) = delete;

static Lexer *Create_PragmaLexer(SourceLocation SpellingLoc,
SourceLocation ExpansionLocStart,
SourceLocation ExpansionLocEnd,
unsigned TokLen, Preprocessor &PP);

const LangOptions &getLangOpts() const { return LangOpts; }

SourceLocation getFileLoc() const { return FileLoc; }

private:
bool Lex(Token &Result);

public:
bool isPragmaLexer() const { return Is_PragmaLexer; }

private:
void IndirectLex(Token &Result) override { Lex(Result); }

public:
bool LexFromRawLexer(Token &Result) {
assert(LexingRawMode && "Not already in raw mode!");
Lex(Result);
return BufferPtr == BufferEnd;
}

bool isKeepWhitespaceMode() const {
return ExtendedTokenMode > 1;
}

void SetKeepWhitespaceMode(bool Val) {
assert((!Val || LexingRawMode || LangOpts.TraditionalCPP) &&
"Can only retain whitespace in raw mode or -traditional-cpp");
ExtendedTokenMode = Val ? 2 : 0;
}

bool inKeepCommentMode() const {
return ExtendedTokenMode > 0;
}

void SetCommentRetentionState(bool Mode) {
assert(!isKeepWhitespaceMode() &&
"Can't play with comment retention state when retaining whitespace");
ExtendedTokenMode = Mode ? 1 : 0;
}

void resetExtendedTokenMode();

StringRef getBuffer() const {
return StringRef(BufferStart, BufferEnd - BufferStart);
}

void ReadToEndOfLine(SmallVectorImpl<char> *Result = nullptr);


DiagnosticBuilder Diag(const char *Loc, unsigned DiagID) const;

SourceLocation getSourceLocation(const char *Loc, unsigned TokLen = 1) const;

SourceLocation getSourceLocation() override {
return getSourceLocation(BufferPtr);
}

const char *getBufferLocation() const { return BufferPtr; }

static std::string Stringify(StringRef Str, bool Charify = false);

static void Stringify(SmallVectorImpl<char> &Str);

static unsigned getSpelling(const Token &Tok, const char *&Buffer,
const SourceManager &SourceMgr,
const LangOptions &LangOpts,
bool *Invalid = nullptr);

static std::string getSpelling(const Token &Tok,
const SourceManager &SourceMgr,
const LangOptions &LangOpts,
bool *Invalid = nullptr);

static StringRef getSpelling(SourceLocation loc,
SmallVectorImpl<char> &buffer,
const SourceManager &SourceMgr,
const LangOptions &LangOpts,
bool *invalid = nullptr);

static unsigned MeasureTokenLength(SourceLocation Loc,
const SourceManager &SM,
const LangOptions &LangOpts);

static bool getRawToken(SourceLocation Loc, Token &Result,
const SourceManager &SM,
const LangOptions &LangOpts,
bool IgnoreWhiteSpace = false);

static SourceLocation GetBeginningOfToken(SourceLocation Loc,
const SourceManager &SM,
const LangOptions &LangOpts);

static unsigned getTokenPrefixLength(SourceLocation TokStart,
unsigned Characters,
const SourceManager &SM,
const LangOptions &LangOpts);

static SourceLocation AdvanceToTokenCharacter(SourceLocation TokStart,
unsigned Characters,
const SourceManager &SM,
const LangOptions &LangOpts) {
return TokStart.getLocWithOffset(
getTokenPrefixLength(TokStart, Characters, SM, LangOpts));
}

static SourceLocation getLocForEndOfToken(SourceLocation Loc, unsigned Offset,
const SourceManager &SM,
const LangOptions &LangOpts);

static CharSourceRange getAsCharRange(SourceRange Range,
const SourceManager &SM,
const LangOptions &LangOpts) {
SourceLocation End = getLocForEndOfToken(Range.getEnd(), 0, SM, LangOpts);
return End.isInvalid() ? CharSourceRange()
: CharSourceRange::getCharRange(
Range.getBegin(), End.getLocWithOffset(-1));
}
static CharSourceRange getAsCharRange(CharSourceRange Range,
const SourceManager &SM,
const LangOptions &LangOpts) {
return Range.isTokenRange()
? getAsCharRange(Range.getAsRange(), SM, LangOpts)
: Range;
}

static bool isAtStartOfMacroExpansion(SourceLocation loc,
const SourceManager &SM,
const LangOptions &LangOpts,
SourceLocation *MacroBegin = nullptr);

static bool isAtEndOfMacroExpansion(SourceLocation loc,
const SourceManager &SM,
const LangOptions &LangOpts,
SourceLocation *MacroEnd = nullptr);

static CharSourceRange makeFileCharRange(CharSourceRange Range,
const SourceManager &SM,
const LangOptions &LangOpts);

static StringRef getSourceText(CharSourceRange Range,
const SourceManager &SM,
const LangOptions &LangOpts,
bool *Invalid = nullptr);

static StringRef getImmediateMacroName(SourceLocation Loc,
const SourceManager &SM,
const LangOptions &LangOpts);

static StringRef getImmediateMacroNameForDiagnostics(
SourceLocation Loc, const SourceManager &SM, const LangOptions &LangOpts);

static PreambleBounds ComputePreamble(StringRef Buffer,
const LangOptions &LangOpts,
unsigned MaxLines = 0);

static Optional<Token> findNextToken(SourceLocation Loc,
const SourceManager &SM,
const LangOptions &LangOpts);

static SourceLocation findLocationAfterToken(SourceLocation loc,
tok::TokenKind TKind,
const SourceManager &SM,
const LangOptions &LangOpts,
bool SkipTrailingWhitespaceAndNewLine);

static bool isIdentifierBodyChar(char c, const LangOptions &LangOpts);

static bool isNewLineEscaped(const char *BufferStart, const char *Str);

static inline char getCharAndSizeNoWarn(const char *Ptr, unsigned &Size,
const LangOptions &LangOpts) {
if (isObviouslySimpleCharacter(Ptr[0])) {
Size = 1;
return *Ptr;
}

Size = 0;
return getCharAndSizeSlowNoWarn(Ptr, Size, LangOpts);
}

static StringRef getIndentationForLine(SourceLocation Loc,
const SourceManager &SM);

private:

bool LexTokenInternal(Token &Result, bool TokAtPhysicalStartOfLine);

bool CheckUnicodeWhitespace(Token &Result, uint32_t C, const char *CurPtr);

bool LexUnicode(Token &Result, uint32_t C, const char *CurPtr);

void FormTokenWithChars(Token &Result, const char *TokEnd,
tok::TokenKind Kind) {
unsigned TokLen = TokEnd-BufferPtr;
Result.setLength(TokLen);
Result.setLocation(getSourceLocation(BufferPtr, TokLen));
Result.setKind(Kind);
BufferPtr = TokEnd;
}

unsigned isNextPPTokenLParen();



static bool isObviouslySimpleCharacter(char C) {
return C != '?' && C != '\\';
}

inline char getAndAdvanceChar(const char *&Ptr, Token &Tok) {
if (isObviouslySimpleCharacter(Ptr[0])) return *Ptr++;

unsigned Size = 0;
char C = getCharAndSizeSlow(Ptr, Size, &Tok);
Ptr += Size;
return C;
}

const char *ConsumeChar(const char *Ptr, unsigned Size, Token &Tok) {
if (Size == 1)
return Ptr+Size;

Size = 0;
getCharAndSizeSlow(Ptr, Size, &Tok);
return Ptr+Size;
}

inline char getCharAndSize(const char *Ptr, unsigned &Size) {
if (isObviouslySimpleCharacter(Ptr[0])) {
Size = 1;
return *Ptr;
}

Size = 0;
return getCharAndSizeSlow(Ptr, Size);
}

char getCharAndSizeSlow(const char *Ptr, unsigned &Size,
Token *Tok = nullptr);

static unsigned getEscapedNewLineSize(const char *P);

static const char *SkipEscapedNewLines(const char *P);

static char getCharAndSizeSlowNoWarn(const char *Ptr, unsigned &Size,
const LangOptions &LangOpts);


void SetByteOffset(unsigned Offset, bool StartOfLine);

void PropagateLineStartLeadingSpaceInfo(Token &Result);

const char *LexUDSuffix(Token &Result, const char *CurPtr,
bool IsStringLiteral);

bool LexIdentifier         (Token &Result, const char *CurPtr);
bool LexNumericConstant    (Token &Result, const char *CurPtr);
bool LexStringLiteral      (Token &Result, const char *CurPtr,
tok::TokenKind Kind);
bool LexRawStringLiteral   (Token &Result, const char *CurPtr,
tok::TokenKind Kind);
bool LexAngledStringLiteral(Token &Result, const char *CurPtr);
bool LexCharConstant       (Token &Result, const char *CurPtr,
tok::TokenKind Kind);
bool LexEndOfFile          (Token &Result, const char *CurPtr);
bool SkipWhitespace        (Token &Result, const char *CurPtr,
bool &TokAtPhysicalStartOfLine);
bool SkipLineComment       (Token &Result, const char *CurPtr,
bool &TokAtPhysicalStartOfLine);
bool SkipBlockComment      (Token &Result, const char *CurPtr,
bool &TokAtPhysicalStartOfLine);
bool SaveLineComment       (Token &Result, const char *CurPtr);

bool IsStartOfConflictMarker(const char *CurPtr);
bool HandleEndOfConflictMarker(const char *CurPtr);

bool lexEditorPlaceholder(Token &Result, const char *CurPtr);

bool isCodeCompletionPoint(const char *CurPtr) const;
void cutOffLexing() { BufferPtr = BufferEnd; }

bool isHexaLiteral(const char *Start, const LangOptions &LangOpts);


uint32_t tryReadUCN(const char *&CurPtr, const char *SlashLoc, Token *Tok);

bool tryConsumeIdentifierUCN(const char *&CurPtr, unsigned Size,
Token &Result);

bool tryConsumeIdentifierUTF8Char(const char *&CurPtr);
};

} 

#endif 
