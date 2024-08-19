
#ifndef LLVM_CLANG_FORMAT_FORMAT_H
#define LLVM_CLANG_FORMAT_FORMAT_H

#include "clang/Basic/LangOptions.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Inclusions/IncludeStyle.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Regex.h"
#include <system_error>

namespace clang {

class Lexer;
class SourceManager;
class DiagnosticConsumer;

namespace vfs {
class FileSystem;
}

namespace format {

enum class ParseError { Success = 0, Error, Unsuitable };
class ParseErrorCategory final : public std::error_category {
public:
const char *name() const noexcept override;
std::string message(int EV) const override;
};
const std::error_category &getParseCategory();
std::error_code make_error_code(ParseError e);

struct FormatStyle {
int AccessModifierOffset;

enum BracketAlignmentStyle {
BAS_Align,
BAS_DontAlign,
BAS_AlwaysBreak,
};

BracketAlignmentStyle AlignAfterOpenBracket;

bool AlignConsecutiveAssignments;

bool AlignConsecutiveDeclarations;

enum EscapedNewlineAlignmentStyle {
ENAS_DontAlign,
ENAS_Left,
ENAS_Right,
};

EscapedNewlineAlignmentStyle AlignEscapedNewlines;

bool AlignOperands;

bool AlignTrailingComments;

bool AllowAllParametersOfDeclarationOnNextLine;

bool AllowShortBlocksOnASingleLine;

bool AllowShortCaseLabelsOnASingleLine;

enum ShortFunctionStyle {
SFS_None,
SFS_InlineOnly,
SFS_Empty,
SFS_Inline,
SFS_All,
};

ShortFunctionStyle AllowShortFunctionsOnASingleLine;

bool AllowShortIfStatementsOnASingleLine;

bool AllowShortLoopsOnASingleLine;

enum DefinitionReturnTypeBreakingStyle {
DRTBS_None,
DRTBS_All,
DRTBS_TopLevel,
};

enum ReturnTypeBreakingStyle {
RTBS_None,
RTBS_All,
RTBS_TopLevel,
RTBS_AllDefinitions,
RTBS_TopLevelDefinitions,
};

DefinitionReturnTypeBreakingStyle AlwaysBreakAfterDefinitionReturnType;

ReturnTypeBreakingStyle AlwaysBreakAfterReturnType;

bool AlwaysBreakBeforeMultilineStrings;

enum BreakTemplateDeclarationsStyle {
BTDS_No,
BTDS_MultiLine,
BTDS_Yes
};

BreakTemplateDeclarationsStyle AlwaysBreakTemplateDeclarations;

bool BinPackArguments;

bool BinPackParameters;

enum BinPackStyle {
BPS_Auto,
BPS_Always,
BPS_Never,
};

enum BinaryOperatorStyle {
BOS_None,
BOS_NonAssignment,
BOS_All,
};

BinaryOperatorStyle BreakBeforeBinaryOperators;

enum BraceBreakingStyle {
BS_Attach,
BS_Linux,
BS_Mozilla,
BS_Stroustrup,
BS_Allman,
BS_GNU,
BS_WebKit,
BS_Custom
};

BraceBreakingStyle BreakBeforeBraces;

struct BraceWrappingFlags {
bool AfterClass;
bool AfterControlStatement;
bool AfterEnum;
bool AfterFunction;
bool AfterNamespace;
bool AfterObjCDeclaration;
bool AfterStruct;
bool AfterUnion;
bool AfterExternBlock;
bool BeforeCatch;
bool BeforeElse;
bool IndentBraces;
bool SplitEmptyFunction;
bool SplitEmptyRecord;
bool SplitEmptyNamespace;
};

BraceWrappingFlags BraceWrapping;

bool BreakBeforeTernaryOperators;

enum BreakConstructorInitializersStyle {
BCIS_BeforeColon,
BCIS_BeforeComma,
BCIS_AfterColon
};

BreakConstructorInitializersStyle BreakConstructorInitializers;

bool BreakAfterJavaFieldAnnotations;

bool BreakStringLiterals;

unsigned ColumnLimit;

std::string CommentPragmas;

enum BreakInheritanceListStyle {
BILS_BeforeColon,
BILS_BeforeComma,
BILS_AfterColon
};

BreakInheritanceListStyle BreakInheritanceList;

bool CompactNamespaces;

bool ConstructorInitializerAllOnOneLineOrOnePerLine;

unsigned ConstructorInitializerIndentWidth;

unsigned ContinuationIndentWidth;

bool Cpp11BracedListStyle;

bool DerivePointerAlignment;

bool DisableFormat;

bool ExperimentalAutoDetectBinPacking;

bool FixNamespaceComments;

std::vector<std::string> ForEachMacros;

tooling::IncludeStyle IncludeStyle;

bool IndentCaseLabels;

enum PPDirectiveIndentStyle {
PPDIS_None,
PPDIS_AfterHash
};

PPDirectiveIndentStyle IndentPPDirectives;

unsigned IndentWidth;

bool IndentWrappedFunctionNames;

enum JavaScriptQuoteStyle {
JSQS_Leave,
JSQS_Single,
JSQS_Double
};

JavaScriptQuoteStyle JavaScriptQuotes;

bool JavaScriptWrapImports;

bool KeepEmptyLinesAtTheStartOfBlocks;

enum LanguageKind {
LK_None,
LK_Cpp,
LK_Java,
LK_JavaScript,
LK_ObjC,
LK_Proto,
LK_TableGen,
LK_TextProto
};
bool isCpp() const { return Language == LK_Cpp || Language == LK_ObjC; }

LanguageKind Language;

std::string MacroBlockBegin;

std::string MacroBlockEnd;

unsigned MaxEmptyLinesToKeep;

enum NamespaceIndentationKind {
NI_None,
NI_Inner,
NI_All
};

NamespaceIndentationKind NamespaceIndentation;

BinPackStyle ObjCBinPackProtocolList;

unsigned ObjCBlockIndentWidth;

bool ObjCSpaceAfterProperty;

bool ObjCSpaceBeforeProtocolList;

unsigned PenaltyBreakAssignment;

unsigned PenaltyBreakBeforeFirstCallParameter;

unsigned PenaltyBreakComment;

unsigned PenaltyBreakFirstLessLess;

unsigned PenaltyBreakString;

unsigned PenaltyBreakTemplateDeclaration;

unsigned PenaltyExcessCharacter;

unsigned PenaltyReturnTypeOnItsOwnLine;

enum PointerAlignmentStyle {
PAS_Left,
PAS_Right,
PAS_Middle
};

PointerAlignmentStyle PointerAlignment;

struct RawStringFormat {
LanguageKind Language;
std::vector<std::string> Delimiters;
std::vector<std::string> EnclosingFunctions;
std::string CanonicalDelimiter;
std::string BasedOnStyle;
bool operator==(const RawStringFormat &Other) const {
return Language == Other.Language && Delimiters == Other.Delimiters &&
EnclosingFunctions == Other.EnclosingFunctions &&
CanonicalDelimiter == Other.CanonicalDelimiter &&
BasedOnStyle == Other.BasedOnStyle;
}
};

std::vector<RawStringFormat> RawStringFormats;

bool ReflowComments;

bool SortIncludes;

bool SortUsingDeclarations;

bool SpaceAfterCStyleCast;

bool SpaceAfterTemplateKeyword;

bool SpaceBeforeAssignmentOperators;

bool SpaceBeforeCpp11BracedList;

bool SpaceBeforeCtorInitializerColon;

bool SpaceBeforeInheritanceColon;

enum SpaceBeforeParensOptions {
SBPO_Never,
SBPO_ControlStatements,
SBPO_Always
};

SpaceBeforeParensOptions SpaceBeforeParens;

bool SpaceBeforeRangeBasedForLoopColon;

bool SpaceInEmptyParentheses;

unsigned SpacesBeforeTrailingComments;

bool SpacesInAngles;

bool SpacesInContainerLiterals;

bool SpacesInCStyleCastParentheses;

bool SpacesInParentheses;

bool SpacesInSquareBrackets;

enum LanguageStandard {
LS_Cpp03,
LS_Cpp11,
LS_Auto
};

LanguageStandard Standard;

unsigned TabWidth;

enum UseTabStyle {
UT_Never,
UT_ForIndentation,
UT_ForContinuationAndIndentation,
UT_Always
};

UseTabStyle UseTab;

bool operator==(const FormatStyle &R) const {
return AccessModifierOffset == R.AccessModifierOffset &&
AlignAfterOpenBracket == R.AlignAfterOpenBracket &&
AlignConsecutiveAssignments == R.AlignConsecutiveAssignments &&
AlignConsecutiveDeclarations == R.AlignConsecutiveDeclarations &&
AlignEscapedNewlines == R.AlignEscapedNewlines &&
AlignOperands == R.AlignOperands &&
AlignTrailingComments == R.AlignTrailingComments &&
AllowAllParametersOfDeclarationOnNextLine ==
R.AllowAllParametersOfDeclarationOnNextLine &&
AllowShortBlocksOnASingleLine == R.AllowShortBlocksOnASingleLine &&
AllowShortCaseLabelsOnASingleLine ==
R.AllowShortCaseLabelsOnASingleLine &&
AllowShortFunctionsOnASingleLine ==
R.AllowShortFunctionsOnASingleLine &&
AllowShortIfStatementsOnASingleLine ==
R.AllowShortIfStatementsOnASingleLine &&
AllowShortLoopsOnASingleLine == R.AllowShortLoopsOnASingleLine &&
AlwaysBreakAfterReturnType == R.AlwaysBreakAfterReturnType &&
AlwaysBreakBeforeMultilineStrings ==
R.AlwaysBreakBeforeMultilineStrings &&
AlwaysBreakTemplateDeclarations ==
R.AlwaysBreakTemplateDeclarations &&
BinPackArguments == R.BinPackArguments &&
BinPackParameters == R.BinPackParameters &&
BreakBeforeBinaryOperators == R.BreakBeforeBinaryOperators &&
BreakBeforeBraces == R.BreakBeforeBraces &&
BreakBeforeTernaryOperators == R.BreakBeforeTernaryOperators &&
BreakConstructorInitializers == R.BreakConstructorInitializers &&
CompactNamespaces == R.CompactNamespaces &&
BreakAfterJavaFieldAnnotations == R.BreakAfterJavaFieldAnnotations &&
BreakStringLiterals == R.BreakStringLiterals &&
ColumnLimit == R.ColumnLimit && CommentPragmas == R.CommentPragmas &&
BreakInheritanceList == R.BreakInheritanceList &&
ConstructorInitializerAllOnOneLineOrOnePerLine ==
R.ConstructorInitializerAllOnOneLineOrOnePerLine &&
ConstructorInitializerIndentWidth ==
R.ConstructorInitializerIndentWidth &&
ContinuationIndentWidth == R.ContinuationIndentWidth &&
Cpp11BracedListStyle == R.Cpp11BracedListStyle &&
DerivePointerAlignment == R.DerivePointerAlignment &&
DisableFormat == R.DisableFormat &&
ExperimentalAutoDetectBinPacking ==
R.ExperimentalAutoDetectBinPacking &&
FixNamespaceComments == R.FixNamespaceComments &&
ForEachMacros == R.ForEachMacros &&
IncludeStyle.IncludeBlocks == R.IncludeStyle.IncludeBlocks &&
IncludeStyle.IncludeCategories == R.IncludeStyle.IncludeCategories &&
IndentCaseLabels == R.IndentCaseLabels &&
IndentPPDirectives == R.IndentPPDirectives &&
IndentWidth == R.IndentWidth && Language == R.Language &&
IndentWrappedFunctionNames == R.IndentWrappedFunctionNames &&
JavaScriptQuotes == R.JavaScriptQuotes &&
JavaScriptWrapImports == R.JavaScriptWrapImports &&
KeepEmptyLinesAtTheStartOfBlocks ==
R.KeepEmptyLinesAtTheStartOfBlocks &&
MacroBlockBegin == R.MacroBlockBegin &&
MacroBlockEnd == R.MacroBlockEnd &&
MaxEmptyLinesToKeep == R.MaxEmptyLinesToKeep &&
NamespaceIndentation == R.NamespaceIndentation &&
ObjCBinPackProtocolList == R.ObjCBinPackProtocolList &&
ObjCBlockIndentWidth == R.ObjCBlockIndentWidth &&
ObjCSpaceAfterProperty == R.ObjCSpaceAfterProperty &&
ObjCSpaceBeforeProtocolList == R.ObjCSpaceBeforeProtocolList &&
PenaltyBreakAssignment == R.PenaltyBreakAssignment &&
PenaltyBreakBeforeFirstCallParameter ==
R.PenaltyBreakBeforeFirstCallParameter &&
PenaltyBreakComment == R.PenaltyBreakComment &&
PenaltyBreakFirstLessLess == R.PenaltyBreakFirstLessLess &&
PenaltyBreakString == R.PenaltyBreakString &&
PenaltyExcessCharacter == R.PenaltyExcessCharacter &&
PenaltyReturnTypeOnItsOwnLine == R.PenaltyReturnTypeOnItsOwnLine &&
PenaltyBreakTemplateDeclaration ==
R.PenaltyBreakTemplateDeclaration &&
PointerAlignment == R.PointerAlignment &&
RawStringFormats == R.RawStringFormats &&
SpaceAfterCStyleCast == R.SpaceAfterCStyleCast &&
SpaceAfterTemplateKeyword == R.SpaceAfterTemplateKeyword &&
SpaceBeforeAssignmentOperators == R.SpaceBeforeAssignmentOperators &&
SpaceBeforeCpp11BracedList == R.SpaceBeforeCpp11BracedList &&
SpaceBeforeCtorInitializerColon ==
R.SpaceBeforeCtorInitializerColon &&
SpaceBeforeInheritanceColon == R.SpaceBeforeInheritanceColon &&
SpaceBeforeParens == R.SpaceBeforeParens &&
SpaceBeforeRangeBasedForLoopColon ==
R.SpaceBeforeRangeBasedForLoopColon &&
SpaceInEmptyParentheses == R.SpaceInEmptyParentheses &&
SpacesBeforeTrailingComments == R.SpacesBeforeTrailingComments &&
SpacesInAngles == R.SpacesInAngles &&
SpacesInContainerLiterals == R.SpacesInContainerLiterals &&
SpacesInCStyleCastParentheses == R.SpacesInCStyleCastParentheses &&
SpacesInParentheses == R.SpacesInParentheses &&
SpacesInSquareBrackets == R.SpacesInSquareBrackets &&
Standard == R.Standard && TabWidth == R.TabWidth &&
UseTab == R.UseTab;
}

llvm::Optional<FormatStyle> GetLanguageStyle(LanguageKind Language) const;

struct FormatStyleSet {
typedef std::map<FormatStyle::LanguageKind, FormatStyle> MapType;

llvm::Optional<FormatStyle> Get(FormatStyle::LanguageKind Language) const;

void Add(FormatStyle Style);

void Clear();

private:
std::shared_ptr<MapType> Styles;
};

static FormatStyleSet BuildStyleSetFromConfiguration(
const FormatStyle &MainStyle,
const std::vector<FormatStyle> &ConfigurationStyles);

private:
FormatStyleSet StyleSet;

friend std::error_code parseConfiguration(StringRef Text, FormatStyle *Style);
};

FormatStyle getLLVMStyle();

FormatStyle getGoogleStyle(FormatStyle::LanguageKind Language);

FormatStyle getChromiumStyle(FormatStyle::LanguageKind Language);

FormatStyle getMozillaStyle();

FormatStyle getWebKitStyle();

FormatStyle getGNUStyle();

FormatStyle getNoStyle();

bool getPredefinedStyle(StringRef Name, FormatStyle::LanguageKind Language,
FormatStyle *Style);

std::error_code parseConfiguration(StringRef Text, FormatStyle *Style);

std::string configurationAsText(const FormatStyle &Style);

tooling::Replacements sortIncludes(const FormatStyle &Style, StringRef Code,
ArrayRef<tooling::Range> Ranges,
StringRef FileName,
unsigned *Cursor = nullptr);

llvm::Expected<tooling::Replacements>
formatReplacements(StringRef Code, const tooling::Replacements &Replaces,
const FormatStyle &Style);

llvm::Expected<tooling::Replacements>
cleanupAroundReplacements(StringRef Code, const tooling::Replacements &Replaces,
const FormatStyle &Style);

struct FormattingAttemptStatus {
bool FormatComplete = true;

unsigned Line = 0;
};

tooling::Replacements reformat(const FormatStyle &Style, StringRef Code,
ArrayRef<tooling::Range> Ranges,
StringRef FileName = "<stdin>",
FormattingAttemptStatus *Status = nullptr);

tooling::Replacements reformat(const FormatStyle &Style, StringRef Code,
ArrayRef<tooling::Range> Ranges,
StringRef FileName,
bool *IncompleteFormat);

tooling::Replacements cleanup(const FormatStyle &Style, StringRef Code,
ArrayRef<tooling::Range> Ranges,
StringRef FileName = "<stdin>");

tooling::Replacements fixNamespaceEndComments(const FormatStyle &Style,
StringRef Code,
ArrayRef<tooling::Range> Ranges,
StringRef FileName = "<stdin>");

tooling::Replacements sortUsingDeclarations(const FormatStyle &Style,
StringRef Code,
ArrayRef<tooling::Range> Ranges,
StringRef FileName = "<stdin>");

LangOptions getFormattingLangOpts(const FormatStyle &Style = getLLVMStyle());

extern const char *StyleOptionHelpDescription;

extern const char *DefaultFormatStyle;

extern const char *DefaultFallbackStyle;

llvm::Expected<FormatStyle> getStyle(StringRef StyleName, StringRef FileName,
StringRef FallbackStyle,
StringRef Code = "",
vfs::FileSystem *FS = nullptr);

FormatStyle::LanguageKind guessLanguage(StringRef FileName, StringRef Code);

inline StringRef getLanguageName(FormatStyle::LanguageKind Language) {
switch (Language) {
case FormatStyle::LK_Cpp:
return "C++";
case FormatStyle::LK_ObjC:
return "Objective-C";
case FormatStyle::LK_Java:
return "Java";
case FormatStyle::LK_JavaScript:
return "JavaScript";
case FormatStyle::LK_Proto:
return "Proto";
case FormatStyle::LK_TextProto:
return "TextProto";
default:
return "Unknown";
}
}

} 
} 

namespace std {
template <>
struct is_error_code_enum<clang::format::ParseError> : std::true_type {};
}

#endif 
