
#ifndef LLVM_CLANG_SERIALIZATION_ASTBITCODES_H
#define LLVM_CLANG_SERIALIZATION_ASTBITCODES_H

#include "clang/AST/DeclarationName.h"
#include "clang/AST/Type.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Bitcode/BitCodes.h"
#include <cassert>
#include <cstdint>

namespace clang {
namespace serialization {

const unsigned VERSION_MAJOR = 7;

const unsigned VERSION_MINOR = 0;

using IdentifierID = uint32_t;

using DeclID = uint32_t;

using LocalDeclID = DeclID;
using GlobalDeclID = DeclID;

using TypeID = uint32_t;

class TypeIdx {
uint32_t Idx = 0;

public:
TypeIdx() = default;
explicit TypeIdx(uint32_t index) : Idx(index) {}

uint32_t getIndex() const { return Idx; }

TypeID asTypeID(unsigned FastQuals) const {
if (Idx == uint32_t(-1))
return TypeID(-1);

return (Idx << Qualifiers::FastWidth) | FastQuals;
}

static TypeIdx fromTypeID(TypeID ID) {
if (ID == TypeID(-1))
return TypeIdx(-1);

return TypeIdx(ID >> Qualifiers::FastWidth);
}
};

struct UnsafeQualTypeDenseMapInfo {
static bool isEqual(QualType A, QualType B) { return A == B; }

static QualType getEmptyKey() {
return QualType::getFromOpaquePtr((void*) 1);
}

static QualType getTombstoneKey() {
return QualType::getFromOpaquePtr((void*) 2);
}

static unsigned getHashValue(QualType T) {
assert(!T.getLocalFastQualifiers() &&
"hash invalid for types with fast quals");
uintptr_t v = reinterpret_cast<uintptr_t>(T.getAsOpaquePtr());
return (unsigned(v) >> 4) ^ (unsigned(v) >> 9);
}
};

using IdentID = uint32_t;

const unsigned int NUM_PREDEF_IDENT_IDS = 1;

using MacroID = uint32_t;

using GlobalMacroID = uint32_t;

using LocalMacroID = uint32_t;

const unsigned int NUM_PREDEF_MACRO_IDS = 1;

using SelectorID = uint32_t;

const unsigned int NUM_PREDEF_SELECTOR_IDS = 1;

using CXXBaseSpecifiersID = uint32_t;

using CXXCtorInitializersID = uint32_t;

using PreprocessedEntityID = uint32_t;

using SubmoduleID = uint32_t;

const unsigned int NUM_PREDEF_SUBMODULE_IDS = 1;

struct PPEntityOffset {
unsigned Begin;

unsigned End;

uint32_t BitOffset;

PPEntityOffset(SourceRange R, uint32_t BitOffset)
: Begin(R.getBegin().getRawEncoding()),
End(R.getEnd().getRawEncoding()), BitOffset(BitOffset) {}

SourceLocation getBegin() const {
return SourceLocation::getFromRawEncoding(Begin);
}

SourceLocation getEnd() const {
return SourceLocation::getFromRawEncoding(End);
}
};

struct PPSkippedRange {
unsigned Begin;
unsigned End;

PPSkippedRange(SourceRange R)
: Begin(R.getBegin().getRawEncoding()),
End(R.getEnd().getRawEncoding()) { }

SourceLocation getBegin() const {
return SourceLocation::getFromRawEncoding(Begin);
}
SourceLocation getEnd() const {
return SourceLocation::getFromRawEncoding(End);
}
};

struct DeclOffset {
unsigned Loc = 0;

uint32_t BitOffset = 0;

DeclOffset() = default;
DeclOffset(SourceLocation Loc, uint32_t BitOffset)
: Loc(Loc.getRawEncoding()), BitOffset(BitOffset) {}

void setLocation(SourceLocation L) {
Loc = L.getRawEncoding();
}

SourceLocation getLocation() const {
return SourceLocation::getFromRawEncoding(Loc);
}
};

const unsigned int NUM_PREDEF_PP_ENTITY_IDS = 1;

enum BlockIDs {
AST_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID,

SOURCE_MANAGER_BLOCK_ID,

PREPROCESSOR_BLOCK_ID,

DECLTYPES_BLOCK_ID,

PREPROCESSOR_DETAIL_BLOCK_ID,

SUBMODULE_BLOCK_ID,

COMMENTS_BLOCK_ID,

CONTROL_BLOCK_ID,

INPUT_FILES_BLOCK_ID,

OPTIONS_BLOCK_ID,

EXTENSION_BLOCK_ID,

UNHASHED_CONTROL_BLOCK_ID,
};

enum ControlRecordTypes {
METADATA = 1,

IMPORTS,

ORIGINAL_FILE,

ORIGINAL_PCH_DIR,

ORIGINAL_FILE_ID,

INPUT_FILE_OFFSETS,

MODULE_NAME,

MODULE_MAP_FILE,

MODULE_DIRECTORY,
};

enum OptionsRecordTypes {
LANGUAGE_OPTIONS = 1,

TARGET_OPTIONS,

FILE_SYSTEM_OPTIONS,

HEADER_SEARCH_OPTIONS,

PREPROCESSOR_OPTIONS,
};

enum UnhashedControlBlockRecordTypes {
SIGNATURE = 1,

DIAGNOSTIC_OPTIONS,

DIAG_PRAGMA_MAPPINGS,
};

enum ExtensionBlockRecordTypes {
EXTENSION_METADATA = 1,

FIRST_EXTENSION_RECORD_ID = 4
};

enum InputFileRecordTypes {
INPUT_FILE = 1
};

enum ASTRecordTypes {
TYPE_OFFSET = 1,

DECL_OFFSET = 2,

IDENTIFIER_OFFSET = 3,

METADATA_OLD_FORMAT = 4,

IDENTIFIER_TABLE = 5,

EAGERLY_DESERIALIZED_DECLS = 6,

SPECIAL_TYPES = 7,

STATISTICS = 8,

TENTATIVE_DEFINITIONS = 9,


SELECTOR_OFFSETS = 11,

METHOD_POOL = 12,

PP_COUNTER_VALUE = 13,

SOURCE_LOCATION_OFFSETS = 14,

SOURCE_LOCATION_PRELOADS = 15,

EXT_VECTOR_DECLS = 16,

UNUSED_FILESCOPED_DECLS = 17,

PPD_ENTITIES_OFFSETS = 18,

VTABLE_USES = 19,


REFERENCED_SELECTOR_POOL = 21,

TU_UPDATE_LEXICAL = 22,


SEMA_DECL_REFS = 24,

WEAK_UNDECLARED_IDENTIFIERS = 25,

PENDING_IMPLICIT_INSTANTIATIONS = 26,


UPDATE_VISIBLE = 28,

DECL_UPDATE_OFFSETS = 29,




CUDA_SPECIAL_DECL_REFS = 33,

HEADER_SEARCH_TABLE = 34,

FP_PRAGMA_OPTIONS = 35,

OPENCL_EXTENSIONS = 36,

DELEGATING_CTORS = 37,

KNOWN_NAMESPACES = 38,

MODULE_OFFSET_MAP = 39,

SOURCE_MANAGER_LINE_TABLE = 40,

OBJC_CATEGORIES_MAP = 41,

FILE_SORTED_DECLS = 42,

IMPORTED_MODULES = 43,


OBJC_CATEGORIES = 46,

MACRO_OFFSET = 47,

INTERESTING_IDENTIFIERS = 48,

UNDEFINED_BUT_USED = 49,

LATE_PARSED_TEMPLATE = 50,

OPTIMIZE_PRAGMA_OPTIONS = 51,

UNUSED_LOCAL_TYPEDEF_NAME_CANDIDATES = 52,


DELETE_EXPRS_TO_ANALYZE = 54,

MSSTRUCT_PRAGMA_OPTIONS = 55,

POINTERS_TO_MEMBERS_PRAGMA_OPTIONS = 56,

CUDA_PRAGMA_FORCE_HOST_DEVICE_DEPTH = 57,

OPENCL_EXTENSION_TYPES = 58,

OPENCL_EXTENSION_DECLS = 59,

MODULAR_CODEGEN_DECLS = 60,

PACK_PRAGMA_OPTIONS = 61,

PP_CONDITIONAL_STACK = 62,

PPD_SKIPPED_RANGES = 63
};

enum SourceManagerRecordTypes {
SM_SLOC_FILE_ENTRY = 1,

SM_SLOC_BUFFER_ENTRY = 2,

SM_SLOC_BUFFER_BLOB = 3,

SM_SLOC_BUFFER_BLOB_COMPRESSED = 4,

SM_SLOC_EXPANSION_ENTRY = 5
};

enum PreprocessorRecordTypes {

PP_MACRO_OBJECT_LIKE = 1,

PP_MACRO_FUNCTION_LIKE = 2,

PP_TOKEN = 3,

PP_MACRO_DIRECTIVE_HISTORY = 4,

PP_MODULE_MACRO = 5,
};

enum PreprocessorDetailRecordTypes {
PPD_MACRO_EXPANSION = 0,

PPD_MACRO_DEFINITION = 1,

PPD_INCLUSION_DIRECTIVE = 2
};

enum SubmoduleRecordTypes {
SUBMODULE_METADATA = 0,

SUBMODULE_DEFINITION = 1,

SUBMODULE_UMBRELLA_HEADER = 2,

SUBMODULE_HEADER = 3,

SUBMODULE_TOPHEADER = 4,

SUBMODULE_UMBRELLA_DIR = 5,

SUBMODULE_IMPORTS = 6,

SUBMODULE_EXPORTS = 7,

SUBMODULE_REQUIRES = 8,

SUBMODULE_EXCLUDED_HEADER = 9,

SUBMODULE_LINK_LIBRARY = 10,

SUBMODULE_CONFIG_MACRO = 11,

SUBMODULE_CONFLICT = 12,

SUBMODULE_PRIVATE_HEADER = 13,

SUBMODULE_TEXTUAL_HEADER = 14,

SUBMODULE_PRIVATE_TEXTUAL_HEADER = 15,

SUBMODULE_INITIALIZERS = 16,

SUBMODULE_EXPORT_AS = 17,
};

enum CommentRecordTypes {
COMMENTS_RAW_COMMENT = 0
};


enum PredefinedTypeIDs {
PREDEF_TYPE_NULL_ID       = 0,

PREDEF_TYPE_VOID_ID       = 1,

PREDEF_TYPE_BOOL_ID       = 2,

PREDEF_TYPE_CHAR_U_ID     = 3,

PREDEF_TYPE_UCHAR_ID      = 4,

PREDEF_TYPE_USHORT_ID     = 5,

PREDEF_TYPE_UINT_ID       = 6,

PREDEF_TYPE_ULONG_ID      = 7,

PREDEF_TYPE_ULONGLONG_ID  = 8,

PREDEF_TYPE_CHAR_S_ID     = 9,

PREDEF_TYPE_SCHAR_ID      = 10,

PREDEF_TYPE_WCHAR_ID      = 11,

PREDEF_TYPE_SHORT_ID      = 12,

PREDEF_TYPE_INT_ID        = 13,

PREDEF_TYPE_LONG_ID       = 14,

PREDEF_TYPE_LONGLONG_ID   = 15,

PREDEF_TYPE_FLOAT_ID      = 16,

PREDEF_TYPE_DOUBLE_ID     = 17,

PREDEF_TYPE_LONGDOUBLE_ID = 18,

PREDEF_TYPE_OVERLOAD_ID   = 19,

PREDEF_TYPE_DEPENDENT_ID  = 20,

PREDEF_TYPE_UINT128_ID    = 21,

PREDEF_TYPE_INT128_ID     = 22,

PREDEF_TYPE_NULLPTR_ID    = 23,

PREDEF_TYPE_CHAR16_ID     = 24,

PREDEF_TYPE_CHAR32_ID     = 25,

PREDEF_TYPE_OBJC_ID       = 26,

PREDEF_TYPE_OBJC_CLASS    = 27,

PREDEF_TYPE_OBJC_SEL      = 28,

PREDEF_TYPE_UNKNOWN_ANY   = 29,

PREDEF_TYPE_BOUND_MEMBER  = 30,

PREDEF_TYPE_AUTO_DEDUCT   = 31,

PREDEF_TYPE_AUTO_RREF_DEDUCT = 32,

PREDEF_TYPE_HALF_ID       = 33,

PREDEF_TYPE_ARC_UNBRIDGED_CAST = 34,

PREDEF_TYPE_PSEUDO_OBJECT = 35,

PREDEF_TYPE_BUILTIN_FN = 36,

PREDEF_TYPE_EVENT_ID      = 37,

PREDEF_TYPE_CLK_EVENT_ID  = 38,

PREDEF_TYPE_SAMPLER_ID    = 39,

PREDEF_TYPE_QUEUE_ID      = 40,

PREDEF_TYPE_RESERVE_ID_ID = 41,

PREDEF_TYPE_OMP_ARRAY_SECTION = 42,

PREDEF_TYPE_FLOAT128_ID = 43,

PREDEF_TYPE_FLOAT16_ID = 44,

PREDEF_TYPE_CHAR8_ID = 45,

PREDEF_TYPE_SHORT_ACCUM_ID    = 46,

PREDEF_TYPE_ACCUM_ID      = 47,

PREDEF_TYPE_LONG_ACCUM_ID = 48,

PREDEF_TYPE_USHORT_ACCUM_ID   = 49,

PREDEF_TYPE_UACCUM_ID     = 50,

PREDEF_TYPE_ULONG_ACCUM_ID    = 51,

PREDEF_TYPE_SHORT_FRACT_ID = 52,

PREDEF_TYPE_FRACT_ID = 53,

PREDEF_TYPE_LONG_FRACT_ID = 54,

PREDEF_TYPE_USHORT_FRACT_ID = 55,

PREDEF_TYPE_UFRACT_ID = 56,

PREDEF_TYPE_ULONG_FRACT_ID = 57,

PREDEF_TYPE_SAT_SHORT_ACCUM_ID = 58,

PREDEF_TYPE_SAT_ACCUM_ID = 59,

PREDEF_TYPE_SAT_LONG_ACCUM_ID = 60,

PREDEF_TYPE_SAT_USHORT_ACCUM_ID = 61,

PREDEF_TYPE_SAT_UACCUM_ID = 62,

PREDEF_TYPE_SAT_ULONG_ACCUM_ID = 63,

PREDEF_TYPE_SAT_SHORT_FRACT_ID = 64,

PREDEF_TYPE_SAT_FRACT_ID = 65,

PREDEF_TYPE_SAT_LONG_FRACT_ID = 66,

PREDEF_TYPE_SAT_USHORT_FRACT_ID = 67,

PREDEF_TYPE_SAT_UFRACT_ID = 68,

PREDEF_TYPE_SAT_ULONG_FRACT_ID = 69,

#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix) \
PREDEF_TYPE_##Id##_ID,
#include "clang/Basic/OpenCLImageTypes.def"
};

const unsigned NUM_PREDEF_TYPE_IDS = 200;

enum TypeCode {
TYPE_EXT_QUAL = 1,

TYPE_COMPLEX = 3,

TYPE_POINTER = 4,

TYPE_BLOCK_POINTER = 5,

TYPE_LVALUE_REFERENCE = 6,

TYPE_RVALUE_REFERENCE = 7,

TYPE_MEMBER_POINTER = 8,

TYPE_CONSTANT_ARRAY = 9,

TYPE_INCOMPLETE_ARRAY = 10,

TYPE_VARIABLE_ARRAY = 11,

TYPE_VECTOR = 12,

TYPE_EXT_VECTOR = 13,

TYPE_FUNCTION_NO_PROTO = 14,

TYPE_FUNCTION_PROTO = 15,

TYPE_TYPEDEF = 16,

TYPE_TYPEOF_EXPR = 17,

TYPE_TYPEOF = 18,

TYPE_RECORD = 19,

TYPE_ENUM = 20,

TYPE_OBJC_INTERFACE = 21,

TYPE_OBJC_OBJECT_POINTER = 22,

TYPE_DECLTYPE = 23,

TYPE_ELABORATED = 24,

TYPE_SUBST_TEMPLATE_TYPE_PARM = 25,

TYPE_UNRESOLVED_USING = 26,

TYPE_INJECTED_CLASS_NAME = 27,

TYPE_OBJC_OBJECT = 28,

TYPE_TEMPLATE_TYPE_PARM = 29,

TYPE_TEMPLATE_SPECIALIZATION = 30,

TYPE_DEPENDENT_NAME = 31,

TYPE_DEPENDENT_TEMPLATE_SPECIALIZATION = 32,

TYPE_DEPENDENT_SIZED_ARRAY = 33,

TYPE_PAREN = 34,

TYPE_PACK_EXPANSION = 35,

TYPE_ATTRIBUTED = 36,

TYPE_SUBST_TEMPLATE_TYPE_PARM_PACK = 37,

TYPE_AUTO = 38,

TYPE_UNARY_TRANSFORM = 39,

TYPE_ATOMIC = 40,

TYPE_DECAYED = 41,

TYPE_ADJUSTED = 42,

TYPE_PIPE = 43,

TYPE_OBJC_TYPE_PARAM = 44,

TYPE_DEDUCED_TEMPLATE_SPECIALIZATION = 45,

TYPE_DEPENDENT_SIZED_EXT_VECTOR = 46,

TYPE_DEPENDENT_ADDRESS_SPACE = 47,

TYPE_DEPENDENT_SIZED_VECTOR = 48
};

enum SpecialTypeIDs {
SPECIAL_TYPE_CF_CONSTANT_STRING          = 0,

SPECIAL_TYPE_FILE                        = 1,

SPECIAL_TYPE_JMP_BUF                     = 2,

SPECIAL_TYPE_SIGJMP_BUF                  = 3,

SPECIAL_TYPE_OBJC_ID_REDEFINITION        = 4,

SPECIAL_TYPE_OBJC_CLASS_REDEFINITION     = 5,

SPECIAL_TYPE_OBJC_SEL_REDEFINITION       = 6,

SPECIAL_TYPE_UCONTEXT_T                  = 7
};

const unsigned NumSpecialTypeIDs = 8;

enum PredefinedDeclIDs {
PREDEF_DECL_NULL_ID = 0,

PREDEF_DECL_TRANSLATION_UNIT_ID = 1,

PREDEF_DECL_OBJC_ID_ID = 2,

PREDEF_DECL_OBJC_SEL_ID = 3,

PREDEF_DECL_OBJC_CLASS_ID = 4,

PREDEF_DECL_OBJC_PROTOCOL_ID = 5,

PREDEF_DECL_INT_128_ID = 6,

PREDEF_DECL_UNSIGNED_INT_128_ID = 7,

PREDEF_DECL_OBJC_INSTANCETYPE_ID = 8,

PREDEF_DECL_BUILTIN_VA_LIST_ID = 9,

PREDEF_DECL_VA_LIST_TAG = 10,

PREDEF_DECL_BUILTIN_MS_VA_LIST_ID = 11,

PREDEF_DECL_EXTERN_C_CONTEXT_ID = 12,

PREDEF_DECL_MAKE_INTEGER_SEQ_ID = 13,

PREDEF_DECL_CF_CONSTANT_STRING_ID = 14,

PREDEF_DECL_CF_CONSTANT_STRING_TAG_ID = 15,

PREDEF_DECL_TYPE_PACK_ELEMENT_ID = 16,
};

const unsigned int NUM_PREDEF_DECL_IDS = 17;

const unsigned int DECL_UPDATES = 49;

const unsigned int LOCAL_REDECLARATIONS = 50;

enum DeclCode {
DECL_TYPEDEF = 51,

DECL_TYPEALIAS,

DECL_ENUM,

DECL_RECORD,

DECL_ENUM_CONSTANT,

DECL_FUNCTION,

DECL_OBJC_METHOD,

DECL_OBJC_INTERFACE,

DECL_OBJC_PROTOCOL,

DECL_OBJC_IVAR,

DECL_OBJC_AT_DEFS_FIELD,

DECL_OBJC_CATEGORY,

DECL_OBJC_CATEGORY_IMPL,

DECL_OBJC_IMPLEMENTATION,

DECL_OBJC_COMPATIBLE_ALIAS,

DECL_OBJC_PROPERTY,

DECL_OBJC_PROPERTY_IMPL,

DECL_FIELD,

DECL_MS_PROPERTY,

DECL_VAR,

DECL_IMPLICIT_PARAM,

DECL_PARM_VAR,

DECL_DECOMPOSITION,

DECL_BINDING,

DECL_FILE_SCOPE_ASM,

DECL_BLOCK,

DECL_CAPTURED,

DECL_CONTEXT_LEXICAL,

DECL_CONTEXT_VISIBLE,

DECL_LABEL,

DECL_NAMESPACE,

DECL_NAMESPACE_ALIAS,

DECL_USING,

DECL_USING_PACK,

DECL_USING_SHADOW,

DECL_CONSTRUCTOR_USING_SHADOW,

DECL_USING_DIRECTIVE,

DECL_UNRESOLVED_USING_VALUE,

DECL_UNRESOLVED_USING_TYPENAME,

DECL_LINKAGE_SPEC,

DECL_EXPORT,

DECL_CXX_RECORD,

DECL_CXX_DEDUCTION_GUIDE,

DECL_CXX_METHOD,

DECL_CXX_CONSTRUCTOR,

DECL_CXX_INHERITED_CONSTRUCTOR,

DECL_CXX_DESTRUCTOR,

DECL_CXX_CONVERSION,

DECL_ACCESS_SPEC,

DECL_FRIEND,

DECL_FRIEND_TEMPLATE,

DECL_CLASS_TEMPLATE,

DECL_CLASS_TEMPLATE_SPECIALIZATION,

DECL_CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,

DECL_VAR_TEMPLATE,

DECL_VAR_TEMPLATE_SPECIALIZATION,

DECL_VAR_TEMPLATE_PARTIAL_SPECIALIZATION,

DECL_FUNCTION_TEMPLATE,

DECL_TEMPLATE_TYPE_PARM,

DECL_NON_TYPE_TEMPLATE_PARM,

DECL_TEMPLATE_TEMPLATE_PARM,

DECL_TYPE_ALIAS_TEMPLATE,

DECL_STATIC_ASSERT,

DECL_CXX_BASE_SPECIFIERS,

DECL_CXX_CTOR_INITIALIZERS,

DECL_INDIRECTFIELD,

DECL_EXPANDED_NON_TYPE_TEMPLATE_PARM_PACK,

DECL_EXPANDED_TEMPLATE_TEMPLATE_PARM_PACK,

DECL_CLASS_SCOPE_FUNCTION_SPECIALIZATION,

DECL_IMPORT,

DECL_OMP_THREADPRIVATE,

DECL_EMPTY,

DECL_OBJC_TYPE_PARAM,

DECL_OMP_CAPTUREDEXPR,

DECL_PRAGMA_COMMENT,

DECL_PRAGMA_DETECT_MISMATCH,

DECL_OMP_DECLARE_REDUCTION,
};

enum StmtCode {
STMT_STOP = 128,

STMT_NULL_PTR,

STMT_REF_PTR,

STMT_NULL,

STMT_COMPOUND,

STMT_CASE,

STMT_DEFAULT,

STMT_LABEL,

STMT_ATTRIBUTED,

STMT_IF,

STMT_SWITCH,

STMT_WHILE,

STMT_DO,

STMT_FOR,

STMT_GOTO,

STMT_INDIRECT_GOTO,

STMT_CONTINUE,

STMT_BREAK,

STMT_RETURN,

STMT_DECL,

STMT_CAPTURED,

STMT_GCCASM,

STMT_MSASM,

EXPR_PREDEFINED,

EXPR_DECL_REF,

EXPR_INTEGER_LITERAL,

EXPR_FLOATING_LITERAL,

EXPR_IMAGINARY_LITERAL,

EXPR_STRING_LITERAL,

EXPR_CHARACTER_LITERAL,

EXPR_PAREN,

EXPR_PAREN_LIST,

EXPR_UNARY_OPERATOR,

EXPR_OFFSETOF,

EXPR_SIZEOF_ALIGN_OF,

EXPR_ARRAY_SUBSCRIPT,

EXPR_CALL,

EXPR_MEMBER,

EXPR_BINARY_OPERATOR,

EXPR_COMPOUND_ASSIGN_OPERATOR,

EXPR_CONDITIONAL_OPERATOR,

EXPR_IMPLICIT_CAST,

EXPR_CSTYLE_CAST,

EXPR_COMPOUND_LITERAL,

EXPR_EXT_VECTOR_ELEMENT,

EXPR_INIT_LIST,

EXPR_DESIGNATED_INIT,

EXPR_DESIGNATED_INIT_UPDATE,

EXPR_NO_INIT,

EXPR_ARRAY_INIT_LOOP,

EXPR_ARRAY_INIT_INDEX,

EXPR_IMPLICIT_VALUE_INIT,

EXPR_VA_ARG,

EXPR_ADDR_LABEL,

EXPR_STMT,

EXPR_CHOOSE,

EXPR_GNU_NULL,

EXPR_SHUFFLE_VECTOR,

EXPR_CONVERT_VECTOR,

EXPR_BLOCK,

EXPR_GENERIC_SELECTION,

EXPR_PSEUDO_OBJECT,

EXPR_ATOMIC,


EXPR_OBJC_STRING_LITERAL,

EXPR_OBJC_BOXED_EXPRESSION,
EXPR_OBJC_ARRAY_LITERAL,
EXPR_OBJC_DICTIONARY_LITERAL,

EXPR_OBJC_ENCODE,

EXPR_OBJC_SELECTOR_EXPR,

EXPR_OBJC_PROTOCOL_EXPR,

EXPR_OBJC_IVAR_REF_EXPR,

EXPR_OBJC_PROPERTY_REF_EXPR,

EXPR_OBJC_SUBSCRIPT_REF_EXPR,

EXPR_OBJC_KVC_REF_EXPR,

EXPR_OBJC_MESSAGE_EXPR,

EXPR_OBJC_ISA,

EXPR_OBJC_INDIRECT_COPY_RESTORE,

STMT_OBJC_FOR_COLLECTION,

STMT_OBJC_CATCH,

STMT_OBJC_FINALLY,

STMT_OBJC_AT_TRY,

STMT_OBJC_AT_SYNCHRONIZED,

STMT_OBJC_AT_THROW,

STMT_OBJC_AUTORELEASE_POOL,

EXPR_OBJC_BOOL_LITERAL,

EXPR_OBJC_AVAILABILITY_CHECK,


STMT_CXX_CATCH,

STMT_CXX_TRY,

STMT_CXX_FOR_RANGE,

EXPR_CXX_OPERATOR_CALL,

EXPR_CXX_MEMBER_CALL,

EXPR_CXX_CONSTRUCT,

EXPR_CXX_INHERITED_CTOR_INIT,

EXPR_CXX_TEMPORARY_OBJECT,

EXPR_CXX_STATIC_CAST,

EXPR_CXX_DYNAMIC_CAST,

EXPR_CXX_REINTERPRET_CAST,

EXPR_CXX_CONST_CAST,

EXPR_CXX_FUNCTIONAL_CAST,

EXPR_USER_DEFINED_LITERAL,

EXPR_CXX_STD_INITIALIZER_LIST,

EXPR_CXX_BOOL_LITERAL,

EXPR_CXX_NULL_PTR_LITERAL,  
EXPR_CXX_TYPEID_EXPR,       
EXPR_CXX_TYPEID_TYPE,       
EXPR_CXX_THIS,              
EXPR_CXX_THROW,             
EXPR_CXX_DEFAULT_ARG,       
EXPR_CXX_DEFAULT_INIT,      
EXPR_CXX_BIND_TEMPORARY,    

EXPR_CXX_SCALAR_VALUE_INIT, 
EXPR_CXX_NEW,               
EXPR_CXX_DELETE,            
EXPR_CXX_PSEUDO_DESTRUCTOR, 

EXPR_EXPR_WITH_CLEANUPS,    

EXPR_CXX_DEPENDENT_SCOPE_MEMBER,   
EXPR_CXX_DEPENDENT_SCOPE_DECL_REF, 
EXPR_CXX_UNRESOLVED_CONSTRUCT,     
EXPR_CXX_UNRESOLVED_MEMBER,        
EXPR_CXX_UNRESOLVED_LOOKUP,        

EXPR_CXX_EXPRESSION_TRAIT,  
EXPR_CXX_NOEXCEPT,          

EXPR_OPAQUE_VALUE,          
EXPR_BINARY_CONDITIONAL_OPERATOR,  
EXPR_TYPE_TRAIT,            
EXPR_ARRAY_TYPE_TRAIT,      

EXPR_PACK_EXPANSION,        
EXPR_SIZEOF_PACK,           
EXPR_SUBST_NON_TYPE_TEMPLATE_PARM, 
EXPR_SUBST_NON_TYPE_TEMPLATE_PARM_PACK,
EXPR_FUNCTION_PARM_PACK,    
EXPR_MATERIALIZE_TEMPORARY, 
EXPR_CXX_FOLD,              

EXPR_CUDA_KERNEL_CALL,       

EXPR_ASTYPE,                 

EXPR_CXX_PROPERTY_REF_EXPR, 
EXPR_CXX_PROPERTY_SUBSCRIPT_EXPR, 
EXPR_CXX_UUIDOF_EXPR,       
EXPR_CXX_UUIDOF_TYPE,       
STMT_SEH_LEAVE,             
STMT_SEH_EXCEPT,            
STMT_SEH_FINALLY,           
STMT_SEH_TRY,               

STMT_OMP_PARALLEL_DIRECTIVE,
STMT_OMP_SIMD_DIRECTIVE,
STMT_OMP_FOR_DIRECTIVE,
STMT_OMP_FOR_SIMD_DIRECTIVE,
STMT_OMP_SECTIONS_DIRECTIVE,
STMT_OMP_SECTION_DIRECTIVE,
STMT_OMP_SINGLE_DIRECTIVE,
STMT_OMP_MASTER_DIRECTIVE,
STMT_OMP_CRITICAL_DIRECTIVE,
STMT_OMP_PARALLEL_FOR_DIRECTIVE,
STMT_OMP_PARALLEL_FOR_SIMD_DIRECTIVE,
STMT_OMP_PARALLEL_SECTIONS_DIRECTIVE,
STMT_OMP_TASK_DIRECTIVE,
STMT_OMP_TASKYIELD_DIRECTIVE,
STMT_OMP_BARRIER_DIRECTIVE,
STMT_OMP_TASKWAIT_DIRECTIVE,
STMT_OMP_FLUSH_DIRECTIVE,
STMT_OMP_ORDERED_DIRECTIVE,
STMT_OMP_ATOMIC_DIRECTIVE,
STMT_OMP_TARGET_DIRECTIVE,
STMT_OMP_TARGET_DATA_DIRECTIVE,
STMT_OMP_TARGET_ENTER_DATA_DIRECTIVE,
STMT_OMP_TARGET_EXIT_DATA_DIRECTIVE,
STMT_OMP_TARGET_PARALLEL_DIRECTIVE,
STMT_OMP_TARGET_PARALLEL_FOR_DIRECTIVE,
STMT_OMP_TEAMS_DIRECTIVE,
STMT_OMP_TASKGROUP_DIRECTIVE,
STMT_OMP_CANCELLATION_POINT_DIRECTIVE,
STMT_OMP_CANCEL_DIRECTIVE,
STMT_OMP_TASKLOOP_DIRECTIVE,
STMT_OMP_TASKLOOP_SIMD_DIRECTIVE,
STMT_OMP_DISTRIBUTE_DIRECTIVE,
STMT_OMP_TARGET_UPDATE_DIRECTIVE,
STMT_OMP_DISTRIBUTE_PARALLEL_FOR_DIRECTIVE,
STMT_OMP_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE,
STMT_OMP_DISTRIBUTE_SIMD_DIRECTIVE,
STMT_OMP_TARGET_PARALLEL_FOR_SIMD_DIRECTIVE,
STMT_OMP_TARGET_SIMD_DIRECTIVE,
STMT_OMP_TEAMS_DISTRIBUTE_DIRECTIVE,
STMT_OMP_TEAMS_DISTRIBUTE_SIMD_DIRECTIVE,
STMT_OMP_TEAMS_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE,
STMT_OMP_TEAMS_DISTRIBUTE_PARALLEL_FOR_DIRECTIVE,
STMT_OMP_TARGET_TEAMS_DIRECTIVE,
STMT_OMP_TARGET_TEAMS_DISTRIBUTE_DIRECTIVE,
STMT_OMP_TARGET_TEAMS_DISTRIBUTE_PARALLEL_FOR_DIRECTIVE,
STMT_OMP_TARGET_TEAMS_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE,
STMT_OMP_TARGET_TEAMS_DISTRIBUTE_SIMD_DIRECTIVE,
EXPR_OMP_ARRAY_SECTION,

EXPR_OBJC_BRIDGED_CAST,     

STMT_MS_DEPENDENT_EXISTS,   
EXPR_LAMBDA,                
STMT_COROUTINE_BODY,
STMT_CORETURN,
EXPR_COAWAIT,
EXPR_COYIELD,
EXPR_DEPENDENT_COAWAIT,
};

enum DesignatorTypes {
DESIG_FIELD_NAME  = 0,

DESIG_FIELD_DECL  = 1,

DESIG_ARRAY       = 2,

DESIG_ARRAY_RANGE = 3
};

enum CtorInitializerType {
CTOR_INITIALIZER_BASE,
CTOR_INITIALIZER_DELEGATING,
CTOR_INITIALIZER_MEMBER,
CTOR_INITIALIZER_INDIRECT_MEMBER
};

struct LocalRedeclarationsInfo {
DeclID FirstID;

unsigned Offset;

friend bool operator<(const LocalRedeclarationsInfo &X,
const LocalRedeclarationsInfo &Y) {
return X.FirstID < Y.FirstID;
}

friend bool operator>(const LocalRedeclarationsInfo &X,
const LocalRedeclarationsInfo &Y) {
return X.FirstID > Y.FirstID;
}

friend bool operator<=(const LocalRedeclarationsInfo &X,
const LocalRedeclarationsInfo &Y) {
return X.FirstID <= Y.FirstID;
}

friend bool operator>=(const LocalRedeclarationsInfo &X,
const LocalRedeclarationsInfo &Y) {
return X.FirstID >= Y.FirstID;
}
};

struct ObjCCategoriesInfo {
DeclID DefinitionID;

unsigned Offset;

friend bool operator<(const ObjCCategoriesInfo &X,
const ObjCCategoriesInfo &Y) {
return X.DefinitionID < Y.DefinitionID;
}

friend bool operator>(const ObjCCategoriesInfo &X,
const ObjCCategoriesInfo &Y) {
return X.DefinitionID > Y.DefinitionID;
}

friend bool operator<=(const ObjCCategoriesInfo &X,
const ObjCCategoriesInfo &Y) {
return X.DefinitionID <= Y.DefinitionID;
}

friend bool operator>=(const ObjCCategoriesInfo &X,
const ObjCCategoriesInfo &Y) {
return X.DefinitionID >= Y.DefinitionID;
}
};

class DeclarationNameKey {
using NameKind = unsigned;

NameKind Kind = 0;
uint64_t Data = 0;

public:
DeclarationNameKey() = default;
DeclarationNameKey(DeclarationName Name);
DeclarationNameKey(NameKind Kind, uint64_t Data)
: Kind(Kind), Data(Data) {}

NameKind getKind() const { return Kind; }

IdentifierInfo *getIdentifier() const {
assert(Kind == DeclarationName::Identifier ||
Kind == DeclarationName::CXXLiteralOperatorName ||
Kind == DeclarationName::CXXDeductionGuideName);
return (IdentifierInfo *)Data;
}

Selector getSelector() const {
assert(Kind == DeclarationName::ObjCZeroArgSelector ||
Kind == DeclarationName::ObjCOneArgSelector ||
Kind == DeclarationName::ObjCMultiArgSelector);
return Selector(Data);
}

OverloadedOperatorKind getOperatorKind() const {
assert(Kind == DeclarationName::CXXOperatorName);
return (OverloadedOperatorKind)Data;
}

unsigned getHash() const;

friend bool operator==(const DeclarationNameKey &A,
const DeclarationNameKey &B) {
return A.Kind == B.Kind && A.Data == B.Data;
}
};


} 
} 

namespace llvm {

template <> struct DenseMapInfo<clang::serialization::DeclarationNameKey> {
static clang::serialization::DeclarationNameKey getEmptyKey() {
return clang::serialization::DeclarationNameKey(-1, 1);
}

static clang::serialization::DeclarationNameKey getTombstoneKey() {
return clang::serialization::DeclarationNameKey(-1, 2);
}

static unsigned
getHashValue(const clang::serialization::DeclarationNameKey &Key) {
return Key.getHash();
}

static bool isEqual(const clang::serialization::DeclarationNameKey &L,
const clang::serialization::DeclarationNameKey &R) {
return L == R;
}
};

} 

#endif 
