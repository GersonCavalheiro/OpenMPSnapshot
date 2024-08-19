
#ifndef LLVM_CLANG_BASIC_PRAGMA_KINDS_H
#define LLVM_CLANG_BASIC_PRAGMA_KINDS_H

namespace clang {

enum PragmaMSCommentKind {
PCK_Unknown,
PCK_Linker,   
PCK_Lib,      
PCK_Compiler, 
PCK_ExeStr,   
PCK_User      
};

enum PragmaMSStructKind {
PMSST_OFF, 
PMSST_ON   
};

}

#endif
