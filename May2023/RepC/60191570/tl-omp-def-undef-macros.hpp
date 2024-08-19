#ifndef OMP_DEF_UNDEF_MACROS_HPP
#define OMP_DEF_UNDEF_MACROS_HPP
#ifndef CLASSNAME
#error "Macro 'CLASSNAME' must be defined before including the 'tl-omp-def-undef-macros.hpp' header"
#endif
#define INVALID_STATEMENT_HANDLER(_func_prefix, _name) \
void CLASSNAME::_func_prefix##_name##_handler_pre(TL::PragmaCustomStatement ctr) { \
error_printf_at(ctr.get_locus(), "invalid '#pragma %s %s'\n",  \
ctr.get_text().c_str(), \
ctr.get_pragma_line().get_text().c_str()); \
} \
void CLASSNAME::_name##_handler_post(TL::PragmaCustomStatement) { }
#define OMP_INVALID_STATEMENT_HANDLER(_name) INVALID_STATEMENT_HANDLER(  , _name)
#define OSS_INVALID_STATEMENT_HANDLER(_name) INVALID_STATEMENT_HANDLER(oss_ , _name)
#define INVALID_DECLARATION_HANDLER(_func_prefix, _name) \
void CLASSNAME::_func_prefix##_name##_handler_pre(TL::PragmaCustomDeclaration ctr) { \
error_printf_at(ctr.get_locus(), "invalid '#pragma %s %s'\n",  \
ctr.get_text().c_str(), \
ctr.get_pragma_line().get_text().c_str()); \
} \
void CLASSNAME::_func_prefix##_name##_handler_post(TL::PragmaCustomDeclaration) { }
#define OMP_INVALID_DECLARATION_HANDLER(_name) INVALID_DECLARATION_HANDLER(  , _name)
#define OSS_INVALID_DECLARATION_HANDLER(_name) INVALID_DECLARATION_HANDLER(oss_ , _name)
#define EMPTY_STATEMENT_HANDLER(_func_prefix,_name) \
void CLASSNAME::_func_prefix##_name##_handler_pre(TL::PragmaCustomStatement) { } \
void CLASSNAME::_func_prefix##_name##_handler_post(TL::PragmaCustomStatement) { }
#define OMP_EMPTY_STATEMENT_HANDLER(_name) EMPTY_STATEMENT_HANDLER(, _name)
#define OSS_EMPTY_STATEMENT_HANDLER(_name) EMPTY_STATEMENT_HANDLER(oss_, _name)
#define EMPTY_DECLARATION_HANDLER(_func_prefix,_name) \
void CLASSNAME::_func_prefix##_name##_handler_pre(TL::PragmaCustomDeclaration) { } \
void CLASSNAME::_func_prefix##_name##_handler_post(TL::PragmaCustomDeclaration) { }
#define OMP_EMPTY_DECLARATION_HANDLER(_name) EMPTY_DECLARATION_HANDLER(, _name)
#define OSS_EMPTY_DECLARATION_HANDLER(_name) EMPTY_DECLARATION_HANDLER(oss_, _name)
#define EMPTY_DIRECTIVE_HANDLER(_func_prefix,_name) \
void CLASSNAME::_func_prefix##_name##_handler_pre(TL::PragmaCustomDirective) { } \
void CLASSNAME::_func_prefix##_name##_handler_post(TL::PragmaCustomDirective) { }
#define OMP_EMPTY_DIRECTIVE_HANDLER(_name) EMPTY_DIRECTIVE_HANDLER(, _name)
#define OSS_EMPTY_DIRECTIVE_HANDLER(_name) EMPTY_DIRECTIVE_HANDLER(oss_, _name)
#define UNIMPLEMENTED_STATEMENT_HANDLER(_func_prefix, _name) \
void CLASSNAME::_func_prefix##_name##_handler_pre(TL::PragmaCustomStatement ctr) { \
error_printf_at(ctr.get_locus(), "OpenMP construct not implemented\n");\
} \
void CLASSNAME::_func_prefix##_name##_handler_post(TL::PragmaCustomStatement) { }
#define OMP_UNIMPLEMENTED_STATEMENT_HANDLER(_name) UNIMPLEMENTED_STATEMENT_HANDLER(, _name)
#define OSS_UNIMPLEMENTED_STATEMENT_HANDLER(_name) UNIMPLEMENTED_STATEMENT_HANDLER(oss_, _name)
#define OSS_TO_OMP_STATEMENT_HANDLER(_name) \
void CLASSNAME::oss_##_name##_handler_pre(TL::PragmaCustomStatement construct) {\
_name##_handler_pre(construct); \
} \
void CLASSNAME::oss_##_name##_handler_post(TL::PragmaCustomStatement construct) {\
_name##_handler_post(construct); \
}
#define OSS_TO_OMP_DECLARATION_HANDLER(_name) \
void CLASSNAME::oss_##_name##_handler_pre(TL::PragmaCustomDeclaration construct) {\
_name##_handler_pre(construct); \
} \
void CLASSNAME::oss_##_name##_handler_post(TL::PragmaCustomDeclaration construct) {\
_name##_handler_post(construct); \
}
#define OSS_TO_OMP_DIRECTIVE_HANDLER(_name) \
void CLASSNAME::oss_##_name##_handler_pre(TL::PragmaCustomDirective construct) {\
_name##_handler_pre(construct); \
} \
void CLASSNAME::oss_##_name##_handler_post(TL::PragmaCustomDirective construct) {\
_name##_handler_post(construct); \
}
#else
#undef OSS_TO_OMP_DIRECTIVE_HANDLER
#undef OSS_TO_OMP_DECLARATION_HANDLER
#undef OSS_TO_OMP_STATEMENT_HANDLER
#undef OSS_UNIMPLEMENTED_STATEMENT_HANDLER
#undef OMP_UNIMPLEMENTED_STATEMENT_HANDLER
#undef     UNIMPLEMENTED_STATEMENT_HANDLER
#undef OSS_EMPTY_DIRECTIVE_HANDLER
#undef OMP_EMPTY_DIRECTIVE_HANDLER
#undef     EMPTY_DIRECTIVE_HANDLER
#undef OSS_EMPTY_DECLARATION_HANDLER
#undef OMP_EMPTY_DECLARATION_HANDLER
#undef     EMPTY_DECLARATION_HANDLER
#undef OSS_EMPTY_STATEMENT_HANDLER
#undef OMP_EMPTY_STATEMENT_HANDLER
#undef     EMPTY_STATEMENT_HANDLER
#undef OSS_INVALID_DECLARATION_HANDLER
#undef OMP_INVALID_DECLARATION_HANDLER
#undef     INVALID_DECLARATION_HANDLER
#undef OSS_INVALID_STATEMENT_HANDLER
#undef OMP_INVALID_STATEMENT_HANDLER
#undef     INVALID_STATEMENT_HANDLER
#undef CLASSNAME
#undef OMP_DEF_UNDEF_MACROS_HPP
#endif 
