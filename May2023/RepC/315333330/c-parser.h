#ifndef GCC_C_PARSER_H
#define GCC_C_PARSER_H
enum c_id_kind {
C_ID_ID,
C_ID_TYPENAME,
C_ID_CLASSNAME,
C_ID_ADDRSPACE,
C_ID_NONE
};
struct GTY (()) c_token {
ENUM_BITFIELD (cpp_ttype) type : 8;
ENUM_BITFIELD (c_id_kind) id_kind : 8;
ENUM_BITFIELD (rid) keyword : 8;
ENUM_BITFIELD (pragma_kind) pragma_kind : 8;
location_t location;
tree value;
unsigned char flags;
source_range get_range () const
{
return get_range_from_loc (line_table, location);
}
location_t get_finish () const
{
return get_range ().m_finish;
}
};
struct c_parser;
enum c_dtr_syn {
C_DTR_NORMAL,
C_DTR_ABSTRACT,
C_DTR_PARM
};
enum c_parser_prec {
PREC_NONE,
PREC_LOGOR,
PREC_LOGAND,
PREC_BITOR,
PREC_BITXOR,
PREC_BITAND,
PREC_EQ,
PREC_REL,
PREC_SHIFT,
PREC_ADD,
PREC_MULT,
NUM_PRECS
};
enum c_lookahead_kind {
cla_prefer_type,
cla_nonabstract_decl,
cla_prefer_id
};
extern c_token * c_parser_peek_token (c_parser *parser);
extern c_token * c_parser_peek_2nd_token (c_parser *parser);
extern c_token * c_parser_peek_nth_token (c_parser *parser, unsigned int n);
extern bool c_parser_require (c_parser *parser, enum cpp_ttype type,
const char *msgid,
location_t matching_location = UNKNOWN_LOCATION,
bool type_is_unique=true);
extern bool c_parser_error (c_parser *parser, const char *gmsgid);
extern void c_parser_consume_token (c_parser *parser);
extern void c_parser_skip_until_found (c_parser *parser, enum cpp_ttype type,
const char *msgid,
location_t = UNKNOWN_LOCATION);
extern bool c_parser_next_token_starts_declspecs (c_parser *parser);
bool c_parser_next_tokens_start_declaration (c_parser *parser);
bool c_token_starts_typename (c_token *token);
extern c_token * c_parser_tokens_buf (c_parser *parser, unsigned n);
extern bool c_parser_error (c_parser *parser);
extern void c_parser_set_error (c_parser *parser, bool);
static inline bool
c_parser_next_token_is (c_parser *parser, enum cpp_ttype type)
{
return c_parser_peek_token (parser)->type == type;
}
static inline bool
c_parser_next_token_is_not (c_parser *parser, enum cpp_ttype type)
{
return !c_parser_next_token_is (parser, type);
}
static inline bool
c_parser_next_token_is_keyword (c_parser *parser, enum rid keyword)
{
return c_parser_peek_token (parser)->keyword == keyword;
}
extern struct c_declarator *
c_parser_declarator (c_parser *parser, bool type_seen_p, c_dtr_syn kind,
bool *seen_id);
extern void c_parser_declspecs (c_parser *, struct c_declspecs *, bool, bool,
bool, bool, bool, enum c_lookahead_kind);
extern struct c_type_name *c_parser_type_name (c_parser *, bool = false);
#endif
