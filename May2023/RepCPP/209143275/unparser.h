

#ifndef UNPARSER_H
#define UNPARSER_H



#define       SPLC_GLOBAL_DEBUG_SWITCH             0



#define UNPARSE_FIRST_AS_BLOCK    \
MAKE_INDENT; unparser->ofs << "{\n"; \
INC_INDENT; \
UNPARSE_FIRST; \
DEC_INDENT; \
MAKE_INDENT; unparser->ofs << "}\n"

#define UNPARSE_FIRST_AS_BLOCK_TEST(VT_TYPE)    \
MAKE_INDENT; unparser->ofs << "{\n"; \
INC_INDENT; \
UNPARSE_FIRST_TEST(VT_TYPE); \
DEC_INDENT; \
MAKE_INDENT; unparser->ofs << "}"

#define FIRST  (*(v->get_children().begin()))
#define SECOND (*(++v->get_children().begin()))
#define THIRD  (*(++++v->get_children().begin()))
#define LAST          (*(--v->get_children().end()))
#define SECOND_LAST   (*(----v->get_children().end()))
#define FIRSTOF(v)         (*((v)->get_children().begin()))
#define SECONDOF(v)        (*(++(v)->get_children().begin()))
#define THIRDOF(v)         (*(++++(v)->get_children().begin()))
#define LASTOF(v)          (*(--(v)->get_children().end()))
#define SECOND_LASTOF(v)   (*(----(v)->get_children().end()))

#define MPFR_FLOAT_PRECISION  200
#define MAKE_INDENT  for(int _i_=0;_i_<unparser->indent;_i_++) unparser->ofs << " "
#define INC_INDENT   unparser->indent+=TAB_WIDTH;
#define DEC_INDENT   unparser->indent-=TAB_WIDTH;
#define UNPARSE_ALL         for(list<ast_vertex*>::const_iterator it=unparser->v->get_children().begin();it!=unparser->v->get_children().end();it++) { assert(*it); unparser->v = *it; (*it)->unparse(argv); unparser->v=v; }
#define UNPARSE_FIRST       assert(unparser->v->get_children().size()>0); assert(FIRST); unparser->v = FIRST; FIRST->unparse(argv); unparser->v=v
#define UNPARSE_SECOND      assert(unparser->v->get_children().size()>1); assert(SECOND); unparser->v = SECOND; SECOND->unparse(argv); unparser->v=v
#define UNPARSE_THIRD       assert(unparser->v->get_children().size()>2); assert(THIRD); unparser->v = THIRD; THIRD->unparse(argv); unparser->v=v
#define UNPARSE_FIRST_TEST(VT_TYPE)       if(FIRST->get_type()==VT_TYPE) { unparser->v = FIRST; FIRST->unparse(argv); unparser->v=v; }
#define UNPARSE_SECOND_TEST(VT_TYPE)      if(SECOND->get_type()==VT_TYPE) { unparser->v = SECOND; SECOND->unparse(argv); unparser->v=v; }
#define UNPARSE_THIRD_TEST(VT_TYPE)       if(THIRD->get_type()==VT_TYPE) { unparser->v = THIRD; THIRD->unparse(argv); unparser->v=v; }
#define DEFINE_UNPARSER           Unparser* unparser=(Unparser*)argv; ast_vertex* v =unparser->v; assert(v)
#define TL_TRANSF      "\\tl"
#define ADJ_TRANSF_F   "\\adjf"
#define ADJ_TRANSF_R   "\\adjr"





#define REDIRECT_UNPARSING(ORIGINAL,TL,PASSIVE,AUGMENTED_FORWARD,REVERSE)    \
switch(unparser->current_mode) {  \
case unparse_original:          \
ORIGINAL(argv);                 \
break;                    \
case unparse_tangent_linear:    \
TL(argv);                       \
break;                    \
case unparse_passive:           \
PASSIVE(argv);	              \
break;                    \
case unparse_augmented_forward: \
AUGMENTED_FORWARD(argv);        \
break;                    \
case unparse_reverse:           \
REVERSE(argv);                  \
break;                    \
default: cerr << "error: The type " << (unparser->current_mode) << " is unknown." << endl;assert(0);             \
}


#define  OMP_GET_WTIME_START        \
if( option_papi || option_time ) { \
MAKE_INDENT; unparser->ofs << "start=omp_get_wtime();\n"; 	\
}

#define  OMP_GET_WTIME_END       \
if( option_papi || option_time ) { \
MAKE_INDENT; unparser->ofs << "end=omp_get_wtime();\n"; \
MAKE_INDENT; unparser->ofs << "elapsed_time=end-start;\n"; \
MAKE_INDENT; unparser->ofs << "fprintf(stderr, \"parallel region took : %18.2F sec\\n\", elapsed_time);\n";  \
}


#define PAPI_SHARED_VARIABLES   \
if(option_papi) {  \
MAKE_INDENT; unparser->ofs << "int p = omp_get_max_threads() ;\n";  \
MAKE_INDENT; unparser->ofs << "const int number_of_events = 2;\n";  \
MAKE_INDENT; unparser->ofs << "int* EventSets = new int [p];\n";  \
MAKE_INDENT; unparser->ofs << "for(i=0;i<p;i++) {EventSets[i] = PAPI_NULL;\n";  \
MAKE_INDENT; unparser->ofs << "long long **values = new long long* [p];\n";  \
MAKE_INDENT; unparser->ofs << "long long fp_ops=0;\n";  \
MAKE_INDENT; unparser->ofs << "long long mflops=0;\n";  \
MAKE_INDENT; unparser->ofs << "long long max_cyc=0;\n";  \
MAKE_INDENT; unparser->ofs << "long long elapsed_cyc;\n";  \
MAKE_INDENT; unparser->ofs << "double elapsed_time=0;\n";  \
MAKE_INDENT; unparser->ofs << "int mhz=-1;\n";  \
MAKE_INDENT; unparser->ofs << "\n";  \
MAKE_INDENT; unparser->ofs << "int global_retval = PAPI_library_init( PAPI_VER_CURRENT ); assert(global_retval == PAPI_VER_CURRENT );\n";  \
MAKE_INDENT; unparser->ofs << "assert( omp_get_nested()==0 ); 
MAKE_INDENT; unparser->ofs << "global_retval = PAPI_thread_init( ( unsigned long (*)( void ) ) ( omp_get_thread_num ) ); assert(global_retval==PAPI_OK);\n";  \
}

#define PAPI_HEADER   \
if(option_papi) {  \
MAKE_INDENT; unparser->ofs << "int PAPI_tid=omp_get_thread_num();\n";  \
MAKE_INDENT; unparser->ofs << "int PAPI_retval;\n";  \
MAKE_INDENT; unparser->ofs << "\n";  \
MAKE_INDENT; unparser->ofs << "PAPI_retval = PAPI_create_eventset( &EventSets[PAPI_tid] ); assert(PAPI_retval==PAPI_OK);\n";  \
MAKE_INDENT; unparser->ofs << "PAPI_values[PAPI_tid] = new long long [number_of_events]; assert(PAPI_values[PAPI_tid]);\n";  \
MAKE_INDENT; unparser->ofs << "assert( PAPI_query_event( PAPI_FP_OPS ) == PAPI_OK );\n";  \
MAKE_INDENT; unparser->ofs << "PAPI_retval = PAPI_add_named_event( EventSets[PAPI_tid], (char*)\"PAPI_FP_OPS\" ); assert(PAPI_retval==PAPI_OK);\n";  \
MAKE_INDENT; unparser->ofs << "assert( PAPI_query_event( PAPI_TOT_CYC ) == PAPI_OK );\n";  \
MAKE_INDENT; unparser->ofs << "PAPI_retval = PAPI_add_named_event( EventSets[PAPI_tid], (char*)\"PAPI_TOT_CYC\" ); assert(PAPI_retval==PAPI_OK);\n";  \
MAKE_INDENT; unparser->ofs << "\n";  \
MAKE_INDENT; unparser->ofs << "#pragma omp master\n";  \
MAKE_INDENT; unparser->ofs << "{\n";  \
INC_INDENT; \
MAKE_INDENT; unparser->ofs << "PAPI_retval = PAPI_start( EventSets[PAPI_tid] ); assert(PAPI_retval==PAPI_OK);\n";  \
MAKE_INDENT; unparser->ofs << "PAPI_elapsed_cyc = PAPI_get_real_cyc();\n";  \
MAKE_INDENT; unparser->ofs << "usleep(1e6);\n";  \
MAKE_INDENT; unparser->ofs << "PAPI_elapsed_cyc = PAPI_get_real_cyc() - PAPI_elapsed_cyc;\n";  \
MAKE_INDENT; unparser->ofs << "PAPI_mhz = PAPI_elapsed_cyc / 1e6; assert(PAPI_mhz>1000); \n";  \
MAKE_INDENT; unparser->ofs << "PAPI_retval = PAPI_stop( EventSets[PAPI_tid], PAPI_values[PAPI_tid] ); assert(PAPI_retval==PAPI_OK);\n";  \
MAKE_INDENT; unparser->ofs << "PAPI_retval = PAPI_reset( EventSets[PAPI_tid] ); assert(PAPI_retval==PAPI_OK);\n";  \
DEC_INDENT; \
MAKE_INDENT; unparser->ofs << "}\n";  \
MAKE_INDENT; unparser->ofs << "\n";  \
MAKE_INDENT; unparser->ofs << "PAPI_retval = PAPI_start( EventSets[PAPI_tid] ); assert(PAPI_retval==PAPI_OK);\n";  \
}

#define PAPI_FOOTER   \
if(option_papi) {  \
MAKE_INDENT; unparser->ofs << "\n";  \
MAKE_INDENT; unparser->ofs << "PAPI_retval = PAPI_stop( EventSets[PAPI_tid], PAPI_values[PAPI_tid] ); assert(PAPI_retval==PAPI_OK);\n";  \
}

#define PAPI_PRINT_MFLOPS  \
if(option_papi) {   \
MAKE_INDENT; unparser->ofs << "fp_ops=0;max_cyc=0;\n"; \
\
MAKE_INDENT; unparser->ofs << "cerr << \"MHz                : \" << setw(17) << PAPI_mhz     << endl;\n";  \
MAKE_INDENT; unparser->ofs << "for(i=0;i<p;i++) { \n"; \
MAKE_INDENT; unparser->ofs << "\tfp_ops+=PAPI_values[i][0];\n"; \
MAKE_INDENT; unparser->ofs << "\tif(PAPI_values[i][1]>max_cyc) max_cyc=PAPI_values[i][1];\n"; \
MAKE_INDENT; unparser->ofs << "\tcerr << \"Thread \" << setw(3) << i << \": FP_OPS : \" << setw(17) << PAPI_values[i][0] << \" (\"<< setw(10) <<PAPI_values[i][0]/1e6<<\"*10^6)           TOT_CYC : \" << setw(17) << PAPI_values[i][1] << \" (\"<< setw(10) <<PAPI_values[i][1]/1e6<<\"*10^6)\" << endl;\n";  \
MAKE_INDENT; unparser->ofs << "}\n"; \
MAKE_INDENT; unparser->ofs << "cerr << \"overall FP_OPS     : \" << setw(17) << fp_ops  << \" (\"<< setw(10) <<fp_ops/1e9<<\"*10^9)    max of TOT_CYC : \" << setw(17) << max_cyc << \" (\"<< setw(10) <<max_cyc/1e9<<\"*10^9)\" << endl;\n";  \
MAKE_INDENT; unparser->ofs << "if(PAPI_mhz>0 && max_cyc/PAPI_mhz>0)  mflops=fp_ops/(max_cyc/PAPI_mhz); else mflops=0;\n";  \
MAKE_INDENT; unparser->ofs << "cerr << \"MFLOPS             : \" << setw(17) << mflops << endl;\n";  \
MAKE_INDENT; unparser->ofs << "fprintf(stderr, \"\\n\\n\");\n";  \
MAKE_INDENT; unparser->ofs << "for(i=0;i<p;i++) { global_PAPI_retval=PAPI_cleanup_eventset(EventSets[i]); assert(global_PAPI_retval==PAPI_OK); PAPI_destroy_eventset(&EventSets[i]); assert(global_PAPI_retval==PAPI_OK); }\n"; \
} 


enum unparse_modes {
unparse_none=10000,
unparse_original,
unparse_tangent_linear,
unparse_adjoint,
unparse_passive,
unparse_augmented_forward,
unparse_reverse
};

extern bool option_long_double;
extern bool option_mpfr;
extern bool option_suppress_global_region;
extern bool option_papi;
extern bool option_time;
extern bool option_prg_analysis;
extern bool option_mem_statistic;
extern bool option_suppress_atomic;


class ast_vertex;

class Unparser {
public:
unparse_modes current_mode;
ast_vertex* v;
ostringstream oss;
ofstream ofs;
list<string> code;
int indent;
bool suppress_linefeed;
std::string float_type;

Unparser() : current_mode(unparse_none), indent(0), suppress_linefeed(false), float_type("double")  
{
if(option_long_double)
float_type="long double";
else if(option_mpfr)
float_type="mpfr_t";
}
};

extern std::string src_filename;

extern enum unparse_modes current_mode;


void unparser_spl_code(void* argv);
void unparser_global_declarations(void* argv);

void unparser_spl_STACKf(void* argv);
void unparser_spl_STACKi(void* argv);
void unparser_spl_STACKc(void* argv);

void unparser_spl_STACKfpush(void* argv);
void unparser_spl_STACKipush(void* argv);
void unparser_spl_STACKcpush(void* argv);
void unparser_spl_STACKcempty(void* argv);

void unparser_spl_STACKfpop(void* argv);
void unparser_spl_STACKipop(void* argv);
void unparser_spl_STACKcpop(void* argv);

void unparser_spl_int_assign(void* argv);
void unparser_spl_float_assign(void* argv);
void unparser_spl_float_plus_assign(void* argv);
void unparser_spl_cond_if(void* argv);
void unparser_spl_cond_while(void* argv);

void unparser_spl_identifier(void* argv);
void unparser_spl_array_index(void* argv);
void unparser_spl_expr_in_brackets(void* argv);
void unparser_spl_expr_negative(void* argv);

void unparser_spl_float_const(void* argv);
void unparser_spl_float_sin(void* argv);
void unparser_spl_float_cos(void* argv);
void unparser_spl_float_exp(void* argv);
void unparser_spl_float_mul(void* argv);
void unparser_spl_float_div(void* argv);
void unparser_spl_float_add(void* argv);
void unparser_spl_float_sub(void* argv);
void unparser_spl_STACKftop(void* argv);

void unparser_spl_int_add(void* argv);
void unparser_spl_int_sub(void* argv);
void unparser_spl_int_mul(void* argv);
void unparser_spl_int_div(void* argv);
void unparser_spl_int_mudolo(void* argv);
void unparser_spl_int_const(void* argv);
void unparser_spl_STACKitop(void* argv);
void unparser_spl_STACKctop(void* argv);

void unparser_spl_boolean_or(void* argv);
void unparser_spl_boolean_and(void* argv);
void unparser_spl_eq(void* argv);
void unparser_spl_neq(void* argv);
void unparser_spl_leq(void* argv);
void unparser_spl_geq(void* argv);
void unparser_spl_lower(void* argv);
void unparser_spl_greater(void* argv);
void unparser_spl_not(void* argv);

void unparser_PARALLEL_REGION(void* argv);
void unparser_PARALLEL_FOR_REGION(void* argv);

void unparser_SEQ_DECLARATIONS(void* argv);
void unparser_INT_DECLARATION(void* argv);
void unparser_UINT_DECLARATION(void* argv);
void unparser_FLOAT_DECLARATION(void* argv);
void unparser_INT_POINTER_DECLARATION(void* argv);
void unparser_FLOAT_POINTER_DECLARATION(void* argv);
void unparser_STACKc_DECLARATION(void* argv);
void unparser_STACKi_DECLARATION(void* argv);
void unparser_STACKf_DECLARATION(void* argv);

void unparser_S(void* argv);

void unparser_omp_ATOMIC(void* argv) ;
void unparser_spl_omp_threadprivate(void* argv) ;
void unparser_spl_omp_for(void* argv) ;

void unparser_clauses(void* argv);

void unparser_FOR_LOOP_HEADER(void* argv);
void unparser_FOR_LOOP_HEADER_INIT(void* argv);
void unparser_FOR_LOOP_HEADER_VAR_DEF(void* argv);
void unparser_FOR_LOOP_HEADER_TEST(void* argv);
void unparser_FOR_LOOP_HEADER_UPDATE(void* argv);

void unparser_clauses(void* argv);
void unparser_list_of_vars(void* argv);
void unparser_cfg_entry(void* argv);
void unparser_cfg_exit(void* argv);

void unparser_OMP_RUNTIME_ROUTINE(void* argv);
void unparser_barrier(void* argv);
void unparser_master(void* argv);
void unparser_critical(void* argv);
void unparser_ad_exclusive_read_failure(void* argv);

void unparser_dummy(void* argv);
void unparser_assert(void* argv);
void unparser_before_reverse_mode_hook(void* argv);

#endif
