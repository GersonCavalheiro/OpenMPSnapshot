#ifndef TL_ANALYSIS_CHECK_PHASE_HPP
#define TL_ANALYSIS_CHECK_PHASE_HPP
#include "tl-analysis-base.hpp"
#include "tl-extensible-graph.hpp"
#include "tl-nodecl-visitor.hpp"
#include "tl-pragmasupport.hpp"
namespace TL {
namespace Analysis {
class LIBTL_CLASS AnalysisCheckPhase : public PragmaCustomCompilerPhase
{
private:
WhichAnalysis _analysis_mask;
std::string _correctness_log_path;
void check_pragma_clauses(
PragmaCustomLine pragma_line, const locus_t* loc,
Nodecl::List& environment);
public:
AnalysisCheckPhase( );
void assert_handler_pre( TL::PragmaCustomStatement directive );
void assert_handler_post( TL::PragmaCustomStatement directive );
void assert_decl_handler_pre( TL::PragmaCustomDeclaration directive );
void assert_decl_handler_post( TL::PragmaCustomDeclaration directive );
void check_pcfg_consistency( ExtensibleGraph* graph );
void check_analysis_assertions( ExtensibleGraph* graph );
std::string _ompss_mode_str;
bool _ompss_mode_enabled;
void set_ompss_mode( const std::string& ompss_mode_str);
virtual void run( TL::DTO& dto );
virtual ~AnalysisCheckPhase( ) { }
};
class LIBTL_CLASS AnalysisCheckVisitor : public Nodecl::ExhaustiveVisitor<void>
{
private:
public:
Ret visit( const Nodecl::Analysis::Assert& n );
Ret visit( const Nodecl::Analysis::AssertDecl& n );
};
}
}
#endif  
