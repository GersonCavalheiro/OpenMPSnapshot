#ifndef TL_OMP_LINT_HPP
#define TL_OMP_LINT_HPP
#include "tl-analysis-base.hpp"
#include "tl-compilerphase.hpp"
#include "tl-nodecl-visitor.hpp"
namespace TL {
namespace OpenMP {
void launch_correctness(
const TL::Analysis::AnalysisBase& analysis,
std::string log_file_path);
class WritesVisitor : public Nodecl::ExhaustiveVisitor<void>
{
private:
ObjectList<Nodecl::NodeclBase> _defined_vars;
bool _define;
void visit_assignment( const Nodecl::NodeclBase& lhs, const Nodecl::NodeclBase& rhs );
void visit_xx_crement( const Nodecl::NodeclBase& rhs );
public:
WritesVisitor( );
ObjectList<Nodecl::NodeclBase> get_defined_symbols( );
void clear( );
Ret visit( const Nodecl::AddAssignment& n );
Ret visit( const Nodecl::ArithmeticShrAssignment& n );
Ret visit( const Nodecl::ArraySubscript& n );
Ret visit( const Nodecl::Assignment& n );
Ret visit( const Nodecl::BitwiseAndAssignment& n );
Ret visit( const Nodecl::BitwiseOrAssignment& n );
Ret visit( const Nodecl::BitwiseShlAssignment& n );
Ret visit( const Nodecl::BitwiseShrAssignment& n );
Ret visit( const Nodecl::BitwiseXorAssignment& n );
Ret visit( const Nodecl::ClassMemberAccess& n );
Ret visit( const Nodecl::Dereference& n );
Ret visit( const Nodecl::DivAssignment& n );
Ret visit( const Nodecl::MinusAssignment& n );
Ret visit( const Nodecl::ModAssignment& n );
Ret visit( const Nodecl::MulAssignment& n );
Ret visit( const Nodecl::ObjectInit& n );
Ret visit( const Nodecl::Postdecrement& n );
Ret visit( const Nodecl::Postincrement& n );
Ret visit( const Nodecl::Predecrement& n );
Ret visit( const Nodecl::Preincrement& n );
Ret visit( const Nodecl::Reference& n );
Ret visit( const Nodecl::Symbol& n );
};
class Lint : public TL::CompilerPhase
{
private:
std::string _disable_phase;
std::string _correctness_log_path;
std::string _lint_deprecated_flag;
std::string _ompss_mode_str;
bool _ompss_mode_enabled;
void set_ompss_mode( const std::string& ompss_mode_str);
void set_lint_deprecated_flag(const std::string& lint_deprecated_flag_str);
public:
Lint();
virtual void run(TL::DTO& dto);
virtual void pre_run(TL::DTO& dto);
virtual ~Lint() { }
};
}
}
#endif 
