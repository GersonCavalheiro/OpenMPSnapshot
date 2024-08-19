
#pragma once



#include <string>
#include <iostream>




#include "includes/define.h"
#include "includes/kratos_application.h"


#include "custom_elements/vms.h"
#include "custom_elements/qs_vms.h"
#include "custom_elements/qs_vms_dem_coupled.h"
#include "custom_elements/alternative_qs_vms_dem_coupled.h"
#include "custom_elements/d_vms.h"
#include "custom_elements/d_vms_dem_coupled.h"
#include "custom_elements/alternative_d_vms_dem_coupled.h"
#include "custom_elements/fic.h"
#include "custom_elements/symbolic_stokes.h"
#include "custom_elements/weakly_compressible_navier_stokes.h"
#include "custom_elements/embedded_fluid_element.h"
#include "custom_elements/embedded_fluid_element_discontinuous.h"
#include "custom_elements/two_fluid_vms.h"
#include "custom_elements/two_fluid_vms_linearized_darcy.h"
#include "custom_elements/stationary_stokes.h"
#include "custom_elements/fractional_step.h"
#include "custom_elements/fractional_step_discontinuous.h"
#include "custom_elements/spalart_allmaras.h"
#include "custom_conditions/wall_condition.h"
#include "custom_conditions/fs_werner_wengle_wall_condition.h"
#include "custom_conditions/fs_generalized_wall_condition.h"
#include "custom_conditions/wall_condition_discontinuous.h"
#include "custom_conditions/monolithic_wall_condition.h"
#include "custom_conditions/stokes_wall_condition.h"
#include "custom_conditions/two_fluid_navier_stokes_wall_condition.h"
#include "custom_conditions/fs_periodic_condition.h"
#include "custom_conditions/navier_stokes_wall_condition.h"
#include "custom_conditions/embedded_ausas_navier_stokes_wall_condition.h"

#include "custom_elements/dpg_vms.h"
#include "custom_elements/bingham_fluid.h"
#include "custom_elements/herschel_bulkley_fluid.h"
#include "custom_elements/stokes_3D.h"
#include "custom_elements/stokes_3D_twofluid.h"
#include "custom_elements/navier_stokes.h"
#include "custom_elements/embedded_navier_stokes.h"
#include "custom_elements/embedded_ausas_navier_stokes.h"
#include "custom_elements/compressible_navier_stokes_explicit.h"
#include "custom_elements/two_fluid_navier_stokes.h"
#include "custom_elements/two_fluid_navier_stokes_alpha_method.h"

#include "custom_utilities/qsvms_data.h"
#include "custom_utilities/time_integrated_qsvms_data.h"
#include "custom_utilities/qsvms_dem_coupled_data.h"
#include "custom_utilities/fic_data.h"
#include "custom_utilities/time_integrated_fic_data.h"
#include "custom_utilities/symbolic_stokes_data.h"
#include "custom_utilities/two_fluid_navier_stokes_data.h"
#include "custom_utilities/two_fluid_navier_stokes_alpha_method_data.h"
#include "custom_utilities/weakly_compressible_navier_stokes_data.h"

#include "custom_constitutive/bingham_3d_law.h"
#include "custom_constitutive/euler_2d_law.h"
#include "custom_constitutive/euler_3d_law.h"
#include "custom_constitutive/herschel_bulkley_3d_law.h"
#include "custom_constitutive/newtonian_2d_law.h"
#include "custom_constitutive/newtonian_3d_law.h"
#include "custom_constitutive/newtonian_two_fluid_2d_law.h"
#include "custom_constitutive/newtonian_two_fluid_3d_law.h"
#include "custom_constitutive/newtonian_temperature_dependent_2d_law.h"
#include "custom_constitutive/newtonian_temperature_dependent_3d_law.h"

#include "custom_conditions/wall_laws/linear_log_wall_law.h"
#include "custom_conditions/wall_laws/navier_slip_wall_law.h"

#include "custom_elements/vms_adjoint_element.h"
#include "custom_elements/fluid_adjoint_element.h"
#include "custom_elements/data_containers/qs_vms/qs_vms_adjoint_element_data.h"

#include "custom_conditions/adjoint_monolithic_wall_condition.h"

namespace Kratos
{






class KRATOS_API(FLUID_DYNAMICS_APPLICATION) KratosFluidDynamicsApplication : public KratosApplication
{
public:


KRATOS_CLASS_POINTER_DEFINITION(KratosFluidDynamicsApplication);


KratosFluidDynamicsApplication();

~KratosFluidDynamicsApplication() override {}





void Register() override;








std::string Info() const override
{
return "KratosFluidDynamicsApplication";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
PrintData(rOStream);
}

void PrintData(std::ostream& rOStream) const override
{
KRATOS_WATCH("in Fluid Dynamics application");
KRATOS_WATCH(KratosComponents<VariableData>::GetComponents().size() );
rOStream << "Variables:" << std::endl;
KratosComponents<VariableData>().PrintData(rOStream);
rOStream << std::endl;
rOStream << "Elements:" << std::endl;
KratosComponents<Element>().PrintData(rOStream);
rOStream << std::endl;
rOStream << "Conditions:" << std::endl;
KratosComponents<Condition>().PrintData(rOStream);
}





protected:















private:





const VMS<2> mVMS2D;
const VMS<3> mVMS3D;
const QSVMS< QSVMSData<2,3> > mQSVMS2D3N;
const QSVMS< QSVMSData<3,4> > mQSVMS3D4N;
const QSVMS< QSVMSData<2,4> > mQSVMS2D4N;
const QSVMS< QSVMSData<3,8> > mQSVMS3D8N;
const QSVMSDEMCoupled< QSVMSDEMCoupledData<2,3> > mQSVMSDEMCoupled2D3N;
const QSVMSDEMCoupled< QSVMSDEMCoupledData<3,4> > mQSVMSDEMCoupled3D4N;
const QSVMSDEMCoupled< QSVMSDEMCoupledData<2,4> > mQSVMSDEMCoupled2D4N;
const QSVMSDEMCoupled< QSVMSDEMCoupledData<3,8> > mQSVMSDEMCoupled3D8N;
const AlternativeQSVMSDEMCoupled< QSVMSDEMCoupledData<2,3> > mAlternativeQSVMSDEMCoupled2D3N;
const AlternativeQSVMSDEMCoupled< QSVMSDEMCoupledData<3,4> > mAlternativeQSVMSDEMCoupled3D4N;
const AlternativeQSVMSDEMCoupled< QSVMSDEMCoupledData<2,4> > mAlternativeQSVMSDEMCoupled2D4N;
const AlternativeQSVMSDEMCoupled< QSVMSDEMCoupledData<3,8> > mAlternativeQSVMSDEMCoupled3D8N;
const QSVMS< TimeIntegratedQSVMSData<2,3> > mTimeIntegratedQSVMS2D3N;
const QSVMS< TimeIntegratedQSVMSData<3,4> > mTimeIntegratedQSVMS3D4N;
const DVMS< QSVMSData<2,3> > mDVMS2D3N;
const DVMS< QSVMSData<3,4> > mDVMS3D4N;
const DVMSDEMCoupled< QSVMSDEMCoupledData<2,3> > mDVMSDEMCoupled2D3N;
const DVMSDEMCoupled< QSVMSDEMCoupledData<3,4> > mDVMSDEMCoupled3D4N;
const DVMSDEMCoupled< QSVMSDEMCoupledData<2,4> > mDVMSDEMCoupled2D4N;
const DVMSDEMCoupled< QSVMSDEMCoupledData<3,8> > mDVMSDEMCoupled3D8N;
const AlternativeDVMSDEMCoupled< QSVMSDEMCoupledData<2,3> > mAlternativeDVMSDEMCoupled2D3N;
const AlternativeDVMSDEMCoupled< QSVMSDEMCoupledData<3,4> > mAlternativeDVMSDEMCoupled3D4N;
const AlternativeDVMSDEMCoupled< QSVMSDEMCoupledData<2,4> > mAlternativeDVMSDEMCoupled2D4N;
const AlternativeDVMSDEMCoupled< QSVMSDEMCoupledData<3,8> > mAlternativeDVMSDEMCoupled3D8N;
const FIC< FICData<2,3> > mFIC2D3N;
const FIC< FICData<2,4> > mFIC2D4N;
const FIC< FICData<3,4> > mFIC3D4N;
const FIC< FICData<3,8> > mFIC3D8N;
const FIC< TimeIntegratedFICData<2,3> > mTimeIntegratedFIC2D3N;
const FIC< TimeIntegratedFICData<3,4> > mTimeIntegratedFIC3D4N;
const SymbolicStokes< SymbolicStokesData<2,3> > mSymbolicStokes2D3N;
const SymbolicStokes< SymbolicStokesData<2,4> > mSymbolicStokes2D4N;
const SymbolicStokes< SymbolicStokesData<3,4> > mSymbolicStokes3D4N;
const SymbolicStokes< SymbolicStokesData<3,6> > mSymbolicStokes3D6N;
const SymbolicStokes< SymbolicStokesData<3,8> > mSymbolicStokes3D8N;
const WeaklyCompressibleNavierStokes< WeaklyCompressibleNavierStokesData<2,3> > mWeaklyCompressibleNavierStokes2D3N;
const WeaklyCompressibleNavierStokes< WeaklyCompressibleNavierStokesData<3,4> > mWeaklyCompressibleNavierStokes3D4N;
const EmbeddedFluidElement< WeaklyCompressibleNavierStokes< WeaklyCompressibleNavierStokesData<2,3> > > mEmbeddedWeaklyCompressibleNavierStokes2D3N;
const EmbeddedFluidElement< WeaklyCompressibleNavierStokes< WeaklyCompressibleNavierStokesData<3,4> > > mEmbeddedWeaklyCompressibleNavierStokes3D4N;
const EmbeddedFluidElementDiscontinuous< WeaklyCompressibleNavierStokes< WeaklyCompressibleNavierStokesData<2,3> > > mEmbeddedWeaklyCompressibleNavierStokesDiscontinuous2D3N;
const EmbeddedFluidElementDiscontinuous< WeaklyCompressibleNavierStokes< WeaklyCompressibleNavierStokesData<3,4> > > mEmbeddedWeaklyCompressibleNavierStokesDiscontinuous3D4N;
const EmbeddedFluidElement< QSVMS< TimeIntegratedQSVMSData<2,3> > > mEmbeddedQSVMS2D3N;
const EmbeddedFluidElement< QSVMS< TimeIntegratedQSVMSData<3,4> > > mEmbeddedQSVMS3D4N;
const EmbeddedFluidElementDiscontinuous< QSVMS< TimeIntegratedQSVMSData<2,3> > > mEmbeddedQSVMSDiscontinuous2D3N;
const EmbeddedFluidElementDiscontinuous< QSVMS< TimeIntegratedQSVMSData<3,4> > > mEmbeddedQSVMSDiscontinuous3D4N;

const TwoFluidVMS<3,4> mTwoFluidVMS3D;
const TwoFluidVMSLinearizedDarcy<3,4> mTwoFluidVMSLinearizedDarcy3D;

const StationaryStokes<2> mStationaryStokes2D;
const StationaryStokes<3> mStationaryStokes3D;

const FractionalStep<2> mFractionalStep2D;
const FractionalStep<3> mFractionalStep3D;
const FractionalStepDiscontinuous<2> mFractionalStepDiscontinuous2D;
const FractionalStepDiscontinuous<3> mFractionalStepDiscontinuous3D;

const SpalartAllmaras mSpalartAllmaras2D;
const SpalartAllmaras mSpalartAllmaras3D;

const WallCondition<2,2> mWallCondition2D;
const WallCondition<3,3> mWallCondition3D;

const FSWernerWengleWallCondition<2,2> mFSWernerWengleWallCondition2D;
const FSWernerWengleWallCondition<3,3> mFSWernerWengleWallCondition3D;

const FSGeneralizedWallCondition<2,2> mFSGeneralizedWallCondition2D;
const FSGeneralizedWallCondition<3,3> mFSGeneralizedWallCondition3D;

const WallConditionDiscontinuous<2,2> mWallConditionDiscontinuous2D;
const WallConditionDiscontinuous<3,3> mWallConditionDiscontinuous3D;

const MonolithicWallCondition<2,2> mMonolithicWallCondition2D;
const MonolithicWallCondition<3,3> mMonolithicWallCondition3D;
const StokesWallCondition<3,3> mStokesWallCondition3D;
const StokesWallCondition<3,4> mStokesWallCondition3D4N;

const FSPeriodicCondition<2> mFSPeriodicCondition2D;
const FSPeriodicCondition<3> mFSPeriodicCondition3D;
const FSPeriodicCondition<2> mFSPeriodicConditionEdge2D;
const FSPeriodicCondition<3> mFSPeriodicConditionEdge3D;


const DPGVMS<2> mDPGVMS2D;
const DPGVMS<3> mDPGVMS3D;



const BinghamFluid< VMS<2> > mBinghamVMS2D;
const BinghamFluid< VMS<3> > mBinghamVMS3D;

const BinghamFluid< FractionalStep<2> > mBinghamFractionalStep2D;
const BinghamFluid< FractionalStep<3> > mBinghamFractionalStep3D;

const BinghamFluid< FractionalStepDiscontinuous<2> > mBinghamFractionalStepDiscontinuous2D;
const BinghamFluid< FractionalStepDiscontinuous<3> > mBinghamFractionalStepDiscontinuous3D;

const HerschelBulkleyFluid< VMS<2> > mHerschelBulkleyVMS2D;
const HerschelBulkleyFluid< VMS<3> > mHerschelBulkleyVMS3D;

const Stokes3D mStokes3D;
const Stokes3DTwoFluid mStokes3DTwoFluid;

const NavierStokes<2> mNavierStokes2D;
const NavierStokes<3> mNavierStokes3D;
const NavierStokesWallCondition<2,2> mNavierStokesWallCondition2D;
const NavierStokesWallCondition<3,3> mNavierStokesWallCondition3D;
const NavierStokesWallCondition<2,2,LinearLogWallLaw<2,2>> mNavierStokesLinearLogWallCondition2D;
const NavierStokesWallCondition<3,3,LinearLogWallLaw<3,3>> mNavierStokesLinearLogWallCondition3D;
const NavierStokesWallCondition<2,2,NavierSlipWallLaw<2,2>> mNavierStokesNavierSlipWallCondition2D;
const NavierStokesWallCondition<3,3,NavierSlipWallLaw<3,3>> mNavierStokesNavierSlipWallCondition3D;

const EmbeddedNavierStokes<2> mEmbeddedNavierStokes2D;
const EmbeddedNavierStokes<3> mEmbeddedNavierStokes3D;

const EmbeddedAusasNavierStokes<2> mEmbeddedAusasNavierStokes2D;
const EmbeddedAusasNavierStokes<3> mEmbeddedAusasNavierStokes3D;
const EmbeddedAusasNavierStokesWallCondition<2> mEmbeddedAusasNavierStokesWallCondition2D;
const EmbeddedAusasNavierStokesWallCondition<3> mEmbeddedAusasNavierStokesWallCondition3D;

const CompressibleNavierStokesExplicit<2, 3> mCompressibleNavierStokesExplicit2D3N;
const CompressibleNavierStokesExplicit<2, 4> mCompressibleNavierStokesExplicit2D4N;
const CompressibleNavierStokesExplicit<3, 4> mCompressibleNavierStokesExplicit3D4N;

const TwoFluidNavierStokes< TwoFluidNavierStokesData<2, 3> > mTwoFluidNavierStokes2D3N;
const TwoFluidNavierStokes< TwoFluidNavierStokesData<3, 4> > mTwoFluidNavierStokes3D4N;
const TwoFluidNavierStokesAlphaMethod< TwoFluidNavierStokesAlphaMethodData<2, 3> > mTwoFluidNavierStokesAlphaMethod2D3N;
const TwoFluidNavierStokesAlphaMethod< TwoFluidNavierStokesAlphaMethodData<3, 4> > mTwoFluidNavierStokesAlphaMethod3D4N;
const TwoFluidNavierStokesWallCondition<2,2> mTwoFluidNavierStokesWallCondition2D;
const TwoFluidNavierStokesWallCondition<3,3> mTwoFluidNavierStokesWallCondition3D;

const Bingham3DLaw mBingham3DLaw;
const Euler2DLaw mEuler2DLaw;
const Euler3DLaw mEuler3DLaw;
const HerschelBulkley3DLaw mHerschelBulkley3DLaw;
const Newtonian2DLaw mNewtonian2DLaw;
const Newtonian3DLaw mNewtonian3DLaw;
const NewtonianTwoFluid2DLaw mNewtonianTwoFluid2DLaw;
const NewtonianTwoFluid3DLaw mNewtonianTwoFluid3DLaw;
const NewtonianTemperatureDependent2DLaw mNewtonianTemperatureDependent2DLaw;
const NewtonianTemperatureDependent3DLaw mNewtonianTemperatureDependent3DLaw;

const VMSAdjointElement<2> mVMSAdjointElement2D;
const VMSAdjointElement<3> mVMSAdjointElement3D;

const FluidAdjointElement<2, 3, QSVMSAdjointElementData<2, 3>> mQSVMSAdjoint2D3N;
const FluidAdjointElement<2, 4, QSVMSAdjointElementData<2, 4>> mQSVMSAdjoint2D4N;
const FluidAdjointElement<3, 4, QSVMSAdjointElementData<3, 4>> mQSVMSAdjoint3D4N;
const FluidAdjointElement<3, 8, QSVMSAdjointElementData<3, 8>> mQSVMSAdjoint3D8N;

const AdjointMonolithicWallCondition<2, 2> mAdjointMonolithicWallCondition2D2N;
const AdjointMonolithicWallCondition<3, 3> mAdjointMonolithicWallCondition3D3N;










KratosFluidDynamicsApplication& operator=(KratosFluidDynamicsApplication const& rOther);

KratosFluidDynamicsApplication(KratosFluidDynamicsApplication const& rOther);



}; 








}  
