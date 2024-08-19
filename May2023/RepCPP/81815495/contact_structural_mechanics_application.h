
#pragma once



#include "includes/define.h"
#include "includes/kratos_application.h"
#include "includes/variables.h"


#include "geometries/triangle_3d_3.h"
#include "geometries/quadrilateral_3d_4.h"
#include "geometries/line_2d_2.h"


#include "custom_conditions/mesh_tying_mortar_condition.h"
#include "custom_conditions/ALM_frictionless_mortar_contact_condition.h"
#include "custom_conditions/ALM_frictionless_components_mortar_contact_condition.h"
#include "custom_conditions/penalty_frictionless_mortar_contact_condition.h"
#include "custom_conditions/ALM_frictionless_mortar_contact_axisym_condition.h"
#include "custom_conditions/penalty_frictionless_mortar_contact_axisym_condition.h"
#include "custom_conditions/ALM_frictional_mortar_contact_condition.h"
#include "custom_conditions/penalty_frictional_mortar_contact_condition.h"
#include "custom_conditions/ALM_frictional_mortar_contact_axisym_condition.h"
#include "custom_conditions/penalty_frictional_mortar_contact_axisym_condition.h"
#include "custom_conditions/mpc_mortar_contact_condition.h"


#include "custom_master_slave_constraints/contact_master_slave_constraint.h"

namespace Kratos
{







class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) KratosContactStructuralMechanicsApplication
: public KratosApplication
{
public:

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

typedef GeometryType::Pointer GeometryPointerType;

typedef GeometryType::PointsArrayType PointsArrayType;

typedef Line2D2<NodeType> LineType;

typedef Triangle3D3<NodeType> TriangleType;

typedef Quadrilateral3D4<NodeType> QuadrilateralType;

KRATOS_CLASS_POINTER_DEFINITION(KratosContactStructuralMechanicsApplication);


KratosContactStructuralMechanicsApplication();

~KratosContactStructuralMechanicsApplication() override = default;




void Register() override;






std::string Info() const override
{
return "KratosContactStructuralMechanicsApplication";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
PrintData(rOStream);
}

void PrintData(std::ostream& rOStream) const override
{
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



const MeshTyingMortarCondition<2, 3> mMeshTyingMortarCondition2D2NTriangle;                   
const MeshTyingMortarCondition<2, 4> mMeshTyingMortarCondition2D2NQuadrilateral;              
const MeshTyingMortarCondition<3, 4> mMeshTyingMortarCondition3D3NTetrahedron;                
const MeshTyingMortarCondition<3, 8> mMeshTyingMortarCondition3D4NHexahedron;                 
const MeshTyingMortarCondition<3, 4, 8> mMeshTyingMortarCondition3D3NTetrahedron4NHexahedron; 
const MeshTyingMortarCondition<3, 8, 4> mMeshTyingMortarCondition3D4NHexahedron3NTetrahedron; 

const AugmentedLagrangianMethodFrictionlessMortarContactCondition<2, 2, false> mALMFrictionlessMortarContactCondition2D2N;
const AugmentedLagrangianMethodFrictionlessMortarContactCondition<2, 2, true> mALMNVFrictionlessMortarContactCondition2D2N;
const AugmentedLagrangianMethodFrictionlessMortarContactAxisymCondition<2, false> mALMFrictionlessAxisymMortarContactCondition2D2N;
const AugmentedLagrangianMethodFrictionlessMortarContactAxisymCondition<2, true> mALMNVFrictionlessAxisymMortarContactCondition2D2N;
const AugmentedLagrangianMethodFrictionlessMortarContactCondition<3, 3, false, 3> mALMFrictionlessMortarContactCondition3D3N;
const AugmentedLagrangianMethodFrictionlessMortarContactCondition<3, 3, true,  3> mALMNVFrictionlessMortarContactCondition3D3N;
const AugmentedLagrangianMethodFrictionlessMortarContactCondition<3, 4, false, 4> mALMFrictionlessMortarContactCondition3D4N;
const AugmentedLagrangianMethodFrictionlessMortarContactCondition<3, 4, true,  4> mALMNVFrictionlessMortarContactCondition3D4N;
const AugmentedLagrangianMethodFrictionlessMortarContactCondition<3, 3, false, 4> mALMFrictionlessMortarContactCondition3D3N4N;
const AugmentedLagrangianMethodFrictionlessMortarContactCondition<3, 3, true,  4> mALMNVFrictionlessMortarContactCondition3D3N4N;
const AugmentedLagrangianMethodFrictionlessMortarContactCondition<3, 4, false, 3> mALMFrictionlessMortarContactCondition3D4N3N;
const AugmentedLagrangianMethodFrictionlessMortarContactCondition<3, 4, true,  3> mALMNVFrictionlessMortarContactCondition3D4N3N;
const AugmentedLagrangianMethodFrictionlessComponentsMortarContactCondition<2, 2, false> mALMFrictionlessComponentsMortarContactCondition2D2N;
const AugmentedLagrangianMethodFrictionlessComponentsMortarContactCondition<2, 2, true> mALMNVFrictionlessComponentsMortarContactCondition2D2N;
const AugmentedLagrangianMethodFrictionlessComponentsMortarContactCondition<3, 3, false, 3> mALMFrictionlessComponentsMortarContactCondition3D3N;
const AugmentedLagrangianMethodFrictionlessComponentsMortarContactCondition<3, 3, true,  3> mALMNVFrictionlessComponentsMortarContactCondition3D3N;
const AugmentedLagrangianMethodFrictionlessComponentsMortarContactCondition<3, 3, false, 4> mALMFrictionlessComponentsMortarContactCondition3D3N4N;
const AugmentedLagrangianMethodFrictionlessComponentsMortarContactCondition<3, 3, true,  4> mALMNVFrictionlessComponentsMortarContactCondition3D3N4N;
const AugmentedLagrangianMethodFrictionlessComponentsMortarContactCondition<3, 4, false, 4> mALMFrictionlessComponentsMortarContactCondition3D4N;
const AugmentedLagrangianMethodFrictionlessComponentsMortarContactCondition<3, 4, true,  4> mALMNVFrictionlessComponentsMortarContactCondition3D4N;
const AugmentedLagrangianMethodFrictionlessComponentsMortarContactCondition<3, 4, false, 3> mALMFrictionlessComponentsMortarContactCondition3D4N3N;
const AugmentedLagrangianMethodFrictionlessComponentsMortarContactCondition<3, 4, true,  3> mALMNVFrictionlessComponentsMortarContactCondition3D4N3N;
const AugmentedLagrangianMethodFrictionalMortarContactCondition<2, 2, false> mALMFrictionalMortarContactCondition2D2N;
const AugmentedLagrangianMethodFrictionalMortarContactCondition<2, 2, true> mALMNVFrictionalMortarContactCondition2D2N;
const AugmentedLagrangianMethodFrictionalMortarContactAxisymCondition<2, false> mALMFrictionalAxisymMortarContactCondition2D2N;
const AugmentedLagrangianMethodFrictionalMortarContactAxisymCondition<2, true> mALMNVFrictionalAxisymMortarContactCondition2D2N;
const AugmentedLagrangianMethodFrictionalMortarContactCondition<3, 3, false, 3> mALMFrictionalMortarContactCondition3D3N;
const AugmentedLagrangianMethodFrictionalMortarContactCondition<3, 3, true,  3> mALMNVFrictionalMortarContactCondition3D3N;
const AugmentedLagrangianMethodFrictionalMortarContactCondition<3, 4, false, 4> mALMFrictionalMortarContactCondition3D4N;
const AugmentedLagrangianMethodFrictionalMortarContactCondition<3, 4, true,  4> mALMNVFrictionalMortarContactCondition3D4N;
const AugmentedLagrangianMethodFrictionalMortarContactCondition<3, 3, false, 4> mALMFrictionalMortarContactCondition3D3N4N;
const AugmentedLagrangianMethodFrictionalMortarContactCondition<3, 3, true,  4> mALMNVFrictionalMortarContactCondition3D3N4N;
const AugmentedLagrangianMethodFrictionalMortarContactCondition<3, 4, false, 3> mALMFrictionalMortarContactCondition3D4N3N;
const AugmentedLagrangianMethodFrictionalMortarContactCondition<3, 4, true,  3> mALMNVFrictionalMortarContactCondition3D4N3N;
const PenaltyMethodFrictionlessMortarContactCondition<2, 2, false> mPenaltyFrictionlessMortarContactCondition2D2N;
const PenaltyMethodFrictionlessMortarContactCondition<2, 2, true> mPenaltyNVFrictionlessMortarContactCondition2D2N;
const PenaltyMethodFrictionlessMortarContactAxisymCondition<2, false> mPenaltyFrictionlessAxisymMortarContactCondition2D2N;
const PenaltyMethodFrictionlessMortarContactAxisymCondition<2, true> mPenaltyNVFrictionlessAxisymMortarContactCondition2D2N;
const PenaltyMethodFrictionlessMortarContactCondition<3, 3, false, 3> mPenaltyFrictionlessMortarContactCondition3D3N;
const PenaltyMethodFrictionlessMortarContactCondition<3, 3, true,  3> mPenaltyNVFrictionlessMortarContactCondition3D3N;
const PenaltyMethodFrictionlessMortarContactCondition<3, 4, false, 4> mPenaltyFrictionlessMortarContactCondition3D4N;
const PenaltyMethodFrictionlessMortarContactCondition<3, 4, true,  4> mPenaltyNVFrictionlessMortarContactCondition3D4N;
const PenaltyMethodFrictionlessMortarContactCondition<3, 3, false, 4> mPenaltyFrictionlessMortarContactCondition3D3N4N;
const PenaltyMethodFrictionlessMortarContactCondition<3, 3, true,  4> mPenaltyNVFrictionlessMortarContactCondition3D3N4N;
const PenaltyMethodFrictionlessMortarContactCondition<3, 4, false, 3> mPenaltyFrictionlessMortarContactCondition3D4N3N;
const PenaltyMethodFrictionlessMortarContactCondition<3, 4, true,  3> mPenaltyNVFrictionlessMortarContactCondition3D4N3N;
const PenaltyMethodFrictionalMortarContactCondition<2, 2, false> mPenaltyFrictionalMortarContactCondition2D2N;
const PenaltyMethodFrictionalMortarContactCondition<2, 2, true> mPenaltyNVFrictionalMortarContactCondition2D2N;
const PenaltyMethodFrictionalMortarContactAxisymCondition<2, false> mPenaltyFrictionalAxisymMortarContactCondition2D2N;
const PenaltyMethodFrictionalMortarContactAxisymCondition<2, true> mPenaltyNVFrictionalAxisymMortarContactCondition2D2N;
const PenaltyMethodFrictionalMortarContactCondition<3, 3, false, 3> mPenaltyFrictionalMortarContactCondition3D3N;
const PenaltyMethodFrictionalMortarContactCondition<3, 3, true,  3> mPenaltyNVFrictionalMortarContactCondition3D3N;
const PenaltyMethodFrictionalMortarContactCondition<3, 4, false, 4> mPenaltyFrictionalMortarContactCondition3D4N;
const PenaltyMethodFrictionalMortarContactCondition<3, 4, true,  4> mPenaltyNVFrictionalMortarContactCondition3D4N;
const PenaltyMethodFrictionalMortarContactCondition<3, 3, false, 4> mPenaltyFrictionalMortarContactCondition3D3N4N;
const PenaltyMethodFrictionalMortarContactCondition<3, 3, true,  4> mPenaltyNVFrictionalMortarContactCondition3D3N4N;
const PenaltyMethodFrictionalMortarContactCondition<3, 4, false, 3> mPenaltyFrictionalMortarContactCondition3D4N3N;
const PenaltyMethodFrictionalMortarContactCondition<3, 4, true,  3> mPenaltyNVFrictionalMortarContactCondition3D4N3N;

const MPCMortarContactCondition<2, 2> mMPCMortarContactCondition2D2N;
const MPCMortarContactCondition<3, 3, 3> mMPCMortarContactCondition3D3N;
const MPCMortarContactCondition<3, 4, 4> mMPCMortarContactCondition3D4N;
const MPCMortarContactCondition<3, 3, 4> mMPCMortarContactCondition3D3N4N;
const MPCMortarContactCondition<3, 4, 3> mMPCMortarContactCondition3D4N3N;

const ContactMasterSlaveConstraint mContactMasterSlaveConstraint;










KratosContactStructuralMechanicsApplication& operator=(KratosContactStructuralMechanicsApplication const& rOther);

KratosContactStructuralMechanicsApplication(KratosContactStructuralMechanicsApplication const& rOther);



}; 





}  


