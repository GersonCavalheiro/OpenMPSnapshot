
#pragma once



#include "includes/model_part.h"
#include "utilities/parallel_utilities.h"

namespace Kratos
{

namespace EntitiesUtilities
{

void KRATOS_API(KRATOS_CORE) InitializeAllEntities(ModelPart& rModelPart);


void KRATOS_API(KRATOS_CORE) InitializeSolutionStepAllEntities(ModelPart& rModelPart);


void KRATOS_API(KRATOS_CORE) FinalizeSolutionStepAllEntities(ModelPart& rModelPart);


void KRATOS_API(KRATOS_CORE) InitializeNonLinearIterationAllEntities(ModelPart& rModelPart);


void KRATOS_API(KRATOS_CORE) FinalizeNonLinearIterationAllEntities(ModelPart& rModelPart);


template<class TEntityType>
KRATOS_API(KRATOS_CORE) PointerVectorSet<TEntityType, IndexedObject>& GetEntities(ModelPart& rModelPart);


template<class TEntityType>
void InitializeEntities(ModelPart& rModelPart)
{
KRATOS_TRY

auto& r_entities_array = GetEntities<TEntityType>(rModelPart);

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

block_for_each(
r_entities_array,
[&r_current_process_info](TEntityType& rEntity) {
if (rEntity.IsActive()) {
rEntity.Initialize(r_current_process_info);
}
}
);

KRATOS_CATCH("")
}


template<class TEntityType>
void InitializeSolutionStepEntities(ModelPart& rModelPart)
{
KRATOS_TRY

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

block_for_each(
GetEntities<TEntityType>(rModelPart),
[&r_current_process_info](TEntityType& rEntity){
rEntity.InitializeSolutionStep(r_current_process_info);
}
);

KRATOS_CATCH("")
}


template<class TEntityType>
void FinalizeSolutionStepEntities(ModelPart& rModelPart)
{
KRATOS_TRY

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

block_for_each(
GetEntities<TEntityType>(rModelPart),
[&r_current_process_info](TEntityType& rEntity){
rEntity.FinalizeSolutionStep(r_current_process_info);
}
);

KRATOS_CATCH("")
}


template<class TEntityType>
void InitializeNonLinearIterationEntities(ModelPart& rModelPart)
{
KRATOS_TRY

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

block_for_each(
GetEntities<TEntityType>(rModelPart),
[&r_current_process_info](TEntityType& rEntity){
rEntity.InitializeNonLinearIteration(r_current_process_info);
}
);

KRATOS_CATCH("")
}


template<class TEntityType>
void FinalizeNonLinearIterationEntities(ModelPart& rModelPart)
{
KRATOS_TRY

const ProcessInfo& r_current_process_info = rModelPart.GetProcessInfo();

block_for_each(
GetEntities<TEntityType>(rModelPart),
[&r_current_process_info](TEntityType& rEntity){
rEntity.FinalizeNonLinearIteration(r_current_process_info);
}
);

KRATOS_CATCH("")
}


}; 
}  
