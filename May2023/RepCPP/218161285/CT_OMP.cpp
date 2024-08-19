
#include "CT_OMP.h"

#ifdef _CT_OMP_H_

CT_OMP::CT_OMP(CTBaseImpl::CTBenchType B,
CTBaseImpl::CTAtomType A) : CTBaseImpl("OMP",B,A),
Array(nullptr),
Idx(nullptr),
memSize(0),
pes(0),
iters(0),
elems(0),
stride(0) {
}

CT_OMP::~CT_OMP(){
}

bool CT_OMP::Execute(double &Timing, double &GAMS){

CTBaseImpl::CTBenchType BType   = this->GetBenchType(); 
CTBaseImpl::CTAtomType AType    = this->GetAtomType();  
double StartTime  = 0.; 
double EndTime    = 0.; 
double OPS        = 0.; 

if( BType == CT_RAND ){
switch( AType ){
case CT_ADD:
StartTime = this->MySecond();
RAND_ADD( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(1,iters,pes);
break;
case CT_CAS:
StartTime = this->MySecond();
RAND_CAS( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(1,iters,pes);
break;
default:
this->ReportBenchError();
return false;
}
}else if( BType == CT_STRIDE1 ){
switch( AType ){
case CT_ADD:
StartTime = this->MySecond();
STRIDE1_ADD( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(1,iters,pes);
break;
case CT_CAS:
StartTime = this->MySecond();
STRIDE1_CAS( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(1,iters,pes);
break;
default:
this->ReportBenchError();
return false;
}
}else if( BType == CT_STRIDEN ){
switch( AType ){
case CT_ADD:
StartTime = this->MySecond();
STRIDEN_ADD( Array, Idx, iters, pes, stride );
EndTime   = this->MySecond();
OPS = this->GAM(1,iters,pes);
break;
case CT_CAS:
StartTime = this->MySecond();
STRIDEN_CAS( Array, Idx, iters, pes, stride );
EndTime   = this->MySecond();
OPS = this->GAM(1,iters,pes);
break;
default:
this->ReportBenchError();
return false;
}
}else if( BType == CT_PTRCHASE ){
switch( AType ){
case CT_ADD:
StartTime = this->MySecond();
PTRCHASE_ADD( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(1,iters,pes);
break;
case CT_CAS:
StartTime = this->MySecond();
PTRCHASE_CAS( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(1,iters,pes);
break;
default:
this->ReportBenchError();
return false;
}
}else if( BType == CT_SG ){
switch( AType ){
case CT_ADD:
StartTime = this->MySecond();
SG_ADD( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(4,iters,pes);
break;
case CT_CAS:
StartTime = this->MySecond();
SG_CAS( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(4,iters,pes);
break;
default:
this->ReportBenchError();
return false;
}
}else if( BType == CT_CENTRAL ){
switch( AType ){
case CT_ADD:
StartTime = this->MySecond();
CENTRAL_ADD( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(1,iters,pes);
break;
case CT_CAS:
StartTime = this->MySecond();
CENTRAL_CAS( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(1,iters,pes);
break;
default:
this->ReportBenchError();
return false;
}
}else if( BType == CT_SCATTER ){
switch( AType ){
case CT_ADD:
StartTime = this->MySecond();
SCATTER_ADD( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(3,iters,pes);
break;
case CT_CAS:
StartTime = this->MySecond();
SCATTER_CAS( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(3,iters,pes);
break;
default:
this->ReportBenchError();
return false;
}
}else if( BType == CT_GATHER ){
switch( AType ){
case CT_ADD:
StartTime = this->MySecond();
GATHER_ADD( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(3,iters,pes);
break;
case CT_CAS:
StartTime = this->MySecond();
GATHER_CAS( Array, Idx, iters, pes );
EndTime   = this->MySecond();
OPS = this->GAM(3,iters,pes);
break;
default:
this->ReportBenchError();
return false;
}
}else{
this->ReportBenchError();
return false;
}

Timing = this->Runtime(StartTime,EndTime);
GAMS   = OPS/Timing;

return true;
}

bool CT_OMP::AllocateData( uint64_t m,
uint64_t p,
uint64_t i,
uint64_t s){
memSize = m;
pes = p;
iters = i;
stride = s;

if( pes == 0 ){
std::cout << "CT_OMP::AllocateData : 'pes' cannot be 0" << std::endl;
return false;
}
if( iters == 0 ){
std::cout << "CT_OMP::AllocateData : 'iters' cannot be 0" << std::endl;
return false;
}
if( stride == 0 ){
std::cout << "CT_OMP::AllocateData : 'stride' cannot be 0" << std::endl;
return false;
}

elems = (memSize/8);

uint64_t end = (pes * iters * stride) - stride;
if( end >= elems ){
std::cout << "CT_OMP::AllocateData : 'Array' is not large enough for pes="
<< pes << "; iters=" << iters << "; stride =" << stride
<< std::endl;
return false;
}

Array = (uint64_t *)(malloc( memSize ));
if( Array == nullptr ){
std::cout << "CT_OMP::AllocateData : 'Array' could not be allocated" << std::endl;
return false;
}

Idx = (uint64_t *)(malloc( sizeof(uint64_t) * (pes+1) * iters ));
if( Idx == nullptr ){
std::cout << "CT_OMP::AllocateData : 'Idx' could not be allocated" << std::endl;
free( Array );
return false;
}

srand(time(NULL));
if( this->GetBenchType() == CT_PTRCHASE ){
for( unsigned i=0; i<((pes+1)*iters); i++ ){
Idx[i] = (uint64_t)(rand()%((pes+1)*iters));
}
}else{
for( unsigned i=0; i<((pes+1)*iters); i++ ){
Idx[i] = (uint64_t)(rand()%(elems-1));
}
}
for( unsigned i=0; i<elems; i++ ){
Array[i] = (uint64_t)(rand());
}

omp_set_num_threads(pes);

#pragma omp parallel
{
#pragma omp single
{
std::cout << "RUNNING WITH NUM_THREADS = " << omp_get_num_threads() << std::endl;
}
}

return true;
}

bool CT_OMP::FreeData(){
if( Array ){
free( Array );
}
if( Idx ){
free( Idx );
}
return true;
}

#endif

