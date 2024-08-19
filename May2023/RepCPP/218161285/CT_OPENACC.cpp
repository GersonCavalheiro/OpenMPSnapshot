
#include "CT_OPENACC.h"

#ifdef _CT_OPENACC_H_

CT_OPENACC::CT_OPENACC(CTBaseImpl::CTBenchType B,
CTBaseImpl::CTAtomType A) : CTBaseImpl("OPENACC",B,A),
Array(nullptr),
Idx(nullptr),
memSize(0),
pes(0),
iters(0),
elems(0),
stride(0) {
}

CT_OPENACC::~CT_OPENACC(){
}

bool CT_OPENACC::Execute(double &Timing, double &GAMS){

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

bool CT_OPENACC::AllocateData( uint64_t m,
uint64_t p,
uint64_t i,
uint64_t s){
memSize = m;
pes = p;
iters = i;
stride = s;

if( pes == 0 ){
std::cout << "CT_OPENACC::AllocateData : 'pes' cannot be 0" << std::endl;
return false;
}
if( iters == 0 ){
std::cout << "CT_OPENACC::AllocateData : 'iters' cannot be 0" << std::endl;
return false;
}
if( stride == 0 ){
std::cout << "CT_OPENACC::AllocateData : 'stride' cannot be 0" << std::endl;
return false;
}

elems = (memSize/8);

uint64_t end = (pes * iters * stride) - stride;
if( end >= elems ){
std::cout << "CT_OPENACC::AllocateData : 'Array' is not large enough for pes="
<< pes << "; iters=" << iters << "; stride =" << stride
<< std::endl;
return false;
}

Array = (uint64_t *) acc_malloc(memSize);
uint64_t *HostArray = (uint64_t *) malloc(memSize);
if( ( Array == nullptr ) || ( HostArray == nullptr ) ){
std::cout << "CT_OPENACC::AllocateData : 'Array' could not be allocated" << std::endl;
acc_free(Array);
if(HostArray != nullptr){
free(HostArray);
}
return false;
}

Idx = (uint64_t *) acc_malloc(sizeof(uint64_t)*(pes+1)*iters);
uint64_t *HostIdx = (uint64_t *) malloc(sizeof(uint64_t)*(pes+1)*iters);
if( ( Idx == nullptr ) || ( HostIdx == nullptr ) ){
std::cout << "CT_OPENACC::AllocateData : 'Idx' could not be allocated" << std::endl;
acc_free(Array);
acc_free(Idx);
if(HostArray != nullptr){
free(HostArray);
}
if(HostIdx != nullptr){
free(HostIdx);
}
return false;
}

srand(time(NULL));
if( this->GetBenchType() == CT_PTRCHASE ){
for( unsigned i=0; i<((pes+1)*iters); i++ ){
HostIdx[i] = (uint64_t)(rand()%((pes+1)*iters));
}
}else{
for( unsigned i=0; i<((pes+1)*iters); i++ ){
HostIdx[i] = (uint64_t)(rand()%(elems-1));
}
}
for( unsigned i=0; i<elems; i++ ){
HostArray[i] = (uint64_t)(rand());
}

acc_memcpy_to_device(Array, HostArray, memSize);
acc_memcpy_to_device(Idx, HostIdx, sizeof(uint64_t)*(pes+1)*iters);

free(HostArray);
free(HostIdx);

uint64_t gangCtr = 0;
#pragma acc parallel num_gangs(pes) copyin(gangCtr) copyout(gangCtr)
{
#pragma acc atomic update
{
gangCtr++;
}
}
std::cout << "RUNNING WITH NUM_GANGS = " << gangCtr << std::endl;

return true;
}

bool CT_OPENACC::SetDevice(){

std::string devName, devVendor;
int selectedDevID;

if(getenv("ACC_DEVICE_TYPE") == nullptr){
std::cout << "CT_OPENACC::SetDevice : ACC_DEVICE_TYPE is not set, using default." << std::endl;
}
else{
std::cout << "CT_OPENACC::SetDevice : ACC_DEVICE_TYPE set to " << getenv("ACC_DEVICE_TYPE") << std::endl;
}

if(getenv("ACC_DEVICE_NUM") == nullptr){
std::cout << "CT_OPENACC::SetDevice : ACC_DEVICE_NUM is not set, using default." << std::endl;
}
else{
std::cout << "CT_OPENACC::SetDevice : ACC_DEVICE_NUM set to " << getenv("ACC_DEVICE_NUM") << std::endl;
}


deviceTypeEnum = acc_get_device_type();
selectedDevID = acc_get_device_num(deviceTypeEnum);
devName.assign(acc_get_property_string(selectedDevID, deviceTypeEnum, acc_property_name));
devVendor.assign(acc_get_property_string(selectedDevID, deviceTypeEnum, acc_property_vendor));

std::cout << "Running on Vendor: " << devVendor << " "
<< "Device: " << devName << std::endl;

acc_init(deviceTypeEnum);

return true;
}

bool CT_OPENACC::FreeData(){
acc_free(Array);
acc_free(Idx);

acc_shutdown(deviceTypeEnum);

return true;
}

#endif

