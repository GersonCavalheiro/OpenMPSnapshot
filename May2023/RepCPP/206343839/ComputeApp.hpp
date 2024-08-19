
#pragma once
#include <MoltenVK/mvk_vulkan.h>
#include "HelperMethods.hpp"
#include "HelperAliasis.hpp"
#include "ComputeGraph.hpp"

namespace nVK {

struct sQueue;
struct sTask;
struct sTaskSequenceCompute;
struct sDevice;

struct sFuncCompiled :
public std::enable_shared_from_this<sFuncCompiled> {
SShared<sDevice> mDevicePtr;
SWeak<nFlow::sFuncDectriptor> mDescriptor;
VkDescriptorSetLayout mVkDescriptorLayout;
VkShaderModule mShader;

~sFuncCompiled();
void fDealloc();
VkDevice fDeviceID() const;
void fInitShader();
void fInitDescriptorLayout();

SShared<sTask> fNewTask();
};

struct sVarWBuffer :
public std::enable_shared_from_this<sVarWBuffer> {

VkBuffer mVkBuffer;
VkDeviceMemory mVkMemory;
bSize mSize = 0;
bPtrUntyped mSourceOnHost = nullptr;
SShared<sTask> mTaskPtr;

~sVarWBuffer();
void fDealloc();
VkDevice fDeviceID() const;
void fInitBuffer();

bBool fResetConstant(bPtrUntypedC, bSize);
bBool fResetVariable(bPtrUntyped, bSize);
bBool fResetIndependantBuffer(bSize);
bBool fPullInto(bPtrUntyped, bSize);

private:
bBool fExchangeWith(bPtrUntyped, bSize, nAV::nDisk::eMemoryAccess);
};


struct sTask :
public std::enable_shared_from_this<sTask> {

VkDescriptorSet mVkDescriptorSet;

VkDescriptorPool mVkDescriptorPool;

SShared<sFuncCompiled> mFuncPtr;
SArr<SShared<sVarWBuffer>> mVarPtrs;
SShared<sTaskSequenceCompute> mGroupPtr;
SArr<bSize> mWorkDimensions;

~sTask();
void fDealloc();
VkDevice fDeviceID() const;
void fInitDescriptorPool();
void fInitDescriptorSet();

SShared<sVarWBuffer> fReallocVar(bSize lIdx);
};

struct sQueueFeatures {
bBool mSupportsGraphics = false;
bBool mSupportsCompute = false;
};
struct sQueueFamily {
bUInt32 mFamilyID = 0;
sQueueFeatures mFeatures;
};
struct sDeviceFeatures {
bSize mMemoryTotal = 0;
bBool mSupportsFlt16 = false;
SArr<sQueueFamily> mQueueFamilies;
sStr mName;
};
struct sQueue {
~sQueue();
void fDealloc();
VkDevice fDeviceID() const;
void fInitCommandPool();
void fRunNow(SShared<sTaskSequenceCompute>);

SArr<SShared<sTaskSequenceCompute>> mTasks;
SShared<sDevice> mDevicePtr;
VkQueue mVk;

sQueueFamily mFamily;

VkCommandPool mVkPoolCommands;
};

struct sDevice :
public std::enable_shared_from_this<sDevice> {

static constexpr bFlt32 kQueuePriorities[] = { 1.0, 0.8, 0.6 };

VkDevice mLogical;

SArr<VkPhysicalDevice> mPhysicals;

SArr<SShared<sQueue>> mQueuesCompute;

SArr<SShared<sQueue>> mQueuesGraphics;

sDeviceFeatures mFeatures;

~sDevice();
void fDealloc();
void fInitNewQueues(SArr<SPair<sQueueFamily, bSize>> lFamsAndCounts);
void fInitNewQueues(bSize lCntCompute, bSize lCntGraphics);
bUInt32 fFindMemoryGroupOfType(bUInt32 lMemoryTypeBits,
VkMemoryPropertyFlags);

SShared<sFuncCompiled> fCompile(SShared<nFlow::sFuncDectriptor>);

void fRunOnAnyQueue(SShared<sTask> const &);
};


struct sVulkan {

static sVulkan & gShared() {
static sVulkan lVk;
return lVk;
}


constexpr static bool kEnableValidationLayers = false;


VkInstance mVkInstance;
VkDebugReportCallbackEXT mDebugReportCallback;

SArr<SShared<sDevice>> mDevices;
SArr<sCStrC> mEnabledLayers;
SArr<sCStrC> mEnabledExtensions;


bUInt32 mQueuesComputePerDevice = 1;
bUInt32 mQueuesGraphicsPerDevice = 0;

static void gCheckCode(VkResult);

sVulkan();
~sVulkan();
void fDealloc();
void fRun(sTask);

private:
void fInitValidation();
void fInitInstance();
void fInitWarningsCallback();
void fInitDevices();
public:

SArr<SShared<sDevice>> fAddAllDevices();
static void gExportDevice(VkPhysicalDevice, sDeviceFeatures &);
static void gExportComputeQueueFamilies(VkPhysicalDevice, SArr<sQueueFamily> &);

};

}
