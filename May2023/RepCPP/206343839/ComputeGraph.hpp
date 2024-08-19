
#pragma once
#include "HelperAliasis.hpp"

namespace nAV {

enum class eArgPurpose {
kUnknown = 0,
kInput,
kOutput,
kInNOut,
kBuffer
};
}

namespace nFlow {
using namespace nAV;

struct sFuncDectriptor;
struct sArgDescriptor;

struct sExeSource :
public std::enable_shared_from_this<sExeSource> {
sStr mContent;
sStr mPath;
SArr<SShared<sFuncDectriptor>> mFuncs;

void fInitText();
SShared<sFuncDectriptor> fMakeFunc(sStr, SArr<sArgDescriptor> = { });
};

struct sFuncDectriptor :
public std::enable_shared_from_this<sFuncDectriptor> {
SArr<SShared<sArgDescriptor>> mArgs;
sStr mName;
SShared<sExeSource> mSourceFile;
};


struct sArgDescriptor :
public std::enable_shared_from_this<sArgDescriptor> {
sStr mName;
sHash32 mType = sHash32 { 0 };
eArgPurpose mPurpose = eArgPurpose::kUnknown;
bSize mFixedSize = 0;
};


struct sVarSymbol {
sStr mName;
};

struct sTaskDependancies {
enum class eStatus {
kReady,
kRunning,
kCompleted,
kRestructuingItself
};
std::atomic<bBool> mCompleted = false;
SArr<SShared<sTaskDependancies>> mDependencies;
SArr<SWeak<sTaskDependancies>> mFollowers;

bBool fCanStart() const;
bBool fMayStillNeedBuffers() const;
};

}


