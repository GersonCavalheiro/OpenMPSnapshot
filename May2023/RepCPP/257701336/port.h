


#ifndef TENSORFLOW_STREAM_EXECUTOR_PLATFORM_PORT_H_
#define TENSORFLOW_STREAM_EXECUTOR_PLATFORM_PORT_H_

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace stream_executor {

using tensorflow::int8;
using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;

using tensorflow::uint8;
using tensorflow::uint16;
using tensorflow::uint32;
using tensorflow::uint64;

#if !defined(PLATFORM_GOOGLE)
using std::string;
#endif

using tensorflow::LinkerInitialized;
using tensorflow::LINKER_INITIALIZED;

#define SE_FALLTHROUGH_INTENDED TF_FALLTHROUGH_INTENDED

}  

#define SE_DISALLOW_COPY_AND_ASSIGN TF_DISALLOW_COPY_AND_ASSIGN
#define SE_MUST_USE_RESULT TF_MUST_USE_RESULT
#define SE_PREDICT_TRUE TF_PREDICT_TRUE
#define SE_PREDICT_FALSE TF_PREDICT_FALSE

#endif  
