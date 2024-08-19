

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/model/StreamDescription.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace Utils
{
namespace Json
{
class JsonValue;
} 
} 
namespace Kinesis
{
namespace Model
{

class AWS_KINESIS_API DescribeStreamResult
{
public:
DescribeStreamResult();
DescribeStreamResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
DescribeStreamResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline const StreamDescription& GetStreamDescription() const{ return m_streamDescription; }


inline void SetStreamDescription(const StreamDescription& value) { m_streamDescription = value; }


inline void SetStreamDescription(StreamDescription&& value) { m_streamDescription = std::move(value); }


inline DescribeStreamResult& WithStreamDescription(const StreamDescription& value) { SetStreamDescription(value); return *this;}


inline DescribeStreamResult& WithStreamDescription(StreamDescription&& value) { SetStreamDescription(std::move(value)); return *this;}

private:

StreamDescription m_streamDescription;
};

} 
} 
} 
