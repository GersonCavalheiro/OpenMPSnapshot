

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/model/StreamDescriptionSummary.h>
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
class AWS_KINESIS_API DescribeStreamSummaryResult
{
public:
DescribeStreamSummaryResult();
DescribeStreamSummaryResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
DescribeStreamSummaryResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline const StreamDescriptionSummary& GetStreamDescriptionSummary() const{ return m_streamDescriptionSummary; }


inline void SetStreamDescriptionSummary(const StreamDescriptionSummary& value) { m_streamDescriptionSummary = value; }


inline void SetStreamDescriptionSummary(StreamDescriptionSummary&& value) { m_streamDescriptionSummary = std::move(value); }


inline DescribeStreamSummaryResult& WithStreamDescriptionSummary(const StreamDescriptionSummary& value) { SetStreamDescriptionSummary(value); return *this;}


inline DescribeStreamSummaryResult& WithStreamDescriptionSummary(StreamDescriptionSummary&& value) { SetStreamDescriptionSummary(std::move(value)); return *this;}

private:

StreamDescriptionSummary m_streamDescriptionSummary;
};

} 
} 
} 
