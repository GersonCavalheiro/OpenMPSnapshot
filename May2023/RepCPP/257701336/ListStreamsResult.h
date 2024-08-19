

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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

class AWS_KINESIS_API ListStreamsResult
{
public:
ListStreamsResult();
ListStreamsResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
ListStreamsResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);



inline const Aws::Vector<Aws::String>& GetStreamNames() const{ return m_streamNames; }


inline void SetStreamNames(const Aws::Vector<Aws::String>& value) { m_streamNames = value; }


inline void SetStreamNames(Aws::Vector<Aws::String>&& value) { m_streamNames = std::move(value); }


inline ListStreamsResult& WithStreamNames(const Aws::Vector<Aws::String>& value) { SetStreamNames(value); return *this;}


inline ListStreamsResult& WithStreamNames(Aws::Vector<Aws::String>&& value) { SetStreamNames(std::move(value)); return *this;}


inline ListStreamsResult& AddStreamNames(const Aws::String& value) { m_streamNames.push_back(value); return *this; }


inline ListStreamsResult& AddStreamNames(Aws::String&& value) { m_streamNames.push_back(std::move(value)); return *this; }


inline ListStreamsResult& AddStreamNames(const char* value) { m_streamNames.push_back(value); return *this; }



inline bool GetHasMoreStreams() const{ return m_hasMoreStreams; }


inline void SetHasMoreStreams(bool value) { m_hasMoreStreams = value; }


inline ListStreamsResult& WithHasMoreStreams(bool value) { SetHasMoreStreams(value); return *this;}

private:

Aws::Vector<Aws::String> m_streamNames;

bool m_hasMoreStreams;
};

} 
} 
} 
