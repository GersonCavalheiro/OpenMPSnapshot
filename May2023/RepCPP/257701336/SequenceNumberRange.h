

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <utility>

namespace Aws
{
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


class AWS_KINESIS_API SequenceNumberRange
{
public:
SequenceNumberRange();
SequenceNumberRange(const Aws::Utils::Json::JsonValue& jsonValue);
SequenceNumberRange& operator=(const Aws::Utils::Json::JsonValue& jsonValue);
Aws::Utils::Json::JsonValue Jsonize() const;



inline const Aws::String& GetStartingSequenceNumber() const{ return m_startingSequenceNumber; }


inline void SetStartingSequenceNumber(const Aws::String& value) { m_startingSequenceNumberHasBeenSet = true; m_startingSequenceNumber = value; }


inline void SetStartingSequenceNumber(Aws::String&& value) { m_startingSequenceNumberHasBeenSet = true; m_startingSequenceNumber = std::move(value); }


inline void SetStartingSequenceNumber(const char* value) { m_startingSequenceNumberHasBeenSet = true; m_startingSequenceNumber.assign(value); }


inline SequenceNumberRange& WithStartingSequenceNumber(const Aws::String& value) { SetStartingSequenceNumber(value); return *this;}


inline SequenceNumberRange& WithStartingSequenceNumber(Aws::String&& value) { SetStartingSequenceNumber(std::move(value)); return *this;}


inline SequenceNumberRange& WithStartingSequenceNumber(const char* value) { SetStartingSequenceNumber(value); return *this;}



inline const Aws::String& GetEndingSequenceNumber() const{ return m_endingSequenceNumber; }


inline void SetEndingSequenceNumber(const Aws::String& value) { m_endingSequenceNumberHasBeenSet = true; m_endingSequenceNumber = value; }


inline void SetEndingSequenceNumber(Aws::String&& value) { m_endingSequenceNumberHasBeenSet = true; m_endingSequenceNumber = std::move(value); }


inline void SetEndingSequenceNumber(const char* value) { m_endingSequenceNumberHasBeenSet = true; m_endingSequenceNumber.assign(value); }


inline SequenceNumberRange& WithEndingSequenceNumber(const Aws::String& value) { SetEndingSequenceNumber(value); return *this;}


inline SequenceNumberRange& WithEndingSequenceNumber(Aws::String&& value) { SetEndingSequenceNumber(std::move(value)); return *this;}


inline SequenceNumberRange& WithEndingSequenceNumber(const char* value) { SetEndingSequenceNumber(value); return *this;}

private:

Aws::String m_startingSequenceNumber;
bool m_startingSequenceNumberHasBeenSet;

Aws::String m_endingSequenceNumber;
bool m_endingSequenceNumberHasBeenSet;
};

} 
} 
} 
