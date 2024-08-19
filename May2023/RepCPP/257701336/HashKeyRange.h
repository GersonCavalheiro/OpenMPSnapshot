

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


class AWS_KINESIS_API HashKeyRange
{
public:
HashKeyRange();
HashKeyRange(const Aws::Utils::Json::JsonValue& jsonValue);
HashKeyRange& operator=(const Aws::Utils::Json::JsonValue& jsonValue);
Aws::Utils::Json::JsonValue Jsonize() const;



inline const Aws::String& GetStartingHashKey() const{ return m_startingHashKey; }


inline void SetStartingHashKey(const Aws::String& value) { m_startingHashKeyHasBeenSet = true; m_startingHashKey = value; }


inline void SetStartingHashKey(Aws::String&& value) { m_startingHashKeyHasBeenSet = true; m_startingHashKey = std::move(value); }


inline void SetStartingHashKey(const char* value) { m_startingHashKeyHasBeenSet = true; m_startingHashKey.assign(value); }


inline HashKeyRange& WithStartingHashKey(const Aws::String& value) { SetStartingHashKey(value); return *this;}


inline HashKeyRange& WithStartingHashKey(Aws::String&& value) { SetStartingHashKey(std::move(value)); return *this;}


inline HashKeyRange& WithStartingHashKey(const char* value) { SetStartingHashKey(value); return *this;}



inline const Aws::String& GetEndingHashKey() const{ return m_endingHashKey; }


inline void SetEndingHashKey(const Aws::String& value) { m_endingHashKeyHasBeenSet = true; m_endingHashKey = value; }


inline void SetEndingHashKey(Aws::String&& value) { m_endingHashKeyHasBeenSet = true; m_endingHashKey = std::move(value); }


inline void SetEndingHashKey(const char* value) { m_endingHashKeyHasBeenSet = true; m_endingHashKey.assign(value); }


inline HashKeyRange& WithEndingHashKey(const Aws::String& value) { SetEndingHashKey(value); return *this;}


inline HashKeyRange& WithEndingHashKey(Aws::String&& value) { SetEndingHashKey(std::move(value)); return *this;}


inline HashKeyRange& WithEndingHashKey(const char* value) { SetEndingHashKey(value); return *this;}

private:

Aws::String m_startingHashKey;
bool m_startingHashKeyHasBeenSet;

Aws::String m_endingHashKey;
bool m_endingHashKeyHasBeenSet;
};

} 
} 
} 
