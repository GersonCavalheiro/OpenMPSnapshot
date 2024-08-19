

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/EncodingType.h>
#include <aws/s3/model/Object.h>
#include <aws/s3/model/CommonPrefix.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace Utils
{
namespace Xml
{
class XmlDocument;
} 
} 
namespace S3
{
namespace Model
{
class AWS_S3_API ListObjectsV2Result
{
public:
ListObjectsV2Result();
ListObjectsV2Result(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
ListObjectsV2Result& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline bool GetIsTruncated() const{ return m_isTruncated; }


inline void SetIsTruncated(bool value) { m_isTruncated = value; }


inline ListObjectsV2Result& WithIsTruncated(bool value) { SetIsTruncated(value); return *this;}



inline const Aws::Vector<Object>& GetContents() const{ return m_contents; }


inline void SetContents(const Aws::Vector<Object>& value) { m_contents = value; }


inline void SetContents(Aws::Vector<Object>&& value) { m_contents = std::move(value); }


inline ListObjectsV2Result& WithContents(const Aws::Vector<Object>& value) { SetContents(value); return *this;}


inline ListObjectsV2Result& WithContents(Aws::Vector<Object>&& value) { SetContents(std::move(value)); return *this;}


inline ListObjectsV2Result& AddContents(const Object& value) { m_contents.push_back(value); return *this; }


inline ListObjectsV2Result& AddContents(Object&& value) { m_contents.push_back(std::move(value)); return *this; }



inline const Aws::String& GetName() const{ return m_name; }


inline void SetName(const Aws::String& value) { m_name = value; }


inline void SetName(Aws::String&& value) { m_name = std::move(value); }


inline void SetName(const char* value) { m_name.assign(value); }


inline ListObjectsV2Result& WithName(const Aws::String& value) { SetName(value); return *this;}


inline ListObjectsV2Result& WithName(Aws::String&& value) { SetName(std::move(value)); return *this;}


inline ListObjectsV2Result& WithName(const char* value) { SetName(value); return *this;}



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefix.assign(value); }


inline ListObjectsV2Result& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline ListObjectsV2Result& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline ListObjectsV2Result& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const Aws::String& GetDelimiter() const{ return m_delimiter; }


inline void SetDelimiter(const Aws::String& value) { m_delimiter = value; }


inline void SetDelimiter(Aws::String&& value) { m_delimiter = std::move(value); }


inline void SetDelimiter(const char* value) { m_delimiter.assign(value); }


inline ListObjectsV2Result& WithDelimiter(const Aws::String& value) { SetDelimiter(value); return *this;}


inline ListObjectsV2Result& WithDelimiter(Aws::String&& value) { SetDelimiter(std::move(value)); return *this;}


inline ListObjectsV2Result& WithDelimiter(const char* value) { SetDelimiter(value); return *this;}



inline int GetMaxKeys() const{ return m_maxKeys; }


inline void SetMaxKeys(int value) { m_maxKeys = value; }


inline ListObjectsV2Result& WithMaxKeys(int value) { SetMaxKeys(value); return *this;}



inline const Aws::Vector<CommonPrefix>& GetCommonPrefixes() const{ return m_commonPrefixes; }


inline void SetCommonPrefixes(const Aws::Vector<CommonPrefix>& value) { m_commonPrefixes = value; }


inline void SetCommonPrefixes(Aws::Vector<CommonPrefix>&& value) { m_commonPrefixes = std::move(value); }


inline ListObjectsV2Result& WithCommonPrefixes(const Aws::Vector<CommonPrefix>& value) { SetCommonPrefixes(value); return *this;}


inline ListObjectsV2Result& WithCommonPrefixes(Aws::Vector<CommonPrefix>&& value) { SetCommonPrefixes(std::move(value)); return *this;}


inline ListObjectsV2Result& AddCommonPrefixes(const CommonPrefix& value) { m_commonPrefixes.push_back(value); return *this; }


inline ListObjectsV2Result& AddCommonPrefixes(CommonPrefix&& value) { m_commonPrefixes.push_back(std::move(value)); return *this; }



inline const EncodingType& GetEncodingType() const{ return m_encodingType; }


inline void SetEncodingType(const EncodingType& value) { m_encodingType = value; }


inline void SetEncodingType(EncodingType&& value) { m_encodingType = std::move(value); }


inline ListObjectsV2Result& WithEncodingType(const EncodingType& value) { SetEncodingType(value); return *this;}


inline ListObjectsV2Result& WithEncodingType(EncodingType&& value) { SetEncodingType(std::move(value)); return *this;}



inline int GetKeyCount() const{ return m_keyCount; }


inline void SetKeyCount(int value) { m_keyCount = value; }


inline ListObjectsV2Result& WithKeyCount(int value) { SetKeyCount(value); return *this;}



inline const Aws::String& GetContinuationToken() const{ return m_continuationToken; }


inline void SetContinuationToken(const Aws::String& value) { m_continuationToken = value; }


inline void SetContinuationToken(Aws::String&& value) { m_continuationToken = std::move(value); }


inline void SetContinuationToken(const char* value) { m_continuationToken.assign(value); }


inline ListObjectsV2Result& WithContinuationToken(const Aws::String& value) { SetContinuationToken(value); return *this;}


inline ListObjectsV2Result& WithContinuationToken(Aws::String&& value) { SetContinuationToken(std::move(value)); return *this;}


inline ListObjectsV2Result& WithContinuationToken(const char* value) { SetContinuationToken(value); return *this;}



inline const Aws::String& GetNextContinuationToken() const{ return m_nextContinuationToken; }


inline void SetNextContinuationToken(const Aws::String& value) { m_nextContinuationToken = value; }


inline void SetNextContinuationToken(Aws::String&& value) { m_nextContinuationToken = std::move(value); }


inline void SetNextContinuationToken(const char* value) { m_nextContinuationToken.assign(value); }


inline ListObjectsV2Result& WithNextContinuationToken(const Aws::String& value) { SetNextContinuationToken(value); return *this;}


inline ListObjectsV2Result& WithNextContinuationToken(Aws::String&& value) { SetNextContinuationToken(std::move(value)); return *this;}


inline ListObjectsV2Result& WithNextContinuationToken(const char* value) { SetNextContinuationToken(value); return *this;}



inline const Aws::String& GetStartAfter() const{ return m_startAfter; }


inline void SetStartAfter(const Aws::String& value) { m_startAfter = value; }


inline void SetStartAfter(Aws::String&& value) { m_startAfter = std::move(value); }


inline void SetStartAfter(const char* value) { m_startAfter.assign(value); }


inline ListObjectsV2Result& WithStartAfter(const Aws::String& value) { SetStartAfter(value); return *this;}


inline ListObjectsV2Result& WithStartAfter(Aws::String&& value) { SetStartAfter(std::move(value)); return *this;}


inline ListObjectsV2Result& WithStartAfter(const char* value) { SetStartAfter(value); return *this;}

private:

bool m_isTruncated;

Aws::Vector<Object> m_contents;

Aws::String m_name;

Aws::String m_prefix;

Aws::String m_delimiter;

int m_maxKeys;

Aws::Vector<CommonPrefix> m_commonPrefixes;

EncodingType m_encodingType;

int m_keyCount;

Aws::String m_continuationToken;

Aws::String m_nextContinuationToken;

Aws::String m_startAfter;
};

} 
} 
} 
