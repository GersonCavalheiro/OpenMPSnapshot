

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/EncodingType.h>
#include <aws/s3/model/ObjectVersion.h>
#include <aws/s3/model/DeleteMarkerEntry.h>
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
class AWS_S3_API ListObjectVersionsResult
{
public:
ListObjectVersionsResult();
ListObjectVersionsResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
ListObjectVersionsResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline bool GetIsTruncated() const{ return m_isTruncated; }


inline void SetIsTruncated(bool value) { m_isTruncated = value; }


inline ListObjectVersionsResult& WithIsTruncated(bool value) { SetIsTruncated(value); return *this;}



inline const Aws::String& GetKeyMarker() const{ return m_keyMarker; }


inline void SetKeyMarker(const Aws::String& value) { m_keyMarker = value; }


inline void SetKeyMarker(Aws::String&& value) { m_keyMarker = std::move(value); }


inline void SetKeyMarker(const char* value) { m_keyMarker.assign(value); }


inline ListObjectVersionsResult& WithKeyMarker(const Aws::String& value) { SetKeyMarker(value); return *this;}


inline ListObjectVersionsResult& WithKeyMarker(Aws::String&& value) { SetKeyMarker(std::move(value)); return *this;}


inline ListObjectVersionsResult& WithKeyMarker(const char* value) { SetKeyMarker(value); return *this;}



inline const Aws::String& GetVersionIdMarker() const{ return m_versionIdMarker; }


inline void SetVersionIdMarker(const Aws::String& value) { m_versionIdMarker = value; }


inline void SetVersionIdMarker(Aws::String&& value) { m_versionIdMarker = std::move(value); }


inline void SetVersionIdMarker(const char* value) { m_versionIdMarker.assign(value); }


inline ListObjectVersionsResult& WithVersionIdMarker(const Aws::String& value) { SetVersionIdMarker(value); return *this;}


inline ListObjectVersionsResult& WithVersionIdMarker(Aws::String&& value) { SetVersionIdMarker(std::move(value)); return *this;}


inline ListObjectVersionsResult& WithVersionIdMarker(const char* value) { SetVersionIdMarker(value); return *this;}



inline const Aws::String& GetNextKeyMarker() const{ return m_nextKeyMarker; }


inline void SetNextKeyMarker(const Aws::String& value) { m_nextKeyMarker = value; }


inline void SetNextKeyMarker(Aws::String&& value) { m_nextKeyMarker = std::move(value); }


inline void SetNextKeyMarker(const char* value) { m_nextKeyMarker.assign(value); }


inline ListObjectVersionsResult& WithNextKeyMarker(const Aws::String& value) { SetNextKeyMarker(value); return *this;}


inline ListObjectVersionsResult& WithNextKeyMarker(Aws::String&& value) { SetNextKeyMarker(std::move(value)); return *this;}


inline ListObjectVersionsResult& WithNextKeyMarker(const char* value) { SetNextKeyMarker(value); return *this;}



inline const Aws::String& GetNextVersionIdMarker() const{ return m_nextVersionIdMarker; }


inline void SetNextVersionIdMarker(const Aws::String& value) { m_nextVersionIdMarker = value; }


inline void SetNextVersionIdMarker(Aws::String&& value) { m_nextVersionIdMarker = std::move(value); }


inline void SetNextVersionIdMarker(const char* value) { m_nextVersionIdMarker.assign(value); }


inline ListObjectVersionsResult& WithNextVersionIdMarker(const Aws::String& value) { SetNextVersionIdMarker(value); return *this;}


inline ListObjectVersionsResult& WithNextVersionIdMarker(Aws::String&& value) { SetNextVersionIdMarker(std::move(value)); return *this;}


inline ListObjectVersionsResult& WithNextVersionIdMarker(const char* value) { SetNextVersionIdMarker(value); return *this;}



inline const Aws::Vector<ObjectVersion>& GetVersions() const{ return m_versions; }


inline void SetVersions(const Aws::Vector<ObjectVersion>& value) { m_versions = value; }


inline void SetVersions(Aws::Vector<ObjectVersion>&& value) { m_versions = std::move(value); }


inline ListObjectVersionsResult& WithVersions(const Aws::Vector<ObjectVersion>& value) { SetVersions(value); return *this;}


inline ListObjectVersionsResult& WithVersions(Aws::Vector<ObjectVersion>&& value) { SetVersions(std::move(value)); return *this;}


inline ListObjectVersionsResult& AddVersions(const ObjectVersion& value) { m_versions.push_back(value); return *this; }


inline ListObjectVersionsResult& AddVersions(ObjectVersion&& value) { m_versions.push_back(std::move(value)); return *this; }



inline const Aws::Vector<DeleteMarkerEntry>& GetDeleteMarkers() const{ return m_deleteMarkers; }


inline void SetDeleteMarkers(const Aws::Vector<DeleteMarkerEntry>& value) { m_deleteMarkers = value; }


inline void SetDeleteMarkers(Aws::Vector<DeleteMarkerEntry>&& value) { m_deleteMarkers = std::move(value); }


inline ListObjectVersionsResult& WithDeleteMarkers(const Aws::Vector<DeleteMarkerEntry>& value) { SetDeleteMarkers(value); return *this;}


inline ListObjectVersionsResult& WithDeleteMarkers(Aws::Vector<DeleteMarkerEntry>&& value) { SetDeleteMarkers(std::move(value)); return *this;}


inline ListObjectVersionsResult& AddDeleteMarkers(const DeleteMarkerEntry& value) { m_deleteMarkers.push_back(value); return *this; }


inline ListObjectVersionsResult& AddDeleteMarkers(DeleteMarkerEntry&& value) { m_deleteMarkers.push_back(std::move(value)); return *this; }



inline const Aws::String& GetName() const{ return m_name; }


inline void SetName(const Aws::String& value) { m_name = value; }


inline void SetName(Aws::String&& value) { m_name = std::move(value); }


inline void SetName(const char* value) { m_name.assign(value); }


inline ListObjectVersionsResult& WithName(const Aws::String& value) { SetName(value); return *this;}


inline ListObjectVersionsResult& WithName(Aws::String&& value) { SetName(std::move(value)); return *this;}


inline ListObjectVersionsResult& WithName(const char* value) { SetName(value); return *this;}



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefix.assign(value); }


inline ListObjectVersionsResult& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline ListObjectVersionsResult& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline ListObjectVersionsResult& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const Aws::String& GetDelimiter() const{ return m_delimiter; }


inline void SetDelimiter(const Aws::String& value) { m_delimiter = value; }


inline void SetDelimiter(Aws::String&& value) { m_delimiter = std::move(value); }


inline void SetDelimiter(const char* value) { m_delimiter.assign(value); }


inline ListObjectVersionsResult& WithDelimiter(const Aws::String& value) { SetDelimiter(value); return *this;}


inline ListObjectVersionsResult& WithDelimiter(Aws::String&& value) { SetDelimiter(std::move(value)); return *this;}


inline ListObjectVersionsResult& WithDelimiter(const char* value) { SetDelimiter(value); return *this;}



inline int GetMaxKeys() const{ return m_maxKeys; }


inline void SetMaxKeys(int value) { m_maxKeys = value; }


inline ListObjectVersionsResult& WithMaxKeys(int value) { SetMaxKeys(value); return *this;}



inline const Aws::Vector<CommonPrefix>& GetCommonPrefixes() const{ return m_commonPrefixes; }


inline void SetCommonPrefixes(const Aws::Vector<CommonPrefix>& value) { m_commonPrefixes = value; }


inline void SetCommonPrefixes(Aws::Vector<CommonPrefix>&& value) { m_commonPrefixes = std::move(value); }


inline ListObjectVersionsResult& WithCommonPrefixes(const Aws::Vector<CommonPrefix>& value) { SetCommonPrefixes(value); return *this;}


inline ListObjectVersionsResult& WithCommonPrefixes(Aws::Vector<CommonPrefix>&& value) { SetCommonPrefixes(std::move(value)); return *this;}


inline ListObjectVersionsResult& AddCommonPrefixes(const CommonPrefix& value) { m_commonPrefixes.push_back(value); return *this; }


inline ListObjectVersionsResult& AddCommonPrefixes(CommonPrefix&& value) { m_commonPrefixes.push_back(std::move(value)); return *this; }



inline const EncodingType& GetEncodingType() const{ return m_encodingType; }


inline void SetEncodingType(const EncodingType& value) { m_encodingType = value; }


inline void SetEncodingType(EncodingType&& value) { m_encodingType = std::move(value); }


inline ListObjectVersionsResult& WithEncodingType(const EncodingType& value) { SetEncodingType(value); return *this;}


inline ListObjectVersionsResult& WithEncodingType(EncodingType&& value) { SetEncodingType(std::move(value)); return *this;}

private:

bool m_isTruncated;

Aws::String m_keyMarker;

Aws::String m_versionIdMarker;

Aws::String m_nextKeyMarker;

Aws::String m_nextVersionIdMarker;

Aws::Vector<ObjectVersion> m_versions;

Aws::Vector<DeleteMarkerEntry> m_deleteMarkers;

Aws::String m_name;

Aws::String m_prefix;

Aws::String m_delimiter;

int m_maxKeys;

Aws::Vector<CommonPrefix> m_commonPrefixes;

EncodingType m_encodingType;
};

} 
} 
} 
