

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/EncodingType.h>
#include <aws/s3/model/MultipartUpload.h>
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
class AWS_S3_API ListMultipartUploadsResult
{
public:
ListMultipartUploadsResult();
ListMultipartUploadsResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
ListMultipartUploadsResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::String& GetBucket() const{ return m_bucket; }


inline void SetBucket(const Aws::String& value) { m_bucket = value; }


inline void SetBucket(Aws::String&& value) { m_bucket = std::move(value); }


inline void SetBucket(const char* value) { m_bucket.assign(value); }


inline ListMultipartUploadsResult& WithBucket(const Aws::String& value) { SetBucket(value); return *this;}


inline ListMultipartUploadsResult& WithBucket(Aws::String&& value) { SetBucket(std::move(value)); return *this;}


inline ListMultipartUploadsResult& WithBucket(const char* value) { SetBucket(value); return *this;}



inline const Aws::String& GetKeyMarker() const{ return m_keyMarker; }


inline void SetKeyMarker(const Aws::String& value) { m_keyMarker = value; }


inline void SetKeyMarker(Aws::String&& value) { m_keyMarker = std::move(value); }


inline void SetKeyMarker(const char* value) { m_keyMarker.assign(value); }


inline ListMultipartUploadsResult& WithKeyMarker(const Aws::String& value) { SetKeyMarker(value); return *this;}


inline ListMultipartUploadsResult& WithKeyMarker(Aws::String&& value) { SetKeyMarker(std::move(value)); return *this;}


inline ListMultipartUploadsResult& WithKeyMarker(const char* value) { SetKeyMarker(value); return *this;}



inline const Aws::String& GetUploadIdMarker() const{ return m_uploadIdMarker; }


inline void SetUploadIdMarker(const Aws::String& value) { m_uploadIdMarker = value; }


inline void SetUploadIdMarker(Aws::String&& value) { m_uploadIdMarker = std::move(value); }


inline void SetUploadIdMarker(const char* value) { m_uploadIdMarker.assign(value); }


inline ListMultipartUploadsResult& WithUploadIdMarker(const Aws::String& value) { SetUploadIdMarker(value); return *this;}


inline ListMultipartUploadsResult& WithUploadIdMarker(Aws::String&& value) { SetUploadIdMarker(std::move(value)); return *this;}


inline ListMultipartUploadsResult& WithUploadIdMarker(const char* value) { SetUploadIdMarker(value); return *this;}



inline const Aws::String& GetNextKeyMarker() const{ return m_nextKeyMarker; }


inline void SetNextKeyMarker(const Aws::String& value) { m_nextKeyMarker = value; }


inline void SetNextKeyMarker(Aws::String&& value) { m_nextKeyMarker = std::move(value); }


inline void SetNextKeyMarker(const char* value) { m_nextKeyMarker.assign(value); }


inline ListMultipartUploadsResult& WithNextKeyMarker(const Aws::String& value) { SetNextKeyMarker(value); return *this;}


inline ListMultipartUploadsResult& WithNextKeyMarker(Aws::String&& value) { SetNextKeyMarker(std::move(value)); return *this;}


inline ListMultipartUploadsResult& WithNextKeyMarker(const char* value) { SetNextKeyMarker(value); return *this;}



inline const Aws::String& GetPrefix() const{ return m_prefix; }


inline void SetPrefix(const Aws::String& value) { m_prefix = value; }


inline void SetPrefix(Aws::String&& value) { m_prefix = std::move(value); }


inline void SetPrefix(const char* value) { m_prefix.assign(value); }


inline ListMultipartUploadsResult& WithPrefix(const Aws::String& value) { SetPrefix(value); return *this;}


inline ListMultipartUploadsResult& WithPrefix(Aws::String&& value) { SetPrefix(std::move(value)); return *this;}


inline ListMultipartUploadsResult& WithPrefix(const char* value) { SetPrefix(value); return *this;}



inline const Aws::String& GetDelimiter() const{ return m_delimiter; }


inline void SetDelimiter(const Aws::String& value) { m_delimiter = value; }


inline void SetDelimiter(Aws::String&& value) { m_delimiter = std::move(value); }


inline void SetDelimiter(const char* value) { m_delimiter.assign(value); }


inline ListMultipartUploadsResult& WithDelimiter(const Aws::String& value) { SetDelimiter(value); return *this;}


inline ListMultipartUploadsResult& WithDelimiter(Aws::String&& value) { SetDelimiter(std::move(value)); return *this;}


inline ListMultipartUploadsResult& WithDelimiter(const char* value) { SetDelimiter(value); return *this;}



inline const Aws::String& GetNextUploadIdMarker() const{ return m_nextUploadIdMarker; }


inline void SetNextUploadIdMarker(const Aws::String& value) { m_nextUploadIdMarker = value; }


inline void SetNextUploadIdMarker(Aws::String&& value) { m_nextUploadIdMarker = std::move(value); }


inline void SetNextUploadIdMarker(const char* value) { m_nextUploadIdMarker.assign(value); }


inline ListMultipartUploadsResult& WithNextUploadIdMarker(const Aws::String& value) { SetNextUploadIdMarker(value); return *this;}


inline ListMultipartUploadsResult& WithNextUploadIdMarker(Aws::String&& value) { SetNextUploadIdMarker(std::move(value)); return *this;}


inline ListMultipartUploadsResult& WithNextUploadIdMarker(const char* value) { SetNextUploadIdMarker(value); return *this;}



inline int GetMaxUploads() const{ return m_maxUploads; }


inline void SetMaxUploads(int value) { m_maxUploads = value; }


inline ListMultipartUploadsResult& WithMaxUploads(int value) { SetMaxUploads(value); return *this;}



inline bool GetIsTruncated() const{ return m_isTruncated; }


inline void SetIsTruncated(bool value) { m_isTruncated = value; }


inline ListMultipartUploadsResult& WithIsTruncated(bool value) { SetIsTruncated(value); return *this;}



inline const Aws::Vector<MultipartUpload>& GetUploads() const{ return m_uploads; }


inline void SetUploads(const Aws::Vector<MultipartUpload>& value) { m_uploads = value; }


inline void SetUploads(Aws::Vector<MultipartUpload>&& value) { m_uploads = std::move(value); }


inline ListMultipartUploadsResult& WithUploads(const Aws::Vector<MultipartUpload>& value) { SetUploads(value); return *this;}


inline ListMultipartUploadsResult& WithUploads(Aws::Vector<MultipartUpload>&& value) { SetUploads(std::move(value)); return *this;}


inline ListMultipartUploadsResult& AddUploads(const MultipartUpload& value) { m_uploads.push_back(value); return *this; }


inline ListMultipartUploadsResult& AddUploads(MultipartUpload&& value) { m_uploads.push_back(std::move(value)); return *this; }



inline const Aws::Vector<CommonPrefix>& GetCommonPrefixes() const{ return m_commonPrefixes; }


inline void SetCommonPrefixes(const Aws::Vector<CommonPrefix>& value) { m_commonPrefixes = value; }


inline void SetCommonPrefixes(Aws::Vector<CommonPrefix>&& value) { m_commonPrefixes = std::move(value); }


inline ListMultipartUploadsResult& WithCommonPrefixes(const Aws::Vector<CommonPrefix>& value) { SetCommonPrefixes(value); return *this;}


inline ListMultipartUploadsResult& WithCommonPrefixes(Aws::Vector<CommonPrefix>&& value) { SetCommonPrefixes(std::move(value)); return *this;}


inline ListMultipartUploadsResult& AddCommonPrefixes(const CommonPrefix& value) { m_commonPrefixes.push_back(value); return *this; }


inline ListMultipartUploadsResult& AddCommonPrefixes(CommonPrefix&& value) { m_commonPrefixes.push_back(std::move(value)); return *this; }



inline const EncodingType& GetEncodingType() const{ return m_encodingType; }


inline void SetEncodingType(const EncodingType& value) { m_encodingType = value; }


inline void SetEncodingType(EncodingType&& value) { m_encodingType = std::move(value); }


inline ListMultipartUploadsResult& WithEncodingType(const EncodingType& value) { SetEncodingType(value); return *this;}


inline ListMultipartUploadsResult& WithEncodingType(EncodingType&& value) { SetEncodingType(std::move(value)); return *this;}

private:

Aws::String m_bucket;

Aws::String m_keyMarker;

Aws::String m_uploadIdMarker;

Aws::String m_nextKeyMarker;

Aws::String m_prefix;

Aws::String m_delimiter;

Aws::String m_nextUploadIdMarker;

int m_maxUploads;

bool m_isTruncated;

Aws::Vector<MultipartUpload> m_uploads;

Aws::Vector<CommonPrefix> m_commonPrefixes;

EncodingType m_encodingType;
};

} 
} 
} 
