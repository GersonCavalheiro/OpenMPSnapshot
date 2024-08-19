

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/RequestCharged.h>
#include <aws/s3/model/DeletedObject.h>
#include <aws/s3/model/Error.h>
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
class AWS_S3_API DeleteObjectsResult
{
public:
DeleteObjectsResult();
DeleteObjectsResult(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);
DeleteObjectsResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>& result);



inline const Aws::Vector<DeletedObject>& GetDeleted() const{ return m_deleted; }


inline void SetDeleted(const Aws::Vector<DeletedObject>& value) { m_deleted = value; }


inline void SetDeleted(Aws::Vector<DeletedObject>&& value) { m_deleted = std::move(value); }


inline DeleteObjectsResult& WithDeleted(const Aws::Vector<DeletedObject>& value) { SetDeleted(value); return *this;}


inline DeleteObjectsResult& WithDeleted(Aws::Vector<DeletedObject>&& value) { SetDeleted(std::move(value)); return *this;}


inline DeleteObjectsResult& AddDeleted(const DeletedObject& value) { m_deleted.push_back(value); return *this; }


inline DeleteObjectsResult& AddDeleted(DeletedObject&& value) { m_deleted.push_back(std::move(value)); return *this; }



inline const RequestCharged& GetRequestCharged() const{ return m_requestCharged; }


inline void SetRequestCharged(const RequestCharged& value) { m_requestCharged = value; }


inline void SetRequestCharged(RequestCharged&& value) { m_requestCharged = std::move(value); }


inline DeleteObjectsResult& WithRequestCharged(const RequestCharged& value) { SetRequestCharged(value); return *this;}


inline DeleteObjectsResult& WithRequestCharged(RequestCharged&& value) { SetRequestCharged(std::move(value)); return *this;}



inline const Aws::Vector<Error>& GetErrors() const{ return m_errors; }


inline void SetErrors(const Aws::Vector<Error>& value) { m_errors = value; }


inline void SetErrors(Aws::Vector<Error>&& value) { m_errors = std::move(value); }


inline DeleteObjectsResult& WithErrors(const Aws::Vector<Error>& value) { SetErrors(value); return *this;}


inline DeleteObjectsResult& WithErrors(Aws::Vector<Error>&& value) { SetErrors(std::move(value)); return *this;}


inline DeleteObjectsResult& AddErrors(const Error& value) { m_errors.push_back(value); return *this; }


inline DeleteObjectsResult& AddErrors(Error&& value) { m_errors.push_back(std::move(value)); return *this; }

private:

Aws::Vector<DeletedObject> m_deleted;

RequestCharged m_requestCharged;

Aws::Vector<Error> m_errors;
};

} 
} 
} 
