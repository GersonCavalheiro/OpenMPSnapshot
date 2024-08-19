

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/S3Errors.h>
#include <aws/core/client/AWSError.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/client/AWSClient.h>
#include <aws/core/auth/AWSAuthSigner.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/xml/XmlSerializer.h>
#include <aws/core/utils/DNS.h>
#include <aws/s3/model/AbortMultipartUploadResult.h>
#include <aws/s3/model/CompleteMultipartUploadResult.h>
#include <aws/s3/model/CopyObjectResult.h>
#include <aws/s3/model/CreateBucketResult.h>
#include <aws/s3/model/CreateMultipartUploadResult.h>
#include <aws/s3/model/DeleteObjectResult.h>
#include <aws/s3/model/DeleteObjectTaggingResult.h>
#include <aws/s3/model/DeleteObjectsResult.h>
#include <aws/s3/model/GetBucketAccelerateConfigurationResult.h>
#include <aws/s3/model/GetBucketAclResult.h>
#include <aws/s3/model/GetBucketAnalyticsConfigurationResult.h>
#include <aws/s3/model/GetBucketCorsResult.h>
#include <aws/s3/model/GetBucketEncryptionResult.h>
#include <aws/s3/model/GetBucketInventoryConfigurationResult.h>
#include <aws/s3/model/GetBucketLifecycleConfigurationResult.h>
#include <aws/s3/model/GetBucketLocationResult.h>
#include <aws/s3/model/GetBucketLoggingResult.h>
#include <aws/s3/model/GetBucketMetricsConfigurationResult.h>
#include <aws/s3/model/GetBucketNotificationConfigurationResult.h>
#include <aws/s3/model/GetBucketPolicyResult.h>
#include <aws/s3/model/GetBucketReplicationResult.h>
#include <aws/s3/model/GetBucketRequestPaymentResult.h>
#include <aws/s3/model/GetBucketTaggingResult.h>
#include <aws/s3/model/GetBucketVersioningResult.h>
#include <aws/s3/model/GetBucketWebsiteResult.h>
#include <aws/s3/model/GetObjectResult.h>
#include <aws/s3/model/GetObjectAclResult.h>
#include <aws/s3/model/GetObjectTaggingResult.h>
#include <aws/s3/model/GetObjectTorrentResult.h>
#include <aws/s3/model/HeadObjectResult.h>
#include <aws/s3/model/ListBucketAnalyticsConfigurationsResult.h>
#include <aws/s3/model/ListBucketInventoryConfigurationsResult.h>
#include <aws/s3/model/ListBucketMetricsConfigurationsResult.h>
#include <aws/s3/model/ListBucketsResult.h>
#include <aws/s3/model/ListMultipartUploadsResult.h>
#include <aws/s3/model/ListObjectVersionsResult.h>
#include <aws/s3/model/ListObjectsResult.h>
#include <aws/s3/model/ListObjectsV2Result.h>
#include <aws/s3/model/ListPartsResult.h>
#include <aws/s3/model/PutObjectResult.h>
#include <aws/s3/model/PutObjectAclResult.h>
#include <aws/s3/model/PutObjectTaggingResult.h>
#include <aws/s3/model/RestoreObjectResult.h>
#include <aws/s3/model/UploadPartResult.h>
#include <aws/s3/model/UploadPartCopyResult.h>
#include <aws/core/NoResult.h>
#include <aws/core/client/AsyncCallerContext.h>
#include <aws/core/http/HttpTypes.h>
#include <future>
#include <functional>
namespace Aws
{

namespace Http
{
class HttpClient;
class HttpClientFactory;
} 

namespace Utils
{
template< typename R, typename E> class Outcome;

namespace Threading
{
class Executor;
} 

namespace Xml
{
class XmlDocument;
} 
} 

namespace Auth
{
class AWSCredentials;
class AWSCredentialsProvider;
} 

namespace Client
{
class RetryStrategy;
} 

namespace S3
{
namespace Model
{
class AbortMultipartUploadRequest;
class CompleteMultipartUploadRequest;
class CopyObjectRequest;
class CreateBucketRequest;
class CreateMultipartUploadRequest;
class DeleteBucketRequest;
class DeleteBucketAnalyticsConfigurationRequest;
class DeleteBucketCorsRequest;
class DeleteBucketEncryptionRequest;
class DeleteBucketInventoryConfigurationRequest;
class DeleteBucketLifecycleRequest;
class DeleteBucketMetricsConfigurationRequest;
class DeleteBucketPolicyRequest;
class DeleteBucketReplicationRequest;
class DeleteBucketTaggingRequest;
class DeleteBucketWebsiteRequest;
class DeleteObjectRequest;
class DeleteObjectTaggingRequest;
class DeleteObjectsRequest;
class GetBucketAccelerateConfigurationRequest;
class GetBucketAclRequest;
class GetBucketAnalyticsConfigurationRequest;
class GetBucketCorsRequest;
class GetBucketEncryptionRequest;
class GetBucketInventoryConfigurationRequest;
class GetBucketLifecycleConfigurationRequest;
class GetBucketLocationRequest;
class GetBucketLoggingRequest;
class GetBucketMetricsConfigurationRequest;
class GetBucketNotificationConfigurationRequest;
class GetBucketPolicyRequest;
class GetBucketReplicationRequest;
class GetBucketRequestPaymentRequest;
class GetBucketTaggingRequest;
class GetBucketVersioningRequest;
class GetBucketWebsiteRequest;
class GetObjectRequest;
class GetObjectAclRequest;
class GetObjectTaggingRequest;
class GetObjectTorrentRequest;
class HeadBucketRequest;
class HeadObjectRequest;
class ListBucketAnalyticsConfigurationsRequest;
class ListBucketInventoryConfigurationsRequest;
class ListBucketMetricsConfigurationsRequest;
class ListMultipartUploadsRequest;
class ListObjectVersionsRequest;
class ListObjectsRequest;
class ListObjectsV2Request;
class ListPartsRequest;
class PutBucketAccelerateConfigurationRequest;
class PutBucketAclRequest;
class PutBucketAnalyticsConfigurationRequest;
class PutBucketCorsRequest;
class PutBucketEncryptionRequest;
class PutBucketInventoryConfigurationRequest;
class PutBucketLifecycleConfigurationRequest;
class PutBucketLoggingRequest;
class PutBucketMetricsConfigurationRequest;
class PutBucketNotificationConfigurationRequest;
class PutBucketPolicyRequest;
class PutBucketReplicationRequest;
class PutBucketRequestPaymentRequest;
class PutBucketTaggingRequest;
class PutBucketVersioningRequest;
class PutBucketWebsiteRequest;
class PutObjectRequest;
class PutObjectAclRequest;
class PutObjectTaggingRequest;
class RestoreObjectRequest;
class UploadPartRequest;
class UploadPartCopyRequest;

typedef Aws::Utils::Outcome<AbortMultipartUploadResult, Aws::Client::AWSError<S3Errors>> AbortMultipartUploadOutcome;
typedef Aws::Utils::Outcome<CompleteMultipartUploadResult, Aws::Client::AWSError<S3Errors>> CompleteMultipartUploadOutcome;
typedef Aws::Utils::Outcome<CopyObjectResult, Aws::Client::AWSError<S3Errors>> CopyObjectOutcome;
typedef Aws::Utils::Outcome<CreateBucketResult, Aws::Client::AWSError<S3Errors>> CreateBucketOutcome;
typedef Aws::Utils::Outcome<CreateMultipartUploadResult, Aws::Client::AWSError<S3Errors>> CreateMultipartUploadOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketAnalyticsConfigurationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketCorsOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketEncryptionOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketInventoryConfigurationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketLifecycleOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketMetricsConfigurationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketPolicyOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketReplicationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketTaggingOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> DeleteBucketWebsiteOutcome;
typedef Aws::Utils::Outcome<DeleteObjectResult, Aws::Client::AWSError<S3Errors>> DeleteObjectOutcome;
typedef Aws::Utils::Outcome<DeleteObjectTaggingResult, Aws::Client::AWSError<S3Errors>> DeleteObjectTaggingOutcome;
typedef Aws::Utils::Outcome<DeleteObjectsResult, Aws::Client::AWSError<S3Errors>> DeleteObjectsOutcome;
typedef Aws::Utils::Outcome<GetBucketAccelerateConfigurationResult, Aws::Client::AWSError<S3Errors>> GetBucketAccelerateConfigurationOutcome;
typedef Aws::Utils::Outcome<GetBucketAclResult, Aws::Client::AWSError<S3Errors>> GetBucketAclOutcome;
typedef Aws::Utils::Outcome<GetBucketAnalyticsConfigurationResult, Aws::Client::AWSError<S3Errors>> GetBucketAnalyticsConfigurationOutcome;
typedef Aws::Utils::Outcome<GetBucketCorsResult, Aws::Client::AWSError<S3Errors>> GetBucketCorsOutcome;
typedef Aws::Utils::Outcome<GetBucketEncryptionResult, Aws::Client::AWSError<S3Errors>> GetBucketEncryptionOutcome;
typedef Aws::Utils::Outcome<GetBucketInventoryConfigurationResult, Aws::Client::AWSError<S3Errors>> GetBucketInventoryConfigurationOutcome;
typedef Aws::Utils::Outcome<GetBucketLifecycleConfigurationResult, Aws::Client::AWSError<S3Errors>> GetBucketLifecycleConfigurationOutcome;
typedef Aws::Utils::Outcome<GetBucketLocationResult, Aws::Client::AWSError<S3Errors>> GetBucketLocationOutcome;
typedef Aws::Utils::Outcome<GetBucketLoggingResult, Aws::Client::AWSError<S3Errors>> GetBucketLoggingOutcome;
typedef Aws::Utils::Outcome<GetBucketMetricsConfigurationResult, Aws::Client::AWSError<S3Errors>> GetBucketMetricsConfigurationOutcome;
typedef Aws::Utils::Outcome<GetBucketNotificationConfigurationResult, Aws::Client::AWSError<S3Errors>> GetBucketNotificationConfigurationOutcome;
typedef Aws::Utils::Outcome<GetBucketPolicyResult, Aws::Client::AWSError<S3Errors>> GetBucketPolicyOutcome;
typedef Aws::Utils::Outcome<GetBucketReplicationResult, Aws::Client::AWSError<S3Errors>> GetBucketReplicationOutcome;
typedef Aws::Utils::Outcome<GetBucketRequestPaymentResult, Aws::Client::AWSError<S3Errors>> GetBucketRequestPaymentOutcome;
typedef Aws::Utils::Outcome<GetBucketTaggingResult, Aws::Client::AWSError<S3Errors>> GetBucketTaggingOutcome;
typedef Aws::Utils::Outcome<GetBucketVersioningResult, Aws::Client::AWSError<S3Errors>> GetBucketVersioningOutcome;
typedef Aws::Utils::Outcome<GetBucketWebsiteResult, Aws::Client::AWSError<S3Errors>> GetBucketWebsiteOutcome;
typedef Aws::Utils::Outcome<GetObjectResult, Aws::Client::AWSError<S3Errors>> GetObjectOutcome;
typedef Aws::Utils::Outcome<GetObjectAclResult, Aws::Client::AWSError<S3Errors>> GetObjectAclOutcome;
typedef Aws::Utils::Outcome<GetObjectTaggingResult, Aws::Client::AWSError<S3Errors>> GetObjectTaggingOutcome;
typedef Aws::Utils::Outcome<GetObjectTorrentResult, Aws::Client::AWSError<S3Errors>> GetObjectTorrentOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> HeadBucketOutcome;
typedef Aws::Utils::Outcome<HeadObjectResult, Aws::Client::AWSError<S3Errors>> HeadObjectOutcome;
typedef Aws::Utils::Outcome<ListBucketAnalyticsConfigurationsResult, Aws::Client::AWSError<S3Errors>> ListBucketAnalyticsConfigurationsOutcome;
typedef Aws::Utils::Outcome<ListBucketInventoryConfigurationsResult, Aws::Client::AWSError<S3Errors>> ListBucketInventoryConfigurationsOutcome;
typedef Aws::Utils::Outcome<ListBucketMetricsConfigurationsResult, Aws::Client::AWSError<S3Errors>> ListBucketMetricsConfigurationsOutcome;
typedef Aws::Utils::Outcome<ListBucketsResult, Aws::Client::AWSError<S3Errors>> ListBucketsOutcome;
typedef Aws::Utils::Outcome<ListMultipartUploadsResult, Aws::Client::AWSError<S3Errors>> ListMultipartUploadsOutcome;
typedef Aws::Utils::Outcome<ListObjectVersionsResult, Aws::Client::AWSError<S3Errors>> ListObjectVersionsOutcome;
typedef Aws::Utils::Outcome<ListObjectsResult, Aws::Client::AWSError<S3Errors>> ListObjectsOutcome;
typedef Aws::Utils::Outcome<ListObjectsV2Result, Aws::Client::AWSError<S3Errors>> ListObjectsV2Outcome;
typedef Aws::Utils::Outcome<ListPartsResult, Aws::Client::AWSError<S3Errors>> ListPartsOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketAccelerateConfigurationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketAclOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketAnalyticsConfigurationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketCorsOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketEncryptionOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketInventoryConfigurationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketLifecycleConfigurationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketLoggingOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketMetricsConfigurationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketNotificationConfigurationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketPolicyOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketReplicationOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketRequestPaymentOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketTaggingOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketVersioningOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<S3Errors>> PutBucketWebsiteOutcome;
typedef Aws::Utils::Outcome<PutObjectResult, Aws::Client::AWSError<S3Errors>> PutObjectOutcome;
typedef Aws::Utils::Outcome<PutObjectAclResult, Aws::Client::AWSError<S3Errors>> PutObjectAclOutcome;
typedef Aws::Utils::Outcome<PutObjectTaggingResult, Aws::Client::AWSError<S3Errors>> PutObjectTaggingOutcome;
typedef Aws::Utils::Outcome<RestoreObjectResult, Aws::Client::AWSError<S3Errors>> RestoreObjectOutcome;
typedef Aws::Utils::Outcome<UploadPartResult, Aws::Client::AWSError<S3Errors>> UploadPartOutcome;
typedef Aws::Utils::Outcome<UploadPartCopyResult, Aws::Client::AWSError<S3Errors>> UploadPartCopyOutcome;

typedef std::future<AbortMultipartUploadOutcome> AbortMultipartUploadOutcomeCallable;
typedef std::future<CompleteMultipartUploadOutcome> CompleteMultipartUploadOutcomeCallable;
typedef std::future<CopyObjectOutcome> CopyObjectOutcomeCallable;
typedef std::future<CreateBucketOutcome> CreateBucketOutcomeCallable;
typedef std::future<CreateMultipartUploadOutcome> CreateMultipartUploadOutcomeCallable;
typedef std::future<DeleteBucketOutcome> DeleteBucketOutcomeCallable;
typedef std::future<DeleteBucketAnalyticsConfigurationOutcome> DeleteBucketAnalyticsConfigurationOutcomeCallable;
typedef std::future<DeleteBucketCorsOutcome> DeleteBucketCorsOutcomeCallable;
typedef std::future<DeleteBucketEncryptionOutcome> DeleteBucketEncryptionOutcomeCallable;
typedef std::future<DeleteBucketInventoryConfigurationOutcome> DeleteBucketInventoryConfigurationOutcomeCallable;
typedef std::future<DeleteBucketLifecycleOutcome> DeleteBucketLifecycleOutcomeCallable;
typedef std::future<DeleteBucketMetricsConfigurationOutcome> DeleteBucketMetricsConfigurationOutcomeCallable;
typedef std::future<DeleteBucketPolicyOutcome> DeleteBucketPolicyOutcomeCallable;
typedef std::future<DeleteBucketReplicationOutcome> DeleteBucketReplicationOutcomeCallable;
typedef std::future<DeleteBucketTaggingOutcome> DeleteBucketTaggingOutcomeCallable;
typedef std::future<DeleteBucketWebsiteOutcome> DeleteBucketWebsiteOutcomeCallable;
typedef std::future<DeleteObjectOutcome> DeleteObjectOutcomeCallable;
typedef std::future<DeleteObjectTaggingOutcome> DeleteObjectTaggingOutcomeCallable;
typedef std::future<DeleteObjectsOutcome> DeleteObjectsOutcomeCallable;
typedef std::future<GetBucketAccelerateConfigurationOutcome> GetBucketAccelerateConfigurationOutcomeCallable;
typedef std::future<GetBucketAclOutcome> GetBucketAclOutcomeCallable;
typedef std::future<GetBucketAnalyticsConfigurationOutcome> GetBucketAnalyticsConfigurationOutcomeCallable;
typedef std::future<GetBucketCorsOutcome> GetBucketCorsOutcomeCallable;
typedef std::future<GetBucketEncryptionOutcome> GetBucketEncryptionOutcomeCallable;
typedef std::future<GetBucketInventoryConfigurationOutcome> GetBucketInventoryConfigurationOutcomeCallable;
typedef std::future<GetBucketLifecycleConfigurationOutcome> GetBucketLifecycleConfigurationOutcomeCallable;
typedef std::future<GetBucketLocationOutcome> GetBucketLocationOutcomeCallable;
typedef std::future<GetBucketLoggingOutcome> GetBucketLoggingOutcomeCallable;
typedef std::future<GetBucketMetricsConfigurationOutcome> GetBucketMetricsConfigurationOutcomeCallable;
typedef std::future<GetBucketNotificationConfigurationOutcome> GetBucketNotificationConfigurationOutcomeCallable;
typedef std::future<GetBucketPolicyOutcome> GetBucketPolicyOutcomeCallable;
typedef std::future<GetBucketReplicationOutcome> GetBucketReplicationOutcomeCallable;
typedef std::future<GetBucketRequestPaymentOutcome> GetBucketRequestPaymentOutcomeCallable;
typedef std::future<GetBucketTaggingOutcome> GetBucketTaggingOutcomeCallable;
typedef std::future<GetBucketVersioningOutcome> GetBucketVersioningOutcomeCallable;
typedef std::future<GetBucketWebsiteOutcome> GetBucketWebsiteOutcomeCallable;
typedef std::future<GetObjectOutcome> GetObjectOutcomeCallable;
typedef std::future<GetObjectAclOutcome> GetObjectAclOutcomeCallable;
typedef std::future<GetObjectTaggingOutcome> GetObjectTaggingOutcomeCallable;
typedef std::future<GetObjectTorrentOutcome> GetObjectTorrentOutcomeCallable;
typedef std::future<HeadBucketOutcome> HeadBucketOutcomeCallable;
typedef std::future<HeadObjectOutcome> HeadObjectOutcomeCallable;
typedef std::future<ListBucketAnalyticsConfigurationsOutcome> ListBucketAnalyticsConfigurationsOutcomeCallable;
typedef std::future<ListBucketInventoryConfigurationsOutcome> ListBucketInventoryConfigurationsOutcomeCallable;
typedef std::future<ListBucketMetricsConfigurationsOutcome> ListBucketMetricsConfigurationsOutcomeCallable;
typedef std::future<ListBucketsOutcome> ListBucketsOutcomeCallable;
typedef std::future<ListMultipartUploadsOutcome> ListMultipartUploadsOutcomeCallable;
typedef std::future<ListObjectVersionsOutcome> ListObjectVersionsOutcomeCallable;
typedef std::future<ListObjectsOutcome> ListObjectsOutcomeCallable;
typedef std::future<ListObjectsV2Outcome> ListObjectsV2OutcomeCallable;
typedef std::future<ListPartsOutcome> ListPartsOutcomeCallable;
typedef std::future<PutBucketAccelerateConfigurationOutcome> PutBucketAccelerateConfigurationOutcomeCallable;
typedef std::future<PutBucketAclOutcome> PutBucketAclOutcomeCallable;
typedef std::future<PutBucketAnalyticsConfigurationOutcome> PutBucketAnalyticsConfigurationOutcomeCallable;
typedef std::future<PutBucketCorsOutcome> PutBucketCorsOutcomeCallable;
typedef std::future<PutBucketEncryptionOutcome> PutBucketEncryptionOutcomeCallable;
typedef std::future<PutBucketInventoryConfigurationOutcome> PutBucketInventoryConfigurationOutcomeCallable;
typedef std::future<PutBucketLifecycleConfigurationOutcome> PutBucketLifecycleConfigurationOutcomeCallable;
typedef std::future<PutBucketLoggingOutcome> PutBucketLoggingOutcomeCallable;
typedef std::future<PutBucketMetricsConfigurationOutcome> PutBucketMetricsConfigurationOutcomeCallable;
typedef std::future<PutBucketNotificationConfigurationOutcome> PutBucketNotificationConfigurationOutcomeCallable;
typedef std::future<PutBucketPolicyOutcome> PutBucketPolicyOutcomeCallable;
typedef std::future<PutBucketReplicationOutcome> PutBucketReplicationOutcomeCallable;
typedef std::future<PutBucketRequestPaymentOutcome> PutBucketRequestPaymentOutcomeCallable;
typedef std::future<PutBucketTaggingOutcome> PutBucketTaggingOutcomeCallable;
typedef std::future<PutBucketVersioningOutcome> PutBucketVersioningOutcomeCallable;
typedef std::future<PutBucketWebsiteOutcome> PutBucketWebsiteOutcomeCallable;
typedef std::future<PutObjectOutcome> PutObjectOutcomeCallable;
typedef std::future<PutObjectAclOutcome> PutObjectAclOutcomeCallable;
typedef std::future<PutObjectTaggingOutcome> PutObjectTaggingOutcomeCallable;
typedef std::future<RestoreObjectOutcome> RestoreObjectOutcomeCallable;
typedef std::future<UploadPartOutcome> UploadPartOutcomeCallable;
typedef std::future<UploadPartCopyOutcome> UploadPartCopyOutcomeCallable;
} 

class S3Client;

typedef std::function<void(const S3Client*, const Model::AbortMultipartUploadRequest&, const Model::AbortMultipartUploadOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > AbortMultipartUploadResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::CompleteMultipartUploadRequest&, const Model::CompleteMultipartUploadOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > CompleteMultipartUploadResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::CopyObjectRequest&, const Model::CopyObjectOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > CopyObjectResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::CreateBucketRequest&, const Model::CreateBucketOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > CreateBucketResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::CreateMultipartUploadRequest&, const Model::CreateMultipartUploadOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > CreateMultipartUploadResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketRequest&, const Model::DeleteBucketOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketAnalyticsConfigurationRequest&, const Model::DeleteBucketAnalyticsConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketAnalyticsConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketCorsRequest&, const Model::DeleteBucketCorsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketCorsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketEncryptionRequest&, const Model::DeleteBucketEncryptionOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketEncryptionResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketInventoryConfigurationRequest&, const Model::DeleteBucketInventoryConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketInventoryConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketLifecycleRequest&, const Model::DeleteBucketLifecycleOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketLifecycleResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketMetricsConfigurationRequest&, const Model::DeleteBucketMetricsConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketMetricsConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketPolicyRequest&, const Model::DeleteBucketPolicyOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketPolicyResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketReplicationRequest&, const Model::DeleteBucketReplicationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketReplicationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketTaggingRequest&, const Model::DeleteBucketTaggingOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketTaggingResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteBucketWebsiteRequest&, const Model::DeleteBucketWebsiteOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteBucketWebsiteResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteObjectRequest&, const Model::DeleteObjectOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteObjectResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteObjectTaggingRequest&, const Model::DeleteObjectTaggingOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteObjectTaggingResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::DeleteObjectsRequest&, const Model::DeleteObjectsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteObjectsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketAccelerateConfigurationRequest&, const Model::GetBucketAccelerateConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketAccelerateConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketAclRequest&, const Model::GetBucketAclOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketAclResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketAnalyticsConfigurationRequest&, const Model::GetBucketAnalyticsConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketAnalyticsConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketCorsRequest&, const Model::GetBucketCorsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketCorsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketEncryptionRequest&, const Model::GetBucketEncryptionOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketEncryptionResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketInventoryConfigurationRequest&, const Model::GetBucketInventoryConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketInventoryConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketLifecycleConfigurationRequest&, const Model::GetBucketLifecycleConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketLifecycleConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketLocationRequest&, const Model::GetBucketLocationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketLocationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketLoggingRequest&, const Model::GetBucketLoggingOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketLoggingResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketMetricsConfigurationRequest&, const Model::GetBucketMetricsConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketMetricsConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketNotificationConfigurationRequest&, const Model::GetBucketNotificationConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketNotificationConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketPolicyRequest&, const Model::GetBucketPolicyOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketPolicyResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketReplicationRequest&, const Model::GetBucketReplicationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketReplicationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketRequestPaymentRequest&, const Model::GetBucketRequestPaymentOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketRequestPaymentResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketTaggingRequest&, const Model::GetBucketTaggingOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketTaggingResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketVersioningRequest&, const Model::GetBucketVersioningOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketVersioningResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetBucketWebsiteRequest&, const Model::GetBucketWebsiteOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetBucketWebsiteResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetObjectRequest&, const Model::GetObjectOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetObjectResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetObjectAclRequest&, const Model::GetObjectAclOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetObjectAclResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetObjectTaggingRequest&, const Model::GetObjectTaggingOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetObjectTaggingResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::GetObjectTorrentRequest&, const Model::GetObjectTorrentOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetObjectTorrentResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::HeadBucketRequest&, const Model::HeadBucketOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > HeadBucketResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::HeadObjectRequest&, const Model::HeadObjectOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > HeadObjectResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::ListBucketAnalyticsConfigurationsRequest&, const Model::ListBucketAnalyticsConfigurationsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListBucketAnalyticsConfigurationsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::ListBucketInventoryConfigurationsRequest&, const Model::ListBucketInventoryConfigurationsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListBucketInventoryConfigurationsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::ListBucketMetricsConfigurationsRequest&, const Model::ListBucketMetricsConfigurationsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListBucketMetricsConfigurationsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::ListBucketsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListBucketsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::ListMultipartUploadsRequest&, const Model::ListMultipartUploadsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListMultipartUploadsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::ListObjectVersionsRequest&, const Model::ListObjectVersionsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListObjectVersionsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::ListObjectsRequest&, const Model::ListObjectsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListObjectsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::ListObjectsV2Request&, const Model::ListObjectsV2Outcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListObjectsV2ResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::ListPartsRequest&, const Model::ListPartsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListPartsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketAccelerateConfigurationRequest&, const Model::PutBucketAccelerateConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketAccelerateConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketAclRequest&, const Model::PutBucketAclOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketAclResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketAnalyticsConfigurationRequest&, const Model::PutBucketAnalyticsConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketAnalyticsConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketCorsRequest&, const Model::PutBucketCorsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketCorsResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketEncryptionRequest&, const Model::PutBucketEncryptionOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketEncryptionResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketInventoryConfigurationRequest&, const Model::PutBucketInventoryConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketInventoryConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketLifecycleConfigurationRequest&, const Model::PutBucketLifecycleConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketLifecycleConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketLoggingRequest&, const Model::PutBucketLoggingOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketLoggingResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketMetricsConfigurationRequest&, const Model::PutBucketMetricsConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketMetricsConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketNotificationConfigurationRequest&, const Model::PutBucketNotificationConfigurationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketNotificationConfigurationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketPolicyRequest&, const Model::PutBucketPolicyOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketPolicyResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketReplicationRequest&, const Model::PutBucketReplicationOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketReplicationResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketRequestPaymentRequest&, const Model::PutBucketRequestPaymentOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketRequestPaymentResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketTaggingRequest&, const Model::PutBucketTaggingOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketTaggingResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketVersioningRequest&, const Model::PutBucketVersioningOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketVersioningResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutBucketWebsiteRequest&, const Model::PutBucketWebsiteOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutBucketWebsiteResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutObjectRequest&, const Model::PutObjectOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutObjectResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutObjectAclRequest&, const Model::PutObjectAclOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutObjectAclResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::PutObjectTaggingRequest&, const Model::PutObjectTaggingOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutObjectTaggingResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::RestoreObjectRequest&, const Model::RestoreObjectOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > RestoreObjectResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::UploadPartRequest&, const Model::UploadPartOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > UploadPartResponseReceivedHandler;
typedef std::function<void(const S3Client*, const Model::UploadPartCopyRequest&, const Model::UploadPartCopyOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > UploadPartCopyResponseReceivedHandler;

static const unsigned MAX_EXPIRATION_SECONDS = 7 * 24 * 60 * 60;

class AWS_S3_API S3Client : public Aws::Client::AWSXMLClient
{
public:
typedef Aws::Client::AWSXMLClient BASECLASS;


S3Client(const Aws::Client::ClientConfiguration& clientConfiguration = Aws::Client::ClientConfiguration(), Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy signPayloads = Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, bool useVirtualAdressing = true);


S3Client(const Aws::Auth::AWSCredentials& credentials, const Aws::Client::ClientConfiguration& clientConfiguration = Aws::Client::ClientConfiguration(), Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy signPayloads = Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, bool useVirtualAdressing = true);


S3Client(const std::shared_ptr<Aws::Auth::AWSCredentialsProvider>& credentialsProvider,
const Aws::Client::ClientConfiguration& clientConfiguration = Aws::Client::ClientConfiguration(), Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy signPayloads = Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, bool useVirtualAdressing = true);

virtual ~S3Client();

inline virtual const char* GetServiceClientName() const override { return "s3"; }



virtual Model::AbortMultipartUploadOutcome AbortMultipartUpload(const Model::AbortMultipartUploadRequest& request) const;


virtual Model::AbortMultipartUploadOutcomeCallable AbortMultipartUploadCallable(const Model::AbortMultipartUploadRequest& request) const;


virtual void AbortMultipartUploadAsync(const Model::AbortMultipartUploadRequest& request, const AbortMultipartUploadResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::CompleteMultipartUploadOutcome CompleteMultipartUpload(const Model::CompleteMultipartUploadRequest& request) const;


virtual Model::CompleteMultipartUploadOutcomeCallable CompleteMultipartUploadCallable(const Model::CompleteMultipartUploadRequest& request) const;


virtual void CompleteMultipartUploadAsync(const Model::CompleteMultipartUploadRequest& request, const CompleteMultipartUploadResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::CopyObjectOutcome CopyObject(const Model::CopyObjectRequest& request) const;


virtual Model::CopyObjectOutcomeCallable CopyObjectCallable(const Model::CopyObjectRequest& request) const;


virtual void CopyObjectAsync(const Model::CopyObjectRequest& request, const CopyObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::CreateBucketOutcome CreateBucket(const Model::CreateBucketRequest& request) const;


virtual Model::CreateBucketOutcomeCallable CreateBucketCallable(const Model::CreateBucketRequest& request) const;


virtual void CreateBucketAsync(const Model::CreateBucketRequest& request, const CreateBucketResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::CreateMultipartUploadOutcome CreateMultipartUpload(const Model::CreateMultipartUploadRequest& request) const;


virtual Model::CreateMultipartUploadOutcomeCallable CreateMultipartUploadCallable(const Model::CreateMultipartUploadRequest& request) const;


virtual void CreateMultipartUploadAsync(const Model::CreateMultipartUploadRequest& request, const CreateMultipartUploadResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketOutcome DeleteBucket(const Model::DeleteBucketRequest& request) const;


virtual Model::DeleteBucketOutcomeCallable DeleteBucketCallable(const Model::DeleteBucketRequest& request) const;


virtual void DeleteBucketAsync(const Model::DeleteBucketRequest& request, const DeleteBucketResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketAnalyticsConfigurationOutcome DeleteBucketAnalyticsConfiguration(const Model::DeleteBucketAnalyticsConfigurationRequest& request) const;


virtual Model::DeleteBucketAnalyticsConfigurationOutcomeCallable DeleteBucketAnalyticsConfigurationCallable(const Model::DeleteBucketAnalyticsConfigurationRequest& request) const;


virtual void DeleteBucketAnalyticsConfigurationAsync(const Model::DeleteBucketAnalyticsConfigurationRequest& request, const DeleteBucketAnalyticsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketCorsOutcome DeleteBucketCors(const Model::DeleteBucketCorsRequest& request) const;


virtual Model::DeleteBucketCorsOutcomeCallable DeleteBucketCorsCallable(const Model::DeleteBucketCorsRequest& request) const;


virtual void DeleteBucketCorsAsync(const Model::DeleteBucketCorsRequest& request, const DeleteBucketCorsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketEncryptionOutcome DeleteBucketEncryption(const Model::DeleteBucketEncryptionRequest& request) const;


virtual Model::DeleteBucketEncryptionOutcomeCallable DeleteBucketEncryptionCallable(const Model::DeleteBucketEncryptionRequest& request) const;


virtual void DeleteBucketEncryptionAsync(const Model::DeleteBucketEncryptionRequest& request, const DeleteBucketEncryptionResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketInventoryConfigurationOutcome DeleteBucketInventoryConfiguration(const Model::DeleteBucketInventoryConfigurationRequest& request) const;


virtual Model::DeleteBucketInventoryConfigurationOutcomeCallable DeleteBucketInventoryConfigurationCallable(const Model::DeleteBucketInventoryConfigurationRequest& request) const;


virtual void DeleteBucketInventoryConfigurationAsync(const Model::DeleteBucketInventoryConfigurationRequest& request, const DeleteBucketInventoryConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketLifecycleOutcome DeleteBucketLifecycle(const Model::DeleteBucketLifecycleRequest& request) const;


virtual Model::DeleteBucketLifecycleOutcomeCallable DeleteBucketLifecycleCallable(const Model::DeleteBucketLifecycleRequest& request) const;


virtual void DeleteBucketLifecycleAsync(const Model::DeleteBucketLifecycleRequest& request, const DeleteBucketLifecycleResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketMetricsConfigurationOutcome DeleteBucketMetricsConfiguration(const Model::DeleteBucketMetricsConfigurationRequest& request) const;


virtual Model::DeleteBucketMetricsConfigurationOutcomeCallable DeleteBucketMetricsConfigurationCallable(const Model::DeleteBucketMetricsConfigurationRequest& request) const;


virtual void DeleteBucketMetricsConfigurationAsync(const Model::DeleteBucketMetricsConfigurationRequest& request, const DeleteBucketMetricsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketPolicyOutcome DeleteBucketPolicy(const Model::DeleteBucketPolicyRequest& request) const;


virtual Model::DeleteBucketPolicyOutcomeCallable DeleteBucketPolicyCallable(const Model::DeleteBucketPolicyRequest& request) const;


virtual void DeleteBucketPolicyAsync(const Model::DeleteBucketPolicyRequest& request, const DeleteBucketPolicyResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketReplicationOutcome DeleteBucketReplication(const Model::DeleteBucketReplicationRequest& request) const;


virtual Model::DeleteBucketReplicationOutcomeCallable DeleteBucketReplicationCallable(const Model::DeleteBucketReplicationRequest& request) const;


virtual void DeleteBucketReplicationAsync(const Model::DeleteBucketReplicationRequest& request, const DeleteBucketReplicationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketTaggingOutcome DeleteBucketTagging(const Model::DeleteBucketTaggingRequest& request) const;


virtual Model::DeleteBucketTaggingOutcomeCallable DeleteBucketTaggingCallable(const Model::DeleteBucketTaggingRequest& request) const;


virtual void DeleteBucketTaggingAsync(const Model::DeleteBucketTaggingRequest& request, const DeleteBucketTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteBucketWebsiteOutcome DeleteBucketWebsite(const Model::DeleteBucketWebsiteRequest& request) const;


virtual Model::DeleteBucketWebsiteOutcomeCallable DeleteBucketWebsiteCallable(const Model::DeleteBucketWebsiteRequest& request) const;


virtual void DeleteBucketWebsiteAsync(const Model::DeleteBucketWebsiteRequest& request, const DeleteBucketWebsiteResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteObjectOutcome DeleteObject(const Model::DeleteObjectRequest& request) const;


virtual Model::DeleteObjectOutcomeCallable DeleteObjectCallable(const Model::DeleteObjectRequest& request) const;


virtual void DeleteObjectAsync(const Model::DeleteObjectRequest& request, const DeleteObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteObjectTaggingOutcome DeleteObjectTagging(const Model::DeleteObjectTaggingRequest& request) const;


virtual Model::DeleteObjectTaggingOutcomeCallable DeleteObjectTaggingCallable(const Model::DeleteObjectTaggingRequest& request) const;


virtual void DeleteObjectTaggingAsync(const Model::DeleteObjectTaggingRequest& request, const DeleteObjectTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteObjectsOutcome DeleteObjects(const Model::DeleteObjectsRequest& request) const;


virtual Model::DeleteObjectsOutcomeCallable DeleteObjectsCallable(const Model::DeleteObjectsRequest& request) const;


virtual void DeleteObjectsAsync(const Model::DeleteObjectsRequest& request, const DeleteObjectsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketAccelerateConfigurationOutcome GetBucketAccelerateConfiguration(const Model::GetBucketAccelerateConfigurationRequest& request) const;


virtual Model::GetBucketAccelerateConfigurationOutcomeCallable GetBucketAccelerateConfigurationCallable(const Model::GetBucketAccelerateConfigurationRequest& request) const;


virtual void GetBucketAccelerateConfigurationAsync(const Model::GetBucketAccelerateConfigurationRequest& request, const GetBucketAccelerateConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketAclOutcome GetBucketAcl(const Model::GetBucketAclRequest& request) const;


virtual Model::GetBucketAclOutcomeCallable GetBucketAclCallable(const Model::GetBucketAclRequest& request) const;


virtual void GetBucketAclAsync(const Model::GetBucketAclRequest& request, const GetBucketAclResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketAnalyticsConfigurationOutcome GetBucketAnalyticsConfiguration(const Model::GetBucketAnalyticsConfigurationRequest& request) const;


virtual Model::GetBucketAnalyticsConfigurationOutcomeCallable GetBucketAnalyticsConfigurationCallable(const Model::GetBucketAnalyticsConfigurationRequest& request) const;


virtual void GetBucketAnalyticsConfigurationAsync(const Model::GetBucketAnalyticsConfigurationRequest& request, const GetBucketAnalyticsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketCorsOutcome GetBucketCors(const Model::GetBucketCorsRequest& request) const;


virtual Model::GetBucketCorsOutcomeCallable GetBucketCorsCallable(const Model::GetBucketCorsRequest& request) const;


virtual void GetBucketCorsAsync(const Model::GetBucketCorsRequest& request, const GetBucketCorsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketEncryptionOutcome GetBucketEncryption(const Model::GetBucketEncryptionRequest& request) const;


virtual Model::GetBucketEncryptionOutcomeCallable GetBucketEncryptionCallable(const Model::GetBucketEncryptionRequest& request) const;


virtual void GetBucketEncryptionAsync(const Model::GetBucketEncryptionRequest& request, const GetBucketEncryptionResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketInventoryConfigurationOutcome GetBucketInventoryConfiguration(const Model::GetBucketInventoryConfigurationRequest& request) const;


virtual Model::GetBucketInventoryConfigurationOutcomeCallable GetBucketInventoryConfigurationCallable(const Model::GetBucketInventoryConfigurationRequest& request) const;


virtual void GetBucketInventoryConfigurationAsync(const Model::GetBucketInventoryConfigurationRequest& request, const GetBucketInventoryConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketLifecycleConfigurationOutcome GetBucketLifecycleConfiguration(const Model::GetBucketLifecycleConfigurationRequest& request) const;


virtual Model::GetBucketLifecycleConfigurationOutcomeCallable GetBucketLifecycleConfigurationCallable(const Model::GetBucketLifecycleConfigurationRequest& request) const;


virtual void GetBucketLifecycleConfigurationAsync(const Model::GetBucketLifecycleConfigurationRequest& request, const GetBucketLifecycleConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketLocationOutcome GetBucketLocation(const Model::GetBucketLocationRequest& request) const;


virtual Model::GetBucketLocationOutcomeCallable GetBucketLocationCallable(const Model::GetBucketLocationRequest& request) const;


virtual void GetBucketLocationAsync(const Model::GetBucketLocationRequest& request, const GetBucketLocationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketLoggingOutcome GetBucketLogging(const Model::GetBucketLoggingRequest& request) const;


virtual Model::GetBucketLoggingOutcomeCallable GetBucketLoggingCallable(const Model::GetBucketLoggingRequest& request) const;


virtual void GetBucketLoggingAsync(const Model::GetBucketLoggingRequest& request, const GetBucketLoggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketMetricsConfigurationOutcome GetBucketMetricsConfiguration(const Model::GetBucketMetricsConfigurationRequest& request) const;


virtual Model::GetBucketMetricsConfigurationOutcomeCallable GetBucketMetricsConfigurationCallable(const Model::GetBucketMetricsConfigurationRequest& request) const;


virtual void GetBucketMetricsConfigurationAsync(const Model::GetBucketMetricsConfigurationRequest& request, const GetBucketMetricsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketNotificationConfigurationOutcome GetBucketNotificationConfiguration(const Model::GetBucketNotificationConfigurationRequest& request) const;


virtual Model::GetBucketNotificationConfigurationOutcomeCallable GetBucketNotificationConfigurationCallable(const Model::GetBucketNotificationConfigurationRequest& request) const;


virtual void GetBucketNotificationConfigurationAsync(const Model::GetBucketNotificationConfigurationRequest& request, const GetBucketNotificationConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketPolicyOutcome GetBucketPolicy(const Model::GetBucketPolicyRequest& request) const;


virtual Model::GetBucketPolicyOutcomeCallable GetBucketPolicyCallable(const Model::GetBucketPolicyRequest& request) const;


virtual void GetBucketPolicyAsync(const Model::GetBucketPolicyRequest& request, const GetBucketPolicyResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketReplicationOutcome GetBucketReplication(const Model::GetBucketReplicationRequest& request) const;


virtual Model::GetBucketReplicationOutcomeCallable GetBucketReplicationCallable(const Model::GetBucketReplicationRequest& request) const;


virtual void GetBucketReplicationAsync(const Model::GetBucketReplicationRequest& request, const GetBucketReplicationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketRequestPaymentOutcome GetBucketRequestPayment(const Model::GetBucketRequestPaymentRequest& request) const;


virtual Model::GetBucketRequestPaymentOutcomeCallable GetBucketRequestPaymentCallable(const Model::GetBucketRequestPaymentRequest& request) const;


virtual void GetBucketRequestPaymentAsync(const Model::GetBucketRequestPaymentRequest& request, const GetBucketRequestPaymentResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketTaggingOutcome GetBucketTagging(const Model::GetBucketTaggingRequest& request) const;


virtual Model::GetBucketTaggingOutcomeCallable GetBucketTaggingCallable(const Model::GetBucketTaggingRequest& request) const;


virtual void GetBucketTaggingAsync(const Model::GetBucketTaggingRequest& request, const GetBucketTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketVersioningOutcome GetBucketVersioning(const Model::GetBucketVersioningRequest& request) const;


virtual Model::GetBucketVersioningOutcomeCallable GetBucketVersioningCallable(const Model::GetBucketVersioningRequest& request) const;


virtual void GetBucketVersioningAsync(const Model::GetBucketVersioningRequest& request, const GetBucketVersioningResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetBucketWebsiteOutcome GetBucketWebsite(const Model::GetBucketWebsiteRequest& request) const;


virtual Model::GetBucketWebsiteOutcomeCallable GetBucketWebsiteCallable(const Model::GetBucketWebsiteRequest& request) const;


virtual void GetBucketWebsiteAsync(const Model::GetBucketWebsiteRequest& request, const GetBucketWebsiteResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetObjectOutcome GetObject(const Model::GetObjectRequest& request) const;


virtual Model::GetObjectOutcomeCallable GetObjectCallable(const Model::GetObjectRequest& request) const;


virtual void GetObjectAsync(const Model::GetObjectRequest& request, const GetObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetObjectAclOutcome GetObjectAcl(const Model::GetObjectAclRequest& request) const;


virtual Model::GetObjectAclOutcomeCallable GetObjectAclCallable(const Model::GetObjectAclRequest& request) const;


virtual void GetObjectAclAsync(const Model::GetObjectAclRequest& request, const GetObjectAclResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetObjectTaggingOutcome GetObjectTagging(const Model::GetObjectTaggingRequest& request) const;


virtual Model::GetObjectTaggingOutcomeCallable GetObjectTaggingCallable(const Model::GetObjectTaggingRequest& request) const;


virtual void GetObjectTaggingAsync(const Model::GetObjectTaggingRequest& request, const GetObjectTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetObjectTorrentOutcome GetObjectTorrent(const Model::GetObjectTorrentRequest& request) const;


virtual Model::GetObjectTorrentOutcomeCallable GetObjectTorrentCallable(const Model::GetObjectTorrentRequest& request) const;


virtual void GetObjectTorrentAsync(const Model::GetObjectTorrentRequest& request, const GetObjectTorrentResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::HeadBucketOutcome HeadBucket(const Model::HeadBucketRequest& request) const;


virtual Model::HeadBucketOutcomeCallable HeadBucketCallable(const Model::HeadBucketRequest& request) const;


virtual void HeadBucketAsync(const Model::HeadBucketRequest& request, const HeadBucketResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::HeadObjectOutcome HeadObject(const Model::HeadObjectRequest& request) const;


virtual Model::HeadObjectOutcomeCallable HeadObjectCallable(const Model::HeadObjectRequest& request) const;


virtual void HeadObjectAsync(const Model::HeadObjectRequest& request, const HeadObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::ListBucketAnalyticsConfigurationsOutcome ListBucketAnalyticsConfigurations(const Model::ListBucketAnalyticsConfigurationsRequest& request) const;


virtual Model::ListBucketAnalyticsConfigurationsOutcomeCallable ListBucketAnalyticsConfigurationsCallable(const Model::ListBucketAnalyticsConfigurationsRequest& request) const;


virtual void ListBucketAnalyticsConfigurationsAsync(const Model::ListBucketAnalyticsConfigurationsRequest& request, const ListBucketAnalyticsConfigurationsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::ListBucketInventoryConfigurationsOutcome ListBucketInventoryConfigurations(const Model::ListBucketInventoryConfigurationsRequest& request) const;


virtual Model::ListBucketInventoryConfigurationsOutcomeCallable ListBucketInventoryConfigurationsCallable(const Model::ListBucketInventoryConfigurationsRequest& request) const;


virtual void ListBucketInventoryConfigurationsAsync(const Model::ListBucketInventoryConfigurationsRequest& request, const ListBucketInventoryConfigurationsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::ListBucketMetricsConfigurationsOutcome ListBucketMetricsConfigurations(const Model::ListBucketMetricsConfigurationsRequest& request) const;


virtual Model::ListBucketMetricsConfigurationsOutcomeCallable ListBucketMetricsConfigurationsCallable(const Model::ListBucketMetricsConfigurationsRequest& request) const;


virtual void ListBucketMetricsConfigurationsAsync(const Model::ListBucketMetricsConfigurationsRequest& request, const ListBucketMetricsConfigurationsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::ListBucketsOutcome ListBuckets() const;


virtual Model::ListBucketsOutcomeCallable ListBucketsCallable() const;


virtual void ListBucketsAsync(const ListBucketsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;

virtual Model::ListMultipartUploadsOutcome ListMultipartUploads(const Model::ListMultipartUploadsRequest& request) const;


virtual Model::ListMultipartUploadsOutcomeCallable ListMultipartUploadsCallable(const Model::ListMultipartUploadsRequest& request) const;


virtual void ListMultipartUploadsAsync(const Model::ListMultipartUploadsRequest& request, const ListMultipartUploadsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::ListObjectVersionsOutcome ListObjectVersions(const Model::ListObjectVersionsRequest& request) const;


virtual Model::ListObjectVersionsOutcomeCallable ListObjectVersionsCallable(const Model::ListObjectVersionsRequest& request) const;


virtual void ListObjectVersionsAsync(const Model::ListObjectVersionsRequest& request, const ListObjectVersionsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::ListObjectsOutcome ListObjects(const Model::ListObjectsRequest& request) const;


virtual Model::ListObjectsOutcomeCallable ListObjectsCallable(const Model::ListObjectsRequest& request) const;


virtual void ListObjectsAsync(const Model::ListObjectsRequest& request, const ListObjectsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::ListObjectsV2Outcome ListObjectsV2(const Model::ListObjectsV2Request& request) const;


virtual Model::ListObjectsV2OutcomeCallable ListObjectsV2Callable(const Model::ListObjectsV2Request& request) const;


virtual void ListObjectsV2Async(const Model::ListObjectsV2Request& request, const ListObjectsV2ResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::ListPartsOutcome ListParts(const Model::ListPartsRequest& request) const;


virtual Model::ListPartsOutcomeCallable ListPartsCallable(const Model::ListPartsRequest& request) const;


virtual void ListPartsAsync(const Model::ListPartsRequest& request, const ListPartsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketAccelerateConfigurationOutcome PutBucketAccelerateConfiguration(const Model::PutBucketAccelerateConfigurationRequest& request) const;


virtual Model::PutBucketAccelerateConfigurationOutcomeCallable PutBucketAccelerateConfigurationCallable(const Model::PutBucketAccelerateConfigurationRequest& request) const;


virtual void PutBucketAccelerateConfigurationAsync(const Model::PutBucketAccelerateConfigurationRequest& request, const PutBucketAccelerateConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketAclOutcome PutBucketAcl(const Model::PutBucketAclRequest& request) const;


virtual Model::PutBucketAclOutcomeCallable PutBucketAclCallable(const Model::PutBucketAclRequest& request) const;


virtual void PutBucketAclAsync(const Model::PutBucketAclRequest& request, const PutBucketAclResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketAnalyticsConfigurationOutcome PutBucketAnalyticsConfiguration(const Model::PutBucketAnalyticsConfigurationRequest& request) const;


virtual Model::PutBucketAnalyticsConfigurationOutcomeCallable PutBucketAnalyticsConfigurationCallable(const Model::PutBucketAnalyticsConfigurationRequest& request) const;


virtual void PutBucketAnalyticsConfigurationAsync(const Model::PutBucketAnalyticsConfigurationRequest& request, const PutBucketAnalyticsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketCorsOutcome PutBucketCors(const Model::PutBucketCorsRequest& request) const;


virtual Model::PutBucketCorsOutcomeCallable PutBucketCorsCallable(const Model::PutBucketCorsRequest& request) const;


virtual void PutBucketCorsAsync(const Model::PutBucketCorsRequest& request, const PutBucketCorsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketEncryptionOutcome PutBucketEncryption(const Model::PutBucketEncryptionRequest& request) const;


virtual Model::PutBucketEncryptionOutcomeCallable PutBucketEncryptionCallable(const Model::PutBucketEncryptionRequest& request) const;


virtual void PutBucketEncryptionAsync(const Model::PutBucketEncryptionRequest& request, const PutBucketEncryptionResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketInventoryConfigurationOutcome PutBucketInventoryConfiguration(const Model::PutBucketInventoryConfigurationRequest& request) const;


virtual Model::PutBucketInventoryConfigurationOutcomeCallable PutBucketInventoryConfigurationCallable(const Model::PutBucketInventoryConfigurationRequest& request) const;


virtual void PutBucketInventoryConfigurationAsync(const Model::PutBucketInventoryConfigurationRequest& request, const PutBucketInventoryConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketLifecycleConfigurationOutcome PutBucketLifecycleConfiguration(const Model::PutBucketLifecycleConfigurationRequest& request) const;


virtual Model::PutBucketLifecycleConfigurationOutcomeCallable PutBucketLifecycleConfigurationCallable(const Model::PutBucketLifecycleConfigurationRequest& request) const;


virtual void PutBucketLifecycleConfigurationAsync(const Model::PutBucketLifecycleConfigurationRequest& request, const PutBucketLifecycleConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketLoggingOutcome PutBucketLogging(const Model::PutBucketLoggingRequest& request) const;


virtual Model::PutBucketLoggingOutcomeCallable PutBucketLoggingCallable(const Model::PutBucketLoggingRequest& request) const;


virtual void PutBucketLoggingAsync(const Model::PutBucketLoggingRequest& request, const PutBucketLoggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketMetricsConfigurationOutcome PutBucketMetricsConfiguration(const Model::PutBucketMetricsConfigurationRequest& request) const;


virtual Model::PutBucketMetricsConfigurationOutcomeCallable PutBucketMetricsConfigurationCallable(const Model::PutBucketMetricsConfigurationRequest& request) const;


virtual void PutBucketMetricsConfigurationAsync(const Model::PutBucketMetricsConfigurationRequest& request, const PutBucketMetricsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketNotificationConfigurationOutcome PutBucketNotificationConfiguration(const Model::PutBucketNotificationConfigurationRequest& request) const;


virtual Model::PutBucketNotificationConfigurationOutcomeCallable PutBucketNotificationConfigurationCallable(const Model::PutBucketNotificationConfigurationRequest& request) const;


virtual void PutBucketNotificationConfigurationAsync(const Model::PutBucketNotificationConfigurationRequest& request, const PutBucketNotificationConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketPolicyOutcome PutBucketPolicy(const Model::PutBucketPolicyRequest& request) const;


virtual Model::PutBucketPolicyOutcomeCallable PutBucketPolicyCallable(const Model::PutBucketPolicyRequest& request) const;


virtual void PutBucketPolicyAsync(const Model::PutBucketPolicyRequest& request, const PutBucketPolicyResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketReplicationOutcome PutBucketReplication(const Model::PutBucketReplicationRequest& request) const;


virtual Model::PutBucketReplicationOutcomeCallable PutBucketReplicationCallable(const Model::PutBucketReplicationRequest& request) const;


virtual void PutBucketReplicationAsync(const Model::PutBucketReplicationRequest& request, const PutBucketReplicationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketRequestPaymentOutcome PutBucketRequestPayment(const Model::PutBucketRequestPaymentRequest& request) const;


virtual Model::PutBucketRequestPaymentOutcomeCallable PutBucketRequestPaymentCallable(const Model::PutBucketRequestPaymentRequest& request) const;


virtual void PutBucketRequestPaymentAsync(const Model::PutBucketRequestPaymentRequest& request, const PutBucketRequestPaymentResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketTaggingOutcome PutBucketTagging(const Model::PutBucketTaggingRequest& request) const;


virtual Model::PutBucketTaggingOutcomeCallable PutBucketTaggingCallable(const Model::PutBucketTaggingRequest& request) const;


virtual void PutBucketTaggingAsync(const Model::PutBucketTaggingRequest& request, const PutBucketTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketVersioningOutcome PutBucketVersioning(const Model::PutBucketVersioningRequest& request) const;


virtual Model::PutBucketVersioningOutcomeCallable PutBucketVersioningCallable(const Model::PutBucketVersioningRequest& request) const;


virtual void PutBucketVersioningAsync(const Model::PutBucketVersioningRequest& request, const PutBucketVersioningResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutBucketWebsiteOutcome PutBucketWebsite(const Model::PutBucketWebsiteRequest& request) const;


virtual Model::PutBucketWebsiteOutcomeCallable PutBucketWebsiteCallable(const Model::PutBucketWebsiteRequest& request) const;


virtual void PutBucketWebsiteAsync(const Model::PutBucketWebsiteRequest& request, const PutBucketWebsiteResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutObjectOutcome PutObject(const Model::PutObjectRequest& request) const;


virtual Model::PutObjectOutcomeCallable PutObjectCallable(const Model::PutObjectRequest& request) const;


virtual void PutObjectAsync(const Model::PutObjectRequest& request, const PutObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutObjectAclOutcome PutObjectAcl(const Model::PutObjectAclRequest& request) const;


virtual Model::PutObjectAclOutcomeCallable PutObjectAclCallable(const Model::PutObjectAclRequest& request) const;


virtual void PutObjectAclAsync(const Model::PutObjectAclRequest& request, const PutObjectAclResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutObjectTaggingOutcome PutObjectTagging(const Model::PutObjectTaggingRequest& request) const;


virtual Model::PutObjectTaggingOutcomeCallable PutObjectTaggingCallable(const Model::PutObjectTaggingRequest& request) const;


virtual void PutObjectTaggingAsync(const Model::PutObjectTaggingRequest& request, const PutObjectTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::RestoreObjectOutcome RestoreObject(const Model::RestoreObjectRequest& request) const;


virtual Model::RestoreObjectOutcomeCallable RestoreObjectCallable(const Model::RestoreObjectRequest& request) const;


virtual void RestoreObjectAsync(const Model::RestoreObjectRequest& request, const RestoreObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::UploadPartOutcome UploadPart(const Model::UploadPartRequest& request) const;


virtual Model::UploadPartOutcomeCallable UploadPartCallable(const Model::UploadPartRequest& request) const;


virtual void UploadPartAsync(const Model::UploadPartRequest& request, const UploadPartResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::UploadPartCopyOutcome UploadPartCopy(const Model::UploadPartCopyRequest& request) const;


virtual Model::UploadPartCopyOutcomeCallable UploadPartCopyCallable(const Model::UploadPartCopyRequest& request) const;


virtual void UploadPartCopyAsync(const Model::UploadPartCopyRequest& request, const UploadPartCopyResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


Aws::String GeneratePresignedUrl(const Aws::String& bucketName, const Aws::String& key, Http::HttpMethod method, long long expirationInSeconds = MAX_EXPIRATION_SECONDS);
virtual bool MultipartUploadSupported() const;

private:
void init(const Client::ClientConfiguration& clientConfiguration);
Aws::String ComputeEndpointString(const Aws::String& bucket) const;
Aws::String ComputeEndpointString() const;


void AbortMultipartUploadAsyncHelper(const Model::AbortMultipartUploadRequest& request, const AbortMultipartUploadResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void CompleteMultipartUploadAsyncHelper(const Model::CompleteMultipartUploadRequest& request, const CompleteMultipartUploadResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void CopyObjectAsyncHelper(const Model::CopyObjectRequest& request, const CopyObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void CreateBucketAsyncHelper(const Model::CreateBucketRequest& request, const CreateBucketResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void CreateMultipartUploadAsyncHelper(const Model::CreateMultipartUploadRequest& request, const CreateMultipartUploadResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketAsyncHelper(const Model::DeleteBucketRequest& request, const DeleteBucketResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketAnalyticsConfigurationAsyncHelper(const Model::DeleteBucketAnalyticsConfigurationRequest& request, const DeleteBucketAnalyticsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketCorsAsyncHelper(const Model::DeleteBucketCorsRequest& request, const DeleteBucketCorsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketEncryptionAsyncHelper(const Model::DeleteBucketEncryptionRequest& request, const DeleteBucketEncryptionResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketInventoryConfigurationAsyncHelper(const Model::DeleteBucketInventoryConfigurationRequest& request, const DeleteBucketInventoryConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketLifecycleAsyncHelper(const Model::DeleteBucketLifecycleRequest& request, const DeleteBucketLifecycleResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketMetricsConfigurationAsyncHelper(const Model::DeleteBucketMetricsConfigurationRequest& request, const DeleteBucketMetricsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketPolicyAsyncHelper(const Model::DeleteBucketPolicyRequest& request, const DeleteBucketPolicyResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketReplicationAsyncHelper(const Model::DeleteBucketReplicationRequest& request, const DeleteBucketReplicationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketTaggingAsyncHelper(const Model::DeleteBucketTaggingRequest& request, const DeleteBucketTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteBucketWebsiteAsyncHelper(const Model::DeleteBucketWebsiteRequest& request, const DeleteBucketWebsiteResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteObjectAsyncHelper(const Model::DeleteObjectRequest& request, const DeleteObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteObjectTaggingAsyncHelper(const Model::DeleteObjectTaggingRequest& request, const DeleteObjectTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteObjectsAsyncHelper(const Model::DeleteObjectsRequest& request, const DeleteObjectsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketAccelerateConfigurationAsyncHelper(const Model::GetBucketAccelerateConfigurationRequest& request, const GetBucketAccelerateConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketAclAsyncHelper(const Model::GetBucketAclRequest& request, const GetBucketAclResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketAnalyticsConfigurationAsyncHelper(const Model::GetBucketAnalyticsConfigurationRequest& request, const GetBucketAnalyticsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketCorsAsyncHelper(const Model::GetBucketCorsRequest& request, const GetBucketCorsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketEncryptionAsyncHelper(const Model::GetBucketEncryptionRequest& request, const GetBucketEncryptionResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketInventoryConfigurationAsyncHelper(const Model::GetBucketInventoryConfigurationRequest& request, const GetBucketInventoryConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketLifecycleConfigurationAsyncHelper(const Model::GetBucketLifecycleConfigurationRequest& request, const GetBucketLifecycleConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketLocationAsyncHelper(const Model::GetBucketLocationRequest& request, const GetBucketLocationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketLoggingAsyncHelper(const Model::GetBucketLoggingRequest& request, const GetBucketLoggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketMetricsConfigurationAsyncHelper(const Model::GetBucketMetricsConfigurationRequest& request, const GetBucketMetricsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketNotificationConfigurationAsyncHelper(const Model::GetBucketNotificationConfigurationRequest& request, const GetBucketNotificationConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketPolicyAsyncHelper(const Model::GetBucketPolicyRequest& request, const GetBucketPolicyResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketReplicationAsyncHelper(const Model::GetBucketReplicationRequest& request, const GetBucketReplicationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketRequestPaymentAsyncHelper(const Model::GetBucketRequestPaymentRequest& request, const GetBucketRequestPaymentResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketTaggingAsyncHelper(const Model::GetBucketTaggingRequest& request, const GetBucketTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketVersioningAsyncHelper(const Model::GetBucketVersioningRequest& request, const GetBucketVersioningResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetBucketWebsiteAsyncHelper(const Model::GetBucketWebsiteRequest& request, const GetBucketWebsiteResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetObjectAsyncHelper(const Model::GetObjectRequest& request, const GetObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetObjectAclAsyncHelper(const Model::GetObjectAclRequest& request, const GetObjectAclResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetObjectTaggingAsyncHelper(const Model::GetObjectTaggingRequest& request, const GetObjectTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetObjectTorrentAsyncHelper(const Model::GetObjectTorrentRequest& request, const GetObjectTorrentResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void HeadBucketAsyncHelper(const Model::HeadBucketRequest& request, const HeadBucketResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void HeadObjectAsyncHelper(const Model::HeadObjectRequest& request, const HeadObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListBucketAnalyticsConfigurationsAsyncHelper(const Model::ListBucketAnalyticsConfigurationsRequest& request, const ListBucketAnalyticsConfigurationsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListBucketInventoryConfigurationsAsyncHelper(const Model::ListBucketInventoryConfigurationsRequest& request, const ListBucketInventoryConfigurationsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListBucketMetricsConfigurationsAsyncHelper(const Model::ListBucketMetricsConfigurationsRequest& request, const ListBucketMetricsConfigurationsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListBucketsAsyncHelper(const ListBucketsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListMultipartUploadsAsyncHelper(const Model::ListMultipartUploadsRequest& request, const ListMultipartUploadsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListObjectVersionsAsyncHelper(const Model::ListObjectVersionsRequest& request, const ListObjectVersionsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListObjectsAsyncHelper(const Model::ListObjectsRequest& request, const ListObjectsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListObjectsV2AsyncHelper(const Model::ListObjectsV2Request& request, const ListObjectsV2ResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListPartsAsyncHelper(const Model::ListPartsRequest& request, const ListPartsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketAccelerateConfigurationAsyncHelper(const Model::PutBucketAccelerateConfigurationRequest& request, const PutBucketAccelerateConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketAclAsyncHelper(const Model::PutBucketAclRequest& request, const PutBucketAclResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketAnalyticsConfigurationAsyncHelper(const Model::PutBucketAnalyticsConfigurationRequest& request, const PutBucketAnalyticsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketCorsAsyncHelper(const Model::PutBucketCorsRequest& request, const PutBucketCorsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketEncryptionAsyncHelper(const Model::PutBucketEncryptionRequest& request, const PutBucketEncryptionResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketInventoryConfigurationAsyncHelper(const Model::PutBucketInventoryConfigurationRequest& request, const PutBucketInventoryConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketLifecycleConfigurationAsyncHelper(const Model::PutBucketLifecycleConfigurationRequest& request, const PutBucketLifecycleConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketLoggingAsyncHelper(const Model::PutBucketLoggingRequest& request, const PutBucketLoggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketMetricsConfigurationAsyncHelper(const Model::PutBucketMetricsConfigurationRequest& request, const PutBucketMetricsConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketNotificationConfigurationAsyncHelper(const Model::PutBucketNotificationConfigurationRequest& request, const PutBucketNotificationConfigurationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketPolicyAsyncHelper(const Model::PutBucketPolicyRequest& request, const PutBucketPolicyResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketReplicationAsyncHelper(const Model::PutBucketReplicationRequest& request, const PutBucketReplicationResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketRequestPaymentAsyncHelper(const Model::PutBucketRequestPaymentRequest& request, const PutBucketRequestPaymentResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketTaggingAsyncHelper(const Model::PutBucketTaggingRequest& request, const PutBucketTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketVersioningAsyncHelper(const Model::PutBucketVersioningRequest& request, const PutBucketVersioningResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutBucketWebsiteAsyncHelper(const Model::PutBucketWebsiteRequest& request, const PutBucketWebsiteResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutObjectAsyncHelper(const Model::PutObjectRequest& request, const PutObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutObjectAclAsyncHelper(const Model::PutObjectAclRequest& request, const PutObjectAclResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutObjectTaggingAsyncHelper(const Model::PutObjectTaggingRequest& request, const PutObjectTaggingResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void RestoreObjectAsyncHelper(const Model::RestoreObjectRequest& request, const RestoreObjectResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void UploadPartAsyncHelper(const Model::UploadPartRequest& request, const UploadPartResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void UploadPartCopyAsyncHelper(const Model::UploadPartCopyRequest& request, const UploadPartCopyResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;

Aws::String m_baseUri;
Aws::String m_scheme;
std::shared_ptr<Utils::Threading::Executor> m_executor;
bool m_useVirtualAdressing;
};

} 
} 
