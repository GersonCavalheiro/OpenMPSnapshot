

#pragma once
#include <aws/kinesis/Kinesis_EXPORTS.h>
#include <aws/kinesis/KinesisErrors.h>
#include <aws/core/client/AWSError.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/client/AWSClient.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/kinesis/model/DescribeLimitsResult.h>
#include <aws/kinesis/model/DescribeStreamResult.h>
#include <aws/kinesis/model/DescribeStreamSummaryResult.h>
#include <aws/kinesis/model/DisableEnhancedMonitoringResult.h>
#include <aws/kinesis/model/EnableEnhancedMonitoringResult.h>
#include <aws/kinesis/model/GetRecordsResult.h>
#include <aws/kinesis/model/GetShardIteratorResult.h>
#include <aws/kinesis/model/ListStreamsResult.h>
#include <aws/kinesis/model/ListTagsForStreamResult.h>
#include <aws/kinesis/model/PutRecordResult.h>
#include <aws/kinesis/model/PutRecordsResult.h>
#include <aws/kinesis/model/UpdateShardCountResult.h>
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

namespace Json
{
class JsonValue;
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

namespace Kinesis
{

namespace Model
{
class AddTagsToStreamRequest;
class CreateStreamRequest;
class DecreaseStreamRetentionPeriodRequest;
class DeleteStreamRequest;
class DescribeLimitsRequest;
class DescribeStreamRequest;
class DescribeStreamSummaryRequest;
class DisableEnhancedMonitoringRequest;
class EnableEnhancedMonitoringRequest;
class GetRecordsRequest;
class GetShardIteratorRequest;
class IncreaseStreamRetentionPeriodRequest;
class ListStreamsRequest;
class ListTagsForStreamRequest;
class MergeShardsRequest;
class PutRecordRequest;
class PutRecordsRequest;
class RemoveTagsFromStreamRequest;
class SplitShardRequest;
class StartStreamEncryptionRequest;
class StopStreamEncryptionRequest;
class UpdateShardCountRequest;

typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<KinesisErrors>> AddTagsToStreamOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<KinesisErrors>> CreateStreamOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<KinesisErrors>> DecreaseStreamRetentionPeriodOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<KinesisErrors>> DeleteStreamOutcome;
typedef Aws::Utils::Outcome<DescribeLimitsResult, Aws::Client::AWSError<KinesisErrors>> DescribeLimitsOutcome;
typedef Aws::Utils::Outcome<DescribeStreamResult, Aws::Client::AWSError<KinesisErrors>> DescribeStreamOutcome;
typedef Aws::Utils::Outcome<DescribeStreamSummaryResult, Aws::Client::AWSError<KinesisErrors>> DescribeStreamSummaryOutcome;
typedef Aws::Utils::Outcome<DisableEnhancedMonitoringResult, Aws::Client::AWSError<KinesisErrors>> DisableEnhancedMonitoringOutcome;
typedef Aws::Utils::Outcome<EnableEnhancedMonitoringResult, Aws::Client::AWSError<KinesisErrors>> EnableEnhancedMonitoringOutcome;
typedef Aws::Utils::Outcome<GetRecordsResult, Aws::Client::AWSError<KinesisErrors>> GetRecordsOutcome;
typedef Aws::Utils::Outcome<GetShardIteratorResult, Aws::Client::AWSError<KinesisErrors>> GetShardIteratorOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<KinesisErrors>> IncreaseStreamRetentionPeriodOutcome;
typedef Aws::Utils::Outcome<ListStreamsResult, Aws::Client::AWSError<KinesisErrors>> ListStreamsOutcome;
typedef Aws::Utils::Outcome<ListTagsForStreamResult, Aws::Client::AWSError<KinesisErrors>> ListTagsForStreamOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<KinesisErrors>> MergeShardsOutcome;
typedef Aws::Utils::Outcome<PutRecordResult, Aws::Client::AWSError<KinesisErrors>> PutRecordOutcome;
typedef Aws::Utils::Outcome<PutRecordsResult, Aws::Client::AWSError<KinesisErrors>> PutRecordsOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<KinesisErrors>> RemoveTagsFromStreamOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<KinesisErrors>> SplitShardOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<KinesisErrors>> StartStreamEncryptionOutcome;
typedef Aws::Utils::Outcome<Aws::NoResult, Aws::Client::AWSError<KinesisErrors>> StopStreamEncryptionOutcome;
typedef Aws::Utils::Outcome<UpdateShardCountResult, Aws::Client::AWSError<KinesisErrors>> UpdateShardCountOutcome;

typedef std::future<AddTagsToStreamOutcome> AddTagsToStreamOutcomeCallable;
typedef std::future<CreateStreamOutcome> CreateStreamOutcomeCallable;
typedef std::future<DecreaseStreamRetentionPeriodOutcome> DecreaseStreamRetentionPeriodOutcomeCallable;
typedef std::future<DeleteStreamOutcome> DeleteStreamOutcomeCallable;
typedef std::future<DescribeLimitsOutcome> DescribeLimitsOutcomeCallable;
typedef std::future<DescribeStreamOutcome> DescribeStreamOutcomeCallable;
typedef std::future<DescribeStreamSummaryOutcome> DescribeStreamSummaryOutcomeCallable;
typedef std::future<DisableEnhancedMonitoringOutcome> DisableEnhancedMonitoringOutcomeCallable;
typedef std::future<EnableEnhancedMonitoringOutcome> EnableEnhancedMonitoringOutcomeCallable;
typedef std::future<GetRecordsOutcome> GetRecordsOutcomeCallable;
typedef std::future<GetShardIteratorOutcome> GetShardIteratorOutcomeCallable;
typedef std::future<IncreaseStreamRetentionPeriodOutcome> IncreaseStreamRetentionPeriodOutcomeCallable;
typedef std::future<ListStreamsOutcome> ListStreamsOutcomeCallable;
typedef std::future<ListTagsForStreamOutcome> ListTagsForStreamOutcomeCallable;
typedef std::future<MergeShardsOutcome> MergeShardsOutcomeCallable;
typedef std::future<PutRecordOutcome> PutRecordOutcomeCallable;
typedef std::future<PutRecordsOutcome> PutRecordsOutcomeCallable;
typedef std::future<RemoveTagsFromStreamOutcome> RemoveTagsFromStreamOutcomeCallable;
typedef std::future<SplitShardOutcome> SplitShardOutcomeCallable;
typedef std::future<StartStreamEncryptionOutcome> StartStreamEncryptionOutcomeCallable;
typedef std::future<StopStreamEncryptionOutcome> StopStreamEncryptionOutcomeCallable;
typedef std::future<UpdateShardCountOutcome> UpdateShardCountOutcomeCallable;
} 

class KinesisClient;

typedef std::function<void(const KinesisClient*, const Model::AddTagsToStreamRequest&, const Model::AddTagsToStreamOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > AddTagsToStreamResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::CreateStreamRequest&, const Model::CreateStreamOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > CreateStreamResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::DecreaseStreamRetentionPeriodRequest&, const Model::DecreaseStreamRetentionPeriodOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DecreaseStreamRetentionPeriodResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::DeleteStreamRequest&, const Model::DeleteStreamOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DeleteStreamResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::DescribeLimitsRequest&, const Model::DescribeLimitsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DescribeLimitsResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::DescribeStreamRequest&, const Model::DescribeStreamOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DescribeStreamResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::DescribeStreamSummaryRequest&, const Model::DescribeStreamSummaryOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DescribeStreamSummaryResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::DisableEnhancedMonitoringRequest&, const Model::DisableEnhancedMonitoringOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > DisableEnhancedMonitoringResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::EnableEnhancedMonitoringRequest&, const Model::EnableEnhancedMonitoringOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > EnableEnhancedMonitoringResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::GetRecordsRequest&, const Model::GetRecordsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetRecordsResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::GetShardIteratorRequest&, const Model::GetShardIteratorOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > GetShardIteratorResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::IncreaseStreamRetentionPeriodRequest&, const Model::IncreaseStreamRetentionPeriodOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > IncreaseStreamRetentionPeriodResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::ListStreamsRequest&, const Model::ListStreamsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListStreamsResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::ListTagsForStreamRequest&, const Model::ListTagsForStreamOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > ListTagsForStreamResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::MergeShardsRequest&, const Model::MergeShardsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > MergeShardsResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::PutRecordRequest&, const Model::PutRecordOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutRecordResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::PutRecordsRequest&, const Model::PutRecordsOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > PutRecordsResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::RemoveTagsFromStreamRequest&, const Model::RemoveTagsFromStreamOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > RemoveTagsFromStreamResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::SplitShardRequest&, const Model::SplitShardOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > SplitShardResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::StartStreamEncryptionRequest&, const Model::StartStreamEncryptionOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > StartStreamEncryptionResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::StopStreamEncryptionRequest&, const Model::StopStreamEncryptionOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > StopStreamEncryptionResponseReceivedHandler;
typedef std::function<void(const KinesisClient*, const Model::UpdateShardCountRequest&, const Model::UpdateShardCountOutcome&, const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) > UpdateShardCountResponseReceivedHandler;


class AWS_KINESIS_API KinesisClient : public Aws::Client::AWSJsonClient
{
public:
typedef Aws::Client::AWSJsonClient BASECLASS;


KinesisClient(const Aws::Client::ClientConfiguration& clientConfiguration = Aws::Client::ClientConfiguration());


KinesisClient(const Aws::Auth::AWSCredentials& credentials, const Aws::Client::ClientConfiguration& clientConfiguration = Aws::Client::ClientConfiguration());


KinesisClient(const std::shared_ptr<Aws::Auth::AWSCredentialsProvider>& credentialsProvider,
const Aws::Client::ClientConfiguration& clientConfiguration = Aws::Client::ClientConfiguration());

virtual ~KinesisClient();

inline virtual const char* GetServiceClientName() const override { return "kinesis"; }



virtual Model::AddTagsToStreamOutcome AddTagsToStream(const Model::AddTagsToStreamRequest& request) const;


virtual Model::AddTagsToStreamOutcomeCallable AddTagsToStreamCallable(const Model::AddTagsToStreamRequest& request) const;


virtual void AddTagsToStreamAsync(const Model::AddTagsToStreamRequest& request, const AddTagsToStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::CreateStreamOutcome CreateStream(const Model::CreateStreamRequest& request) const;


virtual Model::CreateStreamOutcomeCallable CreateStreamCallable(const Model::CreateStreamRequest& request) const;


virtual void CreateStreamAsync(const Model::CreateStreamRequest& request, const CreateStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DecreaseStreamRetentionPeriodOutcome DecreaseStreamRetentionPeriod(const Model::DecreaseStreamRetentionPeriodRequest& request) const;


virtual Model::DecreaseStreamRetentionPeriodOutcomeCallable DecreaseStreamRetentionPeriodCallable(const Model::DecreaseStreamRetentionPeriodRequest& request) const;


virtual void DecreaseStreamRetentionPeriodAsync(const Model::DecreaseStreamRetentionPeriodRequest& request, const DecreaseStreamRetentionPeriodResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DeleteStreamOutcome DeleteStream(const Model::DeleteStreamRequest& request) const;


virtual Model::DeleteStreamOutcomeCallable DeleteStreamCallable(const Model::DeleteStreamRequest& request) const;


virtual void DeleteStreamAsync(const Model::DeleteStreamRequest& request, const DeleteStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DescribeLimitsOutcome DescribeLimits(const Model::DescribeLimitsRequest& request) const;


virtual Model::DescribeLimitsOutcomeCallable DescribeLimitsCallable(const Model::DescribeLimitsRequest& request) const;


virtual void DescribeLimitsAsync(const Model::DescribeLimitsRequest& request, const DescribeLimitsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DescribeStreamOutcome DescribeStream(const Model::DescribeStreamRequest& request) const;


virtual Model::DescribeStreamOutcomeCallable DescribeStreamCallable(const Model::DescribeStreamRequest& request) const;


virtual void DescribeStreamAsync(const Model::DescribeStreamRequest& request, const DescribeStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DescribeStreamSummaryOutcome DescribeStreamSummary(const Model::DescribeStreamSummaryRequest& request) const;


virtual Model::DescribeStreamSummaryOutcomeCallable DescribeStreamSummaryCallable(const Model::DescribeStreamSummaryRequest& request) const;


virtual void DescribeStreamSummaryAsync(const Model::DescribeStreamSummaryRequest& request, const DescribeStreamSummaryResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::DisableEnhancedMonitoringOutcome DisableEnhancedMonitoring(const Model::DisableEnhancedMonitoringRequest& request) const;


virtual Model::DisableEnhancedMonitoringOutcomeCallable DisableEnhancedMonitoringCallable(const Model::DisableEnhancedMonitoringRequest& request) const;


virtual void DisableEnhancedMonitoringAsync(const Model::DisableEnhancedMonitoringRequest& request, const DisableEnhancedMonitoringResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::EnableEnhancedMonitoringOutcome EnableEnhancedMonitoring(const Model::EnableEnhancedMonitoringRequest& request) const;


virtual Model::EnableEnhancedMonitoringOutcomeCallable EnableEnhancedMonitoringCallable(const Model::EnableEnhancedMonitoringRequest& request) const;


virtual void EnableEnhancedMonitoringAsync(const Model::EnableEnhancedMonitoringRequest& request, const EnableEnhancedMonitoringResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetRecordsOutcome GetRecords(const Model::GetRecordsRequest& request) const;


virtual Model::GetRecordsOutcomeCallable GetRecordsCallable(const Model::GetRecordsRequest& request) const;


virtual void GetRecordsAsync(const Model::GetRecordsRequest& request, const GetRecordsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::GetShardIteratorOutcome GetShardIterator(const Model::GetShardIteratorRequest& request) const;


virtual Model::GetShardIteratorOutcomeCallable GetShardIteratorCallable(const Model::GetShardIteratorRequest& request) const;


virtual void GetShardIteratorAsync(const Model::GetShardIteratorRequest& request, const GetShardIteratorResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::IncreaseStreamRetentionPeriodOutcome IncreaseStreamRetentionPeriod(const Model::IncreaseStreamRetentionPeriodRequest& request) const;


virtual Model::IncreaseStreamRetentionPeriodOutcomeCallable IncreaseStreamRetentionPeriodCallable(const Model::IncreaseStreamRetentionPeriodRequest& request) const;


virtual void IncreaseStreamRetentionPeriodAsync(const Model::IncreaseStreamRetentionPeriodRequest& request, const IncreaseStreamRetentionPeriodResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::ListStreamsOutcome ListStreams(const Model::ListStreamsRequest& request) const;


virtual Model::ListStreamsOutcomeCallable ListStreamsCallable(const Model::ListStreamsRequest& request) const;


virtual void ListStreamsAsync(const Model::ListStreamsRequest& request, const ListStreamsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::ListTagsForStreamOutcome ListTagsForStream(const Model::ListTagsForStreamRequest& request) const;


virtual Model::ListTagsForStreamOutcomeCallable ListTagsForStreamCallable(const Model::ListTagsForStreamRequest& request) const;


virtual void ListTagsForStreamAsync(const Model::ListTagsForStreamRequest& request, const ListTagsForStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::MergeShardsOutcome MergeShards(const Model::MergeShardsRequest& request) const;


virtual Model::MergeShardsOutcomeCallable MergeShardsCallable(const Model::MergeShardsRequest& request) const;


virtual void MergeShardsAsync(const Model::MergeShardsRequest& request, const MergeShardsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutRecordOutcome PutRecord(const Model::PutRecordRequest& request) const;


virtual Model::PutRecordOutcomeCallable PutRecordCallable(const Model::PutRecordRequest& request) const;


virtual void PutRecordAsync(const Model::PutRecordRequest& request, const PutRecordResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::PutRecordsOutcome PutRecords(const Model::PutRecordsRequest& request) const;


virtual Model::PutRecordsOutcomeCallable PutRecordsCallable(const Model::PutRecordsRequest& request) const;


virtual void PutRecordsAsync(const Model::PutRecordsRequest& request, const PutRecordsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::RemoveTagsFromStreamOutcome RemoveTagsFromStream(const Model::RemoveTagsFromStreamRequest& request) const;


virtual Model::RemoveTagsFromStreamOutcomeCallable RemoveTagsFromStreamCallable(const Model::RemoveTagsFromStreamRequest& request) const;


virtual void RemoveTagsFromStreamAsync(const Model::RemoveTagsFromStreamRequest& request, const RemoveTagsFromStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::SplitShardOutcome SplitShard(const Model::SplitShardRequest& request) const;


virtual Model::SplitShardOutcomeCallable SplitShardCallable(const Model::SplitShardRequest& request) const;


virtual void SplitShardAsync(const Model::SplitShardRequest& request, const SplitShardResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::StartStreamEncryptionOutcome StartStreamEncryption(const Model::StartStreamEncryptionRequest& request) const;


virtual Model::StartStreamEncryptionOutcomeCallable StartStreamEncryptionCallable(const Model::StartStreamEncryptionRequest& request) const;


virtual void StartStreamEncryptionAsync(const Model::StartStreamEncryptionRequest& request, const StartStreamEncryptionResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::StopStreamEncryptionOutcome StopStreamEncryption(const Model::StopStreamEncryptionRequest& request) const;


virtual Model::StopStreamEncryptionOutcomeCallable StopStreamEncryptionCallable(const Model::StopStreamEncryptionRequest& request) const;


virtual void StopStreamEncryptionAsync(const Model::StopStreamEncryptionRequest& request, const StopStreamEncryptionResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


virtual Model::UpdateShardCountOutcome UpdateShardCount(const Model::UpdateShardCountRequest& request) const;


virtual Model::UpdateShardCountOutcomeCallable UpdateShardCountCallable(const Model::UpdateShardCountRequest& request) const;


virtual void UpdateShardCountAsync(const Model::UpdateShardCountRequest& request, const UpdateShardCountResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context = nullptr) const;


private:
void init(const Aws::Client::ClientConfiguration& clientConfiguration);


void AddTagsToStreamAsyncHelper(const Model::AddTagsToStreamRequest& request, const AddTagsToStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void CreateStreamAsyncHelper(const Model::CreateStreamRequest& request, const CreateStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DecreaseStreamRetentionPeriodAsyncHelper(const Model::DecreaseStreamRetentionPeriodRequest& request, const DecreaseStreamRetentionPeriodResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DeleteStreamAsyncHelper(const Model::DeleteStreamRequest& request, const DeleteStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DescribeLimitsAsyncHelper(const Model::DescribeLimitsRequest& request, const DescribeLimitsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DescribeStreamAsyncHelper(const Model::DescribeStreamRequest& request, const DescribeStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DescribeStreamSummaryAsyncHelper(const Model::DescribeStreamSummaryRequest& request, const DescribeStreamSummaryResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void DisableEnhancedMonitoringAsyncHelper(const Model::DisableEnhancedMonitoringRequest& request, const DisableEnhancedMonitoringResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void EnableEnhancedMonitoringAsyncHelper(const Model::EnableEnhancedMonitoringRequest& request, const EnableEnhancedMonitoringResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetRecordsAsyncHelper(const Model::GetRecordsRequest& request, const GetRecordsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void GetShardIteratorAsyncHelper(const Model::GetShardIteratorRequest& request, const GetShardIteratorResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void IncreaseStreamRetentionPeriodAsyncHelper(const Model::IncreaseStreamRetentionPeriodRequest& request, const IncreaseStreamRetentionPeriodResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListStreamsAsyncHelper(const Model::ListStreamsRequest& request, const ListStreamsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void ListTagsForStreamAsyncHelper(const Model::ListTagsForStreamRequest& request, const ListTagsForStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void MergeShardsAsyncHelper(const Model::MergeShardsRequest& request, const MergeShardsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutRecordAsyncHelper(const Model::PutRecordRequest& request, const PutRecordResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void PutRecordsAsyncHelper(const Model::PutRecordsRequest& request, const PutRecordsResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void RemoveTagsFromStreamAsyncHelper(const Model::RemoveTagsFromStreamRequest& request, const RemoveTagsFromStreamResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void SplitShardAsyncHelper(const Model::SplitShardRequest& request, const SplitShardResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void StartStreamEncryptionAsyncHelper(const Model::StartStreamEncryptionRequest& request, const StartStreamEncryptionResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void StopStreamEncryptionAsyncHelper(const Model::StopStreamEncryptionRequest& request, const StopStreamEncryptionResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;
void UpdateShardCountAsyncHelper(const Model::UpdateShardCountRequest& request, const UpdateShardCountResponseReceivedHandler& handler, const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) const;

Aws::String m_uri;
std::shared_ptr<Aws::Utils::Threading::Executor> m_executor;
};

} 
} 
