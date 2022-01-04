#include "../../header/CudaClass.h"



__global__ 
void match_kernel(const uint32_t* const __restrict__ query_descriptors,
	const uint32_t* const __restrict__ train_descriptors,
	uint16_t* const __restrict__ best_idxs,
	const uint16_t num_query_descriptors,
	const uint16_t num_train_descriptors) 
{
	// blockdim.x should be 32 to occupy the full warp
	// griddim.x should be num_query_descriptors/32 + 1
	const auto query_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (query_idx >= num_query_descriptors) {
		return;
	}
	uint16_t best_idx = 0;
	uint8_t best_distance = 255;

	uint32_t localArr[8] = {
		query_descriptors[query_idx * 8 + 0],
		query_descriptors[query_idx * 8 + 1],
		query_descriptors[query_idx * 8 + 2],
		query_descriptors[query_idx * 8 + 3],
		query_descriptors[query_idx * 8 + 4],
		query_descriptors[query_idx * 8 + 5],
		query_descriptors[query_idx * 8 + 6],
		query_descriptors[query_idx * 8 + 7],
	};

	uint8_t distance = 0;
	for (uint16_t train_idx = 0; train_idx < num_train_descriptors; train_idx++) {

		#pragma unroll
		for (uint8_t i = 0; i < 8; i++) {
			distance +=
				__popc(static_cast<int>(train_descriptors[train_idx * 8 + i] ^ localArr[i]));
				//__popc(static_cast<int>(train_descriptors[train_idx * 8 + i] ^ query_descriptors[query_idx * 8 + i]));
		}
		if (distance < best_distance) {
			best_distance = distance;
			best_idx = train_idx;
		}
	}
	best_idxs[query_idx] = best_idx;
}

__global__
void multi_match_kernel(const uint32_t* const __restrict__ query_descriptors,
	const uint32_t* const __restrict__* const __restrict__ train_descriptors,
	uint16_t* const __restrict__* const __restrict__ best_idxs,
	const uint16_t num_query_descriptors,
	const uint16_t* const __restrict__ num_train_descriptors) 
{
	// blockdim.x should be 32 to occupy the full warp
	// griddim.x should be num_query_descriptors/32 + 1
	const auto query_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (query_idx >= num_query_descriptors) {
		return;
	}

	uint64_t frame = blockIdx.y;

	uint8_t best_distance = 255;
	uint16_t best_idx = 0;

	uint32_t localArr[8] = {
		query_descriptors[query_idx * 8 + 0],
		query_descriptors[query_idx * 8 + 1],
		query_descriptors[query_idx * 8 + 2],
		query_descriptors[query_idx * 8 + 3],
		query_descriptors[query_idx * 8 + 4],
		query_descriptors[query_idx * 8 + 5],
		query_descriptors[query_idx * 8 + 6],
		query_descriptors[query_idx * 8 + 7],
	};

	for (uint16_t train_idx = 0; train_idx < num_train_descriptors[frame]; train_idx++) 
	{
#pragma unroll
		uint8_t distance = 0;
		for (uint8_t i = 0; i < 8; i++) {
			
			distance += __popc(static_cast<int>(train_descriptors[frame][train_idx * 8 + i] ^ localArr[i]));
				//static_cast<int>(train_descriptors[frame][train_idx * 8 + i] ^ query_descriptors[query_idx * 8 + i]));
		}
		if (distance < best_distance) {
			best_distance = distance;
			best_idx = train_idx;
		}
	}
	best_idxs[frame][query_idx] = best_idx;
}

__host__ void CudaKeypoints::match_gpu_caller(const cudaStream_t &providedStream, int queryCount, int trainCount)
{
	
	match_kernel << <(queryCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, providedStream >> > (d_query, d_train, d_resMatcher,
		queryCount, trainCount);
}



//constexpr uint64_t NUM_KPTS = 8192;// limit of pva implementation of harris corner detection
//using keypoints_t = std::vector<float2>;
//using descriptors_t = std::vector<descriptor_t>;


//void match_gpu(const cudaStream_t& stream, const descriptors_t& query_descriptors,
//	const descriptors_t& train_descriptors, ManagedArray<NUM_KPTS, uint16_t>& best_idxs) {
//	if (query_descriptors.empty() || train_descriptors.empty()) {
//		best_idxs.resize(0);
//		// should not launch cuda kernels with 0 blocks
//		return;
//	}
//	query_descriptors.to_gpu_async(stream);
//	train_descriptors.to_gpu_async(stream);
//	best_idxs.to_gpu_async(stream);
//	match_gpu_caller(stream, reinterpret_cast<const uint32_t*>(query_descriptors.data()),
//		reinterpret_cast<const uint32_t*>(train_descriptors.data()), best_idxs.data(),
//		static_cast<uint16_t>(query_descriptors.length()),
//		static_cast<uint16_t>(train_descriptors.length()));
//	query_descriptors.to_cpu_async(stream);
//	train_descriptors.to_cpu_async(stream);
//	best_idxs.to_cpu_async(stream);
//
//	best_idxs.resize(query_descriptors.length());
//}