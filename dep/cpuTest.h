#include "Dep.h"

//std::vector<Point2f> kpts2vec(const keypoints_t& kpts) // this simply makes keypts to vector	
//{																		//of the specif type
//	std::vector<Point2f> result;
//	result.reserve(kpts.length());
//	for (uint64_t i = 0; i < kpts.length(); ++i) {
//		result.emplace_back(kpts[i].x, kpts[i].y);
//	}
//	return result;
//}
//
//void vec2kpts(const std::vector<cv::Point2f>& vec, keypoints_t& kpts) {
//	kpts.resize(vec.size());
//	for (uint64_t i = 0; i < kpts.length(); ++i) {
//		kpts[i].x = vec[i].x;
//		kpts[i].y = vec[i].y;
//	}
//}
//
//void UndistortPoints(const keypoints_t& points, keypoints_t& points_undistorted, const cv::Matx33d& K,
//	const cv::Matx14d& distortion) {
//	NVPROF_SCOPE("undistort");
//	points_undistorted.resize(points.length());
//	if (points.empty()) {
//		return;
//	}
//	std::vector<cv::Point2f> points_v = kpts2vec(points), points_undistorted_v;
//	cv::undistortPoints(points_v, points_undistorted_v, K, distortion, cv::noArray(), K); // TODO: serach what is this func doing
//	vec2kpts(points_undistorted_v, points_undistorted);
//}
//
//
//
//void NewFunc()
//{
//
//	{
//		NVPROF_SCOPE("undistort");
//		UndistortPoints(m_keypoints.l_keypoints, m_keypoints.l_points_undistorted, m_cameras.intrinsic_l,
//			m_cameras.distortion_l);
//		UndistortPoints(m_keypoints.r_keypoints2, m_keypoints.r_points_undistorted, m_cameras.intrinsic_r,
//			m_cameras.distortion_r);
//	}
//
//	EpiFilterPoints(m_keypoints.l_points_undistorted, m_keypoints.r_points_undistorted, m_cameras.fundamental_matrix,
//		m_config.fundamental_distance_threshold, m_indices.epi_constraint_indices);
//
//	CorrectMatches(m_keypoints.l_points_undistorted, m_keypoints.r_points_undistorted, m_keypoints.l_points_corrected,
//		m_cameras.fundamental_matrix.t(), m_indices.correct_match_indices);
//
//	triangulate(m_cameras.rotation, m_cameras.translation, m_cameras.intrinsic_l, m_cameras.intrinsic_r,
//		m_keypoints.l_points_corrected, m_keypoints.r_points_undistorted, m_points3d.points3d);
//
//	Filter3dPointsByZ(m_points3d.points3d, m_keypoints.l_points_corrected, m_keypoints.r_points_undistorted,
//		m_descriptors.l_descriptors, m_descriptors.r_descriptors, m_indices.mask, m_config.vml.z_close,
//		m_config.vml.z_far);
//}



//EpiFilter

//Filter by index
template <uint64_t C2, typename I> void filter_by_index(const ManagedArray<C2, I>& indices) {
	static_assert(is_valid_index<C, I>());

	if (this->length() < indices.length()) {
		throw std::runtime_error("Error, index array is longer than filtered array!!");
	}

	for (uint64_t i = 0; i < indices.length(); ++i) {
		(*this)[i] = (*this)[indices[i]];
	}

	this->resize(indices.length());
}
void vec2idx(const std::vector<size_t>& vec, ManagedArray<NUM_KPTS, uint64_t>& idxs) {
	idxs.resize(vec.size());
	for (uint64_t i = 0; i < idxs.length(); ++i) {
		idxs[i] = vec[i];
	}
}
void VisionPipeline::EpiFilterPoints(const keypoints_t& l_points, const keypoints_t& r_points, const cv::Matx33d& F,
	double thresh, ManagedArray<NUM_KPTS, uint64_t>& idxs) {
	nvtx_scope s("epi-filter");
	std::vector<cv::Point2f> l_points_v = kpts2vec(l_points), r_points_v = kpts2vec(r_points);
	std::vector<size_t> inliner_indices;
	cv::Point2f p1, p2;
	cv::Matx13d p1_m;
	cv::Matx31d p2_m;
	for (size_t i = 0; i < l_points_v.size(); i++) {
		p1 = l_points_v[i], p2 = r_points_v[i];
		// transform to homogeneous coordinate.
		p1_m = cv::Matx13d{ p1.x, p1.y, 1. };
		p2_m = cv::Matx31d{ p2.x, p2.y, 1. };
		// calculate the epi-polar constraint error : p1^T * F * p2
		const double score = ((p1_m * F) * p2_m)(0, 0);
		if (fabs(score) < thresh) {
			inliner_indices.push_back(i);
		}
	}
	vec2idx(inliner_indices, idxs);
	m_descriptors.l_descriptors.filter_by_index(idxs);
	m_descriptors.r_descriptors.filter_by_index(idxs);
	m_keypoints.l_points_undistorted.filter_by_index(idxs);
	m_keypoints.r_points_undistorted.filter_by_index(idxs);
	idxs.resize(0);
}