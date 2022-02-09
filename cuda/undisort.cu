#include "../header/CudaClass.h"
#include "../dep/Dep.h"

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)

//std::vector<Point2f> kpts2vec(const keypoints_t & kpts) // this simply makes keypts to vector	
//{																		//of the specif type
//	std::vector<Point2f> result;
//	result.reserve(kpts.length());
//	for (uint64_t i = 0; i < kpts.length(); ++i) {
//		result.emplace_back(kpts[i].x, kpts[i].y);
//	}
//	return result;
//}

//void vec2kpts(const std::vector<Point2f>& vec, keypoints_t& kpts) {
//	kpts.resize(vec.size());
//	for (uint64_t i = 0; i < kpts.length(); ++i) {
//		kpts[i].x = vec[i].x;
//		kpts[i].y = vec[i].y;
//	}
//}


//void vec2idx(const std::vector<size_t>& vec, ManagedArray<NUM_KPTS, uint64_t>& idxs) {
//	idxs.resize(vec.size());
//	for (uint64_t i = 0; i < idxs.length(); ++i) {
//		idxs[i] = vec[i];
//	}
//}
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
void EpiFilterPoints(const keypoints_t& l_points, const keypoints_t& r_points, const cv::Matx33d& F,
	double thresh, ManagedArray<NUM_KPTS, uint64_t>& idxs) {
	nvtx_scope s("epi-filter");
	std::vector<cv::Point2f> l_points_v = kpts2vec(l_points), r_points_v = kpts2vec(r_points);
	std::vector<size_t> inliner_indices;
	Point2f p1, p2;
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



void NewFunc()
{

	{
		NVPROF_SCOPE("undistort");
		UndistortPoints(m_keypoints.l_keypoints, m_keypoints.l_points_undistorted, m_cameras.intrinsic_l,
			m_cameras.distortion_l);
		UndistortPoints(m_keypoints.r_keypoints2, m_keypoints.r_points_undistorted, m_cameras.intrinsic_r,
			m_cameras.distortion_r);
	}

	EpiFilterPoints(m_keypoints.l_points_undistorted, m_keypoints.r_points_undistorted, m_cameras.fundamental_matrix,
		m_config.fundamental_distance_threshold, m_indices.epi_constraint_indices);

	CorrectMatches(m_keypoints.l_points_undistorted, m_keypoints.r_points_undistorted, m_keypoints.l_points_corrected,
		m_cameras.fundamental_matrix.t(), m_indices.correct_match_indices);

	triangulate(m_cameras.rotation, m_cameras.translation, m_cameras.intrinsic_l, m_cameras.intrinsic_r,
		m_keypoints.l_points_corrected, m_keypoints.r_points_undistorted, m_points3d.points3d);

	Filter3dPointsByZ(m_points3d.points3d, m_keypoints.l_points_corrected, m_keypoints.r_points_undistorted,
		m_descriptors.l_descriptors, m_descriptors.r_descriptors, m_indices.mask, m_config.vml.z_close,
		m_config.vml.z_far);
}




//Mat getDefaultNewCameraMatrix(const Mat& cameraMatrix, Size imgsize,
//	bool centerPrincipalPoint)
//{
//	if (!centerPrincipalPoint && cameraMatrix.type() == CV_64F)
//		return cameraMatrix;
//
//	Mat newCameraMatrix;
//	cameraMatrix.convertTo(newCameraMatrix, CV_64F);
//	if (centerPrincipalPoint)
//	{
//		((double*)newCameraMatrix.data)[2] = (imgsize.width - 1) * 0.5;
//		((double*)newCameraMatrix.data)[5] = (imgsize.height - 1) * 0.5;
//	}
//	return newCameraMatrix;
//}


///CorrectMatches // TODO:
bool ProjectPointOnLine(const cv::Point2d& point, const cv::Matx31d& line, cv::Point2d& result) {
	// line could be represented by a*X + b*Y + c = 0, where:
	double a = line(0, 0), b = line(1, 0), c = line(2, 0);
	double eps = 1e-6;
	// check if the epi-line is degenerate
	if (std::max({ std::abs(a), std::abs(b) }) < eps) {
		return false;
	}
	// check if the line is horizontal or vertical
	if (std::abs(a) < eps) {
		result = { point.x, -c / b };
		return true;
	}
	else if (std::abs(b) < eps) {
		result = { -c / a, point.y };
		return true;
	}
	cv::Point2d p1_on_l, p2_on_l;
	// check if the line goes through origin
	if (std::abs(c) < eps) {
		// choose first point as origin
		p1_on_l = { 0., 0. };
		// choose second point as x = 1., then a + b * y = 0
		p2_on_l = { 1., -a / b };
	}
	else {
		// get a 2D point on the line a*x + b*y + c = 0 with y-coordinate 0:  (-c/a, 0)
		p1_on_l = { -c / a, 0. };
		// get a 2D point on the line with x-coordinate 0:  (0, -b/c)
		p2_on_l = { 0., -c / b };
	}

	// vector connecting p1 and p2
	auto p1p2 = p2_on_l - p1_on_l;
	// vector connecting p1 and point
	auto p1p = point - p1_on_l;

	// vector projection of p1p onto p1p2
	auto dot = p1p.dot(p1p2);
	auto norm_p1p2 = p1p2.x * p1p2.x + p1p2.y * p1p2.y;
	auto p1p_on_p1p2 = p1p2 * dot / norm_p1p2;

	// get the final point projection of p on p1p2
	result = p1_on_l + p1p_on_p1p2;
	return true;
}

void VisionPipeline::CorrectMatches(const keypoints_t& l_points, const keypoints_t& r_points,
	keypoints_t& l_points_corrected, const cv::Matx33d& F,
	ManagedArray<NUM_KPTS, uint64_t>& idxs) {
	nvtx_scope s("correct");
	std::vector<size_t> inliner_indices;
	l_points_corrected.resize(l_points.length());
	if (l_points_corrected.empty()) { // in this case, also the right points should be empty
		return;
	}
	std::vector<cv::Point2f> l_points_v = kpts2vec(l_points), r_points_v = kpts2vec(r_points);
	auto l_points_corrected_v = l_points_v;

	cv::Point2d p1, p2, p1_corrected;
	cv::Matx31d p2_m;
	cv::Matx13d p1_m, p1_corrected_m;
	for (size_t i = 0; i < l_points_v.size(); i++) {
		p1 = l_points_v[i], p2 = r_points_v[i];
		p2_m = cv::Matx31d{ p2.x, p2.y, 1. };
		// corresponding epi-line of p2 on left image
		cv::Matx31d epi_line_l = F * p2_m;

		// check if the left point already lies on the epi line
		p1_m = cv::Matx13d{ p1.x, p1.y, 1. };
		if ((p1_m * epi_line_l)(0, 0) < 1e-6) {
			inliner_indices.push_back(i);
			continue;
		}
		// project p1 onto epi-line to get corrected point pair
		if (ProjectPointOnLine(p1, epi_line_l, p1_corrected)) {
			p1_corrected_m = cv::Matx13d{ p1_corrected.x, p1_corrected.y, 1. };
			auto res = p1_corrected_m * epi_line_l;
			if (res(0, 0) < 1e-6) {
				l_points_corrected_v[i] = p1_corrected;
				inliner_indices.push_back(i);
			}
		}
	}
	vec2kpts(l_points_corrected_v, l_points_corrected);
	vec2idx(inliner_indices, idxs);
	m_descriptors.l_descriptors.filter_by_index(idxs);
	m_descriptors.r_descriptors.filter_by_index(idxs);
	m_keypoints.l_points_corrected.filter_by_index(idxs);
	m_keypoints.r_points_undistorted.filter_by_index(idxs);
	idxs.resize(0);
}


///triangulate //TODO:
std::vector<cv::Point3d> triangulate(const cv::Matx33d& R, const cv::Matx31d& T, const cv::Matx33d& K_l,
	const cv::Matx33d& K_r, const std::vector<cv::Point2f>& points_l,
	const std::vector<cv::Point2f>& points_r) {
	// triangulate based on intersecting lines
	// http://mathforum.org/library/drmath/view/62814.html

	std::vector<cv::Point3d> points3d(points_l.size());

	// denominator = 0 sanity check
	double zero = 0.0;
	if (std::memcmp(&K_l(0, 0), &zero, sizeof(double)) == 0 || std::memcmp(&K_l(1, 1), &zero, sizeof(double)) == 0) {
		throw std::overflow_error("Division by zero. Focal length not loaded correctly");
	}

	// inverses of the x/y focal lengths of the left/right camera matrices.
	const auto flxi = 1. / K_l(0, 0);
	const auto flyi = 1. / K_l(1, 1);
	const auto frxi = 1. / K_r(0, 0);
	const auto fryi = 1. / K_r(1, 1);

	for (size_t i = 0; i < points_l.size(); ++i) {
		const auto lh = cv::Point3d{ (points_l[i].x - K_l(0, 2)) * flxi, (points_l[i].y - K_l(1, 2)) * flyi, 1 };

		const auto _rh = cv::Point2d{ (points_r[i].x - K_r(0, 2)) * frxi, (points_r[i].y - K_r(1, 2)) * fryi };

		const auto rh =
			cv::Point3d{ R(0, 0) * _rh.x + R(0, 1) * _rh.y + R(0, 2), R(1, 0) * _rh.x + R(1, 1) * _rh.y + R(1, 2),
						R(2, 0) * _rh.x + R(2, 1) * _rh.y + R(2, 2) };

		const auto cA = lh.cross(rh);
		const auto cB = cv::Point3d{ T(0, 0), T(0, 1), T(0, 2) }.cross(rh);
		const auto A = sqrt(cA.x * cA.x + cA.y * cA.y + cA.z * cA.z);
		const auto B = sqrt(cB.x * cB.x + cB.y * cB.y + cB.z * cB.z);

		const auto d = B / A;

		points3d[i] = lh * d;
	}

	return points3d;
}



///Filter By Index //TODO:
void UndistortPoints(const std::vector<float>& points, keypoints_t& points_undistorted, const cv::Matx33d& K,
	const cv::Matx14d& distortion)
{
	if (points.size() == 0) { throw "keypoint vector is empty"; }

	//std::vector<float> points_v = kpts2vec(points), points_undistorted_v;// oroginal code
	std::vector<float> points_v = points, points_undistorted_v;
	cv::undistortPoints(points_v, points_undistorted_v, K, distortion, cv::noArray(), K); // TODO: serach what is this func doing
	vec2kpts(points_undistorted_v, points_undistorted);
}

//to invoke
__host__
void Cuda::undistortPoints(cudaStream_t providedStream, const Mat<float>& K, const Mat<float>& distortion)
{
	UndistortPoints(points_v, points_undistorted_v, K, distortion, cv::noArray(), K); // TODO: serach what is this func doing
	vec2kpts(points_undistorted_v, points_undistorted);
}