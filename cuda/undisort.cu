#include "../header/CudaClass.h"
#include "../dep/Dep.h"

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


//template <uint64_t C2, typename I> void filter_by_index(const ManagedArray<C2, I>& indices) {
//	static_assert(is_valid_index<C, I>());
//
//	if (this->length() < indices.length()) {
//		throw std::runtime_error("Error, index array is longer than filtered array!!");
//	}
//
//	for (uint64_t i = 0; i < indices.length(); ++i) {
//		(*this)[i] = (*this)[indices[i]];
//	}
//
//	this->resize(indices.length());
//}
//void vec2idx(const std::vector<size_t>& vec, ManagedArray<NUM_KPTS, uint64_t>& idxs) {
//	idxs.resize(vec.size());
//	for (uint64_t i = 0; i < idxs.length(); ++i) {
//		idxs[i] = vec[i];
//	}
//}
//void VisionPipeline::EpiFilterPoints(const keypoints_t& l_points, const keypoints_t& r_points, const cv::Matx33d& F,
//	double thresh, ManagedArray<NUM_KPTS, uint64_t>& idxs) {
//	nvtx_scope s("epi-filter");
//	std::vector<cv::Point2f> l_points_v = kpts2vec(l_points), r_points_v = kpts2vec(r_points);
//	std::vector<size_t> inliner_indices;
//	cv::Point2f p1, p2;
//	cv::Matx13d p1_m;
//	cv::Matx31d p2_m;
//	for (size_t i = 0; i < l_points_v.size(); i++) {
//		p1 = l_points_v[i], p2 = r_points_v[i];
//		// transform to homogeneous coordinate.
//		p1_m = cv::Matx13d{ p1.x, p1.y, 1. };
//		p2_m = cv::Matx31d{ p2.x, p2.y, 1. };
//		// calculate the epi-polar constraint error : p1^T * F * p2
//		const double score = ((p1_m * F) * p2_m)(0, 0);
//		if (fabs(score) < thresh) {
//			inliner_indices.push_back(i);
//		}
//	}
//	vec2idx(inliner_indices, idxs);
//	m_descriptors.l_descriptors.filter_by_index(idxs);
//	m_descriptors.r_descriptors.filter_by_index(idxs);
//	m_keypoints.l_points_undistorted.filter_by_index(idxs);
//	m_keypoints.r_points_undistorted.filter_by_index(idxs);
//	idxs.resize(0);
//}

void remap(Matf _src, Matf _dst,
	Matf _map1, Matf _map2,
	int interpolation, int borderType, const Scalar& borderValue);



__host__
void Cuda::undistortPoints(cudaStream_t providedStream, const Mat<float>& K, const Mat<float>& distortion)
{
	undistortPoints(points_v, points_undistorted_v, K, distortion, cv::noArray(), K); // TODO: serach what is this func doing
	vec2kpts(points_undistorted_v, points_undistorted);
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


void remap(Matf _src, Matf _dst,
	Matf _map1, Matf _map2,
	int interpolation, int borderType, const Scalar& borderValue)
{
	CV_INSTRUMENT_REGION();

	static RemapNNFunc nn_tab[] =
	{
		remapNearest<uchar>, remapNearest<schar>, remapNearest<ushort>, remapNearest<short>,
		remapNearest<int>, remapNearest<float>, remapNearest<double>, 0
	};

	static RemapFunc linear_tab[] =
	{
		remapBilinear<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, RemapVec_8u, short>, 0,
		remapBilinear<Cast<float, ushort>, RemapNoVec, float>,
		remapBilinear<Cast<float, short>, RemapNoVec, float>, 0,
		remapBilinear<Cast<float, float>, RemapNoVec, float>,
		remapBilinear<Cast<double, double>, RemapNoVec, float>, 0
	};

	static RemapFunc cubic_tab[] =
	{
		remapBicubic<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE>, 0,
		remapBicubic<Cast<float, ushort>, float, 1>,
		remapBicubic<Cast<float, short>, float, 1>, 0,
		remapBicubic<Cast<float, float>, float, 1>,
		remapBicubic<Cast<double, double>, float, 1>, 0
	};

	static RemapFunc lanczos4_tab[] =
	{
		remapLanczos4<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE>, 0,
		remapLanczos4<Cast<float, ushort>, float, 1>,
		remapLanczos4<Cast<float, short>, float, 1>, 0,
		remapLanczos4<Cast<float, float>, float, 1>,
		remapLanczos4<Cast<double, double>, float, 1>, 0
	};

	CV_Assert(!_map1.empty());
	CV_Assert(_map2.empty() || (_map2.size() == _map1.size()));

	CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
		ocl_remap(_src, _dst, _map1, _map2, interpolation, borderType, borderValue))

		Mat src = _src.getMat(), map1 = _map1.getMat(), map2 = _map2.getMat();
	_dst.create(map1.size(), src.type());
	Mat dst = _dst.getMat();


	CV_OVX_RUN(
		src.type() == CV_8UC1 && dst.type() == CV_8UC1 &&
		!ovx::skipSmallImages<VX_KERNEL_REMAP>(src.cols, src.rows) &&
		(borderType & ~BORDER_ISOLATED) == BORDER_CONSTANT &&
		((map1.type() == CV_32FC2 && map2.empty() && map1.size == dst.size) ||
			(map1.type() == CV_32FC1 && map2.type() == CV_32FC1 && map1.size == dst.size && map2.size == dst.size) ||
			(map1.empty() && map2.type() == CV_32FC2 && map2.size == dst.size)) &&
		((borderType & BORDER_ISOLATED) != 0 || !src.isSubmatrix()),
		openvx_remap(src, dst, map1, map2, interpolation, borderValue));

	CV_Assert(dst.cols < SHRT_MAX&& dst.rows < SHRT_MAX&& src.cols < SHRT_MAX&& src.rows < SHRT_MAX);

	if (dst.data == src.data)
		src = src.clone();

	if (interpolation == INTER_AREA)
		interpolation = INTER_LINEAR;

	int type = src.type(), depth = CV_MAT_DEPTH(type);

//#if defined HAVE_IPP && !IPP_DISABLE_REMAP
//	CV_IPP_CHECK()
//	{
//		if ((interpolation == INTER_LINEAR || interpolation == INTER_CUBIC || interpolation == INTER_NEAREST) &&
//			map1.type() == CV_32FC1 && map2.type() == CV_32FC1 &&
//			(borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT))
//		{
//			int ippInterpolation =
//				interpolation == INTER_NEAREST ? IPPI_INTER_NN :
//				interpolation == INTER_LINEAR ? IPPI_INTER_LINEAR : IPPI_INTER_CUBIC;
//
//			ippiRemap ippFunc =
//				type == CV_8UC1 ? (ippiRemap)ippiRemap_8u_C1R :
//				type == CV_8UC3 ? (ippiRemap)ippiRemap_8u_C3R :
//				type == CV_8UC4 ? (ippiRemap)ippiRemap_8u_C4R :
//				type == CV_16UC1 ? (ippiRemap)ippiRemap_16u_C1R :
//				type == CV_16UC3 ? (ippiRemap)ippiRemap_16u_C3R :
//				type == CV_16UC4 ? (ippiRemap)ippiRemap_16u_C4R :
//				type == CV_32FC1 ? (ippiRemap)ippiRemap_32f_C1R :
//				type == CV_32FC3 ? (ippiRemap)ippiRemap_32f_C3R :
//				type == CV_32FC4 ? (ippiRemap)ippiRemap_32f_C4R : 0;
//
//			if (ippFunc)
//			{
//				bool ok;
//				IPPRemapInvoker invoker(src, dst, map1, map2, ippFunc, ippInterpolation,
//					borderType, borderValue, &ok);
//				Range range(0, dst.rows);
//				parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
//
//				if (ok)
//				{
//					CV_IMPL_ADD(CV_IMPL_IPP | CV_IMPL_MT);
//					return;
//				}
//				setIppErrorStatus();
//			}
//		}
//	}
//#endif

	RemapNNFunc nnfunc = 0;
	RemapFunc ifunc = 0;
	const void* ctab = 0;
	bool fixpt = depth == CV_8U;
	bool planar_input = false;

	if (interpolation == INTER_NEAREST)
	{
		nnfunc = nn_tab[depth];
		CV_Assert(nnfunc != 0);
	}
	else
	{
		if (interpolation == INTER_LINEAR)
			ifunc = linear_tab[depth];
		else if (interpolation == INTER_CUBIC) {
			ifunc = cubic_tab[depth];
			CV_Assert(_src.channels() <= 4);
		}
		else if (interpolation == INTER_LANCZOS4) {
			ifunc = lanczos4_tab[depth];
			CV_Assert(_src.channels() <= 4);
		}
		else
			CV_Error(CV_StsBadArg, "Unknown interpolation method");
		CV_Assert(ifunc != 0);
		ctab = initInterTab2D(interpolation, fixpt);
	}

	const Mat* m1 = &map1, * m2 = &map2;

	if ((map1.type() == CV_16SC2 && (map2.type() == CV_16UC1 || map2.type() == CV_16SC1 || map2.empty())) ||
		(map2.type() == CV_16SC2 && (map1.type() == CV_16UC1 || map1.type() == CV_16SC1 || map1.empty())))
	{
		if (map1.type() != CV_16SC2)
			std::swap(m1, m2);
	}
	else
	{
		CV_Assert(((map1.type() == CV_32FC2 || map1.type() == CV_16SC2) && map2.empty()) ||
			(map1.type() == CV_32FC1 && map2.type() == CV_32FC1));
		planar_input = map1.channels() == 1;
	}

	RemapInvoker invoker(src, dst, m1, m2,
		borderType, borderValue, planar_input, nnfunc, ifunc,
		ctab);
	parallel_for_(Range(0, dst.rows), invoker, dst.total() / (double)(1 << 16));
}


///Filter By Index //TODO:
//void UndistortPoints(const std::vector<float>& points, keypoints_t& points_undistorted, const cv::Matx33d& K,
//	const cv::Matx14d& distortion) 
//{
//	if (points.size() == 0) { throw "keypoint vector is empty"; }
//
//	//std::vector<float> points_v = kpts2vec(points), points_undistorted_v;// oroginal code
//	std::vector<float> points_v = points, points_undistorted_v;
//	cv::undistortPoints(points_v, points_undistorted_v, K, distortion, cv::noArray(), K); // TODO: serach what is this func doing
//	vec2kpts(points_undistorted_v, points_undistorted);
//}
//
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
//
//void initUndistortRectifyMap(const Matf& _cameraMatrix, const Matf& _distCoeffs, // i presume that we are working with cv_32FC1, because of the previous assigment
//	const Matf& matR, const Matf& _newCameraMatrix,
//	int size, int m1type, Matf& map1, Matf& map2) // mat2 has to be from floats -> cv_32FC1
//{
//	assert(m1type == CV_16SC2 || m1type == CV_32FC1 || m1type == CV_32FC2);
//
//
//	Mat<float> R(3, 3);
//	R.eye(3, 3);
//	//Mat <float> R(3,3) = Mat<double>::eye(3, 3), distCoeffs; // why would we pass the distCoeffs if we create indentity matrix
//	Mat_<double> A = Mat_<double>(_cameraMatrix), Ar;
//
//	if (_newCameraMatrix.data)
//		Ar = Mat_<double>(_newCameraMatrix);
//	else
//		Ar = getDefaultNewCameraMatrix(A, size, true);
//
//	if (matR.data)
//		R = Mat_<double>(matR);
//
//	if (_distCoeffs.data)
//		distCoeffs = Mat_<double>(_distCoeffs);
//	else
//	{
//		distCoeffs.create(8, 1);
//		distCoeffs = 0.;
//	}
//
//	CV_Assert(A.size() == Size(3, 3) && A.size() == R.size());
//	CV_Assert(Ar.size() == Size(3, 3) || Ar.size() == Size(4, 3));
//	Mat_<double> iR = (Ar.colRange(0, 3) * R).inv(DECOMP_LU);
//	const double* ir = &iR(0, 0);
//
//	double u0 = A(0, 2), v0 = A(1, 2);
//	double fx = A(0, 0), fy = A(1, 1);
//
//	CV_Assert(distCoeffs.size() == Size(1, 4) || distCoeffs.size() == Size(4, 1) ||
//		distCoeffs.size() == Size(1, 5) || distCoeffs.size() == Size(5, 1) ||
//		distCoeffs.size() == Size(1, 8) || distCoeffs.size() == Size(8, 1));
//
//	if (distCoeffs.rows != 1 && !distCoeffs.isContinuous())
//		distCoeffs = distCoeffs.t();
//
//	double k1 = ((double*)distCoeffs.data)[0];
//	double k2 = ((double*)distCoeffs.data)[1];
//	double p1 = ((double*)distCoeffs.data)[2];
//	double p2 = ((double*)distCoeffs.data)[3];
//	double k3 = distCoeffs.cols + distCoeffs.rows - 1 >= 5 ? ((double*)distCoeffs.data)[4] : 0.;
//	double k4 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? ((double*)distCoeffs.data)[5] : 0.;
//	double k5 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? ((double*)distCoeffs.data)[6] : 0.;
//	double k6 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? ((double*)distCoeffs.data)[7] : 0.;
//
//	for (int i = 0; i < size.height; i++)
//	{
//		float* m1f = (float*)(map1.data + map1.step * i);
//		float* m2f = (float*)(map2.data + map2.step * i);
//		short* m1 = (short*)m1f;
//		ushort* m2 = (ushort*)m2f;
//		double _x = i * ir[1] + ir[2], _y = i * ir[4] + ir[5], _w = i * ir[7] + ir[8];
//
//		for (int j = 0; j < size.width; j++, _x += ir[0], _y += ir[3], _w += ir[6])
//		{
//			double w = 1. / _w, x = _x * w, y = _y * w;
//			double x2 = x * x, y2 = y * y;
//			double r2 = x2 + y2, _2xy = 2 * x * y;
//			double kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) * r2);
//			double u = fx * (x * kr + p1 * _2xy + p2 * (r2 + 2 * x2)) + u0;
//			double v = fy * (y * kr + p1 * (r2 + 2 * y2) + p2 * _2xy) + v0;
//			if (m1type == CV_16SC2)
//			{
//				int iu = saturate_cast<int>(u * INTER_TAB_SIZE);
//				int iv = saturate_cast<int>(v * INTER_TAB_SIZE);
//				m1[j * 2] = (short)(iu >> INTER_BITS);
//				m1[j * 2 + 1] = (short)(iv >> INTER_BITS);
//				m2[j] = (ushort)((iv & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (iu & (INTER_TAB_SIZE - 1)));
//			}
//			else if (m1type == CV_32FC1)
//			{
//				m1f[j] = (float)u;
//				m2f[j] = (float)v;
//			}
//			else
//			{
//				m1f[j * 2] = (float)u;
//				m1f[j * 2 + 1] = (float)v;
//			}
//		}
//	}
//}
//
//
//void undistort(const Matf& src, Matf& dst, const Matf& _cameraMatrix,
//	const Matf& _distCoeffs, const Matf& _newCameraMatrix)
//{
//	//dst has to be the same type as src 
//	assert(dst.data != src.data);
//
//	int stripe_size0 = std::min(std::max(1, (1 << 12) / std::max(src.cols(), 1)), src.rows());
//	Mat16s map1(stripe_size0, src.cols(), 2);
//	Mat16u map2(stripe_size0, src.cols());
//
//	Matf A, distCoeffs, Ar;
//	Matf I = eye<float>(3, 3);
//
//	//_cameraMatrix.convertTo(A, CV_64F); // it is said to convert the A to double, but i did not do it 
//	/*if (_distCoeffs.data())
//		distCoeffs = Mat_<double>(_distCoeffs);*/
//	//else
//	//{
//	//	distCoeffs.create(5, 1);
//	//	distCoeffs = 0.;
//	//}
//
//	//if (_newCameraMatrix.data)
//	//	_newCameraMatrix.convertTo(Ar, CV_64F);
//	//else
//	//	A.copyTo(Ar);
//
//	double v0 = Ar[1 * Ar.cols() + 2]; // not sure if opencv counts from 1 or 0
//	int Size = 0;
//	for (int y = 0; y < src.rows(); y += stripe_size0)
//	{
//		int stripe_size = std::min(stripe_size0, src.rows - y);
//		Ar[1*Ar.cols() + 2] = v0 - y;
//		Matf map1_part = map1.rowRange(0, stripe_size),//TODO: find out what is this rowRange doing!
//		     map2_part = map2.rowRange(0, stripe_size),
//			 dst_part = dst.rowRange(y, y + stripe_size);
//
//		Size = src.cols() * stripe_size * 4; // it might not be needed to multiply by 4, i did it because it was said that this creates a class specifying the size of an image
//		initUndistortRectifyMap(A, distCoeffs, I, Ar, Size,
//			map1_part.type(), map1_part, map2_part);
//		remap(src, dst_part, map1_part, map2_part, INTER_LINEAR, BORDER_CONSTANT);
//	}
//}



///CorrectMatches // TODO:
//bool ProjectPointOnLine(const cv::Point2d& point, const cv::Matx31d& line, cv::Point2d& result) {
//	// line could be represented by a*X + b*Y + c = 0, where:
//	double a = line(0, 0), b = line(1, 0), c = line(2, 0);
//	double eps = 1e-6;
//	// check if the epi-line is degenerate
//	if (std::max({ std::abs(a), std::abs(b) }) < eps) {
//		return false;
//	}
//	// check if the line is horizontal or vertical
//	if (std::abs(a) < eps) {
//		result = { point.x, -c / b };
//		return true;
//	}
//	else if (std::abs(b) < eps) {
//		result = { -c / a, point.y };
//		return true;
//	}
//	cv::Point2d p1_on_l, p2_on_l;
//	// check if the line goes through origin
//	if (std::abs(c) < eps) {
//		// choose first point as origin
//		p1_on_l = { 0., 0. };
//		// choose second point as x = 1., then a + b * y = 0
//		p2_on_l = { 1., -a / b };
//	}
//	else {
//		// get a 2D point on the line a*x + b*y + c = 0 with y-coordinate 0:  (-c/a, 0)
//		p1_on_l = { -c / a, 0. };
//		// get a 2D point on the line with x-coordinate 0:  (0, -b/c)
//		p2_on_l = { 0., -c / b };
//	}
//
//	// vector connecting p1 and p2
//	auto p1p2 = p2_on_l - p1_on_l;
//	// vector connecting p1 and point
//	auto p1p = point - p1_on_l;
//
//	// vector projection of p1p onto p1p2
//	auto dot = p1p.dot(p1p2);
//	auto norm_p1p2 = p1p2.x * p1p2.x + p1p2.y * p1p2.y;
//	auto p1p_on_p1p2 = p1p2 * dot / norm_p1p2;
//
//	// get the final point projection of p on p1p2
//	result = p1_on_l + p1p_on_p1p2;
//	return true;
//}
//void VisionPipeline::CorrectMatches(const keypoints_t& l_points, const keypoints_t& r_points,
//	keypoints_t& l_points_corrected, const cv::Matx33d& F,
//	ManagedArray<NUM_KPTS, uint64_t>& idxs) {
//	nvtx_scope s("correct");
//	std::vector<size_t> inliner_indices;
//	l_points_corrected.resize(l_points.length());
//	if (l_points_corrected.empty()) { // in this case, also the right points should be empty
//		return;
//	}
//	std::vector<cv::Point2f> l_points_v = kpts2vec(l_points), r_points_v = kpts2vec(r_points);
//	auto l_points_corrected_v = l_points_v;
//
//	cv::Point2d p1, p2, p1_corrected;
//	cv::Matx31d p2_m;
//	cv::Matx13d p1_m, p1_corrected_m;
//	for (size_t i = 0; i < l_points_v.size(); i++) {
//		p1 = l_points_v[i], p2 = r_points_v[i];
//		p2_m = cv::Matx31d{ p2.x, p2.y, 1. };
//		// corresponding epi-line of p2 on left image
//		cv::Matx31d epi_line_l = F * p2_m;
//
//		// check if the left point already lies on the epi line
//		p1_m = cv::Matx13d{ p1.x, p1.y, 1. };
//		if ((p1_m * epi_line_l)(0, 0) < 1e-6) {
//			inliner_indices.push_back(i);
//			continue;
//		}
//		// project p1 onto epi-line to get corrected point pair
//		if (ProjectPointOnLine(p1, epi_line_l, p1_corrected)) {
//			p1_corrected_m = cv::Matx13d{ p1_corrected.x, p1_corrected.y, 1. };
//			auto res = p1_corrected_m * epi_line_l;
//			if (res(0, 0) < 1e-6) {
//				l_points_corrected_v[i] = p1_corrected;
//				inliner_indices.push_back(i);
//			}
//		}
//	}
//	vec2kpts(l_points_corrected_v, l_points_corrected);
//	vec2idx(inliner_indices, idxs);
//	m_descriptors.l_descriptors.filter_by_index(idxs);
//	m_descriptors.r_descriptors.filter_by_index(idxs);
//	m_keypoints.l_points_corrected.filter_by_index(idxs);
//	m_keypoints.r_points_undistorted.filter_by_index(idxs);
//	idxs.resize(0);
//}


///triangulate //TODO:
//std::vector<cv::Point3d> triangulate(const cv::Matx33d& R, const cv::Matx31d& T, const cv::Matx33d& K_l,
//	const cv::Matx33d& K_r, const std::vector<cv::Point2f>& points_l,
//	const std::vector<cv::Point2f>& points_r) {
//	// triangulate based on intersecting lines
//	// http://mathforum.org/library/drmath/view/62814.html
//
//	std::vector<cv::Point3d> points3d(points_l.size());
//
//	// denominator = 0 sanity check
//	double zero = 0.0;
//	if (std::memcmp(&K_l(0, 0), &zero, sizeof(double)) == 0 || std::memcmp(&K_l(1, 1), &zero, sizeof(double)) == 0) {
//		throw std::overflow_error("Division by zero. Focal length not loaded correctly");
//	}
//
//	// inverses of the x/y focal lengths of the left/right camera matrices.
//	const auto flxi = 1. / K_l(0, 0);
//	const auto flyi = 1. / K_l(1, 1);
//	const auto frxi = 1. / K_r(0, 0);
//	const auto fryi = 1. / K_r(1, 1);
//
//	for (size_t i = 0; i < points_l.size(); ++i) {
//		const auto lh = cv::Point3d{ (points_l[i].x - K_l(0, 2)) * flxi, (points_l[i].y - K_l(1, 2)) * flyi, 1 };
//
//		const auto _rh = cv::Point2d{ (points_r[i].x - K_r(0, 2)) * frxi, (points_r[i].y - K_r(1, 2)) * fryi };
//
//		const auto rh =
//			cv::Point3d{ R(0, 0) * _rh.x + R(0, 1) * _rh.y + R(0, 2), R(1, 0) * _rh.x + R(1, 1) * _rh.y + R(1, 2),
//						R(2, 0) * _rh.x + R(2, 1) * _rh.y + R(2, 2) };
//
//		const auto cA = lh.cross(rh);
//		const auto cB = cv::Point3d{ T(0, 0), T(0, 1), T(0, 2) }.cross(rh);
//		const auto A = sqrt(cA.x * cA.x + cA.y * cA.y + cA.z * cA.z);
//		const auto B = sqrt(cB.x * cB.x + cB.y * cB.y + cB.z * cB.z);
//
//		const auto d = B / A;
//
//		points3d[i] = lh * d;
//	}
//
//	return points3d;
//}

