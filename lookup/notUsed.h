#include "../lookup/remapBilinear.h"
#include "../lookup/remapNearest.h"
#include "../lookup/RemapInvoker.h"
#include "../lookup/Rect.h"
#define INTER_MAX 7
#define CV_CN_SHIFT   3
#define CV_DEPTH_MAX	(1 << CV_CN_SHIFT)
#define CV_MAT_DEPTH 	( flags	) 	   ((flags) & CV_MAT_DEPTH_MASK)
#define CV_MAT_DEPTH_MASK	(CV_DEPTH_MAX - 1)
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
//void EpiFilterPoints(const keypoints_t& l_points, const keypoints_t& r_points, const cv::Matx33d& F,
//	double thresh, ManagedArray<NUM_KPTS, uint64_t>& idxs) {
//	nvtx_scope s("epi-filter");
//	std::vector<cv::Point2f> l_points_v = kpts2vec(l_points), r_points_v = kpts2vec(r_points);
//	std::vector<size_t> inliner_indices;
//	Point2f p1, p2;
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

/////CorrectMatches // TODO:
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
//
//
/////triangulate //TODO:
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



//template<typename _Tp> inline
//Scalar_<_Tp>::Scalar_()
//{
//	this->val[0] = this->val[1] = this->val[2] = this->val[3] = 0;
//}

template<typename ST, typename DT> struct Cast
{
	typedef ST type1;
	typedef DT rtype;

	DT operator()(ST val) const { return static_cast<DT>(val); } // it might trow an exception, look for saturate_cast
};
//enum InterpolationMasks {
//	INTER_BITS = 5,
//	INTER_BITS2 = INTER_BITS * 2,
//	INTER_TAB_SIZE = 1 << INTER_BITS,
//	INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
//};

const int INTER_REMAP_COEF_BITS = 15;
const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

static unsigned char NNDeltaTab_i[INTER_TAB_SIZE2][2];

static float BilinearTab_f[INTER_TAB_SIZE2][2][2];
static short BilinearTab_i[INTER_TAB_SIZE2][2][2];

template <typename FLOAT>
void computeTiltProjectionMatrix(
	FLOAT tauX,
	FLOAT tauY,
	Mat<FLOAT>* matTilt = 0) //					TODO: not sure if it will work(was like this btw
	//Mat<FLOAT, 3, 3>* dMatTiltdTauX = 0,
	//Mat<FLOAT, 3, 3>* dMatTiltdTauY = 0,
	//Mat<FLOAT, 3, 3>* invMatTilt = 0)
{
	FLOAT cTauX = cos(tauX);
	FLOAT sTauX = sin(tauX);
	FLOAT cTauY = cos(tauY);
	FLOAT sTauY = sin(tauY);
	Mat<FLOAT> matRotX(3, 3, std::vector<FLOAT>{1, 0, 0, 0, cTauX, sTauX, 0, -sTauX, cTauX}); // lil hacks here
	Mat<FLOAT> matRotY(3, 3, std::vector<FLOAT>{cTauY, 0, -sTauY, 0, 1, 0, sTauY, 0, cTauY});
	Mat<FLOAT> matRotXY;
	matRotXY = matRotY * matRotX;
	Mat<FLOAT> matProjZ(3, 3, std::vector<FLOAT>{matRotXY(2, 2), 0, -matRotXY(0, 2), 0, matRotXY(2, 2), -matRotXY(1, 2), 0, 0, 1});
	if (matTilt)
	{
		*matTilt = matProjZ * matRotXY;
	}
}
struct Size
{
	Size() = delete;
	Size(int cols, int rows) :cols(cols), rows(rows) {}
	int cols;
	int rows;
};
void initUndistortRectifyMap(Matf& cameraMatrix, Matf& distCoeffs,
	Matf& matR, Matf& newCameraMatrix,
	Size size, Mat16s& map1, Mat16u& map2)
{
	//Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat(); // imma use mine
	//Mat matR = _matR.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

	assert(m1type == CV_16SC2 || m1type == CV_32FC1 || m1type == CV_32FC2);

	Matf R = eye<float>(3, 3);
	Matf A = cameraMatrix, Ar;

	if (newCameraMatrix.hasData())
		Ar = newCameraMatrix;
	//else
	//	Ar = getDefaultNewCameraMatrix(A, size, true);

	if (matR.hasData())
		R = matR;

	if (!distCoeffs.hasData())
		distCoeffs = distCoeffs;

	assert(A.size() == Size(3, 3) && A.size() == R.size());
	assert(Ar.size() == Size(3, 3) || Ar.size() == Size(4, 3));
	//float iR = (Ar.colRange(0, 3) * R).inv(0); // inverse matrix function... motherFuc..		TODO:
	const float* ir = &iR(0, 0);

	float u0 = A[2], v0 = A[1 * A.rows() + 2];
	float fx = A[0], fy = A[1 * A.rows() + 1];

	assert(distCoeffs.size() == Size(1, 4) || distCoeffs.size() == Size(4, 1) ||
		distCoeffs.size() == Size(1, 5) || distCoeffs.size() == Size(5, 1) ||
		distCoeffs.size() == Size(1, 8) || distCoeffs.size() == Size(8, 1) ||
		distCoeffs.size() == Size(1, 12) || distCoeffs.size() == Size(12, 1) ||
		distCoeffs.size() == Size(1, 14) || distCoeffs.size() == Size(14, 1));

#define distPtr distCoeffs.m_matrix // i know, i know...
	float k1 = distPtr[0];
	float k2 = distPtr[1];
	float p1 = distPtr[2];
	float p2 = distPtr[3];
	float k3 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 5 ? distPtr[4] : 0.;
	float k4 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 8 ? distPtr[5] : 0.;
	float k5 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 8 ? distPtr[6] : 0.;
	float k6 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 8 ? distPtr[7] : 0.;
	float s1 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 12 ? distPtr[8] : 0.;
	float s2 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 12 ? distPtr[9] : 0.;
	float s3 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 12 ? distPtr[10] : 0.;
	float s4 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 12 ? distPtr[11] : 0.;
	float tauX = distCoeffs.cols() + distCoeffs.rows() - 1 >= 14 ? distPtr[12] : 0.;
	float tauY = distCoeffs.cols() + distCoeffs.rows() - 1 >= 14 ? distPtr[13] : 0.;

	// Matrix for trapezoidal distortion of tilted image sensor
	Matf matTilt = eye<float>(3, 3);
	//Matx33d matTilt = Matx33d::eye();
	computeTiltProjectionMatrix(tauX, tauY, &matTilt);

	//parallel_for_(Range(0, size.height), *getInitUndistortRectifyMapComputer( // TODO: parallel_for
	//	size, map1, map2, m1type, ir, matTilt, u0, v0,
	//	fx, fy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4));
}

static inline void interpolateLinear(float x, float* coeffs)
{
	coeffs[0] = 1.f - x;
	coeffs[1] = x;
}
static void initInterTab1D(int method, float* tab, int tabsz)
{
	float scale = 1.f / tabsz;
	for (int i = 0; i < tabsz; i++, tab += 2)
		interpolateLinear(i * scale, tab);
}
enum  	InterpolationFlags {
	INTER_NEAREST = 0,
	INTER_LINEAR = 1,
	INTER_CUBIC = 2,
	INTER_AREA = 3,
	INTER_LANCZOS4 = 4,
	INTER_LINEAR_EXACT = 5,
	INTER_NEAREST_EXACT = 6,
	WARP_FILL_OUTLIERS = 8,
	WARP_INVERSE_MAP = 16
};
enum  	BorderTypes {
	BORDER_CONSTANT = 0,
	BORDER_REPLICATE = 1,
	BORDER_REFLECT = 2,
	BORDER_WRAP = 3,
	BORDER_REFLECT_101 = 4,
	BORDER_TRANSPARENT = 5,
	BORDER_REFLECT101 = BORDER_REFLECT_101,
	BORDER_DEFAULT = BORDER_REFLECT_101,
	BORDER_ISOLATED = 16
};
static const void* initInterTab2D(int method, bool fixpt)
{
	static bool inittab[INTER_MAX + 1] = { false };
	float* tab = 0;
	short* itab = 0;
	int ksize = 0;

	tab = BilinearTab_f[0][0], itab = BilinearTab_i[0][0], ksize = 2;

#if 1 // it did not execute in the example code, that's why i commented it
	if (!inittab[method])
	{
		//AutoBuffer<float> _tab(8 * INTER_TAB_SIZE);
		std::vector<float> _tab(8 * INTER_TAB_SIZE);
		int i, j, k1, k2;
		initInterTab1D(method, _tab.data(), INTER_TAB_SIZE);
		for (i = 0; i < INTER_TAB_SIZE; i++)
			for (j = 0; j < INTER_TAB_SIZE; j++, tab += ksize * ksize, itab += ksize * ksize)
			{
				int isum = 0;
				NNDeltaTab_i[i * INTER_TAB_SIZE + j][0] = j < INTER_TAB_SIZE / 2;
				NNDeltaTab_i[i * INTER_TAB_SIZE + j][1] = i < INTER_TAB_SIZE / 2;

				for (k1 = 0; k1 < ksize; k1++)
				{
					float vy = _tab[i * ksize + k1];
					for (k2 = 0; k2 < ksize; k2++)
					{
						float v = vy * _tab[j * ksize + k2];
						tab[k1 * ksize + k2] = v;
						isum += itab[k1 * ksize + k2] = static_cast<short>(v * INTER_REMAP_COEF_SCALE); // expands to 2^15 so i may not have loses(was saturate_cast)
					}
				}

				if (isum != INTER_REMAP_COEF_SCALE)
				{
					int diff = isum - INTER_REMAP_COEF_SCALE;
					int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
					for (k1 = ksize2; k1 < ksize2 + 2; k1++)
						for (k2 = ksize2; k2 < ksize2 + 2; k2++)
						{
							if (itab[k1 * ksize + k2] < itab[mk1 * ksize + mk2])
								mk1 = k1, mk2 = k2;
							else if (itab[k1 * ksize + k2] > itab[Mk1 * ksize + Mk2])
								Mk1 = k1, Mk2 = k2;
						}
					if (diff < 0)
						itab[Mk1 * ksize + Mk2] = (short)(itab[Mk1 * ksize + Mk2] - diff);
					else
						itab[mk1 * ksize + mk2] = (short)(itab[mk1 * ksize + mk2] - diff);
				}
			}
		tab -= INTER_TAB_SIZE2 * ksize * ksize;
		itab -= INTER_TAB_SIZE2 * ksize * ksize;
#if CV_SIMD128
		if (method == INTER_LINEAR)
		{
			for (i = 0; i < INTER_TAB_SIZE2; i++)
				for (j = 0; j < 4; j++)
				{
					BilinearTab_iC4[i][0][j * 2] = BilinearTab_i[i][0][0];
					BilinearTab_iC4[i][0][j * 2 + 1] = BilinearTab_i[i][0][1];
					BilinearTab_iC4[i][1][j * 2] = BilinearTab_i[i][1][0];
					BilinearTab_iC4[i][1][j * 2 + 1] = BilinearTab_i[i][1][1];
				}
		}
#endif
		inittab[method] = true;
	}
#endif
	return fixpt ? (const void*)itab : (const void*)tab;
}
//------------------remap
void remap(const Matf& _src, Matf _dst, // mat16s has to be 2 channeled TODO:
	Mat16s& _map1, Mat16u& _map2,
	int interpolation, int borderType/*, const Scalar& borderValue*/)  // TODO: ASK SASHO FOR SATURATE_CAST
{

	static RemapNNFunc nn_tab[] =
	{
		remapNearest<unsigned char>, remapNearest<char>, remapNearest<unsigned short>, remapNearest<short>,
		remapNearest<int>, remapNearest<float>, remapNearest<double>, 0
	};

	static RemapFunc linear_tab[] =
	{
		/*remapBilinear<FixedPtCast<int, unsigned char, INTER_REMAP_COEF_BITS>, RemapVec_8u, short>, 0,
		remapBilinear<Cast<float, ushort>, RemapNoVec, float>,
		remapBilinear<Cast<float, short>, RemapNoVec, float>, 0,*/
		remapBilinear<Cast<float, float>, 0, float>											// imo the only one needed    //RemapNoVec => 0
		//remapBilinear<Cast<double, double>, RemapNoVec, float>, 0
	};

	assert(_map1.hasData());
	assert(!_map2.hasData() || (_map2.size() == _map1.size()));

	Matf dst(_map1.cols(), _map1.rows(), _src.channels());//_dst.create(map1.size(), src.type()); //TODO: remember to set _dst to dst




	if (dst == _src)
		//_src = src.clone(); // mist try 

	//if (interpolation == INTER_AREA)
	//	interpolation = INTER_LINEAR;

		int depth1 = CV_MAT_DEPTH(16); ////type = src.type(), // after debugging i saw that type was 16, not quite sure though



	RemapNNFunc nnfunc = 0;
	RemapFunc ifunc = 0;
	const void* ctab = 0;
	bool fixpt = (depth1 == 0);
	bool planar_input = false;


	{

		ifunc = linear_tab[depth1];
		assert(ifunc != 0);
		ctab = initInterTab2D(interpolation, fixpt);
	}
	const Mat16s* m1 = &_map1;
	const Mat16u* m2 = &_map2;
	//(map1.type() == CV_16SC2 && (map2.type() == CV_16UC1 || map2.type() == CV_16SC1 || map2.empty())) || // it is 16sc2 (map1), and 16uc1 // so yes
	//(map2.type() == CV_16SC2 && (map1.type() == CV_16UC1 || map1.type() == CV_16SC1 || map1.empty()))

		//if (map1.type() != CV_16SC2)
	std::swap(m1, m2);


	RemapInvoker invoker(_src, dst, m1, m2,
		borderType, planar_input, nnfunc, ifunc, // Matf src, Matf dst, Matf m1, Matf m2, int broderType, bool planar_input, RemapNNFunc nnfunc, RemapFunc ifunc, const void * ctab
		ctab);
	/*parallel_for_(Range(0, dst.rows), invoker, dst.total() / (double)(1 << 16));*/
}



//TODO: rowRange, RemapInvoker
void undistort(const Matf& _src, Matf& _dst, const Matf& K, //dst has to be allocated and with the type of src
	const Matf& distortion, int noArray, Matf& k_result)
{
	Matf src = _src, cameraMatrix = K;
	Matf distCoeffs = distortion, newCameraMatrix = k_result;

	Matf dst = _dst;

	assert(dst != src);

	int stripe_size0 = std::min(std::max(1, (1 << 12) / std::max(src.cols(), 1)), src.rows());
	Mat16s map1(stripe_size0, src.cols(), 2);
	Mat16u map2(stripe_size0, src.cols(), 1);

	Matf A = cameraMatrix, Ar, I = eye<float>(3, 3); // creating indentity matrix

	if (!distCoeffs.hasData()) {// if does NOT have data
		distCoeffs.create(5, 1); // creates the matrix and fills with 0ros, as it was
	}

	if (!newCameraMatrix.hasData()) {
		Ar = A; // why would we, idk
	}


	float v0 = Ar(1, 2);
	for (int y = 0; y < src.rows(); y += stripe_size0)
	{
		int stripe_size = std::min(stripe_size0, src.rows - y);
		Ar[1, 2] = v0 - y;
		//Mat map1_part = map1.rowRange(0, stripe_size),
		//	map2_part = map2.rowRange(0, stripe_size),
		//	dst_part = dst.rowRange(y, y + stripe_size);

		initUndistortRectifyMap(A, distCoeffs, I, Ar, Size(src.cols(), stripe_size), map1, map2);//TODO: map1_part.type() was here, if it asserts look the type
		remap(src, dst, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
	}
}
