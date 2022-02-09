#pragma once 
#include "../dep/Dep.h"

void remap(Matf _src, Matf _dst,
	Matf _map1, Matf _map2,
	int interpolation, int borderType, const Scalar& borderValue);

void initUndistortRectifyMap(const Matf& _cameraMatrix, const Matf& _distCoeffs, // i presume that we are working with cv_32FC1, because of the previous assigment
	const Matf& matR, const Matf& _newCameraMatrix,
	int size, int height, int width, int m1type, Matf& map1, Matf& map2) // mat2 has to be from floats -> cv_32FC1
{
	assert(m1type == CV_16SC2 || m1type == CV_32FC1 || m1type == CV_32FC2);


	Mat<float> R(3, 3);
	R = eye<float>(3, 3);
	//Mat <float> R(3,3) = Mat<double>::eye(3, 3); // why would we pass the distCoeffs if we create indentity matrix
	Matf distCoeffs;
	Matf A = _cameraMatrix;
	Matf Ar;

	if (isNull<float>(_newCameraMatrix))
		Ar = _newCameraMatrix;
	else
		//Ar = getDefaultNewCameraMatrix(A, size, true);

		if (isNull<float>(matR))
			R = matR;

	if (isNull<float>(_distCoeffs))
		distCoeffs = _distCoeffs;
	else
	{
		distCoeffs.create(8, 1);
	}

	assert(A.size() == Size(3, 3) && A.size() == R.size());
	//CV_Assert(Ar.size() == Size(3, 3) || Ar.size() == Size(4, 3));
	Matf iR = (Ar.colRange(0, 3) * R).inv(DECOMP_LU);//TODO: make colRange and rowRange
	const double* ir = &iR(0, 0);

	double u0 = A(0, 2), v0 = A(1, 2);
	double fx = A(0, 0), fy = A(1, 1);

	assert(distCoeffs.size() == Size(1, 4) || distCoeffs.size() == Size(4, 1) ||
		distCoeffs.size() == Size(1, 5) || distCoeffs.size() == Size(5, 1) ||
		distCoeffs.size() == Size(1, 8) || distCoeffs.size() == Size(8, 1));

	if (distCoeffs.rows != 1 && !distCoeffs.isContinuous()) // we are always using contunuous memory, because of the vector TODO: replace with 0
		distCoeffs = distCoeffs.t();

	float k1 = ((float*)distCoeffs.hasData())[0];
	float k2 = ((float*)distCoeffs.hasData())[1];
	float p1 = ((float*)distCoeffs.hasData())[2];
	float p2 = ((float*)distCoeffs.hasData())[3];
	float k3 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 5 ? ((float*)distCoeffs.hasData())[4] : 0.;//TODO:
	float k4 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 8 ? ((float*)distCoeffs.hasData())[5] : 0.;
	float k5 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 8 ? ((float*)distCoeffs.hasData())[6] : 0.;
	float k6 = distCoeffs.cols() + distCoeffs.rows() - 1 >= 8 ? ((float*)distCoeffs.hasData())[7] : 0.;

	for (int i = 0; i < height; i++)
	{
		float* m1f = (float*)(map1.data + map1.step * i);
		float* m2f = (float*)(map2.data + map2.step * i);
		int16_t* m1 = (int16_t*)m1f;
		uint16_t* m2 = (uint16_t*)m2f;
		float _x = i * ir[1] + ir[2], _y = i * ir[4] + ir[5], _w = i * ir[7] + ir[8];

		for (int j = 0; j < width; j++, _x += ir[0], _y += ir[3], _w += ir[6])
		{
			float w = 1. / _w, x = _x * w, y = _y * w;
			float x2 = x * x, y2 = y * y;
			float r2 = x2 + y2, _2xy = 2 * x * y;
			float kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) * r2);
			float u = fx * (x * kr + p1 * _2xy + p2 * (r2 + 2 * x2)) + u0;
			float v = fy * (y * kr + p1 * (r2 + 2 * y2) + p2 * _2xy) + v0;
			if (m1type == CV_16SC2)
			{
				int iu = saturate_cast<int>(u * INTER_TAB_SIZE);
				int iv = saturate_cast<int>(v * INTER_TAB_SIZE);
				m1[j * 2] = (short)(iu >> INTER_BITS);
				m1[j * 2 + 1] = (short)(iv >> INTER_BITS);
				m2[j] = (ushort)((iv & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (iu & (INTER_TAB_SIZE - 1)));
			}
			else if (m1type == CV_32FC1)
			{
				m1f[j] = (float)u;
				m2f[j] = (float)v;
			}
			else
			{
				m1f[j * 2] = (float)u;
				m1f[j * 2 + 1] = (float)v;
			}
		}
	}
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

	CV_Assert(!_map1.empty());
	CV_Assert(_map2.empty() || (_map2.size() == _map1.size()));

	CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
		ocl_remap(_src, _dst, _map1, _map2, interpolation, borderType, borderValue))

		Mat src = _src.getMat(), map1 = _map1.getMat(), map2 = _map2.getMat();
	_dst.create(map1.size(), src.type());
	Mat dst = _dst.getMat();


	/*CV_OVX_RUN(
		src.type() == CV_8UC1 && dst.type() == CV_8UC1 &&
		!ovx::skipSmallImages<VX_KERNEL_REMAP>(src.cols, src.rows) &&
		(borderType & ~BORDER_ISOLATED) == BORDER_CONSTANT &&
		((map1.type() == CV_32FC2 && map2.empty() && map1.size == dst.size) ||
			(map1.type() == CV_32FC1 && map2.type() == CV_32FC1 && map1.size == dst.size && map2.size == dst.size) ||
			(map1.empty() && map2.type() == CV_32FC2 && map2.size == dst.size)) &&
		((borderType & BORDER_ISOLATED) != 0 || !src.isSubmatrix()),
		openvx_remap(src, dst, map1, map2, interpolation, borderValue));

	CV_Assert(dst.cols < SHRT_MAX&& dst.rows < SHRT_MAX&& src.cols < SHRT_MAX&& src.rows < SHRT_MAX);*/

	if (dst.data == src.data)
		src = src.clone();

	if (interpolation == INTER_AREA)
		interpolation = INTER_LINEAR;

	int type = src.type(), depth = CV_MAT_DEPTH(type);

	

	RemapNNFunc nnfunc = 0;
	RemapFunc ifunc = 0;
	const void* ctab = 0;
	bool fixpt = depth == CV_8U;
	bool planar_input = false;


	{
		if (interpolation == INTER_LINEAR)
			ifunc = linear_tab[depth];
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



std::vector<cv::Point2f> kpts2vec(const keypoints_t& kpts) {
	std::vector<cv::Point2f> result;
	result.reserve(kpts.length());
	for (uint64_t i = 0; i < kpts.length(); ++i) {
		result.emplace_back(kpts[i].x, kpts[i].y);
	}
	return result;
}

void vec2kpts(const std::vector<cv::Point2f>& vec, keypoints_t& kpts) {
	kpts.resize(vec.size());
	for (uint64_t i = 0; i < kpts.length(); ++i) {
		kpts[i].x = vec[i].x;
		kpts[i].y = vec[i].y;
	}
}


void undistort(InputArray _src, OutputArray _dst, InputArray _cameraMatrix, // this is used 
	InputArray _distCoeffs, InputArray _newCameraMatrix)
{
	CV_INSTRUMENT_REGION();

	Mat src = _src.getMat(), cameraMatrix = _cameraMatrix.getMat();
	Mat distCoeffs = _distCoeffs.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

	_dst.create(src.size(), src.type());
	Mat dst = _dst.getMat();

	CV_Assert(dst.data != src.data);

	int stripe_size0 = std::min(std::max(1, (1 << 12) / std::max(src.cols, 1)), src.rows);
	Mat map1(stripe_size0, src.cols, CV_16SC2), map2(stripe_size0, src.cols, CV_16UC1);

	Mat_<double> A, Ar, I = Mat_<double>::eye(3, 3);

	cameraMatrix.convertTo(A, CV_64F);
	if (!distCoeffs.empty())
		distCoeffs = Mat_<double>(distCoeffs);
	else
	{
		distCoeffs.create(5, 1, CV_64F);
		distCoeffs = 0.;
	}

	if (!newCameraMatrix.empty())
		newCameraMatrix.convertTo(Ar, CV_64F);
	else
		A.copyTo(Ar);

	double v0 = Ar(1, 2);
	for (int y = 0; y < src.rows; y += stripe_size0)
	{
		int stripe_size = std::min(stripe_size0, src.rows - y);
		Ar(1, 2) = v0 - y;
		Mat map1_part = map1.rowRange(0, stripe_size),
			map2_part = map2.rowRange(0, stripe_size),
			dst_part = dst.rowRange(y, y + stripe_size);

		initUndistortRectifyMap(A, distCoeffs, I, Ar, Size(src.cols, stripe_size),
			map1_part.type(), map1_part, map2_part);
		remap(src, dst_part, map1_part, map2_part, INTER_LINEAR, BORDER_CONSTANT);
	}
}

class CV_EXPORTS TermCriteria
{
public:
	/**
	  Criteria type, can be one of: COUNT, EPS or COUNT + EPS
	*/
	enum Type
	{
		COUNT = 1, //!< the maximum number of iterations or elements to compute
		MAX_ITER = COUNT, //!< ditto
		EPS = 2 //!< the desired accuracy or change in parameters at which the iterative algorithm stops
	};
	TermCriteria();

	TermCriteria(int type, int maxCount, double epsilon);

	inline bool isValid() const
	{
		const bool isCount = (type & COUNT) && maxCount > 0;
		const bool isEps = (type & EPS) && !cvIsNaN(epsilon);
		return isCount || isEps;
	}

	int type; //!< the type of termination criteria: COUNT, EPS or COUNT + EPS
	int maxCount; //!< the maximum number of iterations/elements
	double epsilon; //!< the desired accuracy
};

static void cvUndistortPointsInternal(const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix,
	const CvMat* _distCoeffs,
	const CvMat* matR, const CvMat* matP, cv::TermCriteria criteria)
{
	CV_Assert(criteria.isValid());
	double A[3][3], RR[3][3], k[14] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	CvMat matA = cvMat(3, 3, CV_64F, A), _Dk; 
	CvMat _RR = cvMat(3, 3, CV_64F, RR);
	cv::Matx33d invMatTilt = cv::Matx33d::eye();
	cv::Matx33d matTilt = cv::Matx33d::eye();

	CV_Assert(CV_IS_MAT(_src) && CV_IS_MAT(_dst) &&
		(_src->rows == 1 || _src->cols == 1) &&
		(_dst->rows == 1 || _dst->cols == 1) &&
		_src->cols + _src->rows - 1 == _dst->rows + _dst->cols - 1 &&
		(CV_MAT_TYPE(_src->type) == CV_32FC2 || CV_MAT_TYPE(_src->type) == CV_64FC2) && // CV_32FC2
		(CV_MAT_TYPE(_dst->type) == CV_32FC2 || CV_MAT_TYPE(_dst->type) == CV_64FC2));  // CV_32FC2

	CV_Assert(
		_cameraMatrix->rows == 3 && _cameraMatrix->cols == 3);

	cvConvert(_cameraMatrix, &matA);


	//if (_distCoeffs)
	//{
		CV_Assert(CV_IS_MAT(_distCoeffs) &&
			(_distCoeffs->rows == 1 || _distCoeffs->cols == 1) &&
			(_distCoeffs->rows * _distCoeffs->cols == 4 ||
				_distCoeffs->rows * _distCoeffs->cols == 5 ||
				_distCoeffs->rows * _distCoeffs->cols == 8 ||
				_distCoeffs->rows * _distCoeffs->cols == 12 ||
				_distCoeffs->rows * _distCoeffs->cols == 14));

		_Dk = cvMat(_distCoeffs->rows, _distCoeffs->cols,
			CV_MAKETYPE(CV_64F, CV_MAT_CN(_distCoeffs->type)), k);

		cvConvert(_distCoeffs, &_Dk);
		if (k[12] != 0 || k[13] != 0)
		{
			cv::detail::computeTiltProjectionMatrix<double>(k[12], k[13], NULL, NULL, NULL, &invMatTilt);//TODO: defined in another header
			cv::detail::computeTiltProjectionMatrix<double>(k[12], k[13], &matTilt, NULL, NULL);
		}

	if (matR)// always
	{
		CV_Assert(CV_IS_MAT(matR) && matR->rows == 3 && matR->cols == 3);
		cvConvert(matR, &_RR);
	}
	else
		cvSetIdentity(&_RR);

	if (matP)
	{
		double PP[3][3];
		CvMat _P3x3, _PP = cvMat(3, 3, CV_64F, PP);
		CV_Assert(CV_IS_MAT(matP) && matP->rows == 3 && (matP->cols == 3 || matP->cols == 4));
		cvConvert(cvGetCols(matP, &_P3x3, 0, 3), &_PP);
		cvMatMul(&_PP, &_RR, &_RR);
	}

	const CvPoint2D32f* srcf = (const CvPoint2D32f*)_src->data.ptr;
	//const CvPoint2D64f* srcd = (const CvPoint2D64f*)_src->data.ptr;
	CvPoint2D32f* dstf = (CvPoint2D32f*)_dst->data.ptr;
	//CvPoint2D64f* dstd = (CvPoint2D64f*)_dst->data.ptr;
	int stype = CV_MAT_TYPE(_src->type);
	int dtype = CV_MAT_TYPE(_dst->type);
	int sstep = _src->rows == 1 ? 1 : _src->step / CV_ELEM_SIZE(stype);
	int dstep = _dst->rows == 1 ? 1 : _dst->step / CV_ELEM_SIZE(dtype);

	float fx = A[0][0];
	float fy = A[1][1];
	float ifx = 1. / fx;
	float ify = 1. / fy;
	float cx = A[0][2];
	float cy = A[1][2];

	int n = _src->rows + _src->cols - 1;
	for (int i = 0; i < n; i++)
	{
		float x, y, x0 = 0, y0 = 0, u, v;
		if (stype == CV_32FC2)//here
		{
			x = srcf[i * sstep].x; //datapointer
			y = srcf[i * sstep].y;
		}
		else
		{
			x = srcd[i * sstep].x;
			y = srcd[i * sstep].y;
		}
		u = x; v = y;
		x = (x - cx) * ifx;
		y = (y - cy) * ify;

		if (_distCoeffs) {
			// compensate tilt distortion
			cv::Vec3d vecUntilt = invMatTilt * cv::Vec3d(x, y, 1);//TODO: vec3 invMatTilt is identity matrix, why would we need to multiply by it
			float invProj = vecUntilt(2) ? 1. / vecUntilt(2) : 1;
			x0 = x = invProj * vecUntilt(0);
			y0 = y = invProj * vecUntilt(1);

			float error = std::numeric_limits<float>::max();
			// compensate distortion iteratively

			for (int j = 0; ; j++)
			{
				if ((criteria.type & cv::TermCriteria::COUNT) && j >= criteria.maxCount)
					break;
				if ((criteria.type & cv::TermCriteria::EPS) && error < criteria.epsilon)
					break;
				double r2 = x * x + y * y;
				double icdist = (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) / (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
				if (icdist < 0)  // test: undistortPoints.regression_14583
				{
					x = (u - cx) * ifx;
					y = (v - cy) * ify;
					break;
				}
				double deltaX = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x) + k[8] * r2 + k[9] * r2 * r2;
				double deltaY = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y + k[10] * r2 + k[11] * r2 * r2;
				x = (x0 - deltaX) * icdist;
				y = (y0 - deltaY) * icdist;

				if (criteria.type & cv::TermCriteria::EPS)
				{
					double r4, r6, a1, a2, a3, cdist, icdist2;
					double xd, yd, xd0, yd0;
					cv::Vec3d vecTilt;

					r2 = x * x + y * y;
					r4 = r2 * r2;
					r6 = r4 * r2;
					a1 = 2 * x * y;
					a2 = r2 + 2 * x * x;
					a3 = r2 + 2 * y * y;
					cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
					icdist2 = 1. / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6);
					xd0 = x * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4;
					yd0 = y * cdist * icdist2 + k[2] * a3 + k[3] * a1 + k[10] * r2 + k[11] * r4;

					vecTilt = matTilt * cv::Vec3d(xd0, yd0, 1);
					invProj = vecTilt(2) ? 1. / vecTilt(2) : 1;
					xd = invProj * vecTilt(0);
					yd = invProj * vecTilt(1);

					double x_proj = xd * fx + cx;
					double y_proj = yd * fy + cy;

					error = sqrt(pow(x_proj - u, 2) + pow(y_proj - v, 2));
				}
			}
		}

		double xx = RR[0][0] * x + RR[0][1] * y + RR[0][2];
		double yy = RR[1][0] * x + RR[1][1] * y + RR[1][2];
		double ww = 1. / (RR[2][0] * x + RR[2][1] * y + RR[2][2]);
		x = xx * ww;
		y = yy * ww;

		if (dtype == CV_32FC2)
		{
			dstf[i * dstep].x = (float)x;
			dstf[i * dstep].y = (float)y;
		}
		else
		{
			dstd[i * dstep].x = x;
			dstd[i * dstep].y = y;
		}
	}
}

void cvUndistortPoints(const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix,
	const CvMat* _distCoeffs,
	const CvMat* matR, const CvMat* matP)
{
	cvUndistortPointsInternal(_src, _dst, _cameraMatrix, _distCoeffs, matR, matP,
		cv::TermCriteria(cv::TermCriteria::COUNT, 5, 0.01));
}

void UndistortPoints(InputArray _src, OutputArray _dst,
	InputArray _cameraMatrix,
	InputArray _distCoeffs,
	InputArray _Rmat,
	InputArray _Pmat,
	TermCriteria criteria)
{
	Mat src = _src.getMat(), cameraMatrix = _cameraMatrix.getMat();
	Mat distCoeffs = _distCoeffs.getMat(), R = _Rmat.getMat(), P = _Pmat.getMat();

	int npoints = src.checkVector(2), depth = src.depth();
	if (npoints < 0)
		src = src.t();
	npoints = src.checkVector(2);
	CV_Assert(npoints >= 0 && src.isContinuous() && (depth == CV_32F || depth == CV_64F));

	if (src.cols == 2)
		src = src.reshape(2);

	_dst.create(npoints, 1, CV_MAKETYPE(depth, 2), -1, true);
	Mat dst = _dst.getMat();

	CvMat _csrc = cvMat(src), _cdst = cvMat(dst), _ccameraMatrix = cvMat(cameraMatrix);
	CvMat matR, matP, _cdistCoeffs, * pR = 0, * pP = 0, * pD = 0;
	if (!R.empty())
		pR = &(matR = cvMat(R));
	if (!P.empty())
		pP = &(matP = cvMat(P));
	if (!distCoeffs.empty())
		pD = &(_cdistCoeffs = cvMat(distCoeffs));
	cvUndistortPointsInternal(&_csrc, &_cdst, &_ccameraMatrix, pD, pR, pP, criteria);
}


void UndistortPoints(const keypoints_t& points, keypoints_t& points_undistorted, const cv::Matx33d& K,
	const cv::Matx14d& distortion) {
	nvtx_scope s("undistort");
	points_undistorted.resize(points.length());
	if (points.empty()) {
		return;
	}
	std::vector<cv::Point2f> points_v = kpts2vec(points), points_undistorted_v;
	cv::undistortPoints(points_v, points_undistorted_v, K, distortion, cv::noArray(), K);
	vec2kpts(points_undistorted_v, points_undistorted);
}


__host__
void Cuda::undistortPoints(cudaStream_t providedStream, const Mat<float>& K, const Mat<float>& distortion)
{
	UndistortPoints(points_v, points_undistorted_v, K, distortion, cv::noArray(), K); // TODO: serach what is this func doing
	vec2kpts(points_undistorted_v, points_undistorted);
}