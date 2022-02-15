#include "../header/CudaClass.h"
#include "../dep/Dep.h"

#define 	CV_CN_SHIFT   3
#define 	CV_DEPTH_MAX   (1 << CV_CN_SHIFT)
#define 	CV_8U   0
#define 	CV_8S   1
#define 	CV_16U   2
#define 	CV_16S   3
#define 	CV_32S   4
#define 	CV_32F   5
#define 	CV_64F   6
#define 	CV_USRTYPE1   7
#define 	CV_MAT_DEPTH_MASK   (CV_DEPTH_MAX - 1)
#define 	CV_MAT_DEPTH(flags)   ((flags) & CV_MAT_DEPTH_MASK)
#define 	CV_MAKETYPE(depth, cn)   (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))

typedef union Cv32suf
{
	int i;
	unsigned u;
	float f;
}
Cv32suf;

inline int cvIsNaN(float value)
{

	Cv32suf ieee754;
	ieee754.f = value;
	return (ieee754.u & 0x7fffffff) > 0x7f800000;
}

class TermCriteria
{
public:
	enum Type
	{
		COUNT = 1, //!< the maximum number of iterations or elements to compute
		MAX_ITER = COUNT, //!< ditto
		EPS = 2 //!< the desired accuracy or change in parameters at which the iterative algorithm stops
	};
	TermCriteria();
	TermCriteria(int type, int maxCount, float epsilon);

	inline bool isValid() const
	{
		const bool isCount = (type & COUNT) && maxCount > 0;
		const bool isEps = (type & EPS) && !cvIsNaN(epsilon);
		return isCount || isEps;
	}

	int type; //!< the type of termination criteria: COUNT, EPS or COUNT + EPS
	int maxCount; //!< the maximum number of iterations/elements
	float epsilon; //!< the desired accuracy
};


inline
TermCriteria::TermCriteria()
	: type(0), maxCount(0), epsilon(0) {}

inline
TermCriteria::TermCriteria(int _type, int _maxCount, float _epsilon)
	: type(_type), maxCount(_maxCount), epsilon(_epsilon) {}

template <typename FLOAT>
void computeTiltProjectionMatrix(FLOAT tauX,
	FLOAT tauY,
	Matf* matTilt = 0,
	Matf* dMatTiltdTauX = 0,
	Matf* dMatTiltdTauY = 0,
	Matf* invMatTilt = 0)
{
	FLOAT cTauX = cos(tauX);
	FLOAT sTauX = sin(tauX);
	FLOAT cTauY = cos(tauY);
	FLOAT sTauY = sin(tauY);
	Mat<FLOAT> matRotX(3, 3, std::vector<FLOAT>{1, 0, 0, 0, cTauX, sTauX, 0, -sTauX, cTauX});
	Mat<FLOAT> matRotY(3, 3, std::vector<FLOAT>{cTauY, 0, -sTauY, 0, 1, 0, sTauY, 0, cTauY});
	Mat<FLOAT> matRotXY = matRotY * matRotX;
	Mat<FLOAT> matProjZ(3, 3, std::vector<FLOAT>{matRotXY[2 * matRotXY.cols() + 2], 0, -matRotXY[2], 0, matRotXY[2 * matRotXY.cols() + 2], -matRotXY[1 * matRotXY.cols() + 2], 0, 0, 1});
	if (matTilt)
	{
		// Matrix for trapezoidal distortion of tilted image sensor
		*matTilt = matProjZ * matRotXY;
	}
	if (dMatTiltdTauX)
	{
		// Derivative with respect to tauX
		Mat<FLOAT> dMatRotXYdTauX(3, 3, std::vector<FLOAT>{0, 0, 0, 0, -sTauX, cTauX, 0, -cTauX, -sTauX});
		Mat<FLOAT> dMatProjZdTauX(3, 3, std::vector<FLOAT>{dMatRotXYdTauX[2 * dMatRotXYdTauX.cols() + 2], 0, -dMatRotXYdTauX[0 * dMatRotXYdTauX.cols() + 2],
			0, dMatRotXYdTauX[2 * dMatRotXYdTauX.cols() + 2], -dMatRotXYdTauX[1 * dMatRotXYdTauX.cols() + 2], 0, 0, 0});
		*dMatTiltdTauX = (matProjZ * dMatRotXYdTauX) + (dMatProjZdTauX * matRotXY);
	}
	if (dMatTiltdTauY)
	{
		// Derivative with respect to tauY
		Mat<FLOAT> dMatRotXYdTauY (3, 3, std::vector<FLOAT>{-sTauY, 0, -cTauY, 0, 0, 0, cTauY, 0, -sTauY});
		dMatRotXYdTauY = dMatRotXYdTauY * matRotX;
		Mat<FLOAT> dMatProjZdTauY(3, 3, std::vector<FLOAT>{dMatRotXYdTauY[2 * dMatRotXYdTauY.cols() + 2], 0, -dMatRotXYdTauY[0 * dMatRotXYdTauY.cols() + 2],
			0, dMatRotXYdTauY[2 * dMatRotXYdTauY.cols() + 2], -dMatRotXYdTauY[1 * dMatRotXYdTauY.cols() + 2], 0, 0, 0});
		*dMatTiltdTauY = (matProjZ * dMatRotXYdTauY) + (dMatProjZdTauY * matRotXY);// this pointer, before the mat ???? btw removed
	}
	if (invMatTilt)
	{
		FLOAT inv = 1. / matRotXY[2 * matRotXY.cols() +2];
		Mat<FLOAT> invMatProjZ(3, 3, std::vector<FLOAT>{inv, 0, inv* matRotXY[0* matRotXY.cols() + 2], 0, inv, inv* matRotXY[1 * matRotXY.cols() + 2], 0, 0, 1});
		*invMatTilt = matRotXY.transpose() * invMatProjZ;
	}
}

static void cvUndistortPointsInternal(const Matf* _src, Matf* _dst, const Matf* _cameraMatrix,
	const Matf* _distCoeffs,
	const Matf* matR, const Matf* matP, TermCriteria criteria)
{
	assert(criteria.isValid());
	std::vector<float> k(14,0);
	Matf matA(3, 3, std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0}),	 _Dk;
	Matf _RR(3, 3, std::vector<float> {0, 0, 0, 0, 0, 0, 0, 0, 0});
	Matf invMatTilt = eye<float>(3,3);
	Matf matTilt = eye<float>	(3,3);

	assert(
		(_src->rows() == 1 || _src->cols() == 1) &&
		(_dst->rows() == 1 || _dst->cols() == 1) &&
		_src->cols()+ _src->rows() - 1 == _dst->rows() + _dst->cols() - 1);

		assert(_cameraMatrix->rows() == 3 && _cameraMatrix->cols() == 3);

	//cvConvert(_cameraMatrix, &matA);// already them both are float


	if (_distCoeffs)
	{
			assert((_distCoeffs->rows() == 1 || _distCoeffs->cols() == 1) &&
			(_distCoeffs->rows() *  _distCoeffs->cols() == 4 ||
				_distCoeffs->rows() * _distCoeffs->cols() == 5 ||
				_distCoeffs->rows() * _distCoeffs->cols() == 8 ||
				_distCoeffs->rows() * _distCoeffs->cols() == 12 ||
				_distCoeffs->rows() * _distCoeffs->cols() == 14));

			_Dk.create(_distCoeffs->rows(), _distCoeffs->cols());

		//cvConvert(_distCoeffs, &_Dk);
		if (k[12] != 0 || k[13] != 0)
		{
			computeTiltProjectionMatrix<float>(k[12], k[13], NULL, NULL, NULL, &invMatTilt); // there ain't way of k being != 0, but
			computeTiltProjectionMatrix<float>(k[12], k[13], &matTilt, NULL, NULL);
		}
	}

	if (matR)
	{
		assert(matR->rows() == 3 && matR->cols() == 3);
		//cvConvert(matR, &_RR);
	}
	else
		//cvSetIdentity(&_RR);

	if (matP)
	{
		//float PP[3][3];
		Matf _P3x3, _PP(3, 3, std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0}); //3x3(PP)
		assert( matP->rows() == 3 && (matP->cols() == 3 || matP->cols() == 4));
		//cvConvert(cvGetCols(matP, &_P3x3, 0, 3), &_PP);// wtf TODO:
		//cvMatMul(&_PP, &_RR, &_RR);
	}
	//const Point2f* srcf = (const Point2f*)_src.data().ptr();
	//const Point2f* srcd = (const Point2f*)_src->data().ptr();
	//Point2f* dstf = (Point2f*)_dst->data().ptr();
	//Point2f* dstd = (Point2f*)_dst->data().ptr();
	//int stype = CV_MAT_TYPE(_src->type);
	//int dtype = CV_MAT_TYPE(_dst->type);
	//int sstep = _src->rows == 1 ? 1 : _src->step / CV_ELEM_SIZE(stype);
	//int dstep = _dst->rows == 1 ? 1 : _dst->step / CV_ELEM_SIZE(dtype);

	//double fx = A[0][0];
	//double fy = A[1][1];
	//double ifx = 1. / fx;
	//double ify = 1. / fy;
	//double cx = A[0][2];
	//double cy = A[1][2];

	//int n = _src->rows + _src->cols - 1;
	//for (int i = 0; i < n; i++)
	//{
	//	double x, y, x0 = 0, y0 = 0, u, v;
	//	if (stype == CV_32FC2)
	//	{
	//		x = srcf[i * sstep].x;
	//		y = srcf[i * sstep].y;
	//	}
	//	else
	//	{
	//		x = srcd[i * sstep].x;
	//		y = srcd[i * sstep].y;
	//	}
	//	u = x; v = y;
	//	x = (x - cx) * ifx;
	//	y = (y - cy) * ify;

	//	if (_distCoeffs) {
	//		// compensate tilt distortion
	//		cv::Vec3d vecUntilt = invMatTilt * cv::Vec3d(x, y, 1);
	//		double invProj = vecUntilt(2) ? 1. / vecUntilt(2) : 1;
	//		x0 = x = invProj * vecUntilt(0);
	//		y0 = y = invProj * vecUntilt(1);

	//		double error = std::numeric_limits<double>::max();
	//		// compensate distortion iteratively

	//		for (int j = 0; ; j++)
	//		{
	//			if ((criteria.type & cv::TermCriteria::COUNT) && j >= criteria.maxCount)
	//				break;
	//			if ((criteria.type & cv::TermCriteria::EPS) && error < criteria.epsilon)
	//				break;
	//			double r2 = x * x + y * y;
	//			double icdist = (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) / (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
	//			if (icdist < 0)  // test: undistortPoints.regression_14583
	//			{
	//				x = (u - cx) * ifx;
	//				y = (v - cy) * ify;
	//				break;
	//			}
	//			double deltaX = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x) + k[8] * r2 + k[9] * r2 * r2;
	//			double deltaY = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y + k[10] * r2 + k[11] * r2 * r2;
	//			x = (x0 - deltaX) * icdist;
	//			y = (y0 - deltaY) * icdist;

	//			if (criteria.type & cv::TermCriteria::EPS)
	//			{
	//				double r4, r6, a1, a2, a3, cdist, icdist2;
	//				double xd, yd, xd0, yd0;
	//				cv::Vec3d vecTilt;

	//				r2 = x * x + y * y;
	//				r4 = r2 * r2;
	//				r6 = r4 * r2;
	//				a1 = 2 * x * y;
	//				a2 = r2 + 2 * x * x;
	//				a3 = r2 + 2 * y * y;
	//				cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
	//				icdist2 = 1. / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6);
	//				xd0 = x * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4;
	//				yd0 = y * cdist * icdist2 + k[2] * a3 + k[3] * a1 + k[10] * r2 + k[11] * r4;

	//				vecTilt = matTilt * cv::Vec3d(xd0, yd0, 1);
	//				invProj = vecTilt(2) ? 1. / vecTilt(2) : 1;
	//				xd = invProj * vecTilt(0);
	//				yd = invProj * vecTilt(1);

	//				double x_proj = xd * fx + cx;
	//				double y_proj = yd * fy + cy;

	//				error = sqrt(pow(x_proj - u, 2) + pow(y_proj - v, 2));
	//			}
	//		}
	//	}

	//	double xx = RR[0][0] * x + RR[0][1] * y + RR[0][2];
	//	double yy = RR[1][0] * x + RR[1][1] * y + RR[1][2];
	//	double ww = 1. / (RR[2][0] * x + RR[2][1] * y + RR[2][2]);
	//	x = xx * ww;
	//	y = yy * ww;

	//	if (dtype == CV_32FC2)
	//	{
	//		dstf[i * dstep].x = (float)x;
	//		dstf[i * dstep].y = (float)y;
	//	}
	//	else
	//	{
	//		dstd[i * dstep].x = x;
	//		dstd[i * dstep].y = y;
	//	}
	//}
}


void undistortPoints(Matf _src, Matf _dst,
	Matf _cameraMatrix,
	Matf _distCoeffs,
	Matf _Rmat, // = 0
	Matf _Pmat, // result
	TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER, 5, 0.01))
{
	// I will use mine because i do not have getMat
	//Mat src = _src.getMat(), cameraMatrix = _cameraMatrix.getMat();

	Matf distCoeffs = _distCoeffs, R = _Rmat, P = _Pmat;
	Matf src = src.transpose(); //makes a full copy

	if (src.cols() == 2)// it won't be
		//src = src.reshape(2);

	_dst = src;

	Matf _csrc = src, _cdst = _dst, _ccameraMatrix = _cameraMatrix; // try passing the real matrices
	Matf matR, matP, _cdistCoeffs, * pR = 0, * pP = 0, * pD = 0;
	if (R.hasData())
		pR = &(matR = R); // wtf
	if (P.hasData())
		pP = &(matP = P);
	if (distCoeffs.hasData())
		pD = &(_cdistCoeffs = distCoeffs);
	cvUndistortPointsInternal(&_csrc, &_cdst, &_ccameraMatrix, pD, pR, pP, criteria);
}
__host__
void undistort(cudaStream_t providedStream, const Matf& points, Matf& points_undistorted, const Matf& K, // K = 3,3
	const Matf& distortion)
{
	Matf k_result = K;
	
	undistortPoints(points, points_undistorted, K, distortion, Matf(0,0), k_result); // TODO: invoke directoly the function
}
///Filter By Index //TODO:


//void InvokeUndistort()
//{
//
//	UndistortPoints(m_keypoints.l_keypoints, m_keypoints.l_points_undistorted, m_cameras.intrinsic_l,
//		m_cameras.distortion_l);
//	UndistortPoints(m_keypoints.r_keypoints2, m_keypoints.r_points_undistorted, m_cameras.intrinsic_r,
//		m_cameras.distortion_r);
//
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
