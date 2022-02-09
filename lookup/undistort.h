template <typename FLOAT>
void computeTiltProjectionMatrix(FLOAT tauX,
	FLOAT tauY,
	Matx<FLOAT, 3, 3>* matTilt = 0,
	Matx<FLOAT, 3, 3>* dMatTiltdTauX = 0,
	Matx<FLOAT, 3, 3>* dMatTiltdTauY = 0,
	Matx<FLOAT, 3, 3>* invMatTilt = 0)
{
	FLOAT cTauX = cos(tauX);
	FLOAT sTauX = sin(tauX);
	FLOAT cTauY = cos(tauY);
	FLOAT sTauY = sin(tauY);
	Matx<FLOAT, 3, 3> matRotX = Matx<FLOAT, 3, 3>(1, 0, 0, 0, cTauX, sTauX, 0, -sTauX, cTauX);
	Matx<FLOAT, 3, 3> matRotY = Matx<FLOAT, 3, 3>(cTauY, 0, -sTauY, 0, 1, 0, sTauY, 0, cTauY);
	Matx<FLOAT, 3, 3> matRotXY = matRotY * matRotX;
	Matx<FLOAT, 3, 3> matProjZ = Matx<FLOAT, 3, 3>(matRotXY(2, 2), 0, -matRotXY(0, 2), 0, matRotXY(2, 2), -matRotXY(1, 2), 0, 0, 1);
	if (matTilt)
	{
		// Matrix for trapezoidal distortion of tilted image sensor
		*matTilt = matProjZ * matRotXY;
	}
	if (dMatTiltdTauX)
	{
		// Derivative with respect to tauX
		Matx<FLOAT, 3, 3> dMatRotXYdTauX = matRotY * Matx<FLOAT, 3, 3>(0, 0, 0, 0, -sTauX, cTauX, 0, -cTauX, -sTauX);
		Matx<FLOAT, 3, 3> dMatProjZdTauX = Matx<FLOAT, 3, 3>(dMatRotXYdTauX(2, 2), 0, -dMatRotXYdTauX(0, 2),
			0, dMatRotXYdTauX(2, 2), -dMatRotXYdTauX(1, 2), 0, 0, 0);
		*dMatTiltdTauX = (matProjZ * dMatRotXYdTauX) + (dMatProjZdTauX * matRotXY);
	}
	if (dMatTiltdTauY)
	{
		// Derivative with respect to tauY
		Matx<FLOAT, 3, 3> dMatRotXYdTauY = Matx<FLOAT, 3, 3>(-sTauY, 0, -cTauY, 0, 0, 0, cTauY, 0, -sTauY) * matRotX;
		Matx<FLOAT, 3, 3> dMatProjZdTauY = Matx<FLOAT, 3, 3>(dMatRotXYdTauY(2, 2), 0, -dMatRotXYdTauY(0, 2),
			0, dMatRotXYdTauY(2, 2), -dMatRotXYdTauY(1, 2), 0, 0, 0);
		*dMatTiltdTauY = (matProjZ * dMatRotXYdTauY) + (dMatProjZdTauY * matRotXY);
	}
	if (invMatTilt)
	{
		FLOAT inv = 1. / matRotXY(2, 2);
		Matx<FLOAT, 3, 3> invMatProjZ = Matx<FLOAT, 3, 3>(inv, 0, inv * matRotXY(0, 2), 0, inv, inv * matRotXY(1, 2), 0, 0, 1);
		*invMatTilt = matRotXY.t() * invMatProjZ;
	}
}
}}





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
		(CV_MAT_TYPE(_src->type) == CV_32FC2 || CV_MAT_TYPE(_src->type) == CV_64FC2) &&
		(CV_MAT_TYPE(_dst->type) == CV_32FC2 || CV_MAT_TYPE(_dst->type) == CV_64FC2));

	CV_Assert(CV_IS_MAT(_cameraMatrix) &&
		_cameraMatrix->rows == 3 && _cameraMatrix->cols == 3);

	cvConvert(_cameraMatrix, &matA);


	if (_distCoeffs)
	{
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
			cv::detail::computeTiltProjectionMatrix<double>(k[12], k[13], NULL, NULL, NULL, &invMatTilt);
			cv::detail::computeTiltProjectionMatrix<double>(k[12], k[13], &matTilt, NULL, NULL);
		}
	}

	if (matR)
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
	const CvPoint2D64f* srcd = (const CvPoint2D64f*)_src->data.ptr;
	CvPoint2D32f* dstf = (CvPoint2D32f*)_dst->data.ptr;
	CvPoint2D64f* dstd = (CvPoint2D64f*)_dst->data.ptr;
	int stype = CV_MAT_TYPE(_src->type);
	int dtype = CV_MAT_TYPE(_dst->type);
	int sstep = _src->rows == 1 ? 1 : _src->step / CV_ELEM_SIZE(stype);
	int dstep = _dst->rows == 1 ? 1 : _dst->step / CV_ELEM_SIZE(dtype);

	double fx = A[0][0];
	double fy = A[1][1];
	double ifx = 1. / fx;
	double ify = 1. / fy;
	double cx = A[0][2];
	double cy = A[1][2];

	int n = _src->rows + _src->cols - 1;
	for (int i = 0; i < n; i++)
	{
		double x, y, x0 = 0, y0 = 0, u, v;
		if (stype == CV_32FC2)
		{
			x = srcf[i * sstep].x;
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
			cv::Vec3d vecUntilt = invMatTilt * cv::Vec3d(x, y, 1);
			double invProj = vecUntilt(2) ? 1. / vecUntilt(2) : 1;
			x0 = x = invProj * vecUntilt(0);
			y0 = y = invProj * vecUntilt(1);

			double error = std::numeric_limits<double>::max();
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