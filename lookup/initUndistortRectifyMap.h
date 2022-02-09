void cv::omnidir::initUndistortRectifyMap(InputArray K, InputArray D, InputArray xi, InputArray R, InputArray P,
	const cv::Size& size, int m1type, OutputArray map1, OutputArray map2, int flags)
{
	CV_Assert(m1type == CV_16SC2 || m1type == CV_32F || m1type <= 0);
	map1.create(size, m1type <= 0 ? CV_16SC2 : m1type);
	map2.create(size, map1.type() == CV_16SC2 ? CV_16UC1 : CV_32F);

	CV_Assert((K.depth() == CV_32F || K.depth() == CV_64F) && (D.depth() == CV_32F || D.depth() == CV_64F));
	CV_Assert(K.size() == Size(3, 3) && (D.empty() || D.total() == 4));
	CV_Assert(P.empty() || (P.depth() == CV_32F || P.depth() == CV_64F));
	CV_Assert(P.empty() || P.size() == Size(3, 3) || P.size() == Size(4, 3));
	CV_Assert(R.empty() || (R.depth() == CV_32F || R.depth() == CV_64F));
	CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
	CV_Assert(flags == RECTIFY_PERSPECTIVE || flags == RECTIFY_CYLINDRICAL || flags == RECTIFY_LONGLATI
		|| flags == RECTIFY_STEREOGRAPHIC);
	CV_Assert(xi.total() == 1 && (xi.depth() == CV_32F || xi.depth() == CV_64F));

	cv::Vec2d f, c;
	double s;
	if (K.depth() == CV_32F)
	{
		Matx33f camMat = K.getMat();
		f = Vec2f(camMat(0, 0), camMat(1, 1));
		c = Vec2f(camMat(0, 2), camMat(1, 2));
		s = (double)camMat(0, 1);
	}
	else
	{
		Matx33d camMat = K.getMat();
		f = Vec2d(camMat(0, 0), camMat(1, 1));
		c = Vec2d(camMat(0, 2), camMat(1, 2));
		s = camMat(0, 1);
	}

	Vec4d kp = Vec4d::all(0);
	if (!D.empty())
		kp = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>() : *D.getMat().ptr<Vec4d>();
	double _xi = xi.depth() == CV_32F ? (double)*xi.getMat().ptr<float>() : *xi.getMat().ptr<double>();
	Vec2d k = Vec2d(kp[0], kp[1]);
	Vec2d p = Vec2d(kp[2], kp[3]);
	cv::Matx33d RR = cv::Matx33d::eye();
	if (!R.empty() && R.total() * R.channels() == 3)
	{
		cv::Vec3d rvec;
		R.getMat().convertTo(rvec, CV_64F);
		cv::Rodrigues(rvec, RR);
	}
	else if (!R.empty() && R.size() == Size(3, 3))
		R.getMat().convertTo(RR, CV_64F);

	cv::Matx33d PP = cv::Matx33d::eye();
	if (!P.empty())
		P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
	else
		PP = K.getMat();

	cv::Matx33d iKR = (PP * RR).inv(cv::DECOMP_SVD);
	cv::Matx33d iK = PP.inv(cv::DECOMP_SVD);
	cv::Matx33d iR = RR.inv(cv::DECOMP_SVD);

	if (flags == omnidir::RECTIFY_PERSPECTIVE)
	{
		for (int i = 0; i < size.height; ++i)
		{
			float* m1f = map1.getMat().ptr<float>(i);
			float* m2f = map2.getMat().ptr<float>(i);
			short* m1 = (short*)m1f;
			ushort* m2 = (ushort*)m2f;

			double _x = i * iKR(0, 1) + iKR(0, 2),
				_y = i * iKR(1, 1) + iKR(1, 2),
				_w = i * iKR(2, 1) + iKR(2, 2);
			for (int j = 0; j < size.width; ++j, _x += iKR(0, 0), _y += iKR(1, 0), _w += iKR(2, 0))
			{
				// project back to unit sphere
				double r = sqrt(_x * _x + _y * _y + _w * _w);
				double Xs = _x / r;
				double Ys = _y / r;
				double Zs = _w / r;
				// project to image plane
				double xu = Xs / (Zs + _xi),
					yu = Ys / (Zs + _xi);
				// add distortion
				double r2 = xu * xu + yu * yu;
				double r4 = r2 * r2;
				double xd = (1 + k[0] * r2 + k[1] * r4) * xu + 2 * p[0] * xu * yu + p[1] * (r2 + 2 * xu * xu);
				double yd = (1 + k[0] * r2 + k[1] * r4) * yu + p[0] * (r2 + 2 * yu * yu) + 2 * p[1] * xu * yu;
				// to image pixel
				double u = f[0] * xd + s * yd + c[0];
				double v = f[1] * yd + c[1];

				if (m1type == CV_16SC2)
				{
					int iu = cv::saturate_cast<int>(u * cv::INTER_TAB_SIZE);
					int iv = cv::saturate_cast<int>(v * cv::INTER_TAB_SIZE);
					m1[j * 2 + 0] = (short)(iu >> cv::INTER_BITS);
					m1[j * 2 + 1] = (short)(iv >> cv::INTER_BITS);
					m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE - 1)) * cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE - 1)));
				}
				else if (m1type == CV_32FC1)
				{
					m1f[j] = (float)u;
					m2f[j] = (float)v;
				}
			}
		}
	}
	else if (flags == omnidir::RECTIFY_CYLINDRICAL || flags == omnidir::RECTIFY_LONGLATI ||
		flags == omnidir::RECTIFY_STEREOGRAPHIC)
		flags == omnidir::RECTIFY_STEREOGRAPHIC)
	{
		for (int i = 0; i < size.height; ++i)
		{
			float* m1f = map1.getMat().ptr<float>(i);
			float* m2f = map2.getMat().ptr<float>(i);
			short* m1 = (short*)m1f;
			ushort* m2 = (ushort*)m2f;

			// for RECTIFY_LONGLATI, theta and h are longittude and latitude
			double theta = i * iK(0, 1) + iK(0, 2),
				h = i * iK(1, 1) + iK(1, 2);

			for (int j = 0; j < size.width; ++j, theta += iK(0, 0), h += iK(1, 0))
			{
				double _xt = 0.0, _yt = 0.0, _wt = 0.0;
				if (flags == omnidir::RECTIFY_CYLINDRICAL)
				{
					//_xt = std::sin(theta);
					//_yt = h;
					//_wt = std::cos(theta);
					_xt = std::cos(theta);
					_yt = std::sin(theta);
					_wt = h;
				}
				else if (flags == omnidir::RECTIFY_LONGLATI)
				{
					_xt = -std::cos(theta);
					_yt = -std::sin(theta) * std::cos(h);
					_wt = std::sin(theta) * std::sin(h);
				}
				else if (flags == omnidir::RECTIFY_STEREOGRAPHIC)
				{
					double a = theta * theta + h * h + 4;
					double b = -2 * theta * theta - 2 * h * h;
					double c2 = theta * theta + h * h - 4;

					_yt = (-b - std::sqrt(b * b - 4 * a * c2)) / (2 * a);
					_xt = theta * (1 - _yt) / 2;
					_wt = h * (1 - _yt) / 2;
				}
				double _x = iR(0, 0) * _xt + iR(0, 1) * _yt + iR(0, 2) * _wt;
				double _y = iR(1, 0) * _xt + iR(1, 1) * _yt + iR(1, 2) * _wt;
				double _w = iR(2, 0) * _xt + iR(2, 1) * _yt + iR(2, 2) * _wt;

				double r = sqrt(_x * _x + _y * _y + _w * _w);
				double Xs = _x / r;
				double Ys = _y / r;
				double Zs = _w / r;
				// project to image plane
				double xu = Xs / (Zs + _xi),
					yu = Ys / (Zs + _xi);
				// add distortion
				double r2 = xu * xu + yu * yu;
				double r4 = r2 * r2;
				double xd = (1 + k[0] * r2 + k[1] * r4) * xu + 2 * p[0] * xu * yu + p[1] * (r2 + 2 * xu * xu);
				double yd = (1 + k[0] * r2 + k[1] * r4) * yu + p[0] * (r2 + 2 * yu * yu) + 2 * p[1] * xu * yu;
				// to image pixel
				double u = f[0] * xd + s * yd + c[0];
				double v = f[1] * yd + c[1];

				if (m1type == CV_16SC2)
				{
					int iu = cv::saturate_cast<int>(u * cv::INTER_TAB_SIZE);
					int iv = cv::saturate_cast<int>(v * cv::INTER_TAB_SIZE);
					m1[j * 2 + 0] = (short)(iu >> cv::INTER_BITS);
					m1[j * 2 + 1] = (short)(iv >> cv::INTER_BITS);
					m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE - 1)) * cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE - 1)));
				}
				else if (m1type == CV_32FC1)
				{
					m1f[j] = (float)u;
					m2f[j] = (float)v;
				}
			}
		}
	}
}

//////////