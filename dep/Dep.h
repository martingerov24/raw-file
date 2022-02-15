#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
template<typename T>
struct Point2
{
	Point2() = default;
	Point2(T x, T y)
		:x(x), y(y) {};
	T x;
	T y;
};

typedef Point2<float>   Point2f;
typedef Point2<int32_t> Point2i;


template<typename T>
struct Point3
{
	Point3(T x, T y, T z)
		:x(x), y(y), z(z) {};
	T x;
	T y;
	T z;
};

typedef Point3<float> Point3f;
typedef Point3<int32_t> Point3i;

template<class T>
class Mat
{
public:

	Mat() : m_cols(0), m_rows(0), m_step(0), m_channels(0), m_owner(true) {}

	Mat(int x, int y, std::vector<T> &matrix)
	{
		m_cols = x;
		m_rows = y;
		m_channels = 1;
		std::swap(m_matrix, matrix);
		m_step = m_channels * m_cols * sizeof(T);
		m_owner = false;
	}
	//Mat(Mat const& src, const Rect& roi)
	//	: Mat(src)
	//{
	//	m_rows = roi.height;
	//	m_cols = roi.width;
	//	m_matrix.resize(roi.y* roi.x);
	//}
	Mat(int x, int y, int other_channels = 1)
	{
		create(x, y, other_channels);
	}

	Mat(const Mat& other)
	{
		*this = other;
	};

	Mat& operator=(const Mat& other)
	{
		*this = other;
		return *this;
	}

	Mat& operator=(Mat&& other)
	{
		std::swap(*this, other);
		return *this;
	}
	bool operator!=(const Mat<T>& other)
	{
		return m_matrix == other.m_matrix;
	}
	bool operator==(const Mat<T>& other) // the same as the top
	{
		return m_matrix == other.m_matrix;
	}

	unsigned char* ptr(int row, int col = 0)
	{
		return const_cast<unsigned char*>(const_cast<const Mat*>(this)->ptr(row, col)); // this was copied from opencv, why is it uchar tho...idk
	}

	const unsigned char* ptr(int row, int col = 0) const
	{
		return m_matrix.data() + m_step * row + sizeof(T) * col;
	}
	bool isContinuous()
	{
		return true;// because it is invoked many times
	}
	Mat(Mat&& other) 
	{
		m_cols = other.m_cols;
		m_rows = other.m_rows;
		m_step = other.m_step;
		m_channels = other.m_channels;
		m_owner = other.m_owner;
		std::swap(m_matrix, other.m_matrix);


		other.m_cols = 0;
		other.m_rows = 0;
		other.m_step = 0;
		other.m_channels = 0;
		other.m_owner = false;
	}
	/*int checkVector(int _elemChannels, int _depth, bool _requireContinuous) const
	{
		return data && (depth() == _depth || _depth <= 0) &&
			(isContinuous() || !_requireContinuous) &&
			((dims == 2 && (((rows == 1 || cols == 1) && channels() == _elemChannels) ||
				(cols == _elemChannels && channels() == 1))) ||
				(dims == 3 && channels() == 1 && size.p[2] == _elemChannels && (size.p[0] == 1 || size.p[1] == 1) &&
					(isContinuous() || step.p[1] == step.p[2] * size.p[2])))
			? (int)(total() * channels() / _elemChannels) : -1;
	}*/
	void create(int x, int y, int other_channels = 1)
	{
		m_cols = x;
		m_rows = y;
		m_channels = other_channels;
		m_owner = true;
	
		//mm::deallocate(m_matrix);
		const uint64_t bytesPerLine = m_channels * m_cols * sizeof(T);
		m_step = bytesPerLine;
		m_matrix.resize(m_cols * m_rows * m_channels,0); 
		// we create a matrix with size of the first, filled with 0
		//mm::allocate(reinterpret_cast<void**>(&m_matrix), uint64_t(m_step) * m_rows);
	}

	T& operator[](int where)
	{
		return m_matrix[where];
	}
	int size()
	{
		return m_cols * m_rows;
	}
	bool hasData()
	{
		return this->m_matrix.size() == 0 ? 0 : 1;
	}
	Mat transpose()
	{
		Mat mat(m_cols, m_rows);
		for (int i = 0; i < m_rows; i++)
		{
			for (int k = 0; k < m_cols; k++)
			{
				mat.m_matrix[i + m_cols * k] = m_matrix[k * i + k];
			}
		}
		return std::move(mat);
	}
	T* data()
	{
		return m_matrix.data();
	}
	void createPitched(int x, int y) {
		m_cols = x;
		m_rows = y;
		m_channels = 1;
		m_owner = true;

		//mm::deallocate(m_matrix);

		// The value of texturePitchAlignment should match the value returned by cudaDeviceProp::textureAlignment
		static const uint64_t texturePitchAlignment = 512;
		const uint64_t bytesPerLine = m_channels * m_cols * sizeof(T);
		uint64_t pitch = bytesPerLine;
		if (bytesPerLine % texturePitchAlignment != 0) {
			pitch = (bytesPerLine + texturePitchAlignment - 1) / texturePitchAlignment;
			pitch *= texturePitchAlignment;
		}
		m_step = pitch;

		//mm::allocate(reinterpret_cast<void**>(&m_matrix), uint64_t(m_step) * m_rows);
	}

	~Mat() = default;

	int32_t cols() const { return m_cols; }
	int32_t rows() const { return m_rows; }
	int32_t step() const { return m_step; }
	int32_t channels() const { return m_channels; }

	//T* ptr(int y = 0) { return reinterpret_cast<T*>(reinterpret_cast<char*>(m_matrix) + y * step()); }
	//const T* ptr(int y = 0) const { return reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_matrix) + y * step()); }

	//T operator ()(int y, int x) { return ptr(y)[x]; }
	//const T operator ()(int y, int x) const { return ptr(y)[x]; }


	Mat& operator*(const float a) noexcept
	{
		return operator*=(a);
	}



	Mat& operator*=(const Mat& a) noexcept
	{
		Mat res(m_cols, a.m_rows);

		for (int i = 0; i < m_rows; i++)
		{
			for (int j = 0; j < a.m_cols; j++)
			{
				res[i * j + j] = 0;

				for (int k = 0; k < a.m_rows; k++)
				{
					res.m_matrix[i * j + j] += a.m_matrix[i * k + k] * m_matrix[k*j+j];
				}
			}
		}
		return std::move(res);
	}
	Mat& operator*(const Mat& a) noexcept
	{
		return operator*=(a);
	}


	Mat& operator+(const Mat& a) noexcept
	{
		Mat res(m_cols, a.m_rows);

		for (int i = 0; i < m_rows; i++)
		{
			for (int j = 0; j < a.m_cols; j++)
			{
				res[i * j + j] = m_matrix[i * j + j] + a.m_matrix[i * j + j];
			}
		}
		return std::move(res);
	}


	Mat& operator*=(const float a) noexcept
	{
		for (int i = 0; i < m_rows * m_cols; i++) {
			m_matrix[i] *= a;
		}
		return *this;
	}
	float dm2x2(T in[4])
	{
		return in[0] * in[3] - in[1] * in[2];
	}
	Mat matirxOfMinors(const Mat& in)
	{

	}
	//auto dm3x3(const Mat& in)
	//{   // determinant of 3x3 in matrix
	//	return in.m_channels[0] * (in.m_channels[4] * in.m_channels[8] - in.m_channels[5] * in.m_channels[7])
	//		-  in.m_channels[1] * (in.m_channels[3] * in.m_channels[8] - in.m_channels[5] * in.m_channels[6])
	//		+  in.m_channels[2] * (in.m_channels[3] * in.m_channels[7] - in.m_channels[4] * in.m_channels[6]);
	//}
	
	Mat& inv(const Mat<T>& in)
	{
		float dm = dm3x3(in);

	}
	//for now m_matrix will be public, because it is a lot accessed in undisort.h
	std::vector<T> m_matrix;
private:
	//T* m_matrix;
	int32_t m_cols;
	int32_t m_rows;
	int32_t m_step;
	int32_t m_channels;
	bool m_owner;
};
template<typename T>
inline bool isNull(const Mat<T>& mat)
{
	return mat.m_matrix.size() == 0 ? 1 : 0;
}

template<typename T>
Mat<T> inline eye(int m, int n) //TODO: if that does not work, try removing the inline
{
	Mat<T>M(m, n);
	for (int i = 0; i < std::min(m, n); i++)
	{
		M[n * i + i] = 1; // if i am not mistaken those are the maths
	}
	return std::move(M);
}
typedef Mat<float> Matf;
typedef Mat<int> Mati;
typedef Mat<int16_t> Mat16s;
typedef Mat<uint16_t> Mat16u;

//template<typename T>
//inline unsigned char* ptr(const Mat<T>& m, int row, int col = 0)
//{
//	return const_cast<unsigned char*>(const_cast<const Mat*>(this)->ptr(m, row, col)); // this was copied from opencv, why is it uchar tho...idk
//}
//
//template<typename T>
//inline const T* ptr(const Mat<T>& m, int row, int col = 0)
//{
//	return m.m_matrix.data() + m.m_step * row + sizeof(T) * col;
//}

template<class T>
class MatView
{
public:

	MatView(int x, int y, int step, const T* ptr) : m_cols(x), m_rows(y), m_step(step), m_matrix(const_cast<T*>(ptr))
	{
	}

	~MatView() = default;

	__host__ __device__ int32_t rows() const { return m_rows; }
	__host__ __device__ int32_t cols() const { return m_cols; }
	__host__ __device__ int32_t step() const { return m_step; }


	__host__ __device__ T* ptr(int y = 0) { return reinterpret_cast<T*>(reinterpret_cast<char*>(m_matrix) + y * step()); }
	__host__ __device__ const T* ptr(int y = 0) const { return reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_matrix) + y * step()); }

	__host__ __device__ T& operator ()(int y, int x) { return ptr(y)[x]; }
	__host__ __device__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }

private:
	int32_t m_cols;
	int32_t m_rows;
	int32_t m_step;
	T* m_matrix;
};

