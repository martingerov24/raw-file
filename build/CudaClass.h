#pragma once
#define THREADS_PER_BLOCK 1024
#include "cuda_runtime.h"
#include "cuda/std/cmath"
#include "device_launch_parameters.h"
#include <vector>
#include <cassert>
#include <chrono>

static std::vector<float> keypoints =
{
23.000000, 568.000000,
24.000000, 568.000000,
51.000000, 568.000000,
422.000000, 463.000000,
424.000000, 463.000000,
607.000000, 488.000000,
612.000000, 490.000000,
616.000000, 494.000000,
355.000000, 484.000000,
367.000000, 480.000000,
374.000000, 487.000000,
376.000000, 487.000000,
1350.00000, 228.000000,
607.000000, 486.000000,
608.000000, 485.000000,
423.000000, 467.000000,
424.000000, 467.000000,
63.000000, 551.000000,
64.000000, 550.000000,
77.000000, 544.000000,
612.000000, 496.000000,
616.000000, 496.000000,
624.000000, 498.000000,
70.000000, 495.000000,
74.000000, 495.000000,
62.000000, 504.000000,
64.000000, 504.000000,
72.000000, 504.000000,
76.000000, 541.000000,
295.000000, 568.000000,
297.000000, 568.000000,
319.000000, 575.000000,
320.000000, 573.000000,
333.000000, 568.000000,
342.000000, 575.000000,
350.000000, 573.000000,
352.000000, 573.000000,
364.000000, 568.000000,
46.000000, 530.000000,
48.000000, 530.000000,
671.000000, 552.000000,
672.000000, 552.000000,
701.000000, 552.000000,
704.000000, 554.000000,
295.000000, 566.000000,
296.000000, 567.000000,
304.000000, 562.000000,
334.000000, 564.000000,
336.000000, 564.000000,
364.000000, 563.000000,
368.000000, 563.000000,
357.000000, 488.000000,
367.000000, 491.000000,
373.000000, 489.000000,
376.000000, 489.000000,
70.000000, 496.000000,
74.000000, 496.000000,
550.000000, 639.000000,
552.000000, 639.000000,
561.000000, 636.000000,
574.000000, 639.000000,
579.000000, 638.000000,
584.000000, 639.000000,
634.000000, 632.000000,
653.000000, 632.000000,
656.000000, 632.000000,
667.000000, 637.000000,
672.000000, 638.000000,
693.000000, 639.000000,
696.000000, 639.000000,
707.000000, 639.000000,
719.000000, 636.000000,
721.000000, 638.000000,
15.000000, 563.000000,
23.000000, 567.000000,
24.000000, 567.000000,
52.000000, 565.000000,
62.000000, 560.000000,
64.000000, 562.000000,
1303.000000, 611.000000,
1305.000000, 609.000000,
365.000000, 496.000000,
369.000000, 496.000000,
794.000000, 650.000000,
800.000000, 651.000000,
257.000000, 736.000000,
335.000000, 741.000000,
338.000000, 741.000000,
344.000000, 740.000000,
431.000000, 736.000000,
434.000000, 736.000000,
445.000000, 740.000000,
448.000000, 739.000000,
493.000000, 740.000000,
496.000000, 740.000000,
256.000000, 752.000000,
345.000000, 752.000000,
532.000000, 712.000000,
604.000000, 712.000000,
668.000000, 712.000000,
672.000000, 712.000000,
775.000000, 573.000000,
778.000000, 573.000000,
788.000000, 574.000000,
304.000000, 543.000000,
331.000000, 542.000000,
336.000000, 543.000000,
294.000000, 550.000000,
303.000000, 546.000000,
304.000000, 544.000000,
332.000000, 544.000000,
336.000000, 546.000000,
295.000000, 554.000000,
298.000000, 555.000000,
304.000000, 558.000000,
334.000000, 559.000000,
362.000000, 559.000000,
614.000000, 572.000000,
616.000000, 572.000000,
663.000000, 568.000000,
667.000000, 568.000000,
679.000000, 572.000000,
682.000000, 573.000000,
688.000000, 574.000000,
701.000000, 568.000000,
704.000000, 568.000000,
253.000000, 630.000000,
274.000000, 583.000000,
318.000000, 576.000000,
320.000000, 576.000000,
333.000000, 583.000000,
342.000000, 576.000000,
344.000000, 576.000000,
352.000000, 576.000000,
18.000000, 600.000000,
28.000000, 600.000000,
32.000000, 600.000000,
40.000000, 600.000000,
51.000000, 605.000000,
61.000000, 605.000000,
64.000000, 605.000000,
73.000000, 601.000000,
84.000000, 601.000000,
88.000000, 600.000000,
9.000000, 594.000000,
20.000000, 597.000000,
29.000000, 598.000000,
32.000000, 598.000000,
40.000000, 596.000000,
73.000000, 598.000000,
84.000000, 599.000000,
88.000000, 599.000000,
365.000000, 475.000000,
368.000000, 477.000000,
424.000000, 472.000000,
18.000000, 616.000000,
28.000000, 616.000000,
39.000000, 616.000000,
40.000000, 616.000000,
50.000000, 616.000000,
61.000000, 616.000000,
71.000000, 616.000000,
73.000000, 616.000000,
84.000000, 616.000000,
676.000000, 543.000000,
671.000000, 550.000000,
674.000000, 549.000000,
701.000000, 550.000000,
710.000000, 607.000000,
712.000000, 607.000000,
375.000000, 602.000000,
381.000000, 605.000000,
386.000000, 607.000000,
693.000000, 590.000000,
696.000000, 590.000000,
242.000000, 677.000000,
255.000000, 675.000000,
391.000000, 661.000000,
392.000000, 660.000000,
415.000000, 662.000000,
419.000000, 663.000000,
426.000000, 663.000000,
432.000000, 662.000000,
339.000000, 640.000000,
391.000000, 640.000000,
393.000000, 640.000000,
407.000000, 644.000000,
412.000000, 646.000000,
416.000000, 640.000000,
427.000000, 642.000000,
432.000000, 640.000000,
441.000000, 646.000000,
462.000000, 640.000000,
464.000000, 640.000000,
484.000000, 640.000000,
795.000000, 662.000000,
800.000000, 661.000000,
727.000000, 727.000000,
728.000000, 727.000000,
535.000000, 742.000000,
537.000000, 742.000000,
599.000000, 743.000000,
600.000000, 742.000000,
687.000000, 738.000000,
691.000000, 736.000000,
696.000000, 736.000000,
22.000000, 767.000000,
24.000000, 765.000000,
340.000000, 688.000000,
344.000000, 688.000000,
391.000000, 695.000000,
392.000000, 695.000000,
535.000000, 744.000000,
537.000000, 744.000000,
596.000000, 744.000000,
600.000000, 744.000000,
60.000000, 558.000000,
64.000000, 552.000000,
663.000000, 567.000000,
667.000000, 567.000000,
672.000000, 567.000000,
702.000000, 567.000000,
707.000000, 562.000000,
712.000000, 562.000000,
256.000000, 728.000000,
367.000000, 728.000000,
370.000000, 729.000000,
407.000000, 728.000000,
410.000000, 728.000000,
431.000000, 734.000000,
434.000000, 735.000000,
346.000000, 761.000000,
389.000000, 704.000000,
392.000000, 704.000000,
455.000000, 709.000000,
458.000000, 709.000000,
464.000000, 704.000000,
687.000000, 734.000000,
691.000000, 735.000000,
696.000000, 734.000000,
726.000000, 728.000000,
733.000000, 731.000000,
736.000000, 731.000000,
252.000000, 632.000000,
533.000000, 702.000000,
542.000000, 703.000000,
544.000000, 703.000000,
552.000000, 699.000000,
560.000000, 703.000000,
568.000000, 699.000000,
578.000000, 699.000000,
584.000000, 699.000000,
592.000000, 700.000000,
603.000000, 696.000000,
608.000000, 702.000000,
723.000000, 698.000000,
241.000000, 680.000000,
254.000000, 680.000000,
694.000000, 592.000000,
696.000000, 592.000000,
407.000000, 648.000000,
412.000000, 648.000000,
426.000000, 648.000000,
439.000000, 649.000000,
442.000000, 650.000000,
711.000000, 613.000000,
712.000000, 613.000000,
519.000000, 628.000000,
521.000000, 628.000000,
543.000000, 628.000000,
544.000000, 628.000000,
567.000000, 628.000000,
568.000000, 628.000000,
589.000000, 628.000000,
592.000000, 629.000000,
607.000000, 629.000000,
613.000000, 629.000000,
616.000000, 629.000000,
631.000000, 630.000000,
635.000000, 629.000000,
640.000000, 630.000000,
653.000000, 631.000000,
656.000000, 630.000000,
671.000000, 624.000000,
675.000000, 624.000000,
680.000000, 624.000000,
692.000000, 626.000000,
696.000000, 626.000000,
709.000000, 624.000000,
717.000000, 627.000000,
720.000000, 626.000000,
1159.000000, 620.000000,
1163.000000, 622.000000,
1168.000000, 623.000000,
1159.000000, 624.000000,
1163.000000, 624.000000,
256.000000, 678.000000,
390.000000, 674.000000,
392.000000, 672.000000,
453.000000, 672.000000,
456.000000, 672.000000,
678.000000, 576.000000,
680.000000, 576.000000,
688.000000, 576.000000,
273.000000, 584.000000,
332.000000, 584.000000,
336.000000, 584.000000,
346.000000, 584.000000,
352.000000, 589.000000,
14.000000, 584.000000,
16.000000, 584.000000,
24.000000, 584.000000,
86.000000, 586.000000,
88.000000, 586.000000,
256.000000, 680.000000,
341.000000, 687.000000,
344.000000, 685.000000,
391.000000, 680.000000,
394.000000, 685.000000,
177.000000, 837.000000,
13.000000, 848.000000,
217.000000, 853.000000,
1223.000000, 656.000000,
1224.000000, 657.000000,
723.000000, 677.000000,
257.000000, 748.000000,
335.000000, 744.000000,
338.000000, 744.000000,
346.000000, 747.000000,
421.000000, 745.000000,
424.000000, 746.000000,
444.000000, 744.000000,
494.000000, 744.000000,
496.000000, 744.000000,
375.000000, 618.000000,
379.000000, 618.000000,
384.000000, 617.000000,
399.000000, 619.000000,
403.000000, 620.000000,
408.000000, 620.000000,
423.000000, 623.000000,
426.000000, 621.000000,
439.000000, 620.000000,
446.000000, 621.000000,
452.000000, 621.000000,
456.000000, 621.000000,
532.000000, 687.000000,
716.000000, 686.000000,
722.000000, 680.000000,
390.000000, 702.000000,
392.000000, 698.000000,
460.000000, 703.000000,
464.000000, 700.000000,
390.000000, 671.000000,
392.000000, 671.000000,
415.000000, 664.000000,
419.000000, 664.000000,
425.000000, 665.000000,
432.000000, 664.000000,
454.000000, 670.000000,
456.000000, 670.000000,
256.000000, 632.000000,
340.000000, 638.000000,
391.000000, 639.000000,
394.000000, 638.000000,
411.000000, 637.000000,
416.000000, 637.000000,
428.000000, 638.000000,
432.000000, 638.000000,
455.000000, 637.000000,
462.000000, 639.000000,
464.000000, 639.000000,
472.000000, 632.000000,
484.000000, 637.000000,
497.000000, 632.000000,
252.000000, 726.000000,
255.000000, 748.000000,
533.000000, 708.000000,
538.000000, 704.000000,
544.000000, 704.000000,
556.000000, 704.000000,
560.000000, 704.000000,
577.000000, 704.000000,
597.000000, 704.000000,
600.000000, 706.000000,
668.000000, 710.000000,
672.000000, 710.000000,
723.000000, 704.000000,
147.000000, 824.000000,
15.000000, 582.000000,
23.000000, 578.000000,
25.000000, 578.000000,
85.000000, 582.000000,
88.000000, 582.000000,
671.000000, 623.000000,
675.000000, 623.000000,
680.000000, 623.000000,
691.000000, 623.000000,
696.000000, 622.000000,
711.000000, 616.000000,
712.000000, 616.000000,
720.000000, 622.000000,
793.000000, 664.000000,
800.000000, 664.000000,
202.000000, 968.000000,
549.000000, 643.000000,
552.000000, 642.000000,
561.000000, 640.000000,
575.000000, 647.000000,
576.000000, 645.000000,
584.000000, 640.000000,
673.000000, 640.000000,
693.000000, 640.000000,
696.000000, 640.000000,
707.000000, 640.000000,
719.000000, 640.000000,
720.000000, 640.000000,
18.000000, 613.000000,
29.000000, 612.000000,
39.000000, 613.000000,
40.000000, 613.000000,
50.000000, 614.000000,
61.000000, 615.000000,
64.000000, 608.000000,
72.000000, 615.000000,
83.000000, 615.000000,
219.000000, 610.000000,
256.000000, 726.000000,
366.000000, 722.000000,
369.000000, 725.000000,
407.000000, 727.000000,
410.000000, 727.000000,
431.000000, 726.000000,
433.000000, 727.000000,
440.000000, 720.000000,
533.000000, 691.000000,
536.000000, 691.000000,
546.000000, 690.000000,
556.000000, 689.000000,
560.000000, 689.000000,
568.000000, 691.000000,
603.000000, 688.000000,
716.000000, 688.000000,
720.000000, 688.000000,
695.000000, 663.000000,
698.000000, 663.000000,
704.000000, 663.000000,
720.000000, 663.000000,
1155.000000, 675.000000,
1164.000000, 675.000000,
147.000000, 821.000000,
548.000000, 648.000000,
557.000000, 648.000000,
560.000000, 648.000000,
575.000000, 648.000000,
576.000000, 648.000000,
22.000000, 768.000000,
24.000000, 768.000000,
54.000000, 774.000000,
56.000000, 772.000000,
439.000000, 717.000000,
442.000000, 716.000000,
455.000000, 712.000000,
457.000000, 712.000000,
304.000000, 592.000000,
350.000000, 592.000000,
352.000000, 592.000000,
381.000000, 599.000000,
384.000000, 599.000000,
751.000000, 931.000000,
752.000000, 932.000000,
695.000000, 665.000000,
697.000000, 665.000000,
704.000000, 665.000000,
721.000000, 664.000000,
383.000000, 608.000000,
387.000000, 608.000000,
398.000000, 615.000000,
400.000000, 614.000000,
415.000000, 612.000000,
416.000000, 613.000000,
825.000000, 946.000000,
216.000000, 856.000000,
251.000000, 728.000000,
256.000000, 631.000000,
403.000000, 624.000000,
408.000000, 624.000000,
422.000000, 624.000000,
425.000000, 624.000000,
439.000000, 624.000000,
442.000000, 624.000000,
452.000000, 624.000000,
456.000000, 624.000000,
471.000000, 629.000000,
472.000000, 629.000000,
495.000000, 628.000000,
497.000000, 628.000000,
20.000000, 776.000000,
53.000000, 776.000000,
56.000000, 776.000000,
253.000000, 736.000000,
345.000000, 768.000000,
775.000000, 576.000000,
777.000000, 576.000000,
789.000000, 576.000000,
792.000000, 576.000000,
748.000000, 945.000000,
752.000000, 945.000000,
13.000000, 846.000000,
202.000000, 967.000000,
1102.000000, 1080.000000,
1104.000000, 1080.000000,
980.000000, 1021.000000,
984.000000, 1023.000000,
982.000000, 1024.000000,
984.000000, 1024.000000,
1102.000000, 1078.000000,
1104.000000, 1077.000000 };
class CudaKeypoints
{
public:
	CudaKeypoints(const std::vector<uint8_t>& data,
		const int height, const int width)
		
		: img(data), height(height)
		, width(width), d_image(nullptr)
		, d_result(nullptr), d_kp(nullptr)
	{
		keypointSize = keypoints.size();
		cudaStatus = cudaError_t(0);
		size = height * width;
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		cudaStatus = cudaSetDevice(0);
		assert(cudaStatus == cudaSuccess, "you do not have cuda capable device!");
		cudaStatus = cudaStreamCreate(&stream);
	}

	__host__
		void startup(int size)
	{
		cudaStatus = cudaMalloc((void**)&d_image, sizeof(uint8_t) * size);
		assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");

		cudaStatus = cudaMalloc((void**)&d_result, sizeof(uint8_t) * keypointSize);
		assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");

		cudaStatus = cudaMalloc((void**)&d_kp, sizeof(float) * keypointSize);
		assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");
	}

	__host__
	void Kernel();

	__host__
		void cudaMemcpy()
	{
		cudaStatus = cudaMemcpyAsync(d_image, img.data(), sizeof(uint8_t) * size, cudaMemcpyHostToDevice, stream);
		assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");

		cudaStatus = cudaMemcpyAsync(d_kp, (float2*)keypoints.data(), sizeof(float) * keypointSize, cudaMemcpyHostToDevice, stream);
		assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");
	}
	__host__
		void cudaMemcpyD2H(std::vector<uint8_t> &h_result)
	{
		h_result.resize(keypointSize);
		cudaStatus = cudaMemcpyAsync(h_result.data(), d_result, sizeof(uint8_t) * keypointSize, cudaMemcpyDeviceToHost, stream);
		assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");
	}
	__host__
		void sync()
	{
		cudaStatus = cudaStreamSynchronize(stream);
	}

	__host__
		~CudaKeypoints()
	{
		cudaFree(d_image);
		cudaFree(d_result);// it was said to -> cudaFree ( void* devPtr )Frees memory on the device.
		cudaFree(d_kp);
		cudaStreamDestroy(stream);
	}
private:
	//-------Provided By Main-------
	const std::vector<uint8_t> &img;
	const int height, width;
	int size; int keypointSize;
	//------Provided By Class,------
	//should be deleted after using
	uint8_t* d_image;
	uint8_t* d_result;
	float2* d_kp;
	cudaStream_t stream;
	cudaError_t cudaStatus;
};

class Cuda
{
public:
	Cuda(std::vector<uint8_t>& h_cpy, const std::vector<uint16_t>& data,
		const int& height, const int& width)
		:h_cpy(h_cpy), data(data), height(height), width(width)
		, d_data(nullptr), cpyData(nullptr)
	{
		cudaStatus = cudaError_t(0);
		size = height * width;
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		cudaStatus = cudaSetDevice(0);
		assert(cudaStatus == cudaSuccess, "you do not have cuda capable device!");
		cudaStatus = cudaStreamCreate(&stream);
	}
	
	__host__
		void startup(int size)
	{
		cudaStatus = cudaMalloc((void**)&d_data, size * sizeof(uint16_t));//sizeof(uint16_t) * size
		assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");

		cudaStatus = cudaMalloc((void**)&cpyData, sizeof(uint8_t) * size * 3);
		assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");
	}
	__host__
	void rawValue();

	__host__
	void sync()
	{
		cudaStatus = cudaStreamSynchronize(stream);
	}
	void outPutFile()
	{
		FILE* fileWr;
		fileWr = fopen("writingFile.ppm", "w+");
		fprintf(fileWr, "%s %d %d %d ", "P6", width, height, 255);
		fclose(fileWr);

		fileWr = fopen("writingFile.ppm", "ab+");
		fwrite(reinterpret_cast<const char*>(&h_cpy[0]), 1, sizeof(uint8_t) * width * height * 3, fileWr);
		fclose(fileWr);
		fileWr = nullptr;
	}
	__host__
		~Cuda()
	{
		cudaFree(cpyData);
		cudaFree(d_data);// it was said to -> cudaFree ( void* devPtr )Frees memory on the device.
		cudaStreamDestroy(stream);
	}
protected:
	int height, width, size;
	uint16_t* d_data;
	uint8_t* cpyData;
	std::vector<uint8_t> &h_cpy;
	const std::vector<uint16_t> &data;
	cudaStream_t stream;
	cudaError_t cudaStatus;
};
