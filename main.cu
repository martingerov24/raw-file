#include <stdio.h>
#include <vector>
#include <string>

#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include "cuda_runtime.h"
#include "cuda/std/cmath"
#include "device_launch_parameters.h"

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include "build/CudaClass.h"


std::vector<float> keypoints =
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
void laod(std::vector<uint8_t> &image,  int& width, int& height)
{
	cv::Mat img = cv::imread("C:/Users/marting/Pictures/feature-matching/cathedral.jpg", cv::IMREAD_GRAYSCALE);
	if (img.empty())
	{
		throw "no ellements in cv::Mat buffer";
	}
	image.assign(img.begin<uint8_t>(), img.end<uint8_t>());
	img.deallocate();
	if (image.empty())
	{
		throw "vector is not filled with ellements";
	}
}

std::vector<uint16_t> ReadingFiles(char* fileName, int& height, int& width);

void bindTexture(GLuint texture)
{
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
}
void onNewFrame()
{
	glfwPollEvents();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}
void createContext(GLFWwindow* &window)
{
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
}

void Loop(std::vector<uint8_t>& h_cpy, const std::vector<uint16_t>& data, const int& height, const int& width)
{
	int size = height * width;

	uint16_t* d_data;
	uint8_t* cpyData;
	Cuda cuda(h_cpy, data, height, width, d_data, cpyData);
	cuda.startup(size);

	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	if (!glfwInit()) {
		throw "glfwInit() FAILED!";
	}

	GLFWwindow* window = glfwCreateWindow(800, 600, "Raw-File Viewer", NULL, NULL);

	if (!window) {
		glfwTerminate();
		throw "no window created";
	}

	cuda.sync();
	createContext(window);

	bool is_show = true;
	GLuint texture;
	glGenTextures(1, &texture);

	while (!glfwWindowShouldClose(window))
	{
		cuda.rawValue();
		onNewFrame();
		ImGui::Begin("raw Image", &is_show);
		bindTexture(texture);
		cuda.sync();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, h_cpy.data());
		ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture)), ImVec2(800, 600));
		ImGui::End();
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glDeleteTextures(sizeof(texture), &texture);
		glfwSwapBuffers(window);
	}

	cuda.outPutFile();
	ImGui_ImplGlfw_Shutdown();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
	glfwDestroyWindow(window);
}

void Keypoints(std::vector<uint8_t>& h_result, const std::vector<uint8_t>& data, const int height, const int width)
{
	int size = height * width;
	h_result.resize(size);
	uint8_t* d_image;
	uint8_t* d_result;
	float2* d_kp;
	// have taken care of those pointers in my class, they were used for the kernel
	CudaKeypoints cuda(h_result, (float2*)keypoints.data(), data, height, width , d_image, d_result, d_kp, keypoints.size());
	cuda.startup(size);
	cuda.cudaKernel();
	cuda.cudaMemcpyD2H();
	cuda.sync();
	delete[] d_image;
	delete[] d_result;
	delete[] d_kp;
}
int main()
{
	//char* fileName = "fileToRead.raw";
	int width = 1920, height = 1200;
	//int width = 3840, height = 1920;
	std::vector<uint8_t> data;
	laod(data, height, width);
	//const std::vector<uint16_t> &data = ReadingFiles(fileName, height, width);// data = radingFiles, if you want to read raw File, and must change to uint16_t
	std::vector<uint8_t> h_cpy;

	//Loop(h_cpy, data, height, width);
	Keypoints(h_cpy, data, height, width);
	return 0;
}

std::vector<uint16_t> ReadingFiles(char* fileName, int& height, int& width)
{
	FILE* rdFile = fopen(fileName, "rb+");
	std::vector<uint16_t> data;
	if (rdFile == 0) {
		printf("no file found!");
		return data; 
	}
	data.resize(height * width);
	fread(reinterpret_cast<char*>(&data[0]), 2, height * width, rdFile);
	fclose(rdFile);
	return data;
}