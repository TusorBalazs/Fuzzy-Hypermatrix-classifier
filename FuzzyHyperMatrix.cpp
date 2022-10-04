#include <fstream>
#include <iostream>
#include <string>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <math.h>
#include <time.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "FuzzyHyperMatrix.h"

using namespace cv;
using namespace std;
//--------------------------------------------

//This version of the fuzzy hypermatrix has been simplified for image processing (and speed) 
//thus, the dimensions of the structure are fixed to [256,256,256]


int main( int argc, char** argv )
{

	
	HyperMatrixColorFilter * HM = new HyperMatrixColorFilter();

	Mat trainimg = imread("WL00060.jpg", IMREAD_COLOR);
	Mat trainmask = imread("WL00060mask.png", IMREAD_COLOR);

	int ColorSpaceTO;
	int ColorSpaceFROM;
	//---------------- UNCOMMENT THE DESIRED COLOR SPACE ---------------

	//if we want to use the HSV color space
//	ColorSpaceTO = COLOR_BGR2HSV;
//	ColorSpaceFROM = COLOR_HSV2BGR;
	
	//if we want to use the Lab color space
	ColorSpaceTO = COLOR_BGR2Lab;
	ColorSpaceFROM = COLOR_Lab2BGR;

	//if we want to use the Luv color space
//	ColorSpaceTO = COLOR_BGR2Luv;
//	ColorSpaceFROM = COLOR_Luv2BGR;
	//-------------------------------------------------------------------

	//training mask is always HSV, since it's just manual denotation (class label)
	// (background)class 0: black (0,0,0)
	// class 1: yellow (30,255,255)
	// class 2: red (0,255,255)
	// class 3: magenta (150,255,255)
	// class 4: white (255,255,255)
	cvtColor(trainmask, trainmask, COLOR_BGR2HSV);
	//convert the training image into the desired color space
	cvtColor(trainimg, trainimg, ColorSpaceTO);

	clock_t begin, end;
	double elapsed_secs;
	begin = clock();

	HM->trainHyperMatrix(10, 4, trainimg, trainmask);

	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "\nTime spent on training:" << elapsed_secs;
		
	Mat testImage = imread("WL00064.jpg", IMREAD_COLOR);
	if (!testImage.data)    // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Mat results;
	results.create(testImage.size(), testImage.type());
	results = Scalar::all(0);



	begin = clock();
	cvtColor(testImage, testImage, ColorSpaceTO);

	HM->filterImage(testImage, results, 4, 75);

	cvtColor(results, results, COLOR_HSV2BGR);
	cvtColor(testImage, testImage, ColorSpaceFROM);

	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "\nTime spent on filtering:" << elapsed_secs << " s";

	  imshow( "Test Image", testImage); 
	  imshow( "Train Image", trainimg );  
	  imshow( "Image Mask", trainmask ); 
	  imshow( "Filtered Image", results );  
	  Mat added_image;
	  cv::addWeighted(testImage, 0.5, results, 0.6, 0, added_image);
	  imshow("Overlay", added_image);

	waitKey(0);
    return 0;
}
