#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include "image.h"

using namespace std;
using namespace cv;


// Gradient (and derivatives), Sobel denoising
void sobel(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G1)
{
	Mat I;
	cvtColor(Ic, I, CV_BGR2GRAY);
	
	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G1 = Mat(m, n, CV_32F);

	Sobel(Ic,Ix,-1,1,0);
	Sobel(Ic,Iy,-1,0,1);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			//float?
			G1.at<float>(i,j) = float(floor(Ix.at<uchar>(i,j)) + floor(Iy.at<uchar>(i,j)));	
		}
	}
}

void findseamhorizontal(const Mat&E, Mat& Ih)
{
	int m = E.rows, n = E.cols;
	Ih = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		Ih.at<float>(i,0) = E.at<float>(i,0);
	}

	for (int j = 1; j < n; j++) {
		for (int i = 0; i < m; i++) {
			float minimum = Ih.at<float>(i,j-1);
			if(i>0){
				minimum = min(Ih.at<float>(i-1,j-1),minimum);
			}
			if(i<m-1){
				minimum = min(Ih.at<float>(i+1,j-1),minimum);
			}
			Ih.at<float>(i,j) = E.at<float>(i,j) + minimum;
		}
	}
}




int main() {

	Image<Vec3b> I= Image<Vec3b>(imread("../temple.jpg"));
	imshow("I",I);

	int m = I.rows;
	int n = I.cols;

	Mat Energie;
	Mat Ix,Iy;
	sobel(I,Ix,Iy,Energie);
	imshow("Energie",Energie);

	Mat Mh;
	Mat Mv;
	findseamhorizontal(Energie,Mh);

	//cout << "Mh = "<< endl << " "  << Mh << endl << endl;

	int jmin = 0;
	float Mval = Mh.at<float>(m-1,0);

	for(int i = m-1; i >= 0; i--){
		if(i == m-1){
			for(int j = 0; j < n ; j++){
				if(Mh.at<float>(m-1,j) < Mval){
					Mval = Mh.at<float>(m-1,j);
					jmin = j;
				}
			}
		}
		else{
			for(int j = max(0, jmin-1); j <= min(jmin+1, n-1) ; j++){
				if(Mh.at<float>(m-1,j) < Mval){
					Mval = Mh.at<float>(m-1,j);
					jmin = j;
				}
			}
		}
		//vector<int> myvector
		I.at<Vec3b>(i,jmin) = Vec3b(0,255,0);
	}

	imshow("I",I);
	waitKey(0);
	return 0;
}
