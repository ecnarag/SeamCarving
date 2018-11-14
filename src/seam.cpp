#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <math.h>  

#include "image.h"

using namespace std;
using namespace cv;


// Gradient (and derivatives), Sobel denoising
void sobel(const Mat& Ic, Mat& Ix, Mat& Iy, Mat& G1)
{
	Mat I;
	cvtColor(Ic, I, CV_BGR2GRAY);
	
	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G1 = Mat(m, n, CV_32F);

	Sobel(I,Ix,CV_32F,1,0);
	Sobel(I,Iy,CV_32F,0,1);


	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			//float?
			//cout << "ix = "<< endl << Ix.at<float>(i,j) << endl;
			//cout << "iy = "<< endl << Iy.at<float>(i,j) << endl;
			
			G1.at<float>(i,j) = ( fabs(Ix.at<float>(i,j)) + fabs(Iy.at<float>(i,j)) ) / 2;	
			
		}
	}

	//cout << "g1 = "<< endl << G1 << endl;
}



//Energy matrix to get the seams
void energymatrixhorizontal(const Mat& E, Mat& Ih)
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

void energymatrixvertical(const Mat& E, Mat& Iv)
{
	int m = E.rows, n = E.cols;
	Iv = Mat(m, n, CV_32F);

	for (int j = 0; j < n; j++) {
		Iv.at<float>(0,j) = E.at<float>(0,j);
	}

	for (int i = 1; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float minimum = Iv.at<float>(i-1,j);
			if(j>0){
				minimum = min(Iv.at<float>(i-1,j-1),minimum);
			}
			if(j<n-1){
				minimum = min(Iv.at<float>(i-1,j+1),minimum);
			}
			Iv.at<float>(i,j) = E.at<float>(i,j) + minimum;
		}
	}
}


//Return the first seam from the energy matrix
vector<int> findseamhorizontal(const Mat& I, Mat& Mh){
	int n = I.cols;
	int m = I.rows;
	vector<int> seam(n);
	int imin = 0;
	float Mval = Mh.at<float>(0,n-1);

	for(int j = n-1; j >= 0; j--){
		if(j == n-1){
			for(int i = 0; i < m ; i++){
				if(Mh.at<float>(i,n-1) < Mval){
					Mval = Mh.at<float>(i,n-1);
					imin = i;
				}
			}
		}
		else{
			for(int i = max(0, imin-1); i <= min(imin+1, m-1) ; i++){
				if(Mh.at<float>(i,j) < Mval){
					Mval = Mh.at<float>(i,j);
					imin = i;
				}
			}
		}
		//vector<int> myvector
		seam[j] = imin;
	}
	return(seam);
}


vector<int> findseamvertical(const Mat& I, Mat& Mv){
	int m = I.rows;
	int n = I.cols;
	vector<int> seam(m);
	int jmin = 0;
	float Mval2 = Mv.at<float>(m-1,0);
	for(int i = m-1; i >= 0; i--){
		if(i == m-1){
			for(int j = 0; j < n ; j++){
				if(Mv.at<float>(m-1,j) < Mval2){
					Mval2 = Mv.at<float>(m-1,j);
					jmin = j;
				}
			}
		}
		else{
			for(int j = max(0, jmin-1); j <= min(jmin+1, n-1) ; j++){
				if(Mv.at<float>(i,j) < Mval2){
					Mval2 = Mv.at<float>(i,j);
					jmin = j;
				}
			}
		}
		//vector<int> myvector
		seam[i] = jmin;
	}
	return(seam);
}

void deletehorizontal(Mat& I, vector<int> seamh){
	int m = I.rows;
	int n = I.cols;

	for(int j = 0; j < n; j++ ){
		for(int i = seamh[j]; i < m - 1; i++){
			I.at<Vec3b>(i,j) = I.at<Vec3b>(i+1,j);
		}
		I.at<Vec3b>(m-1,j) = Vec3b(0,0,0);
	}
}

void deletemultiplehorizontal(int n , Mat&Energie, Mat& Mh, Mat& I){

	Mat Ix,Iy;
	vector<int> seam;

	for(int i = 0 ; i<n; i++){
	
	sobel(I,Ix,Iy,Energie);
	
	energymatrixhorizontal(Energie,Mh);

	seam = findseamhorizontal(I,Mh);

	deletehorizontal(I,seam);
	
	}

}


int main() {

	Image<Vec3b> I= Image<Vec3b>(imread("../temple.jpg"));
	
	Mat Iref(I);

	imshow("Iref",Iref);

	int m = I.rows;
	int n = I.cols;

	Mat Energie;
	Mat Ix,Iy;
	sobel(I,Ix,Iy,Energie);
    imshow("Ix",Ix);
    imshow("Iy",Iy);
	imshow("Energie",Energie);

	//cout << "energie = "<< endl << " "  << Energie << endl << endl;

	Mat Mh;
	Mat Mv;
	energymatrixhorizontal(Energie,Mh);
	energymatrixvertical(Energie,Mv);

	//cout << "Mh = "<< endl << " "  << Mh << endl << endl;

	vector<int> seam = findseamhorizontal(I,Mh);

	deletehorizontal(I,seam);

	imshow("I",I);
	waitKey(0);
	return 0;
}
