#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <math.h>  

#include "image.h"

using namespace std;
using namespace cv;


// Gradient (and derivatives), Sobel denoising
void sobel(const Mat& Ic, Mat& Ix, Mat& Iy, Mat& G1, int m, int n)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);
	
	//int m = I.rows, n = I.cols;
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
void energymatrixhorizontal(const Mat& E, Mat& Ih, int m, int n)
{
	//int m = E.rows, n = E.cols;
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

void energymatrixvertical(const Mat& E, Mat& Iv, int m, int n)
{
	//int m = E.rows, n = E.cols;
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


vector <int> findseamhorizontal (const Mat& I, Mat& Mh, int m, int n, int i) {
	vector<int> seam(n);
	float Mval = Mh.at<float>(0,n-1);
	for(int j = n-2; j >= 0; j--){
		for(int ip = max(0, i-1); ip <= min(i+1, m-1) ; ip++){
			if(Mh.at<float>(ip,j) < Mval){
				Mval = Mh.at<float>(ip,j);
				i = ip;
			}
		}
		seam[j] = i;
	}
	return(seam);
}
			
//Return the first seam from the energy matrix
vector<int> findhminimalseam(const Mat& I, Mat& Mh, int m, int n){
	int imin = 0;
	float Mval = Mh.at<float>(0,n-1);
	for(int i = 0; i < m ; i++){
		if(Mh.at<float>(i,n-1) < Mval){
			Mval = Mh.at<float>(i,n-1);
			imin = i;
		}
	}
	return(findseamhorizontal(I, Mh, m, n, imin));
}


vector<int> findseamvertical(const Mat& I, Mat& Mv, int m, int n, int j){
	vector<int> seam(m);
	float Mval2 = Mv.at<float>(m-1,0);
	for(int i = m-2; i >= 0; i--){
		for(int jp = max(0, j-1); jp <= min(j+1, n-1) ; jp++){
			if(Mv.at<float>(i,jp) < Mval2){
				Mval2 = Mv.at<float>(i,jp);
				j = jp;
			}
		}
		seam[i] = j;
	}
	return(seam);
}

vector<int> findvminimalseam(const Mat& I, Mat& Mv, int m, int n) {
	int jmin = 0;
	float Mval2 = Mv.at<float>(m-1,0);
	for(int j = 0; j < n ; j++){
		if(Mv.at<float>(m-1,j) < Mval2){
			Mval2 = Mv.at<float>(m-1,j);
			jmin = j;
		}
	}
	return(findseamvertical(I, Mv, m, n, jmin));
}

void deletehorizontal(Mat& I, vector<int> seamh, int m, int n){
	//int m = I.rows;
	//int n = I.cols;

	for(int j = 0; j < n; j++ ){
		for(int i = seamh[j]; i < m - 1; i++){
			I.at<Vec3b>(i,j) = I.at<Vec3b>(i+1,j);
		}
		I.at<Vec3b>(m-1,j) = Vec3b(0,0,0);
	}
}

void deletevertical(Mat& I, vector<int> seamv, int m, int n){
	//int m = I.rows;
	//int n = I.cols;

	for(int i = 0; i < m; i++ ){
		for(int j = seamv[i]; j < n - 1; j++){
			I.at<Vec3b>(i,j) = I.at<Vec3b>(i,j+1);
		}
		I.at<Vec3b>(i,n-1) = Vec3b(0,0,0);
	}
}

void deletemultiplehorizontal(int k , Mat&Energie, Mat& Mh, Mat& Ix, Mat& Iy, Mat& I){
	int m = I.rows;
	int n = I.cols;
	vector<int> seam;
	for(int i = 0 ; i<k; i++){
		sobel(I,Ix,Iy,Energie,m-i,n);
		energymatrixhorizontal(Energie,Mh,m-i,n);
		seam = findhminimalseam(I,Mh,m-i,n);
		deletehorizontal(I,seam,m-i,n);
	}
}
void deletemultiplevertical(int k , Mat&Energie, Mat& Mv, Mat& Ix, Mat& Iy, Mat& I){
	int m = I.rows;
	int n = I.cols;
	vector<int> seam;
	for(int i = 0 ; i<k; i++){
		sobel(I,Ix,Iy,Energie,m,n-i);
		energymatrixvertical(Energie,Mv,m,n-i);
		seam = findvminimalseam(I,Mv,m,n-i);
		deletevertical(I,seam,m,n-i);
	}
}

void addmultiplehorizontal(int k, int m, int n, Mat& Mh, Mat& I, Mat& Energie, Mat& Ix, Mat& Iy, Mat& J) { 
	vector<vector<int>> seams;
	vector<vector<Vec3b>> values;
	for (int o = 0; o < k; o++) {
		sobel(I,Ix,Iy,Energie,m-o,n);
		energymatrixhorizontal(Energie,Mh,m-o,n);
		vector<int> minseam = findhminimalseam(I, Mh, m-o, n);
		vector<Vec3b> valuesofseam;
		for (int i = 0; i < n; i++) {
			valuesofseam.push_back(I.at<Vec3b>(minseam[i],i));
			for (int j = 0; j < o; j++) {
				if(seams[j][i] > minseam[i]) {
					seams[j][i]++;
				}
			}
		}
		seams.push_back(minseam);
		values.push_back(valuesofseam);
		deletehorizontal(I, minseam, m-o, n);
	}
	for (int i = 0; i < m-k; i++) {
		for (int j = 0; j < n; j++) {
			J.at<Vec3b>(i,j) = I.at<Vec3b>(i,j);
		}
	}
	for (int o = k-1; o >= 0; o--) {
		for (int i= 0; i < n; i++) {
			int begin = seams[o][i];
			for (int j = m+k-2*o-3; j > begin-1; j--) {
				J.at<Vec3b>(j+2,i) = J.at<Vec3b>(j,i); 
			}
			J.at<Vec3b>(begin, i) = values[o][i];
			J.at<Vec3b>(begin+1,i) = values[o][i];
		}
	}
}

void addmultiplevertical(int k, int m, int n, Mat& Mv, Mat& I, Mat& Energie, Mat& Ix, Mat& Iy, Mat& J) { 
	vector<vector<int>> seams;
	vector<vector<Vec3b>> values;
	for (int o = 0; o < k; o++) {
		printf("iteration %d", o);
		sobel(I,Ix,Iy,Energie,m,n-o);
		energymatrixvertical(Energie,Mv,m,n-o);
		printf("blibli");
		vector<int> minseam = findvminimalseam(I, Mv, m, n-o);
		vector<Vec3b> valuesofseam;
		printf("bla");
		for (int j = 0; j < m; j++) {
			printf("blu %d", j);
			valuesofseam.push_back(I.at<Vec3b>(j, minseam[j]));
			for (int i = 0; i < o; i++) {
				if(seams[i][j] > minseam[j]) {
					seams[i][j]++;
				}
			}
		}
		printf("bleh");
		seams.push_back(minseam);
		values.push_back(valuesofseam);
		deletevertical(I, minseam, m, n-o);
	}
	printf("ok till here");
	for (int j = 0; j < n-k; j++) {
		for (int i = 0; i < m; i++) {
			J.at<Vec3b>(i,j) = I.at<Vec3b>(i,j);
		}
	}
	printf("also copied");
	for (int o = k-1; o >= 0; o--) {
		for (int i= 0; i < m; i++) {
			int begin = seams[o][i];
			for (int j = n+k-2*o-3; j > begin-1; j--) {
				J.at<Vec3b>(i, j+2) = J.at<Vec3b>(i,j); 
			}
			J.at<Vec3b>(i,begin) = values[o][i];
			J.at<Vec3b>(i,begin+1) = values[o][i];
		}
	}
}

void deletemultipleverticalthenhorizontal(int p, int q , Mat&Energie, Mat& Mv, Mat& Mh, Mat& Ix, Mat& Iy, Mat& I){
	int m = I.rows;
	int n = I.cols;
	vector<int> seam;
	for(int i = 0 ; i<q; i++){
		sobel(I,Ix,Iy,Energie,m,n-i);
		energymatrixvertical(Energie,Mv,m,n-i);
		seam = findvminimalseam(I,Mv,m,n-i);
		deletevertical(I,seam,m,n-i);
	}
	for(int i = 0 ; i<p; i++){
		sobel(I,Ix,Iy,Energie,m-i,n);
		energymatrixhorizontal(Energie,Mh,m-i,n);
		seam = findhminimalseam(I,Mh,m-i,n);
		deletehorizontal(I,seam,m-i,n);
	}
}


int main() {

	//INITIALISATION (IMAGE ET PARAMÈTRES)
	Image<Vec3b> I= Image<Vec3b>(imread("../../../test_images/temple.jpg"));
	
	Mat Iref(I);

	imshow("Iref",Iref);

	int r = I.rows;
	int c = I.cols;

	int p = 10;
	int q = 5;

	Mat Energie;
	Mat Ix,Iy;
	//sobel(I,Ix,Iy,Energie);
    //imshow("Ix",Ix);
    //imshow("Iy",Iy);
	//imshow("Energie",Energie);

	//cout << "energie = "<< endl << " "  << Energie << endl << endl;

	//energymatrixhorizontal(Energie,Mh);
	//energymatrixvertical(Energie,Mv);

	//cout << "Mh = "<< endl << " "  << Mh << endl << endl;

	//SUPPRESSION DE LIGNES/COLONNES
	Mat Mh;
	//deletemultiplehorizontal(q,Energie,Mh,Ix,Iy,I);
	Mat Mv;
	//deletemultiplevertical(50,Energie,Mv,Ix,Iy,I);
	//deletemultipleverticalthenhorizontal(p,q,Energie,Mv,Mh,Ix,Iy,I);

	//INSERTION DE LIGNES
	Mat J;
	J = Mat(r+20, c, CV_8UC3);
	addmultiplehorizontal(20, r, c, Mh, I, Energie, Ix, Iy, J);
	addmultiplevertical(20, r+20, c, Mv, I, Energie, Ix, Iy, J);

	//Mat roi(I, Rect(0,0,n-q,m-p));
	imshow("added", J);

	//imshow("roi",roi);
	waitKey(0);
	return 0;
}
