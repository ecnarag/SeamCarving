#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <math.h>  

#include "image.h"

using namespace std;
using namespace cv;

static int mouse_x = -1;
static int mouse_y = -1;
static int etat = 0;

void on_mouse( int evt, int x, int y, int d, void *ptr )
{
	Mat* wgt = (Mat*) ptr;
	int m = (*wgt).rows;

	if (d == EVENT_LBUTTONDOWN) 
    	{
		(*wgt).at<float>(y,x) = +1.;
		cout << "lbut avec" << y << "et" << x;
		mouse_x = x;
		mouse_y = y;
		etat = 1;
		}
		else
		{
			if (d == EVENT_RBUTTONDOWN) 
    		{
				(*wgt).at<float>(y,x) = -1.;
				mouse_x = x;
				mouse_y = y;
				etat = 2;
				cout << "rbut avec" << y << "et" << x;
			}
			else
			{
				if(etat != 0)
				{
					etat = 0;
				}
			}
		}
	
}

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
			//cout << "G1 = "<< endl << G1.at<float>(i,j) << endl;
		}
	}

	//cout << "g1 = "<< endl << G1 << endl;
}

void weightenergy(Mat& energy, Mat& zones, int m, int n){
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if(zones.at<float>(i,j)==1.){
					energy.at<float>(i,j) = energy.at<float>(i,j) + 500;
				}
				if(zones.at<float>(i,j)==-1.){
					//energy.at<float>(i,j) = -1000 * energy.at<float>(i,j);
					energy.at<float>(i,j) = -50;
				}
			}
		}
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

void deletehorizontalzones(Mat& I, vector<int> seamv, int m, int n){
	//int m = I.rows;
	//int n = I.cols;

	for(int j = 0; j < n; j++ ){
		for(int i = seamv[j]; i < m - 1; i++){
			I.at<float>(i,j) = I.at<float>(i+1,j);
		}
		I.at<float>(m-1,j) = 0;
	}
}

void deleteverticalzones(Mat& I, vector<int> seamv, int m, int n){
	//int m = I.rows;
	//int n = I.cols;

	for(int i = 0; i < m; i++ ){
		for(int j = seamv[i]; j < n - 1; j++){
			I.at<float>(i,j) = I.at<float>(i,j+1);
		}
		I.at<float>(i,n-1) = 0;
	}
}

void deletemultiplehorizontal(int k , Mat&Energie, Mat& zones, Mat& Mh, Mat& Ix, Mat& Iy, Mat& I){
	int m = I.rows;
	int n = I.cols;
	vector<int> seam;
	for(int i = 0 ; i<k; i++){
		sobel(I,Ix,Iy,Energie,m-i,n);
		energymatrixhorizontal(Energie,Mh,m-i,n);
		seam = findhminimalseam(I,Mh,m-i,n);		
		deletehorizontal(I,seam,m-i,n);
		deleteverticalzones(zones,seam,m-i,n);
	}
}
void deletemultiplevertical(int k , Mat&Energie, Mat& zones, Mat& Mv, Mat& Ix, Mat& Iy, Mat& I){
	int m = I.rows;
	int n = I.cols;
	vector<int> seam;
	for(int i = 0 ; i<k; i++){
		sobel(I,Ix,Iy,Energie,m,n-i);
		energymatrixvertical(Energie,Mv,m,n-i);
		seam = findvminimalseam(I,Mv,m,n-i);
		deletevertical(I,seam,m,n-i);
		deleteverticalzones(zones,seam,m,n-i);
	}
}

void deletemultipleverticalthenhorizontal(int p, int q , Mat&Energie, Mat& zones, Mat& Mv, Mat& Mh, Mat& Ix, Mat& Iy, Mat& I){
	int m = I.rows;
	int n = I.cols;
	vector<int> seam;
	bool vertical = true;
	int i = 0;
	int j = 0;

	while(i<q && j<p){
		sobel(I,Ix,Iy,Energie,m-j,n-i);
		weightenergy(Energie,zones,m-j,n-i);
		if(vertical){
			energymatrixvertical(Energie,Mv,m-j,n-i);
			seam = findvminimalseam(I,Mv,m-j,n-i);
			deletevertical(I,seam,m-j,n-i);
			deleteverticalzones(zones,seam,m-j,n-i);
			vertical = false;
			i++;
			}
		else{
			energymatrixhorizontal(Energie,Mh,m-j,n-i);
			seam = findhminimalseam(I,Mv,m-j,n-i);
			deletehorizontal(I,seam,m-j,n-i);
			deletehorizontalzones(zones,seam,m-j,n-i);
			vertical = true;
			j++;
			}
		//cout << "p " << p << " +q " << q << + " +i " << i << " +j " << j;
		}
	if (i == q) {
		while (j < p) {
			energymatrixhorizontal(Energie, Mh, m - j, n - i);
			seam = findhminimalseam(I, Mv, m - j, n - i);
			deletehorizontal(I, seam, m - j, n - i);
			deletehorizontalzones(zones, seam, m - j, n - i);
			j++;
		}
	}
	if (j == p) {
		while (i < q) {
			energymatrixvertical(Energie, Mv, m - j, n - i);
			seam = findvminimalseam(I, Mv, m - j, n - i);
			deletevertical(I, seam, m - j, n - i);
			deleteverticalzones(zones, seam, m - j, n - i);
			i++;
		}
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
			J.at<Vec3b>(begin + 1, i) = values[o][i];
		}
	}
	for (int o = k - 1; o >= 0; o--) {
		for (int i = 0; i < n; i++) {
			int begin = seams[o][i];
			if (begin > 0) {
				J.at<Vec3b>(begin + 2 * o, i) = Vec3b(
					(J.at<Vec3b>(begin + 2 * o, i).val[0] + J.at<Vec3b>(begin + 2 * o - 1, i).val[0]) / 2,
					(J.at<Vec3b>(begin + 2 * o, i).val[1] + J.at<Vec3b>(begin + 2 * o - 1, i).val[1]) / 2,
					(J.at<Vec3b>(begin + 2 * o, i).val[2] + J.at<Vec3b>(begin + 2 * o - 1, i).val[2]) / 2);
			}
			if (begin + 2 * o + 2 <= m + k) {
				J.at<Vec3b>(begin + 2 * o + 1, i) = Vec3b(
					(J.at<Vec3b>(begin + 2 * o + 1, i).val[0] + J.at<Vec3b>(begin + 2 * o + 2, i).val[0]) / 2,
					(J.at<Vec3b>(begin + 2 * o + 1, i).val[1] + J.at<Vec3b>(begin + 2 * o + 2, i).val[1]) / 2,
					(J.at<Vec3b>(begin + 2 * o + 1, i).val[2] + J.at<Vec3b>(begin + 2 * o + 2, i).val[2]) / 2);
			}
		}
	}
}

void addmultiplevertical(int k, int m, int n, Mat& Mv, Mat& I, Mat& Energie, Mat& Ix, Mat& Iy, Mat& J) { 
	vector<vector<int>> seams;
	vector<vector<Vec3b>> values;
	for (int o = 0; o < k; o++) {
		sobel(I,Ix,Iy,Energie,m,n-o); 
		energymatrixvertical(Energie,Mv,m,n-o);
		vector<int> minseam = findvminimalseam(I, Mv, m, n-o);
		vector<Vec3b> valuesofseam;
		for (int j = 0; j < m; j++) {
			valuesofseam.push_back(I.at<Vec3b>(j, minseam[j]));
			for (int i = 0; i < o; i++) {
				if(seams[i][j] > minseam[j]) {
					seams[i][j]++;
				}
			}
		}
		seams.push_back(minseam);
		values.push_back(valuesofseam);
		deletevertical(I, minseam, m, n-o);
	}
	for (int j = 0; j < n-k; j++) {
		for (int i = 0; i < m; i++) {
			J.at<Vec3b>(i,j) = I.at<Vec3b>(i,j);
		}
	}
	for (int o = k-1; o >= 0; o--) {
		for (int i= 0; i < m; i++) {
			int begin = seams[o][i];
			for (int j = n+k-2*o-3; j > begin-1; j--) {
				J.at<Vec3b>(i,j+2) = J.at<Vec3b>(i,j); 
			}
			
			J.at<Vec3b>(i, begin) = values[o][i];
			J.at<Vec3b>(i, begin + 1) = values[o][i];
		
		}
	}
	for (int o = k - 1; o >= 0; o--) {
		for (int i = 0; i < m; i++) {
			int begin = seams[o][i];

			if (begin > 0) {
				J.at<Vec3b>(i, begin + 2*o) = Vec3b(
					(J.at<Vec3b>(i, begin + 2 * o).val[0] + J.at<Vec3b>(i, begin + 2 * o - 1).val[0]) / 2,
					(J.at<Vec3b>(i, begin + 2 * o).val[1] + J.at<Vec3b>(i, begin + 2 * o - 1).val[1]) / 2,
					(J.at<Vec3b>(i, begin + 2 * o).val[2] + J.at<Vec3b>(i, begin + 2 * o - 1).val[2]) / 2);
			}
			if (begin + 2 + 2 * o <= n + k) {
				J.at<Vec3b>(i, begin + 2 * o + 1) = Vec3b(
					(J.at<Vec3b>(i, begin + 2 * o + 1).val[0] + J.at<Vec3b>(i, begin + 2 * o + 2).val[0]) / 2,
					(J.at<Vec3b>(i, begin + 2 * o + 1).val[1] + J.at<Vec3b>(i, begin + 2 * o + 2).val[1]) / 2,
					(J.at<Vec3b>(i, begin + 2 * o + 1).val[2] + J.at<Vec3b>(i, begin + 2 * o + 2).val[2]) / 2);
			}
		}
	}
}

void addhorizontalandvertical(int k, int l, int r, int c, Mat& Mh, Mat& Mv, Mat& I, Mat& Energie, Mat& Ix, Mat& Iy, Mat& J) {
	J = Mat(r+k, c+l, CV_8UC3);
	addmultiplehorizontal(k, r, c, Mh, I, Energie, Ix, Iy, J);
	addmultiplevertical(l, r+k, c, Mv, J, Energie, Ix, Iy, J);
}	



int main() {

	Image<Vec3b> I= Image<Vec3b>(imread("../../../test_images/temple.jpg"));
	
	printf("size of i %d %d", I.rows, I.cols);
	Mat Iref = I.clone();

	printf("size of iref %d %d", Iref.rows, Iref.cols);
	Mat Iref2 = I.clone();

	Mat Iref3 = I.clone();
	
	int m = I.rows;
	int n = I.cols;

	int p = 100;
	int q = 50;

	Mat zones = Mat(m, n, CV_32F);

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

	Mat Mh;
	//deletemultiplehorizontal(q,Energie,Mh,Ix,Iy,I);
	Mat Mv;
	//deletemultiplevertical(50,Energie,Mv,Ix,Iy,I);

	namedWindow("Choisir les zones",1);

	imshow("Choisir les zones",Iref);

	setMouseCallback("Choisir les zones", on_mouse, &zones);

	while(waitKey(20) != 27){
		imshow("Choisir les zones",Iref);
		//cout << "Sorti de waitKey";
		if(etat==1){
			circle(Iref, Point(mouse_x, mouse_y), 2, Scalar(0,255,0));
			}
		if(etat==2){
			circle(Iref, Point(mouse_x, mouse_y), 2, Scalar(255,0,0));
			}
/* 			for(int j = 0; j < n; j++ ){
				for(int i = 0; i < m; i++){
					if(zones.at<float>(i,j) == 1.){
						Iref.at<Vec3b>(i,j) = Vec3b(255,0,0);
						}
					if(zones.at<float>(i,j) == -1.){
						Iref.at<Vec3b>(i,j) = Vec3b(0,255,0);
						}
					}
				} */
	}
	cout<< "Sorti du while";



	deletemultipleverticalthenhorizontal(p,q,Energie,zones,Mv,Mh,Ix,Iy,I);

	Mat roi(I, Rect(0,0,n-q,m-p));
	imshow("roi",roi);

	
	//INSERTION DE LIGNES

	Mat J;
	int r = m-p;
	int c = n-q;
	cout << "r " << r;
	cout << "c " << c;
	addhorizontalandvertical(100, 50, r, c, Mh, Mv, I, Energie, Ix, Iy, J);
	imshow("added", J);
	//Mat roi(I, Rect(0,0,n-q,m-p));


	waitKey(0);
	return 0;
}
