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

bool lexicographicorder(tuple<float,int> fst, tuple<float,int> snd) { //create an utils.cpp file to clarify ?
	if (get<0>(fst) < get<0>(snd)) {
		return true;
	}
	else if (get<0>(fst) == get<0>(snd)) {
		return (get<1>(fst) < get<1>(snd));
	}
	else return false;
}

bool sndorder (tuple<float,int> fst, tuple<float,int> snd) { //idem
	if (get<1>(fst) < get<1>(snd)) {
		return true;
	}
	else return false;
}

void addmultiplehorizontal(int k, int m, int n, Mat& Mh, Mat& I, Mat& Energie, Mat& Ix, Mat& Iy, Mat& J) { //J will contain the result matrix, it should be at least as large and k rows wider. Mh peut d�j� avoir �t� obtenue avec energymatrixhorizontal
	sobel(I,Ix,Iy,Energie,m,n);
	energymatrixhorizontal(Energie,Mh,m,n);
	vector<tuple<float, int>> last_column;
	printf("taille de I: %d, %d", m, n);
	printf("taille de J: %d, %d\n", J.rows, J.cols);
	printf("taille de Mh: %d, %d\n", Mh.rows, Mh.cols);
	for(int i = 0; i < m ; i++){
		last_column.push_back(make_tuple(Mh.at<float>(i,n-1), i));
	}
	std::sort(last_column.begin(), last_column.begin()+last_column.size(), lexicographicorder);
	printf("sorted\n");
	vector<vector<int>> seams;
	vector<int> to_add;
	for (int o = 0; o < k; o++) {
		to_add.push_back(get<1>(last_column[o]));
	}
	printf("selected\n");
	std::sort(to_add.begin(), to_add.begin()+k); //normalement deux seams ne peuvent pas se croiser (sinon ils suivent la m�me fin) et donc normalement l'ordre est conserv� � chaque ligne (j'esp�re sinon �a complique)
	printf("sorted back\n");
	for (int o = 0; o < k; o++) {
		seams.push_back(findseamhorizontal(I, Mh, m, n, to_add[o])); //on peut donc les mettre dans l'ordre logique et comme �a les ajouter dans l'ordre ce qui permet de ne pas tout d�caler k fois
		printf("to add in row 10 : %d", seams[o][10]);
	}
	printf("seams found\n");
	for(int j = 0; j < n; j++) { //on ajoute k lignes donc colonne par colonne il faut ajouter k pixels
		printf("started iteration%d", j);
		int added = 0; //le nombre d�j� ajout� qui correspond au d�calage
		int begin = 0; //le premier indice quand on a ajout� added pixels sur la colonne j
		while (added != k) {
			//printf("already added %d", added);
			//printf("from %d to %d", begin, seams[j][added]);
			//ici on a un probl�me, en gros tous les seams[j][added] sont �gaux (pour un m�me added)
			for(int i = begin; i < seams[j][added]; i++) { //les indices �a va �te un bordel � d�bugger
				J.at<Vec3b>(i+added, j) = I.at<Vec3b>(i, j);
			}
			J.at<Vec3b>(seams[j][added]+added, j) = I.at<Vec3b>(seams[j][added], j);
			begin = seams[j][added];
			added ++;
		}
	}
	printf("function ended\n");
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

	//INITIALISATION (IMAGE ET PARAM�TRES)
	Image<Vec3b> I= Image<Vec3b>(imread("../../../test_images/temple.jpg"));
	
	Mat Iref(I);

	imshow("Iref",Iref);

	int m = I.rows;
	int n = I.cols;

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
	deletemultipleverticalthenhorizontal(p,q,Energie,Mv,Mh,Ix,Iy,I);

	//INSERTION DE LIGNES
	/*Mat J;
	J = Mat(m+20, n, CV_8U);
	addmultiplehorizontal(20, m, n, Mh, I, Energie, Ix, Iy, J);*/

	//Mat roi(I, Rect(0,0,n-q,m-p));
	imshow("deleted", J);

	//imshow("roi",roi);
	waitKey(0);
	return 0;
}
