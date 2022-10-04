#pragma once
using namespace cv;

class HyperMatrixColorFilter
{
private:
		
	// storing which class is dominant for each point of the color space
	char C[256][256][256];

	// storing the membership function values for each point of the color space
	char MMF[256][256][256][5]; // 4 + 1 = 5 classes, each class takes 16 MB of RAM

	const int Dx = 256; //the size of the 1st dimension
	const int Dy = 256; //the size of the 2nd dimension
	const int Dz = 256; //the size of the 3rd dimension

public:
	HyperMatrixColorFilter()
	{
	}

	// calculating the values around point (x,y,z) using an r x r x r sized cube "window" for class v
	void calculateMMF(short x, short y, short z, short v, short r)
	{
		short mx = max(x - r, 0);
		short my = max(y - r, 0);
		short mz = max(z - r, 0);
		short Mx = min(x + r, 255);
		short My = min(y + r, 255);
		short Mz = min(z + r, 255);
		char cl = C[x][y][z];

		for (int i = mx; i<Mx + 1; i++)
			for (int j = my; j<My + 1; j++)
				for (int k = mz; k<Mz + 1; k++)
				{
					if (C[i][j][k] != cl && C[i][j][k] != 'n')
					{
						if (i <= x) { if (i>mx) mx = i; }
						else { if (i<Mx) Mx = i; }
						if (j <= y) { if (j>my) my = j; }
						else { if (j<My) My = j; }
						if (k <= z) { if (k>mz) mz = k; }
						else { if (k<Mz) Mz = k; }
					}
				}
		short Bx, By, Bz;
		short A[3][2];
		double t[3];
		double X[3];
		double Y[3];
		double Z[3];
		// refreshing the values
		for (int i = mx + 1; i<Mx; i++)
		{
			if (i<x) Bx = mx; else Bx = Mx;
			for (int j = my + 1; j<My; j++)
			{
				if (j<y) By = my; else By = My;
				for (int k = mz + 1; k<Mz; k++)
				{
					if (i != x && j != y && k != z)
					{
						if (k<z) Bz = mz; else Bz = Mz;

						A[0][0] = x; A[0][1] = abs(i - x);
						A[1][0] = y; A[1][1] = abs(j - y);
						A[2][0] = z; A[2][1] = abs(k - z);
						t[0] = (Bx - A[0][0]) / A[0][1];
						t[1] = (By - A[1][0]) / A[1][1];
						t[2] = (Bz - A[2][0]) / A[2][1];
						X[0] = Bx;							Y[0] = A[1][0] + A[1][1] * t[0];		Z[0] = A[2][0] + A[2][1] * t[0];
						X[1] = A[0][0] + A[0][1] * t[1];	Y[1] = By;								Z[1] = A[2][0] + A[2][1] * t[1];
						X[2] = A[0][0] + A[0][1] * t[2];	Y[2] = A[1][0] + A[1][1] * t[2];		Z[2] = Bz;
						//which bordercrossing is the closest? 
						double aamin = 100;
						double temp;
						short ind;
						for (int aa = 0; aa<3; aa++)
						{
							temp = sqrt((x - X[i])*(x - X[i]) + (y - Y[i])*(y - Y[i]) + (z - Z[i])*(z - Z[i]));
							if (temp<aamin) { aamin = temp; ind = i; }
						}
						temp = sqrt((double)((x - i)*(x - i) + (y - j)*(y - j) + (z - k)*(z - k)));
						temp = (1 - temp / aamin) * 100;

						if (MMF[i][j][k][v]<temp)
						{
							MMF[i][j][k][v] = (int)temp;
						}
					}
				}
			}
		}

	}

	void trainHyperMatrix(int r, int Cs, Mat img, Mat mask)
	{

		short x, y, z;
		// pessimistic approach: if a color tone is marked as positive and negative too, then it should be classified as negative => Cs
		for (int i = 0; i < Dx; i++)
			for (int j = 0; j < Dy; j++)
				for (int k = 0; k < Dz; k++)
				{
					for (int l = 0; l < Cs; l++)
					{
						MMF[i][j][k][l] = 0;
					}
					C[i][j][k] = 'n';
				}
		char ci;
		char ct;
		int S[3];
		int t;

		for (int i = 1; i<img.size.p[0]; i++)
			for (int j = 1; j<img.size.p[1]; j++)
			{
				for (int i = 0; i<3; i++)
				{
					S[i] = img.at<Vec3b>(i, j)[i];
				}
				x = img.at<Vec3b>(i, j)[0];
				y = img.at<Vec3b>(i, j)[1];
				z = img.at<Vec3b>(i, j)[2];

				if (mask.at<Vec3b>(i, j)[2] >= 10)
				{
					if (mask.at<Vec3b>(i, j)[0] == 30) //yellow
					{
						t = 1;
						//cout << " 1 ";
					}
					if (mask.at<Vec3b>(i, j)[0] == 0) //red
					{
						t = 2;
						//cout << " 2 ";
					}
					if (mask.at<Vec3b>(i, j)[0] == 150) //magenta
					{
						t = 3;
						//cout << " 3 ";
					}
				}
				else
				{
					t = 0;
				}

				ci = t;
				ct = C[x][y][z];

				if (ct == 'n')
				{
					//if this is a new color tone in the hypermatrix
					C[x][y][z] = ci;
				}
				else
				{
					//if this is NOT a new color tone in the hypermatrix
					if (ct == 0)
					{
						//if the already existing one is marked negative, then skip
					}
					else
					{
						//if the already existing one is marked different from the new one --> inconsistency, set it as negative
						if (ci != ct)
						{
							C[x][y][z] = 0;
						}
					}
				}
			}

		int l;
		//calculate the fuzzy membership function values in the hypermatrix in the areas around all positive markers
		for (int i = 0; i<Dx; i++)
			for (int j = 0; j<Dy; j++)
				for (int k = 0; k<Dz; k++)
				{
					if (C[i][j][k] != 'n' && C[i][j][k] != 0)
					{
						l = (int)C[i][j][k];
						MMF[i][j][k][l] = 100;
						calculateMMF(i, j, k, l, r);
					}
					if (C[i][j][k] == 0) MMF[i][j][k][0] = 100;
				}
	}

	//looks through the membership functions of all classes to find the one with the largest value (0 = background)
	int evaluateMMFs(Vec3b p, int Cs, int THRESH)
	{
		int maxC = 0;
		int maxV = 0;
		for (int c = 1; c<Cs; c++)
		{
			if ((int)MMF[p[0]][p[1]][p[2]][c] > maxV)
			{
				maxV = (int)MMF[p[0]][p[1]][p[2]][c]; maxC = c;
			}

		}
		if (maxV>THRESH)
			return maxC;
		else return 0;
	}
	//input - the image we want to filter (in BGR)
	//results - a blank matrix with the size of the input matrix, should be all zeros, --> the output of the filtering
	//NoC - number of color classes, i.e. how many objects (+the background) we want to differentiate
	//THRESH - 0...100, the threshold of the fuzzy membership function value, we only let pixels above this through (e.g. 75 [%])
	void filterImage(Mat input, Mat results, int NoC, int THRESH)
	{
		for (int i = 0; i < input.size.p[0]; i++)
			for (int j = 0; j < input.size.p[1]; j++)
			{
				switch (evaluateMMFs(input.at<Vec3b>(i, j), NoC, THRESH))
				{
				case 0: break;
				case 1:
					results.at<Vec3b>(i, j)[0] = 30;
					results.at<Vec3b>(i, j)[1] = 255;
					results.at<Vec3b>(i, j)[2] = 255;
					break;
				case 2:
					results.at<Vec3b>(i, j)[0] = 0;
					results.at<Vec3b>(i, j)[1] = 255;
					results.at<Vec3b>(i, j)[2] = 255;
					break;
				case 3:
					results.at<Vec3b>(i, j)[0] = 150;
					results.at<Vec3b>(i, j)[1] = 255;
					results.at<Vec3b>(i, j)[2] = 255;
					break;
				case 4:
					results.at<Vec3b>(i, j)[0] = 255;
					results.at<Vec3b>(i, j)[1] = 255;
					results.at<Vec3b>(i, j)[2] = 255;
					break;
				}
			}
	}
	~HyperMatrixColorFilter()
	{
	}
};
