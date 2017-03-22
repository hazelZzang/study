/*
kookmin university
computer science
20143051
Kim Hye Ji 

Making magic square
*/

#include <iostream>
#include <fstream>
using namespace std; 

void magicSquare(int n) {

	//square matrix
	int** square = new int*[n];
	for (int i = 0; i < n; i++) {
		square[i] = new int[n];
	}
	
	//initialization
	int r = 0, c = (n - 1) / 2;
	int key = 1;
	for (int i = 0; i < n; i++)
		memset(square[i], 0, sizeof(int) * n);
	square[r][c] = key++;
	
	while (key <= n*n) {
		int tmpR, tmpC;
		//move up and left
		//if there is out of range, move to end
		if (r - 1 < 0) tmpR = n - 1; else tmpR = r - 1;
		if (c - 1 < 0) tmpC = n - 1; else tmpC = c - 1;
		
		//if square is full, move down  
		if (square[tmpR][tmpC])	r = (r + 1) % n;
		else r = tmpR, c = tmpC;
		square[r][c] = key++;		
	}

	//print
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			cout << square[i][j] << ' ';
		cout << endl;
	}
}

int main(){
	int num;
	cin >> num;
	magicSquare(num);
}