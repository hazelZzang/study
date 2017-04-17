/*
20143051
자료구조
homework Maze

* 문제
(0,0) 위치에서 시작해 (5,5)의 끝을 찾아 미로의 경로를 출력한다.

* 알고리즘
8방면을 탐색하면서 방문한적이 없고, 길이 있는 경로를 방문한다.
경로를 방문할 때 마다 stack 에 기존 행과 열 정보, 방향 정보를 저장한다.
만약 끝을 만나기 전에 막다른 길에 다다른다면, stack에서 pop한 과거 정보를 통해
탐색을 이어간다.
(5,5)에 도달하거나 stack이 빈 경우에 종료한다.
*/




#include <iostream>
#include <vector>
#include "maze.h"
using namespace std;

int main(){

	//미로 정보
	vector <vector <int>> maze = {{ 0,1,1,1,1,1 },
								  { 1,0,1,1,1,1 },
								  { 1,0,0,0,0,1 },
								  { 1,1,0,1,1,1 },
								  { 1,0,1,0,0,1 },
								  { 1,1,1,1,1,0 }};
	Maze i(maze);
	i.find();
	i.printPath();
	i.printMark();
}