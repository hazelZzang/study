#pragma once
#include <iostream>
#include <vector>
#define MAX 6
using namespace std;


//방향 표현
enum {
	N, NE, E, SE, S, SW, W, NW
};

//Stack의 Node
//row, column, direction을 저장한다.
//direction은 탐색을 완료한 방향 + 1이다.
class Node {
private:
	int row;
	int col;
	int dir;
public:
	Node();
	Node(int r, int c, int d);

	friend class Stack;
	friend class Maze;
};

//스택 자료 구조 구축
//Stack : stack의 길이를 입력받아 초기화한다.
//push : Node data를 stack 에 저장한다.
//pop : top의 data를 삭제한다.
//isEmpty / isFull : 비어있는지, 차있는지 검사한다.
//전체 Node 의 row와 col 값을 출력한다.
class Stack {
private:
	int top;
	int len;
	vector<Node> arr;
public:
	Stack(int d);
	void push(Node v);
	Node pop();
	bool isEmpty();
	bool isFull();
	void print();
};

//미로를 제작할 클래스
//find : 미로의 출구를 확인한다.
//printMark : 방문한 길을 출력한다.
//printMaze : 미로를 출력한다.
//printPath : 경로를 출력한다. 
class Maze {
private:
	vector <vector <int>> maze;
	vector <vector <int>> mark;
	Stack path;
	int found;
public:

	Maze(vector <vector <int>> m);
	void find();
	void printPath();
	void printMaze();
	void printMark();
};

