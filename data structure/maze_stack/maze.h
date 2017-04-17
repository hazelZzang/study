#pragma once
#include <iostream>
#include <vector>
#define MAX 6
using namespace std;


//���� ǥ��
enum {
	N, NE, E, SE, S, SW, W, NW
};

//Stack�� Node
//row, column, direction�� �����Ѵ�.
//direction�� Ž���� �Ϸ��� ���� + 1�̴�.
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

//���� �ڷ� ���� ����
//Stack : stack�� ���̸� �Է¹޾� �ʱ�ȭ�Ѵ�.
//push : Node data�� stack �� �����Ѵ�.
//pop : top�� data�� �����Ѵ�.
//isEmpty / isFull : ����ִ���, ���ִ��� �˻��Ѵ�.
//��ü Node �� row�� col ���� ����Ѵ�.
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

//�̷θ� ������ Ŭ����
//find : �̷��� �ⱸ�� Ȯ���Ѵ�.
//printMark : �湮�� ���� ����Ѵ�.
//printMaze : �̷θ� ����Ѵ�.
//printPath : ��θ� ����Ѵ�. 
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

