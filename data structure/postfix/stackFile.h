#ifndef __STACK__
#define __STACK__

#include <iostream>
using namespace std;


class Stack {
private:
	int topIndex;
	int len;
	char* stackPtr;

public:
	Stack(int l = 100);
	~Stack();
	char pop();
	void push(const char& data);
	void pushAll(const char* data);
	char top();
	bool isFull();
	bool isEmpty();
	void print();
};

#endif