#pragma once
#include <string>
using namespace std;

class Stack {
private:
	int topIndex;
	int len;
	char* stackPtr;

public:
	Stack(int l = 100);
	~Stack();
	int getSize();
	char pop();
	void push(const char& data);
	void pushAll(const string& data);
	char top();
	bool isFull();
	bool isEmpty();
	void print();
};
