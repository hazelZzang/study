#include "stack.h"
#include <iostream>
using namespace std;

Stack::Stack(int l) {
	topIndex = -1;
	len = l;

	stackPtr = new char[len];
	memset(stackPtr, 0, len * sizeof(char));
}


Stack::~Stack() {
	delete[] stackPtr;
}
int Stack::getSize() {
	return topIndex;
}
char Stack::pop() {
	if (isEmpty()) {
		cout << "stack is empty" << endl;
		return 0;
	}
	return stackPtr[topIndex--];
}

void Stack::push(const char& data) {
	if (isFull()) {
		cout << "stack is full" << endl;
		return;
	}
	stackPtr[++topIndex] = data;
}

void Stack::pushAll(const string& data) {

	for (int i = data.length() - 1; i >= 0; i--) {
		push(data[i]);
	}
}

char Stack::top() {
	if (isEmpty())	return 0;
	return stackPtr[topIndex];
}

bool Stack::isFull() {
	return topIndex == len - 1;
}


bool Stack::isEmpty() {
	return topIndex == -1;
}


void Stack::print() {
	for (int i = 0; i <= topIndex; i++) {
		cout << stackPtr[i] << ' ';
	}
	cout << endl;
}
