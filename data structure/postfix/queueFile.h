#ifndef __QUEUE__
#define __QUEUE__


#include <iostream>
using namespace std;

class Queue {
private:
	int front, rear;
	int size;
	int flag;
	char *queuePtr;
public:
	Queue(int size);
	~Queue();
	void enQueue(int value);
	char deQueue();
	bool isFull();
	bool isEmpty();
	void print();
};

#endif