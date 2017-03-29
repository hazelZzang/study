#pragma once

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
