#include "queueFile.h"

Queue::Queue(int s)
:size(s){
	front = 0,rear = 0;
	queuePtr = new char[size];
}
Queue::~Queue() {
	delete[]queuePtr;
}

void Queue::enQueue(int v) {
	if (isFull()) {
		cout << "Queue is Full" << endl;
		return;
	}
	queuePtr[rear++] = v;
	flag = 1;
	rear %= size;
}

char Queue::deQueue() {
	if (isEmpty()) {
		cout << "Queue is Empty!" << endl;
		return 0;	}
	char i = queuePtr[front++];
	front %= size;
	flag = 0;
	return i;
}

bool Queue::isFull() {
	return (front == rear && flag == 1);
}
bool Queue::isEmpty() {
	return (front == rear && flag == 0);
}

void Queue::print() {
	if (isEmpty()){
		cout << "Empty!" << endl;
	}
	else if (front < rear) {
		for (int i = front; i < rear; i++)
			cout << queuePtr[i] << " ";
	}
	else {
		for (int i = front; i < size; i++)
			cout << queuePtr[i] << " ";

		for (int i = 0; i < rear; i++)
			cout << queuePtr[i] << " ";
	}
	cout << endl;
}