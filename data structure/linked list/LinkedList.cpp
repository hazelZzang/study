/*
Linked List
Data Structure
20143051
Kim Hye Ji
*/

#include <iostream> 
using namespace std;

class Node{
private:
	int value;
	Node* next;
public:
	Node(int v)
		:value(v), next(NULL){
	}
	Node(){}

	friend class List;
};

class List {
private:
	Node *head;

public:
	List() { head = 0; }
	void insertNode(int v);
	void deleteNode(int v);
	void printList();
	Node* searchList(int num);
	bool isEmpty();

};

void List::insertNode(int v) {
	Node *temp = new Node(v);
	// no Node in List
	if (head == 0) head = temp;

	// insert value < head value
	else if (temp->value < head->value) {
		temp->next = head;
		head = temp;
	}

	// insert value > head value
	else {
		Node *p, *q;
		p = head;
		while ((p != 0) && (p->value < temp->value)) {
			q = p;
			p = p->next;
		}
		if (p != 0) {
			temp->next = p;
			q->next = temp;
		}
		else {
			q->next = temp;
		}
	}
};

void List::deleteNode(int v) {
	Node *temp, *del;
	temp = searchList(v);

	// No data Value in Linked List
	if (temp == 0) {
		cout << "No delete Data" << endl;
		return;
	}

	del = temp->next;

	temp->next = del->next;
	delete del;

};

void List::printList() {
	Node *temp;
	if (!isEmpty()) {
		temp = head;
		while (temp) {
			cout << temp->value << ' ';
			temp = temp->next;
		}
		cout << endl;
	}
};

Node* List::searchList(int num) {
	Node *p, *q;
	p = head, q = head;

	while (p != 0 && p->value != num) {
		q = p;
		p = p->next;
	}

	if (p != 0) {
		return q;
	}
	else {
		cout << "No data" << endl;
		return NULL;
	}
};

bool List::isEmpty() {
	if (head == 0) return true;
	else return false;
};


int main() {
	List l;
	l.insertNode(5);
	l.insertNode(3);
	l.printList();
}