#include "DLL.h"

template <typename T>
List<T>::List() {
}

template <typename T>
List<T>::~List() {
	Node<T> *temp;
	while (!isEmpty()) {
		temp = head;
		head = head->next;
		delete temp;
	}
}

template <typename T>
void List<T>::insertList(T data, string name) {
	Node<T> *temp = new Node<T>(data, name);
	Node<T> *p, *q;
	//first node
	if (isEmpty()) {
		head = temp;
	}
	//smaller than head
	else if (temp->data < head->data) {
		temp->next = head;
		head->prev = temp;
		head = temp;
	}
	else {
		p = head, q = head;
		while ((p != 0) && (p->data < temp->data)) {
			q = p;
			p = p->next;
		}
		// insert between nodes
		if (p != 0) {
			temp->next = p;
			temp->prev = q;
			q->next = temp;
			p->prev = temp;
		}
		// insert in the end
		else {
			q->next = temp;
			temp->prev = q;
		}
	}
}

template <typename T>
void List<T>::deleteList(T v) {
	Node<T> *p, *q;

	if (isEmpty()) {
		cout << "list is empty!" << endl;
	}
	//value is in head node
	else if (head->data == v) {
		if (head->next != 0) {
			p = head;
			head = head->next;
			head->prev = 0;
			delete p;
		}
		//head node is a only node
		else {
			head = 0;
		}
	}
	// between nodes
	else {
		p = head, q = head;
		while (p != 0 && p->data != v) {
			q = p;
			p = p->next;
		}
		if (p != 0) {
			q->next = p->next;
			(p->next)->prev = q;
			delete p;
		}
		else
			cout << v << "is not in the list" << endl;
	}
}

template <typename T>
void List<T>::forwardList() {
	if (!isEmpty()) {
		Node<T> *temp = head;
		while (temp != 0) {
			cout << temp->data << temp->name << endl;
			temp = temp->next;
		}
	}
}

template <typename T>
void List<T>::backwardList() {
	if (!isEmpty()) {
		Node<T> *temp = head;
		while (temp->next != 0) {
			temp = temp->next;
		}
		while (temp != 0) {
			cout << temp->data << temp->name << endl;
			temp = temp->prev;
		}
	}
}

template <typename T>
void List<T>::searchList(T v) {
	Node<T> *temp = head;
	while (temp != 0 && temp->data != v) {
		temp = temp->next;
	}
	if (temp != 0) {
		cout << temp->data << "is in the list" << endl;
	}
	else
		cout << v << "is not in the list" << endl;
}

template <typename T>
void List<T>::displayList() {
	Node<T> *temp = head;
	while (temp != 0) {
		cout << temp->data << ' ' << temp->name << endl;
		temp = temp->next;
	}
}

template <typename T>
bool List<T>::isEmpty() {
	return (head == 0);
}

