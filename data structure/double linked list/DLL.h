#include <iostream>
#include <string>
using namespace std;

template <typename T>
class Node {
	T data;
	string name;
	Node<T> *prev;
	Node<T> *next;

	template<typename T>friend class List;
public:
	template <typename T>
	Node(T d, string c) {
		data = d;
		name = c;
	}
};

template <typename T>
class List {
private:
	Node<T> *head;

public:
	List();
	~List();
	void insertList(T, string);
	void deleteList(T);
	void forwardList();
	void backwardList();
	void searchList(T);
	void displayList();
	bool isEmpty();
};
