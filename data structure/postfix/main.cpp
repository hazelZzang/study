#include "stack.h"
#include <iostream>
#include <string>
using namespace std;

int token(char t) {
	switch (t) {

	case ')':
		return 3;
		break;

	case '*': case '/': case '%':
		return 2;
		break;

	case '+': case '-':
		return 1;
		break;

	case '(':
		return 0;
		break;
	}
	return 0;
}


string postfix(Stack& s) {
	Stack temp;
	string result;

	int count = 0;
	while (!s.isEmpty()) {
		char input = s.top();

		if (isdigit(input)) { // operand
			result += input;
		}
		else if (input == '(') {	//left paren
			temp.push(input);
		}
		else if (input == ')') {	//right paren
			while (temp.top() != '(') {
				result += temp.pop();
			}
			temp.pop(); //pop '('
		}
		else {	//operator
			if (token(input) > token(temp.top())) {	//input priority is bigger
				temp.push(input);
			}
			else {
				while ((!temp.isEmpty()) && token(input) <= token(temp.top())) {	//stack top priority is bigger
					result += temp.pop();
				}
				temp.push(input);
			}
		}
		s.pop();
	}
	while (!temp.isEmpty()) {	//left in stack
		result += temp.pop();
	}
	return result;
}

char eval(Stack& s) {
	int op1, op2;
	int n;
	int top = -1;
	char token, result;
	Stack temp;
	while (token = s.pop()) {
		if (isdigit(token)) {
			temp.push(token);
		}
		else {
			op2 = temp.pop() - '0';
			op1 = temp.pop() - '0';
			switch (token) {
			case '+':
				result = (op1 + op2) + '0';
				temp.push(result);
				break;
			case '-':
				result = (op1 - op2) + '0';
				temp.push(result);
				break;
			case '*':
				result = (op1 * op2) + '0';
				temp.push(result);
				break;
			case '/':
				result = (op1 / op2) + '0';
				temp.push(result);
				break;
			}
		}
	}
	return temp.pop();
}

int main() {
	string test,pf;
	cout << "Enter Data:";
	cin >> test;
	Stack i(20), p(20);
	i.pushAll(test);

	cout << "Echo Data:" << test << endl;

	//infix to postfix
	pf = postfix(i);
	cout << "Conversion:" << pf << endl;

	p.pushAll(pf);
	
	//value
	cout << "Result:" << int(eval(p) - '0')<<endl;
	
	return 0;
}
