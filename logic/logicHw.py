########
#카르노맵의 구현 과제
#입력 : 카르노맵의 차수, sum-of-product 각 항을 나타내는 숫자
#출력 : 최적화된 식
#########

#입력
input_data = input()
n = input_data.split(' ')

#카르노맵의 차수
degree = int(n[0])

#이진수 문자열
bin_n = [bin(int(i))[2:].zfill(degree) for i in n[1:]]
result = n[1:]

last_num = 0
count = 0
#차수만큼 반복한다.
for i in range(degree):
    before_len = len(bin_n)

    #비교하지 않은 문자만 비교한다
    for num in range(last_num, before_len):
        for next_num in range(num + 1,before_len):
            bit_dif = 0
            same_str = ''
            same_num = []
            for str in range(degree):
                #각 자리수를 비교에 비트차이가 1이 나는지 확인한다.
                if(bin_n[num][str] != bin_n[next_num][str]):
                    bit_dif += 1
                    if bit_dif > 1: break
                    #비트 차이가 1이 나는 문자만 *으로 교체해준다.
                    #항을 줄이는 연산이다. 101, 100 -> 10*
                    same_str = bin_n[num][0:str] + '*' + bin_n[num][str+1:]
                    same_num = [result[num],result[next_num]]

            if (bit_dif == 0):
                same_str = bin_n[num]
                same_num = [result[num] , result[next_num]]

            #비트 차이가 1일 경우에 리스트에 맨 끝에 추가한다.
            if(bit_dif <= 1):
                bin_n.append(same_str)
                result.append(same_num)


    #리스트 맨 끝에 추가한 줄여진 항에 대하여
    #이전에 중복된 항을 삭제한다.
    #101,100,10* 의 경우 101과 100을 삭제한다.
    for num in range(last_num, before_len):
        if(before_len > len(bin_n)): break
        for append_num in range(before_len, len(bin_n)):
            if result[num] in result[append_num]:
                result[num] = 0
                count += 1
                break
    last_num = before_len

#최적화된 식을 출력한다.
char = ['a','b','c','d']
char_n = ['a\'', 'b\'','c\'','d\'']
count = len(bin_n) - count
for i in range(len(bin_n)):
    if result[i] != 0:
        for d in range(degree):
            if(bin_n[i][d] == '1'): print(char[d],end='')
            elif(bin_n[i][d] == '0'): print(char_n[d],end='')
            else: pass
        count -= 1
        if count != 0:
            print('+',end='')